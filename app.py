from flask import Flask, request, send_file, render_template, jsonify
from PIL import Image
import io
import os
import zipfile

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

QUALITY = 82

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def autocrop_white(img, threshold=240):
    pixels = img.load()
    w, h = img.size

    top = 0
    for y in range(h):
        if any(pixels[x, y][c] < threshold for x in range(0, w, 3) for c in range(3)):
            top = y
            break

    bottom = h
    for y in range(h - 1, -1, -1):
        if any(pixels[x, y][c] < threshold for x in range(0, w, 3) for c in range(3)):
            bottom = y + 1
            break

    left = 0
    for x in range(w):
        if any(pixels[x, y][c] < threshold for y in range(0, h, 3) for c in range(3)):
            left = x
            break

    right = w
    for x in range(w - 1, -1, -1):
        if any(pixels[x, y][c] < threshold for y in range(0, h, 3) for c in range(3)):
            right = x + 1
            break

    padding = 20
    top = max(0, top - padding)
    bottom = min(h, bottom + padding)
    left = max(0, left - padding)
    right = min(w, right + padding)

    return img.crop((left, top, right, bottom))

def process_image(img_bytes, target_w, target_h, bg_color_hex, remove_bg):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    bg_rgb = hex_to_rgb(bg_color_hex)

    if remove_bg:
        try:
            from rembg import remove
            img = remove(img)
        except Exception as e:
            print(f"rembg failed: {e}")
            img = img.convert("RGBA")

        # Paste onto background colour
        canvas_rgba = Image.new("RGBA", img.size, bg_rgb + (255,))
        canvas_rgba.paste(img, mask=img.split()[3])
        img = canvas_rgba.convert("RGB")
    else:
        img = img.convert("RGB")
        img = autocrop_white(img)

    w, h = img.size

    # Fill top/bottom by default; fall back to fill sides if overflow
    scale_h = target_h / h
    new_w_by_h = int(w * scale_h)

    if new_w_by_h <= target_w:
        new_w, new_h = new_w_by_h, target_h
    else:
        scale_w = target_w / w
        new_w, new_h = target_w, int(h * scale_w)

    img = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), bg_rgb)
    left = (target_w - new_w) // 2
    top = (target_h - new_h) // 2
    canvas.paste(img, (left, top))

    out = io.BytesIO()
    canvas.save(out, "JPEG", quality=QUALITY, optimize=True, progressive=True)
    out.seek(0)
    return out

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    # Canvas size
    try:
        target_w = int(request.form.get("canvas_w", 800))
        target_h = int(request.form.get("canvas_h", 800))
        target_w = max(100, min(5000, target_w))
        target_h = max(100, min(5000, target_h))
    except:
        target_w, target_h = 800, 800

    bg_color = request.form.get("bg_color", "#ffffff")
    remove_bg = request.form.get("remove_bg", "false").lower() == "true"

    if len(files) == 1:
        f = files[0]
        result = process_image(f.read(), target_w, target_h, bg_color, remove_bg)
        name = os.path.splitext(f.filename)[0] + ".jpg"
        return send_file(result, mimetype="image/jpeg",
                         as_attachment=True, download_name=name)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            result = process_image(f.read(), target_w, target_h, bg_color, remove_bg)
            name = os.path.splitext(f.filename)[0] + ".jpg"
            zf.writestr(name, result.read())
    zip_buf.seek(0)
    return send_file(zip_buf, mimetype="application/zip",
                     as_attachment=True, download_name="resized_images.zip")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
