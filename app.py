from flask import Flask, request, send_file, render_template, jsonify
from PIL import Image
import io
import os
import zipfile
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# Quality 85 + subsampling 0 (4:4:4) = best sharpness at smallest size
# subsampling=0 preserves colour detail (important for product photos)
QUALITY = 85
SUBSAMPLING = 0  # 4:4:4 — no colour downsampling

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def autocrop_transparent(img):
    arr = np.array(img)
    alpha = arr[:, :, 3]
    rows = np.any(alpha > 10, axis=1)
    cols = np.any(alpha > 10, axis=0)
    if not rows.any():
        return img
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    padding = 20
    h, w = arr.shape[:2]
    rmin = max(0, rmin - padding)
    rmax = min(h - 1, rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(w - 1, cmax + padding)
    return img.crop((cmin, rmin, cmax + 1, rmax + 1))

def autocrop_white(img, threshold=240):
    arr = np.array(img)
    mask = np.any(arr < threshold, axis=2)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return img
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    padding = 20
    h, w = arr.shape[:2]
    rmin = max(0, rmin - padding)
    rmax = min(h - 1, rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(w - 1, cmax + padding)
    return img.crop((cmin, rmin, cmax + 1, rmax + 1))

def fill_interior_gaps(rgba_img, bg_rgb):
    from collections import deque
    arr = np.array(rgba_img).copy()
    h, w = arr.shape[:2]
    alpha = arr[:, :, 3]

    visited = np.zeros((h, w), dtype=bool)
    queue = deque()

    for y in range(h):
        for x in [0, w - 1]:
            if alpha[y, x] < 128 and not visited[y, x]:
                visited[y, x] = True
                queue.append((y, x))
    for x in range(w):
        for y in [0, h - 1]:
            if alpha[y, x] < 128 and not visited[y, x]:
                visited[y, x] = True
                queue.append((y, x))

    while queue:
        cy, cx = queue.popleft()
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and alpha[ny, nx] < 128:
                visited[ny, nx] = True
                queue.append((ny, nx))

    interior = (alpha < 128) & (~visited)
    arr[interior, 0] = bg_rgb[0]
    arr[interior, 1] = bg_rgb[1]
    arr[interior, 2] = bg_rgb[2]
    arr[interior, 3] = 255

    return Image.fromarray(arr, 'RGBA')

def fit_and_place(img_rgb, target_w, target_h, bg_rgb):
    w, h = img_rgb.size
    scale_h = target_h / h
    new_w_by_h = int(w * scale_h)

    if new_w_by_h <= target_w:
        new_w, new_h = new_w_by_h, target_h
    else:
        scale_w = target_w / w
        new_w, new_h = target_w, int(h * scale_w)

    img_rgb = img_rgb.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), bg_rgb)
    left = (target_w - new_w) // 2
    top = (target_h - new_h) // 2
    canvas.paste(img_rgb, (left, top))
    return canvas

def save_optimised(img_rgb):
    """Save as JPEG with max optimisation — strips all metadata."""
    out = io.BytesIO()
    img_rgb.save(
        out,
        format="JPEG",
        quality=QUALITY,
        subsampling=SUBSAMPLING,   # 4:4:4 — preserves colour sharpness
        optimize=True,             # extra Huffman optimisation pass
        progressive=True,          # progressive scan — renders top-down on slow connections
        # No exif= param = EXIF is stripped by default in Pillow when not passed
    )
    out.seek(0)
    return out

def process_image(img_bytes, target_w, target_h, bg_color_hex, remove_bg):
    bg_rgb = hex_to_rgb(bg_color_hex)

    # Open without passing through any existing metadata
    img = Image.open(io.BytesIO(img_bytes))
    img = img.convert("RGBA")  # discard all EXIF/metadata in conversion

    if remove_bg:
        try:
            from rembg import remove
            img = remove(img)
        except Exception as e:
            print(f"rembg failed: {e}")

        img = fill_interior_gaps(img, bg_rgb)
        img = autocrop_transparent(img)

        bg_layer = Image.new("RGBA", img.size, bg_rgb + (255,))
        bg_layer.paste(img, mask=img.split()[3])
        img_rgb = bg_layer.convert("RGB")
    else:
        img_rgb = img.convert("RGB")
        img_rgb = autocrop_white(img_rgb)

    canvas = fit_and_place(img_rgb, target_w, target_h, bg_rgb)
    return save_optimised(canvas)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

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
