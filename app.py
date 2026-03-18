from flask import Flask, request, send_file, render_template, jsonify
from PIL import Image
import io
import os
import zipfile
import numpy as np
from collections import deque

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

QUALITY = 85
SUBSAMPLING = 0

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def sample_bg_color(arr, sample_size=40):
    """Sample the original background colour from image corners."""
    h, w = arr.shape[:2]
    corners = [
        arr[0:sample_size, 0:sample_size],
        arr[0:sample_size, w-sample_size:w],
        arr[h-sample_size:h, 0:sample_size],
        arr[h-sample_size:h, w-sample_size:w],
    ]
    all_pixels = np.vstack([c.reshape(-1, c.shape[-1]) for c in corners])
    return tuple(all_pixels.mean(axis=0).astype(int)[:3])

def fill_interior_transparent(rgba_arr, new_bg_rgb):
    """BFS from edges — fill transparent interior gaps (between boot & frame)."""
    h, w = rgba_arr.shape[:2]
    alpha = rgba_arr[:, :, 3]
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
            ny, nx = cy+dy, cx+dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and alpha[ny, nx] < 128:
                visited[ny, nx] = True
                queue.append((ny, nx))

    result = rgba_arr.copy()
    interior = (alpha < 128) & (~visited)
    result[interior, 0] = new_bg_rgb[0]
    result[interior, 1] = new_bg_rgb[1]
    result[interior, 2] = new_bg_rgb[2]
    result[interior, 3] = 255
    return result

def fill_interior_bg_colour(rgba_arr, orig_bg_rgb, new_bg_rgb, tolerance=40):
    """Find opaque pixels that still match the original bg colour (interior remnants
    like the gap between boot and skate frame) and replace them with the new bg colour.
    Uses BFS to exclude edge-connected bg pixels (already handled by rembg)."""
    h, w = rgba_arr.shape[:2]
    rgb = rgba_arr[:, :, :3]
    alpha = rgba_arr[:, :, 3]

    # Pixels that are opaque but look like the original bg colour
    diff = np.abs(rgb.astype(int) - np.array(orig_bg_rgb)).max(axis=2)
    bg_like = (diff < tolerance) & (alpha > 128)

    # BFS from edges to find outer/edge-connected bg remnants
    visited = np.zeros((h, w), dtype=bool)
    queue = deque()
    for y in range(h):
        for x in [0, w-1]:
            if bg_like[y, x] and not visited[y, x]:
                visited[y, x] = True
                queue.append((y, x))
    for x in range(w):
        for y in [0, h-1]:
            if bg_like[y, x] and not visited[y, x]:
                visited[y, x] = True
                queue.append((y, x))
    while queue:
        cy, cx = queue.popleft()
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = cy+dy, cx+dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and bg_like[ny, nx]:
                visited[ny, nx] = True
                queue.append((ny, nx))

    # Interior bg pixels = bg_like but NOT reachable from edges → replace
    interior_bg = bg_like & ~visited
    result = rgba_arr.copy()
    result[interior_bg, 0] = new_bg_rgb[0]
    result[interior_bg, 1] = new_bg_rgb[1]
    result[interior_bg, 2] = new_bg_rgb[2]
    result[interior_bg, 3] = 255
    return result

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
    return img.crop((max(0, cmin-padding), max(0, rmin-padding),
                     min(w, cmax+padding+1), min(h, rmax+padding+1)))

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
    return img.crop((max(0, cmin-padding), max(0, rmin-padding),
                     min(w, cmax+padding+1), min(h, rmax+padding+1)))

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
    canvas.paste(img_rgb, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return canvas

def save_optimised(img_rgb):
    out = io.BytesIO()
    img_rgb.save(out, format="JPEG", quality=QUALITY, subsampling=SUBSAMPLING,
                 optimize=True, progressive=True)
    out.seek(0)
    return out

def process_image(img_bytes, target_w, target_h, bg_color_hex, remove_bg):
    bg_rgb = hex_to_rgb(bg_color_hex)
    img = Image.open(io.BytesIO(img_bytes))

    # Sample original bg colour BEFORE any processing
    orig_arr = np.array(img.convert("RGB"))
    orig_bg_rgb = sample_bg_color(orig_arr)

    img = img.convert("RGBA")

    if remove_bg:
        try:
            from rembg import remove
            img = remove(img)
        except Exception as e:
            print(f"rembg failed: {e}")

        arr = np.array(img)

        # Pass 1: Fill transparent interior gaps (BFS from edges on alpha)
        arr = fill_interior_transparent(arr, bg_rgb)

        # Pass 2: Fill opaque interior pixels that still match the original bg colour
        # (e.g. grey remnants between boot and frame that rembg left as opaque)
        arr = fill_interior_bg_colour(arr, orig_bg_rgb, bg_rgb, tolerance=40)

        img = Image.fromarray(arr, 'RGBA')
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
