import os
import secrets
import json
from datetime import datetime, timedelta, timezone
from functools import wraps
from concurrent.futures import ThreadPoolExecutor

import stripe
import bcrypt
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import (Flask, request, send_file, render_template,
                   jsonify, session, redirect, url_for, make_response)
from PIL import Image
import io
import zipfile
import numpy as np
from collections import deque
import psycopg2
from psycopg2.extras import RealDictCursor

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY")
STRIPE_PRICE_ID = os.environ.get("STRIPE_PRICE_ID")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET")
APP_URL = os.environ.get("APP_URL", "http://localhost:5000")

GMAIL_USER = os.environ.get("GMAIL_USER")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD")

QUALITY = 85
SUBSAMPLING = 0

# Load rembg sessions once at startup — dual model support
REMBG_SESSION_QUALITY = None  # BiRefNet-lite — good quality, faster (~8-10s)
REMBG_SESSION_FAST = None     # u2net — good quality, fast (~3s), max 10/upload
BG_LIMITS = {"birefnet": 5, "u2net": 10, "none": 20}
REMBG_MAX_DIM = 1024  # Downscale images before rembg for speed

def get_rembg_session(model="birefnet"):
    global REMBG_SESSION_QUALITY, REMBG_SESSION_FAST
    if model == "birefnet":
        if REMBG_SESSION_QUALITY is None:
            try:
                from rembg import new_session
                REMBG_SESSION_QUALITY = new_session("birefnet-general-lite")
                print("BiRefNet-lite session loaded")
            except Exception as e:
                print(f"BiRefNet-lite session failed: {e}")
                # Fallback to full model
                try:
                    from rembg import new_session
                    REMBG_SESSION_QUALITY = new_session("birefnet-general")
                    print("BiRefNet full session loaded (fallback)")
                except Exception as e2:
                    print(f"BiRefNet fallback also failed: {e2}")
        return REMBG_SESSION_QUALITY
    else:
        if REMBG_SESSION_FAST is None:
            try:
                from rembg import new_session
                REMBG_SESSION_FAST = new_session("u2net")
                print("u2net session loaded")
            except Exception as e:
                print(f"u2net session failed: {e}")
        return REMBG_SESSION_FAST
TRIAL_DAYS = 7
ANON_FREE_IMAGES = 20

# ── Database ──────────────────────────────────────────────────────────────────

def get_db():
    return psycopg2.connect(os.environ["DATABASE_URL"], cursor_factory=RealDictCursor)

def init_db():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                ALTER TABLE IF EXISTS users ADD COLUMN IF NOT EXISTS trial_images_used INTEGER DEFAULT 0;
                ALTER TABLE IF EXISTS users ADD COLUMN IF NOT EXISTS is_admin BOOLEAN DEFAULT FALSE;
                ALTER TABLE IF EXISTS users ADD COLUMN IF NOT EXISTS email_verified BOOLEAN DEFAULT FALSE;
                CREATE TABLE IF NOT EXISTS email_tokens (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    token TEXT UNIQUE NOT NULL,
                    token_type TEXT NOT NULL,
                    expires_at TIMESTAMPTZ NOT NULL,
                    used BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS anon_usage (
                    id SERIAL PRIMARY KEY,
                    ip TEXT NOT NULL,
                    image_count INTEGER DEFAULT 0,
                    first_seen TIMESTAMPTZ DEFAULT NOW(),
                    last_seen TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE UNIQUE INDEX IF NOT EXISTS anon_usage_ip_idx ON anon_usage(ip);
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    stripe_customer_id TEXT,
                    subscription_status TEXT DEFAULT 'trial',
                    trial_ends_at TIMESTAMPTZ,
                    subscription_ends_at TIMESTAMPTZ,
                    trial_images_used INTEGER DEFAULT 0,
                    is_admin BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
        conn.commit()

def increment_user_trial_usage(user_id, count=1):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET trial_images_used = trial_images_used + %s WHERE id = %s RETURNING trial_images_used",
                (count, user_id)
            )
            result = cur.fetchone()
        conn.commit()
    return result['trial_images_used'] if result else 0

def set_user_trial_usage(user_id, count):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE users SET trial_images_used = %s WHERE id = %s", (count, user_id))
        conn.commit()

def get_user_by_id(user_id):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            return cur.fetchone()

def get_user_by_email(email):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE email = %s", (email.lower(),))
            return cur.fetchone()

def get_user_by_username(username):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE username = %s", (username.lower(),))
            return cur.fetchone()

def create_user(username, email, password):
    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    trial_ends = datetime.now(timezone.utc) + timedelta(days=TRIAL_DAYS)
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO users (username, email, password_hash, trial_ends_at, subscription_status)
                VALUES (%s, %s, %s, %s, 'trial') RETURNING *
            """, (username.lower(), email.lower(), pw_hash, trial_ends))
            result = cur.fetchone()
        conn.commit()
    return result

def check_password(user, password):
    return bcrypt.checkpw(password.encode(), user['password_hash'].encode())

def update_user_subscription(stripe_customer_id, status, ends_at=None):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE users SET subscription_status = %s, subscription_ends_at = %s
                WHERE stripe_customer_id = %s
            """, (status, ends_at, stripe_customer_id))
        conn.commit()

def is_active(user):
    if not user:
        return False
    if user.get('is_admin', False):
        return True
    now = datetime.now(timezone.utc)
    if user['subscription_status'] in ('active', 'cancelled'):
        ends = user['subscription_ends_at']
        return ends is None or ends > now
    if user['subscription_status'] == 'trial':
        ends = user['trial_ends_at']
        time_ok = ends is not None and ends > now
        images_used = user.get('trial_images_used', 0) or 0
        images_ok = images_used < ANON_FREE_IMAGES
        return time_ok and images_ok
    return False

def user_trial_images_left(user):
    used = user.get('trial_images_used', 0) or 0
    return max(0, ANON_FREE_IMAGES - used)

def trial_days_left(user):
    if user['subscription_status'] != 'trial':
        return 0
    ends = user['trial_ends_at']
    if not ends:
        return 0
    return max(0, (ends - datetime.now(timezone.utc)).days)

# ── IP tracking ───────────────────────────────────────────────────────────────

def get_real_ip():
    forwarded = request.headers.get('X-Forwarded-For', '')
    if forwarded:
        return forwarded.split(',')[0].strip()
    return request.remote_addr or '0.0.0.0'

def get_anon_usage(ip):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM anon_usage WHERE ip = %s", (ip,))
            return cur.fetchone()

def increment_anon_usage(ip, count):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO anon_usage (ip, image_count, last_seen)
                VALUES (%s, %s, NOW())
                ON CONFLICT (ip) DO UPDATE
                SET image_count = anon_usage.image_count + %s,
                    last_seen = NOW()
                RETURNING image_count
            """, (ip, count, count))
            result = cur.fetchone()
        conn.commit()
    return result['image_count'] if result else count

def anon_images_remaining(ip):
    usage = get_anon_usage(ip)
    used = usage['image_count'] if usage else 0
    return max(0, ANON_FREE_IMAGES - used)

# ── Auth helpers ──────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated

# ── Image processing ──────────────────────────────────────────────────────────

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def fill_interior_transparent(rgba_arr, new_bg_rgb):
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

def fit_and_place(img_rgb, target_w, target_h, bg_rgb, fill_pct=0.95):
    eff_w = int(target_w * fill_pct)
    eff_h = int(target_h * fill_pct)
    w, h = img_rgb.size
    scale_h = eff_h / h
    new_w_by_h = int(w * scale_h)
    if new_w_by_h <= eff_w:
        new_w, new_h = new_w_by_h, eff_h
    else:
        scale_w = eff_w / w
        new_w, new_h = eff_w, int(h * scale_w)
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

def downscale_for_rembg(img, max_dim=REMBG_MAX_DIM):
    """Downscale image before rembg to speed up inference."""
    w, h = img.size
    if max(w, h) <= max_dim:
        return img, 1.0
    scale = max_dim / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS), scale

def process_image(img_bytes, target_w, target_h, bg_color_hex, remove_bg, fill_pct=0.95, bg_model='birefnet'):
    bg_rgb = hex_to_rgb(bg_color_hex)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    if remove_bg:
        try:
            from rembg import remove
            session_rembg = get_rembg_session(model=bg_model)
            if session_rembg:
                # Downscale for faster inference, then upscale mask
                small_img, scale = downscale_for_rembg(img)
                small_result = remove(small_img, session=session_rembg)
                if scale < 1.0:
                    # Extract alpha from small result, upscale it, apply to original
                    small_alpha = small_result.split()[3]
                    full_alpha = small_alpha.resize(img.size, Image.LANCZOS)
                    img.putalpha(full_alpha)
                else:
                    img = small_result
        except Exception as e:
            print(f"rembg failed: {e}")
        arr = np.array(img)
        arr = fill_interior_transparent(arr, bg_rgb)
        img = Image.fromarray(arr, 'RGBA')
        img = autocrop_transparent(img)
        bg_layer = Image.new("RGBA", img.size, bg_rgb + (255,))
        bg_layer.paste(img, mask=img.split()[3])
        img_rgb = bg_layer.convert("RGB")
    else:
        img_rgb = img.convert("RGB")
        img_rgb = autocrop_white(img_rgb)
    canvas = fit_and_place(img_rgb, target_w, target_h, bg_rgb, fill_pct)
    return save_optimised(canvas)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    # Logged in user
    if 'user_id' in session:
        user = get_user_by_id(session['user_id'])
        if not user:
            session.clear()
            return redirect(url_for('login_page'))
        if not is_active(user):
            return redirect(url_for('subscribe_page'))
        is_admin = user.get('is_admin', False)
        is_trial = user['subscription_status'] == 'trial' and not is_admin
        is_paid = user['subscription_status'] in ('active', 'cancelled') and not is_admin
        images_remaining = user_trial_images_left(user) if is_trial else None
        return render_template("index.html", user=user,
                               images_remaining=images_remaining,
                               is_trial=is_trial,
                               is_admin=is_admin)

    # Anonymous — check IP + cookie
    ip = get_real_ip()
    remaining_ip = anon_images_remaining(ip)
    cookie_used = int(request.cookies.get('anon_used', 0))
    remaining_cookie = max(0, ANON_FREE_IMAGES - cookie_used)
    remaining = min(remaining_ip, remaining_cookie)

    if remaining == 0:
        return redirect(url_for('login_page') + '?expired=1')

    return render_template("index.html", user=None,
                           days_left=None,
                           images_remaining=remaining,
                           is_trial=True)

@app.route("/login")
def login_page():
    if 'user_id' in session:
        return redirect(url_for('index'))
    return render_template("login.html")

@app.route("/auth/login", methods=["POST"])
def do_login():
    data = request.get_json()
    identifier = (data.get("identifier") or "").strip().lower()
    password = data.get("password") or ""
    if not identifier or not password:
        return jsonify({"error": "Please fill in all fields"}), 400
    user = get_user_by_email(identifier) or get_user_by_username(identifier)
    if not user or not check_password(user, password):
        return jsonify({"error": "Incorrect username/email or password"}), 401
    session['user_id'] = user['id']
    session.permanent = True
    ip = get_real_ip()
    ip_used = get_anon_usage(ip)
    if ip_used and ip_used['image_count'] > 0:
        current = user.get('trial_images_used', 0) or 0
        merged = max(current, ip_used['image_count'])
        if merged > current:
            set_user_trial_usage(user['id'], merged)
        user = get_user_by_id(user['id'])
    return jsonify({"ok": True, "redirect": "/subscribe" if not is_active(user) else "/"})

@app.route("/auth/signup", methods=["POST"])
def do_signup():
    data = request.get_json()
    username = (data.get("username") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    if not username or not email or not password:
        return jsonify({"error": "Please fill in all fields"}), 400
    if len(username) < 3:
        return jsonify({"error": "Username must be at least 3 characters"}), 400
    if "@" not in email:
        return jsonify({"error": "Invalid email address"}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400
    if get_user_by_username(username):
        return jsonify({"error": "Username already taken"}), 409
    if get_user_by_email(email):
        return jsonify({"error": "An account with this email already exists"}), 409
    try:
        user = create_user(username, email, password)
    except Exception as e:
        return jsonify({"error": "Could not create account"}), 500
    session['user_id'] = user['id']
    session.permanent = True
    ip = get_real_ip()
    ip_used = get_anon_usage(ip)
    if ip_used and ip_used['image_count'] > 0:
        set_user_trial_usage(user['id'], ip_used['image_count'])
    send_verification_email(user)
    return jsonify({"ok": True, "redirect": "/"})

@app.route("/auth/forgot-password", methods=["POST"])
def forgot_password():
    data = request.get_json()
    email = (data.get("email") or "").strip().lower()
    if not email:
        return jsonify({"error": "Please enter your email"}), 400
    user = get_user_by_email(email)
    # Always return success to prevent email enumeration
    if user:
        send_password_reset_email(user)
    return jsonify({"ok": True})

@app.route("/auth/verify-email")
def verify_email():
    token = request.args.get("token", "")
    row = verify_email_token(token, "verify")
    if not row:
        return render_template("login.html", error="Verification link has expired or already been used.")
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE users SET email_verified = TRUE WHERE id = %s", (row["user_id"],))
        conn.commit()
    if "user_id" not in session:
        session["user_id"] = row["user_id"]
        session.permanent = True
    return redirect("/?verified=1")

@app.route("/auth/reset-password")
def reset_password_page():
    token = request.args.get("token", "")
    row = verify_email_token(token, "reset")
    if not row:
        return render_template("login.html", error="Password reset link has expired or already been used. Please request a new one.")
    return render_template("reset_password.html", token=token)

@app.route("/auth/reset-password", methods=["POST"])
def do_reset_password():
    data = request.get_json()
    token = data.get("token", "")
    password = data.get("password", "")
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400
    row = verify_email_token(token, "reset")
    if not row:
        return jsonify({"error": "Link has expired. Please request a new password reset."}), 400
    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE users SET password_hash = %s WHERE id = %s", (pw_hash, row["user_id"]))
        conn.commit()
    session["user_id"] = row["user_id"]
    session.permanent = True
    return jsonify({"ok": True, "redirect": "/"})

@app.route("/contact")
def contact_page():
    user = get_user_by_id(session['user_id']) if 'user_id' in session else None
    return render_template("contact.html", user=user)

@app.route("/contact", methods=["POST"])
def do_contact():
    data = request.get_json()
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip()
    message = (data.get("message") or "").strip()
    if not name or not email or not message:
        return jsonify({"error": "Please fill in all fields"}), 400
    if "@" not in email:
        return jsonify({"error": "Invalid email address"}), 400
    if len(message) < 10:
        return jsonify({"error": "Message is too short"}), 400
    html = f"""
    <div style="font-family:sans-serif;max-width:600px;margin:0 auto;padding:24px;">
      <h2 style="margin-bottom:16px;">New PixelPrep Support Request</h2>
      <table style="width:100%;border-collapse:collapse;">
        <tr><td style="padding:8px;color:#666;width:100px;">Name</td><td style="padding:8px;font-weight:700;">{name}</td></tr>
        <tr style="background:#f5f5f5;"><td style="padding:8px;color:#666;">Email</td><td style="padding:8px;"><a href="mailto:{email}">{email}</a></td></tr>
        <tr><td style="padding:8px;color:#666;">Message</td><td style="padding:8px;">{message}</td></tr>
      </table>
    </div>"""
    ok = send_email(GMAIL_USER, f"PixelPrep Support: {name}", html)
    # Also send confirmation to user
    confirm_html = f"""
    <div style="font-family:sans-serif;max-width:480px;margin:0 auto;padding:40px 24px;">
      <h2 style="font-size:22px;font-weight:700;margin-bottom:8px;">We got your message</h2>
      <p style="color:#666;margin-bottom:16px;">Hi {name}, thanks for reaching out. We'll get back to you at {email} within 1–2 business days.</p>
      <p style="color:#999;font-size:12px;">Your message: <em>"{message[:200]}{'...' if len(message) > 200 else ''}"</em></p>
    </div>"""
    send_email(email, "We received your PixelPrep support request", confirm_html)
    if not ok:
        return jsonify({"error": "Failed to send message. Please email us directly at support@inlinex.com.sg"}), 500
    return jsonify({"ok": True})

@app.route("/mobile-save")
def mobile_save():
    return """<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Save Image — PixelPrep</title>
  <style>
    * { margin:0; padding:0; box-sizing:border-box; }
    body { background:#111; min-height:100vh; display:flex; flex-direction:column; align-items:center; font-family:-apple-system,sans-serif; }
    .tip { background:#1a1a1a; width:100%; padding:20px; text-align:center; }
    .tip-title { color:#f0a500; font-size:17px; font-weight:700; margin-bottom:6px; }
    .tip-body { color:#ccc; font-size:14px; line-height:1.6; }
    .img-wrap { flex:1; display:flex; align-items:center; justify-content:center; padding:16px; width:100%; }
    #preview { max-width:100%; max-height:70vh; border-radius:8px; object-fit:contain; }
    .back { color:#555; font-size:13px; padding:20px; text-align:center; text-decoration:none; }
    .loading { color:#555; font-size:14px; padding:40px; }
  </style>
</head>
<body>
  <div class="tip">
    <div class="tip-title">Hold down on the image below</div>
    <div class="tip-body">Tap "Save to Photos" or "Save Image" to add it to your camera roll</div>
  </div>
  <div class="img-wrap">
    <img id="preview" src="" alt="Your processed image"/>
    <div class="loading" id="loading">Loading image...</div>
  </div>
  <a class="back" href="/">← Back to PixelPrep</a>
  <script>
    const data = sessionStorage.getItem('pixelprep_img');
    if (data) {
      const img = document.getElementById('preview');
      img.src = data;
      img.style.display = 'block';
      document.getElementById('loading').style.display = 'none';
      sessionStorage.removeItem('pixelprep_img');
    } else {
      document.getElementById('loading').textContent = 'No image found. Please go back and process an image first.';
    }
  </script>
</body>
</html>"""

@app.route("/view-image")
def view_image():
    """Mobile image viewer — shows image full screen for long-press save to Photos"""
    img_url = request.args.get("url", "")
    filename = request.args.get("name", "image.jpg")
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no"/>
  <title>Save Image — PixelPrep</title>
  <style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{ background:#000; min-height:100vh; display:flex; flex-direction:column; align-items:center; justify-content:center; font-family:sans-serif; }}
    .tip {{ color:#fff; font-size:14px; text-align:center; padding:16px; line-height:1.6; }}
    .tip strong {{ color:#f90; display:block; font-size:16px; margin-bottom:6px; }}
    img {{ max-width:100%; max-height:80vh; object-fit:contain; display:block; }}
    .back {{ color:#888; font-size:13px; text-align:center; padding:16px; text-decoration:none; display:block; }}
  </style>
</head>
<body>
  <div class="tip"><strong>Hold down on the image below</strong>Tap "Save to Photos" or "Add to Photos"</div>
  <img src="{img_url}" alt="{filename}"/>
  <a class="back" href="/">← Back to PixelPrep</a>
</body>
</html>"""

@app.route("/google7fd4b51598ab19d5.html")
def google_verify():
    return "google-site-verification: google7fd4b51598ab19d5.html", 200, {"Content-Type": "text/html"}

@app.route("/sitemap.xml")
def sitemap():
    return app.send_static_file("sitemap.xml"), 200, {"Content-Type": "application/xml"}

@app.route("/robots.txt")
def robots():
    return app.send_static_file("robots.txt"), 200, {"Content-Type": "text/plain"}

@app.route("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html")

@app.route("/terms")
def terms():
    return render_template("terms.html")

@app.route("/privacy")
def privacy():
    return render_template("privacy.html")

@app.route("/subscribe")
@login_required
def subscribe_page():
    user = get_user_by_id(session['user_id'])
    # Active subscribers and admins should never see this page
    if user and is_active(user) and user.get('subscription_status') in ('active',):
        return redirect(url_for('index'))
    return render_template("subscribe.html", user=user)

@app.route("/subscribe/checkout", methods=["POST"])
@login_required
def create_checkout():
    user = get_user_by_id(session['user_id'])
    if not user['stripe_customer_id']:
        customer = stripe.Customer.create(email=user['email'], metadata={"username": user['username']})
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE users SET stripe_customer_id = %s WHERE id = %s",
                            (customer.id, user['id']))
            conn.commit()
        customer_id = customer.id
    else:
        customer_id = user['stripe_customer_id']
    checkout = stripe.checkout.Session.create(
        customer=customer_id,
        payment_method_types=["card"],
        line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
        mode="subscription",
        success_url=f"{APP_URL}/subscribe/success",
        cancel_url=f"{APP_URL}/subscribe",
        currency="sgd",
    )
    return jsonify({"url": checkout.url})

@app.route("/subscribe/success")
@login_required
def subscribe_success():
    return render_template("success.html")

@app.route("/subscribe/portal", methods=["POST"])
@login_required
def billing_portal():
    user = get_user_by_id(session['user_id'])
    if not user['stripe_customer_id']:
        return jsonify({"error": "No billing account found"}), 400
    portal = stripe.billing_portal.Session.create(
        customer=user['stripe_customer_id'],
        return_url=f"{APP_URL}/",
    )
    return jsonify({"url": portal.url})

# ── Email ────────────────────────────────────────────────────────────────────

def send_email(to, subject, html):
    if not GMAIL_USER or not GMAIL_APP_PASSWORD:
        print(f"Email not configured. Would send to {to}: {subject}")
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"PixelPrep <{GMAIL_USER}>"
        msg["To"] = to
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_USER, to, msg.as_string())
        return True
    except Exception as e:
        print(f"Email error: {e}")
        return False

def create_email_token(user_id, token_type, expiry_hours=24):
    token = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(hours=expiry_hours)
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE email_tokens SET used = TRUE WHERE user_id = %s AND token_type = %s AND used = FALSE",
                (user_id, token_type)
            )
            cur.execute(
                "INSERT INTO email_tokens (user_id, token, token_type, expires_at) VALUES (%s, %s, %s, %s)",
                (user_id, token, token_type, expires)
            )
        conn.commit()
    return token

def verify_email_token(token, token_type):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM email_tokens WHERE token = %s AND token_type = %s AND used = FALSE AND expires_at > NOW()",
                (token, token_type)
            )
            row = cur.fetchone()
            if row:
                cur.execute("UPDATE email_tokens SET used = TRUE WHERE id = %s", (row["id"],))
            conn.commit()
    return row

def send_verification_email(user):
    token = create_email_token(user["id"], "verify", expiry_hours=48)
    link = f"{APP_URL}/auth/verify-email?token={token}"
    html = f"""
    <div style="font-family:sans-serif;max-width:480px;margin:0 auto;padding:40px 24px;">
      <h2 style="font-size:22px;font-weight:700;margin-bottom:8px;">Verify your email</h2>
      <p style="color:#666;margin-bottom:24px;">Hi {user["username"]}, click below to verify your PixelPrep account. This link expires in 48 hours.</p>
      <a href="{link}" style="display:inline-block;background:#1a1714;color:#fff;text-decoration:none;padding:14px 28px;border-radius:8px;font-weight:700;font-size:14px;">Verify email address</a>
      <p style="color:#999;font-size:12px;margin-top:24px;">If you didn't create a PixelPrep account, you can safely ignore this email.</p>
    </div>"""
    return send_email(user["email"], "Verify your PixelPrep email address", html)

def send_password_reset_email(user):
    token = create_email_token(user["id"], "reset", expiry_hours=1)
    link = f"{APP_URL}/auth/reset-password?token={token}"
    html = f"""
    <div style="font-family:sans-serif;max-width:480px;margin:0 auto;padding:40px 24px;">
      <h2 style="font-size:22px;font-weight:700;margin-bottom:8px;">Reset your password</h2>
      <p style="color:#666;margin-bottom:24px;">Hi {user["username"]}, click below to reset your password. This link expires in 1 hour.</p>
      <a href="{link}" style="display:inline-block;background:#1a1714;color:#fff;text-decoration:none;padding:14px 28px;border-radius:8px;font-weight:700;font-size:14px;">Reset password</a>
      <p style="color:#999;font-size:12px;margin-top:24px;">If you didn't request this, you can safely ignore this email. Your password won't change.</p>
    </div>"""
    return send_email(user["email"], "Reset your PixelPrep password", html)

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login_page'))
        user = get_user_by_id(session['user_id'])
        if not user or not user.get('is_admin'):
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated

@app.route("/admin")
@admin_required
def admin_page():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, username, email, subscription_status, trial_images_used, "
                "trial_ends_at, subscription_ends_at, is_admin, created_at "
                "FROM users ORDER BY created_at DESC"
            )
            users = cur.fetchall()
            cur.execute("SELECT COUNT(*) as c FROM users")
            total = cur.fetchone()['c']
            cur.execute("SELECT COUNT(*) as c FROM users WHERE subscription_status = 'active'")
            active = cur.fetchone()['c']
            cur.execute("SELECT COUNT(*) as c FROM users WHERE subscription_status = 'trial'")
            trial = cur.fetchone()['c']
            cur.execute("SELECT COUNT(*) as c FROM users WHERE subscription_status = 'cancelled'")
            cancelled = cur.fetchone()['c']
    return render_template("admin.html", users=users, total=total,
                           active=active, trial=trial, cancelled=cancelled)

@app.route("/webhook/stripe", methods=["POST"])
def stripe_webhook():
    payload = request.get_data()
    sig = request.headers.get("Stripe-Signature")
    try:
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        return str(e), 400
    if event['type'] in ('customer.subscription.created', 'customer.subscription.updated'):
        sub = event['data']['object']
        ends_at = datetime.fromtimestamp(sub['current_period_end'], tz=timezone.utc)
        status = 'active' if sub['status'] == 'active' else sub['status']
        update_user_subscription(sub['customer'], status, ends_at)
    elif event['type'] == 'customer.subscription.deleted':
        sub = event['data']['object']
        ends_at = datetime.fromtimestamp(sub['current_period_end'], tz=timezone.utc)
        update_user_subscription(sub['customer'], 'cancelled', ends_at)
    return jsonify({"ok": True})

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('login_page'))

@app.route("/process", methods=["POST"])
def process():
    is_anon = False
    cookie_used = 0

    # Get files first so we can use len(files) everywhere
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    if 'user_id' not in session:
        # Anonymous — check IP + cookie
        ip = get_real_ip()
        remaining_ip = anon_images_remaining(ip)
        cookie_used = int(request.cookies.get('anon_used', 0))
        remaining_cookie = max(0, ANON_FREE_IMAGES - cookie_used)
        remaining = min(remaining_ip, remaining_cookie)
        if remaining == 0:
            return jsonify({"error": "trial_expired"}), 402
        is_anon = True
        files = files[:remaining]
    else:
        user = get_user_by_id(session['user_id'])
        if not user:
            return jsonify({"error": "login_required"}), 401
        if not is_active(user):
            return jsonify({"error": "subscription_required"}), 402
        if user['subscription_status'] == 'trial':
            left = user_trial_images_left(user)
            if left <= 0:
                return jsonify({"error": "subscription_required"}), 402
            if len(files) > left:
                files = files[:left]

    try:
        target_w = int(request.form.get("canvas_w", 800))
        target_h = int(request.form.get("canvas_h", 800))
        target_w = max(100, min(5000, target_w))
        target_h = max(100, min(5000, target_h))
    except:
        target_w, target_h = 800, 800

    bg_color = request.form.get("bg_color", "#ffffff")
    remove_bg = request.form.get("remove_bg", "false").lower() == "true"
    bg_model = request.form.get("bg_model", "birefnet") if remove_bg else "none"
    if bg_model not in ("birefnet", "u2net"):
        bg_model = "birefnet"
    try:
        fill_pct = max(10, min(100, int(request.form.get("fill_pct", 95)))) / 100.0
    except:
        fill_pct = 0.95

    # Enforce per-upload image limits based on AI mode (admins are unlimited)
    _user_for_limit = get_user_by_id(session['user_id']) if 'user_id' in session else None
    _is_admin = _user_for_limit.get('is_admin', False) if _user_for_limit else False
    if not _is_admin:
        upload_limit = BG_LIMITS[bg_model]
        if len(files) > upload_limit:
            files = files[:upload_limit]

    image_count = len(files)

    # Read all file data upfront (can't read Flask file objects in threads)
    file_data = [(f.read(), os.path.splitext(f.filename)[0] + ".jpg") for f in files]

    def _process(args):
        data, name = args
        return name, process_image(data, target_w, target_h, bg_color, remove_bg, fill_pct, bg_model)

    if image_count == 1:
        name, result = _process(file_data[0])
        response = send_file(result, mimetype="image/jpeg", as_attachment=True, download_name=name)
    else:
        # Process images in parallel (2 workers to avoid OOM on Railway)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(_process, file_data))
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, result in results:
                zf.writestr(name, result.read())
        zip_buf.seek(0)
        response = send_file(zip_buf, mimetype="application/zip", as_attachment=True, download_name="resized_images.zip")

    # Track usage for anonymous users
    if is_anon:
        increment_anon_usage(ip, image_count)
        new_used = cookie_used + image_count
        remaining_after = max(0, ANON_FREE_IMAGES - new_used)
        response.set_cookie('anon_used', str(new_used),
                            max_age=60*60*24*365, httponly=True, samesite='Lax')
        response.headers['X-Images-Remaining'] = str(remaining_after)

    return response

# Run init_db when module loads (works with gunicorn)
try:
    init_db()
except Exception as e:
    print(f'DB init warning: {e}')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
