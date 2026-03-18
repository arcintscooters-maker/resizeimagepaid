import os
import secrets
import string
from datetime import datetime, timedelta, timezone
from functools import wraps

import stripe
from flask import (Flask, request, send_file, render_template,
                   jsonify, session, redirect, url_for)
from PIL import Image
import io
import zipfile
import numpy as np
from collections import deque
import psycopg2
from psycopg2.extras import RealDictCursor
import requests as http_requests

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# Stripe
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY")
STRIPE_PRICE_ID = os.environ.get("STRIPE_PRICE_ID")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET")

# Resend
RESEND_API_KEY = os.environ.get("RESEND_API_KEY")
FROM_EMAIL = os.environ.get("FROM_EMAIL", "noreply@pixelprep.app")
APP_URL = os.environ.get("APP_URL", "http://localhost:5000")

QUALITY = 85
SUBSAMPLING = 0
TRIAL_DAYS = 7

# ── Database ──────────────────────────────────────────────────────────────────

def get_db():
    return psycopg2.connect(os.environ["DATABASE_URL"], cursor_factory=RealDictCursor)

def init_db():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    stripe_customer_id TEXT,
                    subscription_status TEXT DEFAULT 'trial',
                    trial_ends_at TIMESTAMPTZ,
                    subscription_ends_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS magic_tokens (
                    id SERIAL PRIMARY KEY,
                    email TEXT NOT NULL,
                    token TEXT UNIQUE NOT NULL,
                    expires_at TIMESTAMPTZ NOT NULL,
                    used BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
        conn.commit()

def get_user_by_email(email):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE email = %s", (email,))
            return cur.fetchone()

def get_user_by_id(user_id):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            return cur.fetchone()

def create_user(email):
    trial_ends = datetime.now(timezone.utc) + timedelta(days=TRIAL_DAYS)
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO users (email, trial_ends_at, subscription_status)
                VALUES (%s, %s, 'trial')
                ON CONFLICT (email) DO NOTHING
                RETURNING *
            """, (email, trial_ends))
            result = cur.fetchone()
        conn.commit()
    return result

def upsert_user(email):
    user = get_user_by_email(email)
    if not user:
        user = create_user(email)
        if not user:
            user = get_user_by_email(email)
    return user

def update_user_subscription(stripe_customer_id, status, ends_at=None):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE users
                SET subscription_status = %s, subscription_ends_at = %s
                WHERE stripe_customer_id = %s
            """, (status, ends_at, stripe_customer_id))
        conn.commit()

def create_magic_token(email):
    token = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(minutes=15)
    with get_db() as conn:
        with conn.cursor() as cur:
            # Invalidate old tokens for this email
            cur.execute("UPDATE magic_tokens SET used = TRUE WHERE email = %s AND used = FALSE", (email,))
            cur.execute("""
                INSERT INTO magic_tokens (email, token, expires_at)
                VALUES (%s, %s, %s)
            """, (email, token, expires))
        conn.commit()
    return token

def verify_magic_token(token):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM magic_tokens
                WHERE token = %s AND used = FALSE AND expires_at > NOW()
            """, (token,))
            row = cur.fetchone()
            if row:
                cur.execute("UPDATE magic_tokens SET used = TRUE WHERE id = %s", (row['id'],))
            conn.commit()
    return row

def is_active(user):
    """Check if user has active access — trial or paid subscription."""
    if not user:
        return False
    now = datetime.now(timezone.utc)
    status = user['subscription_status']
    if status == 'active':
        ends = user['subscription_ends_at']
        return ends is None or ends > now
    if status == 'trial':
        trial_ends = user['trial_ends_at']
        return trial_ends is not None and trial_ends > now
    return False

def trial_days_left(user):
    if user['subscription_status'] != 'trial':
        return 0
    ends = user['trial_ends_at']
    if not ends:
        return 0
    diff = ends - datetime.now(timezone.utc)
    return max(0, diff.days)

# ── Auth helpers ──────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated

def subscription_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({"error": "login_required"}), 401
        user = get_user_by_id(session['user_id'])
        if not is_active(user):
            return jsonify({"error": "subscription_required"}), 402
        return f(*args, **kwargs)
    return decorated

# ── Email ─────────────────────────────────────────────────────────────────────

def send_magic_link(email, token):
    link = f"{APP_URL}/auth/verify?token={token}"
    html = f"""
    <div style="font-family:sans-serif;max-width:480px;margin:0 auto;padding:40px 24px;">
      <h2 style="font-size:22px;font-weight:700;margin-bottom:8px;">Sign in to PixelPrep</h2>
      <p style="color:#666;margin-bottom:24px;">Click the button below to sign in. This link expires in 15 minutes.</p>
      <a href="{link}" style="display:inline-block;background:#1a1714;color:#fff;text-decoration:none;padding:14px 28px;border-radius:8px;font-weight:700;font-size:14px;">Sign in to PixelPrep</a>
      <p style="color:#999;font-size:12px;margin-top:24px;">If you didn't request this, you can safely ignore this email.</p>
    </div>
    """
    resp = http_requests.post(
        "https://api.resend.com/emails",
        headers={"Authorization": f"Bearer {RESEND_API_KEY}", "Content-Type": "application/json"},
        json={"from": FROM_EMAIL, "to": email, "subject": "Your PixelPrep sign-in link", "html": html}
    )
    return resp.status_code == 200

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

def process_image(img_bytes, target_w, target_h, bg_color_hex, remove_bg, fill_pct=0.95):
    bg_rgb = hex_to_rgb(bg_color_hex)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    if remove_bg:
        try:
            from rembg import remove, new_session
            session_rembg = new_session("birefnet-general")
            img = remove(img, session=session_rembg)
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
    user = None
    if 'user_id' in session:
        user = get_user_by_id(session['user_id'])
        if user and not is_active(user):
            return redirect(url_for('subscribe_page'))
    if not user:
        return redirect(url_for('login_page'))
    days_left = trial_days_left(user)
    return render_template("index.html", user=user, days_left=days_left,
                           is_trial=user['subscription_status'] == 'trial')

@app.route("/login")
def login_page():
    if 'user_id' in session:
        return redirect(url_for('index'))
    return render_template("login.html")

@app.route("/auth/send", methods=["POST"])
def send_magic():
    data = request.get_json()
    email = (data.get("email") or "").strip().lower()
    if not email or "@" not in email:
        return jsonify({"error": "Invalid email"}), 400
    user = upsert_user(email)
    token = create_magic_token(email)
    ok = send_magic_link(email, token)
    if not ok:
        return jsonify({"error": "Failed to send email"}), 500
    return jsonify({"ok": True})

@app.route("/auth/verify")
def verify_magic():
    token = request.args.get("token", "")
    row = verify_magic_token(token)
    if not row:
        return render_template("login.html", error="This link has expired or already been used. Please request a new one.")
    user = upsert_user(row['email'])
    session['user_id'] = user['id']
    session.permanent = True
    if not is_active(user):
        return redirect(url_for('subscribe_page'))
    return redirect(url_for('index'))

@app.route("/subscribe")
@login_required
def subscribe_page():
    user = get_user_by_id(session['user_id'])
    return render_template("subscribe.html", user=user)

@app.route("/subscribe/checkout", methods=["POST"])
@login_required
def create_checkout():
    user = get_user_by_id(session['user_id'])
    # Create or reuse Stripe customer
    if not user['stripe_customer_id']:
        customer = stripe.Customer.create(email=user['email'])
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
        success_url=f"{APP_URL}/subscribe/success?session_id={{CHECKOUT_SESSION_ID}}",
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
        update_user_subscription(sub['customer'], 'cancelled', None)

    return jsonify({"ok": True})

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('login_page'))

@app.route("/process", methods=["POST"])
@subscription_required
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
    try:
        fill_pct = max(10, min(100, int(request.form.get("fill_pct", 95)))) / 100.0
    except:
        fill_pct = 0.95

    if len(files) == 1:
        f = files[0]
        result = process_image(f.read(), target_w, target_h, bg_color, remove_bg, fill_pct)
        name = os.path.splitext(f.filename)[0] + ".jpg"
        return send_file(result, mimetype="image/jpeg", as_attachment=True, download_name=name)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            result = process_image(f.read(), target_w, target_h, bg_color, remove_bg, fill_pct)
            name = os.path.splitext(f.filename)[0] + ".jpg"
            zf.writestr(name, result.read())
    zip_buf.seek(0)
    return send_file(zip_buf, mimetype="application/zip", as_attachment=True, download_name="resized_images.zip")

if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
