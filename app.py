import os
import secrets
import json
from datetime import datetime, timedelta, timezone
from functools import wraps

import stripe
import bcrypt
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

QUALITY = 85
SUBSAMPLING = 0
TRIAL_DAYS = 7
ANON_FREE_IMAGES = 20

# ── Database ──────────────────────────────────────────────────────────────────

def get_db():
    return psycopg2.connect(os.environ["DATABASE_URL"], cursor_factory=RealDictCursor)

def init_db():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
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
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
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
    now = datetime.now(timezone.utc)
    if user['subscription_status'] == 'active':
        ends = user['subscription_ends_at']
        return ends is None or ends > now
    if user['subscription_status'] == 'trial':
        ends = user['trial_ends_at']
        return ends is not None and ends > now
    return False

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
    # Logged in user
    if 'user_id' in session:
        user = get_user_by_id(session['user_id'])
        if not user:
            session.clear()
            return redirect(url_for('login_page'))
        if not is_active(user):
            return redirect(url_for('subscribe_page'))
        return render_template("index.html", user=user,
                               days_left=trial_days_left(user),
                               images_remaining=None,
                               is_trial=user['subscription_status'] == 'trial')

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
    return jsonify({"ok": True, "redirect": "/"})

@app.route("/subscribe")
@login_required
def subscribe_page():
    user = get_user_by_id(session['user_id'])
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
def process():
    is_anon = False
    cookie_used = 0

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
    else:
        user = get_user_by_id(session['user_id'])
        if not is_active(user):
            return jsonify({"error": "subscription_required"}), 402

    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    # For anon users, cap batch to remaining count
    if is_anon:
        files = files[:remaining]

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

    image_count = len(files)

    if image_count == 1:
        f = files[0]
        result = process_image(f.read(), target_w, target_h, bg_color, remove_bg, fill_pct)
        name = os.path.splitext(f.filename)[0] + ".jpg"
        response = send_file(result, mimetype="image/jpeg", as_attachment=True, download_name=name)
    else:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                result = process_image(f.read(), target_w, target_h, bg_color, remove_bg, fill_pct)
                name = os.path.splitext(f.filename)[0] + ".jpg"
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

if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
