#!/usr/bin/env python3
"""Publish a blog post to PixelPrep through the site's key-protected API.

Usage:
  python publish_blog.py <slug> <title> <meta_description> <content_html_file> [--created-at ISO] [--draft]
  python publish_blog.py --count        # {total, published} from the live site
  python publish_blog.py --list         # slugs + titles currently on /blog

Env:
  PIXELPREP_BLOG_URL   default https://www.pixel-prep.com
  BLOG_PUBLISH_KEY     the X-Blog-Key secret; if unset, read from the gitignored
                       file .blog_publish_key next to this script, or
                       PIXELPREP_BLOG_PUBLISH_KEY= in C:/Users/Admin/inlinex-admin-app/.env
"""
import html
import json
import os
import re
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("PIXELPREP_BLOG_URL", "https://www.pixel-prep.com").rstrip("/")
ADMIN_ENV = "C:/Users/Admin/inlinex-admin-app/.env"
UA = "PixelPrep-publish/1.0"


def load_key():
    key = os.environ.get("BLOG_PUBLISH_KEY", "").strip()
    if key:
        return key
    local = os.path.join(HERE, ".blog_publish_key")
    if os.path.exists(local):
        with open(local, encoding="utf-8") as f:
            key = f.read().strip()
        if key:
            return key
    if os.path.exists(ADMIN_ENV):
        with open(ADMIN_ENV, encoding="utf-8") as f:
            for line in f:
                if line.startswith("PIXELPREP_BLOG_PUBLISH_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    print("ERROR: BLOG_PUBLISH_KEY not set (env, .blog_publish_key, or inlinex-admin-app/.env)")
    sys.exit(1)


def call(method, path, body=None, key=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    req.add_header("User-Agent", UA)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("X-Blog-Key", key)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def main():
    args = sys.argv[1:]
    if args[:1] == ["--count"]:
        status, text = call("GET", "/admin/api/blog-count", key=load_key())
        print(text)
        sys.exit(0 if status == 200 else 1)
    if args[:1] == ["--list"]:
        status, text = call("GET", "/blog")
        if status != 200:
            print(f"ERROR: /blog returned {status}")
            sys.exit(1)
        for slug, title in re.findall(r'href="/blog/([^"#?]+)".*?class="post-title">(.*?)</div>', text, re.S):
            print(f"{slug}\t{html.unescape(re.sub(r'<[^>]+>', '', title)).strip()}")
        sys.exit(0)

    created_at = None
    published = True
    if "--created-at" in args:
        i = args.index("--created-at")
        created_at = args[i + 1]
        del args[i:i + 2]
    if "--draft" in args:
        args.remove("--draft")
        published = False
    if len(args) < 4:
        print(__doc__)
        sys.exit(1)

    slug, title, meta_desc, content_file = args[:4]
    with open(content_file, "r", encoding="utf-8") as f:
        content = f.read()

    body = {"slug": slug, "title": title, "meta_description": meta_desc,
            "content": content, "published": published}
    if created_at:
        body["created_at"] = created_at
    status, text = call("POST", "/admin/api/blog-publish", body, key=load_key())
    if status != 200:
        print(f"ERROR: publish failed HTTP {status}: {text[:300]}")
        sys.exit(1)
    print(f"Published: /blog/{slug}  ({text.strip()})")


if __name__ == "__main__":
    main()
