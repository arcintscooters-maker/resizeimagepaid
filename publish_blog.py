#!/usr/bin/env python3
"""Publish a blog post to the PixelPrep database.
Usage: python publish_blog.py <slug> <title> <meta_description> <content_html_file>
"""
import os, sys, psycopg2

def main():
    if len(sys.argv) < 5:
        print("Usage: python publish_blog.py <slug> <title> <meta_description> <content_html_file>")
        sys.exit(1)

    slug = sys.argv[1]
    title = sys.argv[2]
    meta_desc = sys.argv[3]
    content_file = sys.argv[4]

    with open(content_file, 'r', encoding='utf-8') as f:
        content = f.read()

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set")
        sys.exit(1)

    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO blog_posts (slug, title, meta_description, content, published)
           VALUES (%s, %s, %s, %s, TRUE)
           ON CONFLICT (slug) DO UPDATE SET title=EXCLUDED.title, meta_description=EXCLUDED.meta_description,
           content=EXCLUDED.content, updated_at=NOW()""",
        (slug, title, meta_desc, content)
    )
    conn.commit()
    cur.close()
    conn.close()
    print(f"Published: /blog/{slug}")

if __name__ == "__main__":
    main()
