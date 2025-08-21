#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mass RSS/Atom ingestion focused on UI/UX content.

- Timezone-aware UTC dates
- Optional requests-based fetching
- WordPress pagination (?paged=N) via --wp-pages
- Best-effort image extraction (+ optional OG image fetch)
- Verbose logging and per-feed stats

Usage:
  python ingest_rss.py --feeds feeds_mass.yaml --max 80 --wp-pages 6 --requests --opengraph --no-cutoff
  python ingest_rss.py --one https://www.smashingmagazine.com/feed/ --wp-pages 8 --requests --no-cutoff
"""
import os, sys, re, sqlite3, argparse
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
from pathlib import Path

import feedparser
import requests
from bs4 import BeautifulSoup

DB_PATH = os.path.abspath("trends.db")

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS topics (id INTEGER PRIMARY KEY, name TEXT, keywords TEXT, category TEXT);
CREATE TABLE IF NOT EXISTS posts (
  id INTEGER PRIMARY KEY,
  title TEXT,
  url TEXT UNIQUE,
  source TEXT,
  image_url TEXT,
  published_at TIMESTAMP,
  summary TEXT
);
CREATE TABLE IF NOT EXISTS post_topics (post_id INTEGER, topic_id INTEGER);
CREATE TABLE IF NOT EXISTS topic_timeseries (topic_id INTEGER, week TIMESTAMP, count INTEGER, momentum REAL);
CREATE TABLE IF NOT EXISTS ui_style_trends (style TEXT, week TIMESTAMP, count INTEGER);
CREATE TABLE IF NOT EXISTS ui_framework_trends (framework TEXT, week TIMESTAMP, count INTEGER);
CREATE TABLE IF NOT EXISTS ui_component_tags (post_id INTEGER, component TEXT);
"""

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36"}

def ensure_schema():
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)
    con.commit()
    return con

def load_feeds(path: str):
    data = {}
    cat = None
    text = Path(path).read_text(encoding="utf-8")
    for line in text.splitlines():
        raw = line.rstrip("\n")
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if not raw.startswith(" ") and s.endswith(":"):
            cat = s[:-1]; data[cat] = []
        elif cat and s.startswith("- "):
            data[cat].append(s[2:].strip())
    return data

def iso_from_struct(tup):
    try:
        return datetime(*tup[:6], tzinfo=timezone.utc).isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()

def sanitize_summary(s, max_len=900):
    s = s or ""
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_len: s = s[:max_len] + "…"
    return s

def best_image_from_entry(entry):
    media = entry.get("media_thumbnail") or entry.get("media_content")
    if isinstance(media, list) and media:
        for m in media:
            u = m.get("url")
            if u: return u
    for enc in entry.get("enclosures") or []:
        href = enc.get("href"); typ = (enc.get("type") or "").lower()
        if href and (typ.startswith("image/") or re.search(r"\.(png|jpe?g|webp|gif)(?:$|\?)", href, re.I)):
            return href
    html = ""
    if entry.get("content"):
        try:
            html = " ".join([c.get("value","") for c in entry["content"] if isinstance(c, dict)])
        except Exception:
            pass
    if not html:
        html = entry.get("summary","") or ""
    m = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
    if m: return m.group(1)
    return None

def discover_og_image(url, timeout=10):
    try:
        r = requests.get(url, timeout=timeout, headers=UA)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        og = soup.find("meta", property="og:image") or soup.find("meta", attrs={"name":"og:image"})
        if og and og.get("content"): return og["content"]
        tw = soup.find("meta", attrs={"name":"twitter:image"}) or soup.find("meta", attrs={"property":"twitter:image"})
        if tw and tw.get("content"): return tw["content"]
        img = soup.find("img")
        if img and img.get("src"): return img["src"]
    except Exception:
        return None
    return None

def upsert_post(con, title, url, source, published_at_iso, summary, image_url):
    cur = con.cursor()
    # Try to update existing rows by URL
    cur.execute("""
        UPDATE posts
           SET title = ?,
               source = ?,
               published_at = ?,
               summary = ?,
               image_url = COALESCE(?, image_url)
         WHERE url = ?
    """, (title, source, published_at_iso, summary, image_url, url))

    if cur.rowcount == 0:
        # No existing row with this URL -> insert
        cur.execute("""
            INSERT INTO posts(title, url, source, published_at, summary, image_url)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (title, url, source, published_at_iso, summary, image_url))


def parse_feed(href, use_requests):
    if use_requests:
        try:
            r = requests.get(href, headers=UA, timeout=15)
            r.raise_for_status()
            return feedparser.parse(r.content)
        except Exception as e:
            print(f"  ! requests fetch failed: {e}")
            return feedparser.parse(href)
    else:
        return feedparser.parse(href)

def add_paged(url, page:int):
    parts = urlsplit(url)
    qs = dict(parse_qsl(parts.query, keep_blank_values=True))
    qs["paged"] = str(page)
    new_query = urlencode(qs, doseq=True)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, new_query, parts.fragment))

def is_wordpress_like(feed_meta):
    gen = (feed_meta.get("generator") or "").lower()
    return "wordpress" in gen

def iter_entries_with_paging(feed_url, use_requests, wp_pages):
    d = parse_feed(feed_url, use_requests)
    meta = d.get("feed", {})
    entries = list(d.get("entries", []))
    yield entries, meta, 1

    wpish = is_wordpress_like(meta) or "/feed" in (feed_url or "")
    if not wpish or wp_pages <= 1:
        return

    seen_links = set(e.get("link") for e in entries if e.get("link"))
    for page in range(2, wp_pages+1):
        href = add_paged(feed_url, page)
        d2 = parse_feed(href, use_requests)
        ents = d2.get("entries", []) or []
        fresh = [e for e in ents if e.get("link") and e["link"] not in seen_links]
        if not fresh:
            break
        for e in fresh:
            seen_links.add(e.get("link"))
        yield fresh, meta, page

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feeds", default="feeds_mass.yaml")
    ap.add_argument("--max", type=int, default=80)
    ap.add_argument("--wp-pages", dest="wp_pages", type=int, default=1, help="WordPress pagination pages")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--opengraph", action="store_true")
    ap.add_argument("--no-cutoff", action="store_true")
    ap.add_argument("--one")
    ap.add_argument("--requests", action="store_true")
    args = ap.parse_args()

    con = ensure_schema()
    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)

    feeds = {"debug":[args.one]} if args.one else load_feeds(args.feeds)
    if not feeds:
        print("No feeds found."); sys.exit(0)

    total_new = 0; total_seen = 0
    for cat, urls in feeds.items():
        print(f"[{cat}]")
        for feed_url in urls:
            feed_new = 0; feed_seen = 0
            page_idx = 0
            for entries, meta, page in iter_entries_with_paging(feed_url, args.requests, args.wp_pages):
                page_idx = page
                print(f"  - entries found: {len(entries)}  (page {page} | {feed_url})")
                for e in entries[: args.max]:
                    feed_seen += 1; total_seen += 1
                    title = (e.get("title") or "(untitled)").strip()
                    link  = (e.get("link")  or "").strip()
                    if not link: 
                        continue

                    pub_struct = e.get("published_parsed") or e.get("updated_parsed") or e.get("created_parsed")
                    dt_iso = iso_from_struct(pub_struct) if pub_struct else datetime.now(timezone.utc).isoformat()
                    raw = dt_iso.replace("Z", "+00:00")
                    try:
                        dt = datetime.fromisoformat(raw)
                    except Exception:
                        dt = datetime.now(timezone.utc)
                    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
                    else: dt = dt.astimezone(timezone.utc)

                    if not args.no_cutoff and dt < cutoff:
                        continue

                    summary = e.get("summary") or ""
                    if e.get("content"):
                        try:
                            summary = e["content"][0].get("value", summary)
                        except Exception:
                            pass
                    summary = sanitize_summary(summary)

                    img = best_image_from_entry(e)
                    if not img and args.opengraph:
                        img = discover_og_image(link)

                    before = con.total_changes
                    upsert_post(con, title, link, cat, dt.isoformat(), summary, img)
                    con.commit()
                    after = con.total_changes
                    if after > before:
                        feed_new += 1; total_new += 1

            print(f"    inserted/updated this feed: {feed_new}  (pages scanned: {page_idx})")

    print(f"Done. New/updated rows this run: {total_new}  (from {total_seen} entries scanned)")

if __name__ == "__main__":
    main()
