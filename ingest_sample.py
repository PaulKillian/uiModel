#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ingest many RSS/Atom feeds into trends.db (posts table), aiming for ~N articles.

Usage examples:
  python ingest_sample.py --feeds feeds.json --target 120
  python ingest_sample.py --feeds feeds.json --target 200 --per_feed 60 --no-embed
  python ingest_sample.py --feeds feeds.json --target 150 --days 30

Notes:
- If readability-lxml is installed, will try to extract clean full text.
  pip install readability-lxml lxml beautifulsoup4
- For embeddings (default ON):
  pip install sentence-transformers
"""

import os
import sys
import json
import time
import math
import argparse
import hashlib
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

import sqlite3
import requests
import feedparser

# Optional text extraction
try:
    from readability import Document as ReadabilityDocument  # readability-lxml
    HAVE_READABILITY = True
except Exception:
    HAVE_READABILITY = False

try:
    from bs4 import BeautifulSoup
    HAVE_BS4 = True
except Exception:
    HAVE_BS4 = False

# Optional embeddings
EMBED_DIM = 384  # all-MiniLM-L6-v2
try:
    from sentence_transformers import SentenceTransformer
    HAVE_EMBED = True
except Exception:
    HAVE_EMBED = False

DB_PATH = os.path.abspath("trends.db")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) IngestBot/1.0"
}


# ------------------------------
# Console-safe logging (ASCII)
# ------------------------------
def log(msg: str):
    print(str(msg), flush=True)


# ------------------------------
# DB helpers
# ------------------------------
def ensure_posts_table(con: sqlite3.Connection):
    con.execute("""
    CREATE TABLE IF NOT EXISTS posts (
        id TEXT PRIMARY KEY,
        url TEXT,
        title TEXT,
        source TEXT,
        published_at TIMESTAMP,
        author TEXT,
        summary TEXT,
        image_url TEXT,
        fulltext TEXT,
        text_embedding BLOB,
        image_embedding BLOB,
        labels TEXT
    )
    """)
    con.commit()


def upsert_post(con: sqlite3.Connection, row: dict):
    # Only store fields we know
    fields = ["id","url","title","source","published_at","author",
              "summary","image_url","fulltext","text_embedding","image_embedding","labels"]
    vals = [row.get(k) for k in fields]
    con.execute(f"""
        INSERT OR REPLACE INTO posts ({",".join(fields)})
        VALUES ({",".join(["?"]*len(fields))})
    """, vals)


def has_embedding(con: sqlite3.Connection, pid: str) -> bool:
    cur = con.execute("SELECT text_embedding FROM posts WHERE id = ?", (pid,))
    r = cur.fetchone()
    return bool(r and r[0])


# ------------------------------
# URL canonicalization & IDs
# ------------------------------
STRIP_QUERY_KEYS = {
    "utm_source","utm_medium","utm_campaign","utm_term","utm_content",
    "utm_id","utm_name","utm_cid","utm_reader","utm_viz_id",
    "fbclid","gclid","igshid","mc_cid","mc_eid","ref","ref_src",
}

def canonical_url(raw: str) -> str:
    try:
        u = urlparse(raw)
        q = parse_qs(u.query, keep_blank_values=True)
        q = {k:v for k,v in q.items() if k.lower() not in STRIP_QUERY_KEYS}
        new_query = urlencode(q, doseq=True)
        # normalize scheme/host
        scheme = u.scheme.lower() if u.scheme else "https"
        netloc = u.netloc.lower()
        cleaned = urlunparse((scheme, netloc, u.path or "/", u.params, new_query, ""))  # drop fragment
        return cleaned
    except Exception:
        return raw.strip()


def make_post_id(url: str) -> str:
    # deterministic ID from canonical URL
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


# ------------------------------
# Date parsing
# ------------------------------
def parse_published(entry) -> str:
    # Return ISO UTC string "YYYY-MM-DDTHH:MM:SSZ"
    dt_utc = None
    # feedparser often gives "published_parsed" or "updated_parsed"
    for attr in ("published_parsed", "updated_parsed"):
        tm = entry.get(attr)
        if tm:
            try:
                dt_utc = datetime(*tm[:6], tzinfo=timezone.utc)
                break
            except Exception:
                pass
    if not dt_utc:
        # fallback: now
        dt_utc = datetime.now(timezone.utc)
    return dt_utc.replace(microsecond=0).isoformat().replace("+00:00","Z")


# ------------------------------
# Full text extraction
# ------------------------------
def extract_fulltext(url: str, html: str | None) -> tuple[str, str | None]:
    """
    Returns (text, image_url_or_none). Best-effort, safe on missing deps.
    """
    if not html:
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            html = r.text
        except Exception:
            html = ""

    img_url = None
    text = ""

    try:
        if HAVE_READABILITY and html:
            doc = ReadabilityDocument(html)
            content_html = doc.summary(html_partial=True)
            title = doc.short_title() or ""
            if HAVE_BS4:
                soup = BeautifulSoup(content_html, "html.parser")
                # teaser image if present
                img = soup.find("img")
                if img and img.get("src"):
                    img_url = img["src"]
                # extract text
                paras = [p.get_text(" ", strip=True) for p in soup.find_all(["p","li"])]
                text = "\n".join([title] + paras).strip()
            else:
                # crude fallback: strip tags
                text = title
        elif HAVE_BS4 and html:
            soup = BeautifulSoup(html, "html.parser")
            # meta image
            m = (soup.find("meta", attrs={"property":"og:image"}) or
                 soup.find("meta", attrs={"name":"twitter:image"}))
            if m and m.get("content"):
                img_url = m["content"]
            # text
            paras = [p.get_text(" ", strip=True) for p in soup.find_all(["p","li"])]
            text = "\n".join(paras).strip()
    except Exception:
        pass

    return text, img_url


# ------------------------------
# Embeddings
# ------------------------------
def embed_texts(texts: list[str]) -> list[bytes] | None:
    if not HAVE_EMBED:
        log("Embeddings disabled or sentence-transformers not installed.")
        return None
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embs = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    out = []
    for v in embs:
        arr = (v.astype("float32") if hasattr(v, "dtype") else v)
        out.append(arr.tobytes())
    return out


# ------------------------------
# Main ingest
# ------------------------------
def ingest(feeds_file: str, target: int, per_feed: int, days: int, do_embed: bool):
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError("DB not found at %s. Run db_setup.py first." % DB_PATH)

    with open(feeds_file, "r", encoding="utf-8") as f:
        feeds = json.load(f)
    if isinstance(feeds, dict) and "feeds" in feeds:
        feeds = feeds["feeds"]
    if not isinstance(feeds, list) or not feeds:
        raise ValueError("feeds.json must be a list of objects: [{\"name\":\"...\",\"url\":\"...\"}, ...]")

    con = sqlite3.connect(DB_PATH)
    ensure_posts_table(con)

    seen = set()  # canonical URLs
    added_rows = []
    total = 0

    cutoff = None
    if days and days > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=int(days))

    log("Loading feeds from %s" % feeds_file)
    for feed_item in feeds:
        if total >= target:
            break
        name = str(feed_item.get("name") or "").strip() or "Feed"
        url = str(feed_item.get("url") or "").strip()
        if not url:
            continue

        log("Fetching: %s - %s" % (name, url))
        try:
            fp = feedparser.parse(url, request_headers=HEADERS)
        except Exception as e:
            log("  ERROR: feedparser failed: %s" % e)
            continue

        entries = fp.entries or []
        grabbed = 0

        for entry in entries[:max(per_feed, 0) or 0]:
            if total >= target:
                break

            raw_url = entry.get("link") or entry.get("id") or ""
            if not raw_url:
                continue
            url_c = canonical_url(raw_url)
            if url_c in seen:
                continue
            seen.add(url_c)

            # date filter
            pub_iso = parse_published(entry)
            if cutoff:
                try:
                    pub_dt = datetime.fromisoformat(pub_iso.replace("Z","+00:00"))
                    if pub_dt < cutoff:
                        continue
                except Exception:
                    pass

            title = (entry.get("title") or "").strip()
            author = (entry.get("author") or "")[:200]
            summary = (entry.get("summary") or entry.get("description") or "")
            # strip tags from summary if any
            if "<" in summary and ">" in summary and HAVE_BS4:
                summary = BeautifulSoup(summary, "html.parser").get_text(" ", strip=True)
            summary = summary.strip()

            image_url = None
            # media content image if provided
            media_content = entry.get("media_content") or []
            if isinstance(media_content, list) and media_content:
                try:
                    image_url = media_content[0].get("url")
                except Exception:
                    pass
            if not image_url:
                # try media_thumbnail
                thumbs = entry.get("media_thumbnail") or []
                if isinstance(thumbs, list) and thumbs:
                    try:
                        image_url = thumbs[0].get("url")
                    except Exception:
                        pass

            fulltext = ""
            if HAVE_READABILITY or HAVE_BS4:
                try:
                    html = None
                    # some feeds include content:encoded
                    for k in ("content","summary_detail"):
                        val = entry.get(k)
                        if isinstance(val, list) and val:
                            html = val[0].get("value")
                            break
                        elif isinstance(val, dict) and val.get("value"):
                            html = val["value"]
                            break
                    ft, img2 = extract_fulltext(url_c, html)
                    if ft:
                        fulltext = ft
                    if (not image_url) and img2:
                        image_url = img2
                except Exception:
                    pass

            # Build post record
            pid = make_post_id(url_c)
            row = {
                "id": pid,
                "url": url_c,
                "title": title[:500],
                "source": name[:120],
                "published_at": pub_iso,
                "author": author,
                "summary": summary[:4000],
                "image_url": image_url or "",
                "fulltext": (fulltext or "").strip()[:100000],
                "text_embedding": None,
                "image_embedding": None,
                "labels": None
            }

            added_rows.append(row)
            grabbed += 1
            total += 1

        log("  Added %d entries from this feed (so far total=%d)" % (grabbed, total))
        time.sleep(0.2)  # be polite

    if not added_rows:
        log("No new items gathered. Try adding more feeds or lowering --days filter.")
        return

    # Compute embeddings (optional)
    if do_embed and HAVE_EMBED:
        texts = []
        for r in added_rows:
            txt = (r["fulltext"] or "").strip()
            if len(txt) < 140:
                txt = (r["title"] + " " + (r["summary"] or "")).strip()
            texts.append(txt if txt else (r["title"] or ""))
        log("Computing embeddings for %d documents..." % len(texts))
        embs = embed_texts(texts)
        if embs is not None:
            for r, blob in zip(added_rows, embs):
                r["text_embedding"] = blob
    else:
        if not do_embed:
            log("Skipping embeddings (--no-embed).")
        elif not HAVE_EMBED:
            log("sentence-transformers not installed; skipping embeddings.")

    # Write to DB
    con = sqlite3.connect(DB_PATH)
    ensure_posts_table(con)
    inserted = 0
    for r in added_rows:
        try:
            upsert_post(con, r)
            inserted += 1
        except Exception as e:
            log("  ERROR inserting row %s: %s" % (r.get("id","?"), e))
    con.commit()
    con.close()

    log("Done. Inserted %d posts." % inserted)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feeds", type=str, default="feeds.json", help="Path to feeds.json")
    ap.add_argument("--target", type=int, default=120, help="Target total articles to gather")
    ap.add_argument("--per_feed", type=int, default=50, help="Max items per feed (soft cap)")
    ap.add_argument("--days", type=int, default=0, help="Only include articles from the last N days (0 = all)")
    ap.add_argument("--no-embed", action="store_true", help="Skip computing text embeddings")
    args = ap.parse_args()

    ingest(
        feeds_file=args.feeds,
        target=int(args.target),
        per_feed=int(args.per_feed),
        days=int(args.days),
        do_embed=(not args.no_embed),
    )


if __name__ == "__main__":
    main()
