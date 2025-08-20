# db_setup.py
# Creates the full UI/UX trends schema on first run (idempotent) and inserts a demo row.

import os
import sys
import sqlite3
from datetime import datetime, timezone
import numpy as np

DB = os.path.abspath("trends.db")

FULL_SCHEMA_SQL = """
-- posts table
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
);

-- topics table
CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    description TEXT,
    keywords TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- mapping
CREATE TABLE IF NOT EXISTS post_topics (
    post_id TEXT REFERENCES posts(id),
    topic_id INTEGER REFERENCES topics(id)
);

-- timeseries
CREATE TABLE IF NOT EXISTS topic_timeseries (
    topic_id INTEGER REFERENCES topics(id),
    week DATE,
    count INTEGER,
    momentum FLOAT,
    burst_score FLOAT
);
"""

NEEDED_POSTS_COLS = {
    "id","url","title","source","published_at","author","summary",
    "image_url","fulltext","text_embedding","image_embedding","labels"
}

def log(*a):
    print(*a, flush=True)

def iso_utc_now() -> str:
    # Use timezone-aware UTC and store as ISO-8601 text for portability
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def connect():
    # You can add detect_types for datetime adapters if you later want native timestamps
    return sqlite3.connect(DB)

def ensure_full_schema():
    """Create all tables if missing. Also sanity-check 'posts' columns."""
    con = connect()
    try:
        con.executescript(FULL_SCHEMA_SQL)
        con.commit()

        # Verify posts has all expected columns
        cols = [r[1] for r in con.execute("PRAGMA table_info(posts)")]
        cols_norm = { (c or "").strip().lower() for c in cols }
        missing = NEEDED_POSTS_COLS - cols_norm
        if missing:
            # If you're here, an older posts table existed; safest is to ALTER in the missing columns
            # (SQLite ignores adding a column that already exists, so we'll try each one defensively).
            add_map = {
                "url": "TEXT", "title":"TEXT", "source":"TEXT", "published_at":"TIMESTAMP",
                "author":"TEXT","summary":"TEXT","image_url":"TEXT","fulltext":"TEXT",
                "text_embedding":"BLOB","image_embedding":"BLOB","labels":"TEXT"
            }
            for col in missing:
                coltype = add_map.get(col, "TEXT")
                sql = f'ALTER TABLE posts ADD COLUMN {col} {coltype};'
                try:
                    con.execute(sql)
                except sqlite3.OperationalError:
                    # e.g., very old SQLite without ALTER semantics or odd column state; ignore and continue
                    pass
            con.commit()
            # Re-check
            cols = [r[1] for r in con.execute("PRAGMA table_info(posts)")]
            cols_norm = { (c or "").strip().lower() for c in cols }
            still_missing = NEEDED_POSTS_COLS - cols_norm
            if still_missing:
                raise RuntimeError(
                    f"Schema migration incomplete; missing columns in posts: {still_missing}. "
                    f"Consider recreating DB (delete {DB}) if this persists."
                )
    finally:
        con.close()

def upsert_post(row: dict):
    """Upsert a post with all main columns. Pass embeddings as bytes."""
    con = connect()
    try:
        con.execute("""
        INSERT INTO posts (
            id, url, title, source, published_at, author, summary,
            image_url, fulltext, text_embedding, image_embedding, labels
        ) VALUES (
            :id, :url, :title, :source, :published_at, :author, :summary,
            :image_url, :fulltext, :text_embedding, :image_embedding, :labels
        )
        ON CONFLICT(id) DO UPDATE SET
            url=excluded.url,
            title=excluded.title,
            source=excluded.source,
            published_at=excluded.published_at,
            author=excluded.author,
            summary=excluded.summary,
            image_url=excluded.image_url,
            fulltext=excluded.fulltext,
            text_embedding=excluded.text_embedding,
            image_embedding=excluded.image_embedding,
            labels=excluded.labels
        """, row)
        con.commit()
    finally:
        con.close()

def get_latest_posts(limit=5):
    con = connect()
    try:
        cur = con.cursor()
        cur.execute("""
            SELECT id, title, published_at,
                   LENGTH(text_embedding) AS text_bytes,
                   LENGTH(image_embedding) AS image_bytes
            FROM posts
            ORDER BY rowid DESC
            LIMIT ?
        """, (limit,))
        return cur.fetchall()
    finally:
        con.close()

def get_text_embedding(post_id: str):
    con = connect()
    try:
        row = con.execute("SELECT text_embedding FROM posts WHERE id = ?", (post_id,)).fetchone()
        if not row or row[0] is None:
            return None
        return np.frombuffer(row[0], dtype=np.float32)
    finally:
        con.close()

if __name__ == "__main__":
    log("Python:", sys.executable)
    log("CWD   :", os.getcwd())
    log("DB    :", DB)

    ensure_full_schema()
    log("Schema OK ✅")

    # --- Demo insert
    text_emb = np.random.rand(384).astype("float32").tobytes()
    img_emb  = np.random.rand(512).astype("float32").tobytes()  # optional demo

    demo = {
        "id": "demo1",
        "url": "https://example.com",
        "title": "Sample Title",
        "source": "demo-source",
        "published_at": iso_utc_now(),
        "author": "demo-author",
        "summary": "A short summary about a UI/UX topic.",
        "image_url": "https://example.com/cover.jpg",
        "fulltext": "Full article text goes here...",
        "text_embedding": text_emb,
        "image_embedding": img_emb,
        "labels": "demo,example"
    }
    upsert_post(demo)

    rows = get_latest_posts(3)
    log("\nLatest rows:")
    for r in rows:
        log(r)

    emb = get_text_embedding("demo1")
    if emb is not None:
        log(f"\nRecovered text embedding shape: ({emb.shape[0]},) dtype=float32")
    log("\nAll good ✅")
