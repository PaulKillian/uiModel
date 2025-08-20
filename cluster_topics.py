#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cluster posts into topics and MERGE near-duplicates conservatively so the
dashboard shows one card per real topic (no over-merged 'Misc').

- Reads posts + (optional) precomputed text embeddings from trends.db
- Fits BERTopic
- Builds strict, compact topic names
- Merges topics only when genuinely similar (high cosine)
- Writes:
    * topics (upsert)
    * post_topics (one best topic per post; merged ids)
    * topic_timeseries (counts per week; momentum computed in trends_momentum.py)

Usage:
    python cluster_topics.py --min_topic_size 8 --similarity 0.95 --days 0
"""

import os
import sys
import math
import argparse
import sqlite3
import datetime as dt
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# BERTopic stack
from bertopic import BERTopic
import umap
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional embedding model (only used if DB has no text_embedding)
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


DB = os.path.abspath("trends.db")


# ---------------------------
# Utils
# ---------------------------
def log(msg: str):
    print(str(msg), flush=True)


def week_start(d: dt.datetime) -> dt.datetime:
    if isinstance(d, str):
        try:
            d = dt.datetime.fromisoformat(d.replace("Z", ""))
        except Exception:
            d = dt.datetime.utcnow()
    if isinstance(d, (dt.date,)):
        d = dt.datetime(d.year, d.month, d.day)
    return d - dt.timedelta(days=d.weekday())  # Monday


def connect_db():
    if not os.path.exists(DB):
        raise FileNotFoundError("DB not found: %s" % DB)
    return sqlite3.connect(DB)


# ---------------------------
# Load documents
# ---------------------------
def load_posts(con, days: int = 0) -> pd.DataFrame:
    q = """
    SELECT id, COALESCE(fulltext, '') AS fulltext, COALESCE(title,'') AS title,
           COALESCE(url,'') AS url, COALESCE(published_at,'') AS published_at,
           text_embedding
    FROM posts
    """
    if days and days > 0:
        q += " WHERE DATE(published_at) >= DATE('now', ?)"
        df = pd.read_sql_query(q, con, params=(f'-{int(days)} days',))
    else:
        df = pd.read_sql_query(q, con)
    return df


def pick_text(row: pd.Series) -> str:
    t = (row.get("fulltext") or "").strip()
    if len(t) < 140:
        t = f"{row.get('title','')} {t}".strip()
    return (t or row.get("title") or "").strip()


# ---------------------------
# Strict label building
# ---------------------------
STOP = {
    # function words only (do NOT include domain terms like design/product/ai)
    "the","and","or","a","an","of","in","on","for","to","with","by","from",
    "this","that","these","those","you","your","we","our","it","its",
    "be","is","are","was","were","will","can","could","should","would",
    "about","into","across","at","over","under","more","most","less","least",
    "how","why","what","when","where",
}

CANON_MAP = [
    ("performance", "Performance"),
    ("perf", "Performance"),
    ("latency", "Performance"),
    ("speed", "Performance"),
    ("animation", "Animation"),
    ("motion", "Motion"),
    ("microinteraction", "Motion"),
    ("accessibility", "Accessibility"),
    ("a11y", "Accessibility"),
    ("token", "Design Tokens"),
    ("tokens", "Design Tokens"),
    ("typography", "Typography"),
    ("layout", "Layout"),
    ("grid", "Layout"),
    ("navigation", "Navigation"),
    ("nav", "Navigation"),
    ("form", "Forms"),
    ("forms", "Forms"),
    ("research", "User Research"),
    ("usability", "Usability"),
    ("personalization", "Personalization"),
    ("dark mode", "Theming"),
    ("theme", "Theming"),
    ("branding", "Branding"),
    ("color", "Color"),
    ("colors", "Color"),
    ("image", "Imagery"),
    ("images", "Imagery"),
    ("illustration", "Illustration"),
    ("icons", "Iconography"),
    ("icon", "Iconography"),
]

def canon_category(words_lower: list[str]) -> str | None:
    text = " ".join(words_lower)
    for key, label in CANON_MAP:
        if key in text:
            return label
    return None


def build_strict_label(topic_words: list[tuple[str, float]], top_k: int = 6):
    """
    topic_words: list of (word, weight) from BERTopic.get_topic(id)
    - Remove only function-word stopwords
    - Keep top_k distinct tokens
    - Build signature for dedup (sorted tokens joined by '_')
    - Choose compact label:
        * If a canonical bucket is detected -> that label
        * else join top 2-3 tokens as Title Case
    """
    toks = []
    for w, _ in topic_words[: max(12, top_k)]:
        w = (w or "").strip().lower().replace("-", " ")
        if not w:
            continue
        toks.extend([t for t in w.split() if t and t not in STOP])

    if not toks:
        # fall back to raw top terms (lowercased)
        toks = [ (topic_words[i][0].lower()) for i in range(min(6, len(topic_words))) if topic_words[i][0] ]

    counts = Counter(toks)
    core = [w for w, _ in counts.most_common(top_k)]
    if not core:
        core = ["general"]  # never "misc"

    cat = canon_category(core)

    if cat:
        label = cat
    else:
        label = " ".join([w.title() for w in core[:3]])

    keywords = ", ".join(core)
    category = cat or "Other"
    signature = "_".join(sorted(core))

    return label, keywords, category, signature


# ---------------------------
# Merge topics by similarity
# ---------------------------
def merge_topics_by_similarity(labels: list[str], keywords_list: list[str], threshold: float = 0.95):
    """
    Use TF-IDF over [label + keywords] strings, merge by cosine >= threshold.
    For generic/weak topics require an even higher threshold (0.97).
    Returns: list[int] parent index for each topic (union-find style)
    """
    texts = [(labels[i] + " " + (keywords_list[i] or "")) for i in range(len(labels))]
    if len(texts) <= 1:
        return list(range(len(texts)))

    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1).fit_transform(texts)
    sim = cosine_similarity(vec)

    parent = list(range(len(texts)))

    def is_generic(i: int) -> bool:
        lbl = (labels[i] or "").strip().lower()
        kws = (keywords_list[i] or "").strip().lower()
        generic_labels = {"misc","other","general"}
        few_kws = len([k for k in kws.split(",") if k.strip()]) < 3
        return lbl in generic_labels or few_kws

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    n = len(texts)
    for i in range(n):
        for j in range(i + 1, n):
            thr = threshold
            if is_generic(i) or is_generic(j):
                thr = max(threshold, 0.97)
            if sim[i, j] >= thr:
                union(i, j)

    for i in range(n):
        parent[i] = find(i)
    return parent


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_topic_size", type=int, default=8, help="HDBSCAN min_cluster_size")
    ap.add_argument("--days", type=int, default=0, help="Only include posts from the last N days (0 = all)")
    ap.add_argument("--similarity", type=float, default=0.95, help="Similarity threshold for merging topics")
    ap.add_argument("--dry_run", action="store_true", help="Compute only; do not write DB")
    args = ap.parse_args()

    log("DB:  " + DB)
    con = connect_db()

    # Load posts
    posts = load_posts(con, args.days)
    if posts.empty:
        log("No posts found.")
        sys.exit(0)

    posts["text"] = posts.apply(pick_text, axis=1)
    docs = posts["text"].fillna("").tolist()

    # Embeddings
    emb = None
    if "text_embedding" in posts.columns and posts["text_embedding"].notna().any():
        try:
            emb = np.vstack([
                np.frombuffer(x, dtype=np.float32) if x else np.zeros(384, dtype=np.float32)
                for x in posts["text_embedding"].tolist()
            ])
            log("Loaded %d documents; embedding shape=%s" % (len(docs), str(emb.shape)))
        except Exception:
            emb = None

    if emb is None:
        if SentenceTransformer is None:
            log("No precomputed embeddings and sentence-transformers unavailable. Exiting.")
            sys.exit(1)
        log("Computing embeddings with all-MiniLM-L6-v2 ...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = model.encode(docs, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    # BERTopic
    log("Fitting BERTopic... (this can take a minute)")
    umap_model = umap.UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=args.min_topic_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True
    )
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        top_n_words=10,
        calculate_probabilities=False,
        verbose=True,
        min_topic_size=None,  # handled by HDBSCAN
    )

    topics, _ = topic_model.fit_transform(docs, emb)

    # Collect raw topic metadata (exclude outliers -1)
    topic_ids = sorted([t for t in set(topics) if t != -1])
    if not topic_ids:
        log("No topics discovered (all outliers). Try reducing --min_topic_size.")
        sys.exit(0)

    raw = []
    for tid in topic_ids:
        words = topic_model.get_topic(tid) or []
        label, keywords, category, signature = build_strict_label(words)
        raw.append({
            "topic_id": int(tid),
            "label": label,
            "keywords": keywords,
            "category": category,
            "signature": signature,
        })
    raw_df = pd.DataFrame(raw).sort_values("topic_id").reset_index(drop=True)

    # Merge highly similar topics (conservative)
    parents = merge_topics_by_similarity(
        labels=raw_df["label"].tolist(),
        keywords_list=raw_df["keywords"].tolist(),
        threshold=float(args.similarity),
    )
    raw_df["parent_idx"] = parents

    # Map parent index -> canonical topic_id (smallest original id in that group)
    group_map = {}
    for parent_idx, g in raw_df.groupby("parent_idx"):
        canonical_tid = int(g["topic_id"].min())
        for tid in g["topic_id"]:
            group_map[int(tid)] = canonical_tid

    # Remap per-post topic
    topics_merged = [group_map.get(int(t), -1) if t != -1 else -1 for t in topics]

    # Build final label for each canonical topic id
    final_rows = []
    for canon_idx, g in raw_df.groupby(raw_df["parent_idx"]):
        orig_ids = g["topic_id"].tolist()
        all_kw = []
        for tid in orig_ids:
            ws = topic_model.get_topic(tid) or []
            all_kw.extend([w for w, _ in ws[:10]])

        _, keywords, category, signature = build_strict_label([(w, 1.0) for w in all_kw], top_k=6)
        cat_label = canon_category([w.lower() for w in keywords.split(", ")]) or None
        if cat_label:
            label = cat_label
        else:
            parts = [w.title() for w in keywords.split(", ")[:3] if w]
            label = " ".join(parts) if parts else "General"

        final_rows.append({
            "topic_id": int(min(orig_ids)),
            "name": label,
            "keywords": keywords,
            "category": category,
        })

    final_df = pd.DataFrame(final_rows).sort_values("topic_id").reset_index(drop=True)

    log("Discovered %d merged topics from %d raw topics." % (len(final_df), len(topic_ids)))

    if args.dry_run:
        log("Dry run complete.")
        sys.exit(0)

    # ---------------------------
    # WRITE DB
    # ---------------------------
    cur = con.cursor()

    # 1) Upsert topics
    cur.execute("""
    CREATE TABLE IF NOT EXISTS topics (
        id INTEGER PRIMARY KEY,
        name TEXT,
        keywords TEXT,
        category TEXT
    )
    """)
    for _, r in final_df.iterrows():
        cur.execute("SELECT 1 FROM topics WHERE id = ?", (int(r["topic_id"]),))
        if cur.fetchone():
            cur.execute("""
                UPDATE topics SET name = ?, keywords = ?, category = ?
                WHERE id = ?
            """, (r["name"], r["keywords"], r["category"], int(r["topic_id"])))
        else:
            cur.execute("""
                INSERT INTO topics (id, name, keywords, category)
                VALUES (?, ?, ?, ?)
            """, (int(r["topic_id"]), r["name"], r["keywords"], r["category"]))

    # 2) post_topics (rebuild cleanly)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS post_topics (
        post_id TEXT NOT NULL,
        topic_id INTEGER NOT NULL,
        PRIMARY KEY (post_id)
    )
    """)
    cur.execute("DELETE FROM post_topics")
    for pid, t in zip(posts["id"].tolist(), topics_merged):
        if t == -1:
            continue
        cur.execute("INSERT OR REPLACE INTO post_topics (post_id, topic_id) VALUES (?, ?)", (pid, int(t)))

    # 3) topic_timeseries (rebuild counts per week; momentum computed later)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS topic_timeseries (
        topic_id INTEGER NOT NULL,
        week TIMESTAMP NOT NULL,
        count INTEGER NOT NULL,
        momentum REAL,
        PRIMARY KEY (topic_id, week)
    )
    """)
    cur.execute("DELETE FROM topic_timeseries")

    joined = posts[["id","published_at"]].merge(
        pd.DataFrame({"id": posts["id"].tolist(), "topic": topics_merged}),
        on="id"
    )
    joined = joined[joined["topic"] != -1].copy()

    def _week(x):
        try:
            return week_start(x)
        except Exception:
            return week_start(dt.datetime.utcnow())

    joined["week"] = joined["published_at"].apply(_week)

    grp = joined.groupby(["topic","week"]).size().reset_index(name="count")
    for _, r in grp.iterrows():
        cur.execute("""
        INSERT OR REPLACE INTO topic_timeseries (topic_id, week, count, momentum)
        VALUES (?, ?, ?, COALESCE((
            SELECT momentum FROM topic_timeseries WHERE topic_id = ? AND week = ?
        ), NULL))
        """, (int(r["topic"]), r["week"].isoformat(), int(r["count"]), int(r["topic"]), r["week"].isoformat()))

    con.commit()

    # ---------------------------
    # Report
    # ---------------------------
    log("Updated topics, post_topics, topic_timeseries.")

    show = final_df.copy()
    show["topic_id"] = show["topic_id"].astype(int)
    show = show.sort_values(["category","topic_id"])
    log("\nMerged Topics:")
    for _, r in show.iterrows():
        log("  #%3d  %-18s  [%s]  :: %s" % (
            int(r["topic_id"]),
            (r["name"] or "")[:18],
            r["category"] or "Other",
            r["keywords"] or ""
        ))

    log("\nDone.")
    con.close()


if __name__ == "__main__":
    main()
