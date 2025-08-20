#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build a multimodal index:
- Per-post combined vector (text + image) stored in a temp table
- Per-topic centroid written to topic_vectors(modality='multimodal')

Usage:
  python multimodal_index.py --alpha 1.0 --beta 1.0
"""

import os, sqlite3, argparse, numpy as np

DB = os.path.abspath("trends.db")

def log(m): print(m, flush=True)

def buf_to_vec(b):
    if b is None: return None
    # text emb is 384-d (MiniLM), image emb is 512-d (CLIP)
    # We'll pad the shorter to the longer so we can sum.
    return np.frombuffer(b, dtype=np.float32)

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def pad_to(v, d):
    if v is None: return None
    if v.shape[0] == d: return v
    if v.shape[0] > d: return v[:d]
    out = np.zeros(d, dtype=np.float32)
    out[:v.shape[0]] = v
    return out

def main(alpha, beta):
    if not os.path.exists(DB):
        raise FileNotFoundError(DB)
    con = sqlite3.connect(DB)
    cur = con.cursor()

    # Tables
    cur.execute("""
    CREATE TABLE IF NOT EXISTS topic_vectors (
        topic_id INTEGER PRIMARY KEY,
        dim INTEGER NOT NULL,
        vec BLOB NOT NULL,
        modality TEXT NOT NULL
    )
    """)
    con.commit()

    # Pull posts joined to topics
    q = """
    SELECT p.id, pt.topic_id, p.text_embedding, p.image_embedding
    FROM post_topics pt
    JOIN posts p ON p.id = pt.post_id
    """
    rows = cur.execute(q).fetchall()
    if not rows:
        log("No post_topics rows; run clustering first.")
        return

    # Detect dimensions present
    dim_text = 384  # MiniLM
    dim_image = 512 # CLIP
    dim = max(dim_text, dim_image)

    post_vectors = []
    for pid, tid, tblob, iblob in rows:
        t = buf_to_vec(tblob)
        i = buf_to_vec(iblob)

        if t is not None: t = pad_to(t, dim)
        if i is not None: i = pad_to(i, dim)

        if t is None and i is None:
            continue
        if t is None: v = normalize(beta * i)
        elif i is None: v = normalize(alpha * t)
        else: v = normalize(alpha * t + beta * i)

        post_vectors.append((pid, tid, v))

    if not post_vectors:
        log("No usable post vectors (no embeddings found).")
        return

    # Build topic centroids
    sums = {}
    counts = {}
    for _, tid, v in post_vectors:
        sums[tid] = sums.get(tid, np.zeros(dim, dtype=np.float32)) + v
        counts[tid] = counts.get(tid, 0) + 1

    # Write topic_vectors
    wrote = 0
    for tid, s in sums.items():
        c = counts[tid]
        centroid = normalize(s / max(c, 1))
        cur.execute("""
            INSERT INTO topic_vectors (topic_id, dim, vec, modality)
            VALUES (?, ?, ?, 'multimodal')
            ON CONFLICT(topic_id) DO UPDATE SET
              dim = excluded.dim,
              vec = excluded.vec,
              modality = 'multimodal'
        """, (int(tid), dim, centroid.astype(np.float32).tobytes()))
        wrote += 1

    con.commit()
    con.close()
    log("Wrote %d topic centroids (dim=%d, modality=multimodal)." % (wrote, dim))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=1.0, help="weight for text")
    ap.add_argument("--beta", type=float, default=1.0, help="weight for image")
    args = ap.parse_args()
    main(args.alpha, args.beta)
