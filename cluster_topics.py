#!/usr/bin/env python
print('cluster placeholder')
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cluster posts into topics using TF-IDF + KMeans.
Creates/overwrites `topics` and `post_topics` and drafts weekly counts (topic_timeseries).
Usage:
  python cluster_topics.py [--n_clusters 10] [--min_topic_size 5]
"""
import os, sqlite3, argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

DB_PATH = os.path.abspath("trends.db")

def ensure_tables(con):
    con.execute("CREATE TABLE IF NOT EXISTS topics (id INTEGER PRIMARY KEY, name TEXT, keywords TEXT, category TEXT)")
    con.execute("CREATE TABLE IF NOT EXISTS post_topics (post_id INTEGER, topic_id INTEGER)")
    con.execute("CREATE TABLE IF NOT EXISTS topic_timeseries (topic_id INTEGER, week TIMESTAMP, count INTEGER, momentum REAL)")
    con.commit()

def fetch_posts(con):
    return pd.read_sql_query("SELECT id, title, source, published_at, COALESCE(summary, '') AS content FROM posts ORDER BY published_at ASC", con, parse_dates=["published_at"])

def k_for_size(n, min_topic_size=5):
    if n <= min_topic_size:
        return 1
    return max(1, min(20, n // max(3, min_topic_size)))

def top_terms_for_cluster(X, labels, vocab, k, topn=4):
    names = []
    for c in range(k):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            names.append(f"Topic {c}")
            continue
        sub = X[idx].mean(axis=0).A1
        top_idx = sub.argsort()[::-1][:topn]
        terms = [vocab[i] for i in top_idx]
        names.append(", ".join(terms))
    return names

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_clusters", type=int, default=None)
    ap.add_argument("--min_topic_size", type=int, default=5)
    args = ap.parse_args()

    con = sqlite3.connect(DB_PATH)
    ensure_tables(con)

    posts = fetch_posts(con)
    if posts.empty:
        print("No posts found. Run ingest_rss.py first.")
        return

    texts = (posts["title"].fillna("") + " " + posts["content"].fillna("")).tolist()
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english")
    X = vec.fit_transform(texts)
    k = args.n_clusters if args.n_clusters else k_for_size(len(posts), args.min_topic_size)
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    names = top_terms_for_cluster(X, labels, vec.get_feature_names_out(), k, topn=4)

    cur = con.cursor()
    cur.execute("DELETE FROM topics")
    cur.execute("DELETE FROM post_topics")
    cur.execute("DELETE FROM topic_timeseries")

    for tid, nm in enumerate(names, start=1):
        cur.execute("INSERT INTO topics(id,name,keywords,category) VALUES (?,?,?,?)", (tid, nm, nm, "Auto"))

    for i, r in posts.iterrows():
        cur.execute("INSERT INTO post_topics(post_id,topic_id) VALUES (?,?)", (int(r["id"]), int(labels[i])+1))

    posts["week"] = posts["published_at"].dt.to_period("W").dt.start_time
    tmp = posts.join(pd.Series(labels, name="cluster")).groupby(["cluster","week"]).size().reset_index(name="count")
    for _, row in tmp.iterrows():
        cur.execute("INSERT INTO topic_timeseries(topic_id,week,count,momentum) VALUES (?,?,?,NULL)", (int(row["cluster"])+1, row["week"].isoformat(), int(row["count"])))

    con.commit()
    con.close()
    print(f"Clustering complete. Created {k} topics for {len(posts)} posts.")

if __name__ == "__main__":
    main()
