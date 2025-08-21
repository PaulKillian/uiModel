#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute momentum for each topic's weekly series.
Momentum = slope of counts over last W points (polyfit deg=1), default W=6.
Usage:
  python trends_momentum.py [--window 6]
"""
import os, sqlite3, argparse
import numpy as np
import pandas as pd

DB_PATH = os.path.abspath("trends.db")

def compute_momentum(counts, window=6):
    if len(counts) < 2:
        return 0.0
    w = min(window, len(counts))
    y = np.array(counts[-w:], dtype=float)
    x = np.arange(w, dtype=float)
    try:
        m = np.polyfit(x, y, 1)[0]
    except Exception:
        m = 0.0
    return float(m)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=6)
    args = ap.parse_args()

    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT topic_id, week, count FROM topic_timeseries ORDER BY week ASC", con, parse_dates=["week"])
    if df.empty:
        print("No topic_timeseries. Run cluster_topics.py first.")
        return
    out = []
    for tid, g in df.groupby("topic_id"):
        g = g.sort_values("week")
        mom = compute_momentum(g["count"].tolist(), window=args.window)
        for _, r in g.iterrows():
            out.append((int(tid), r["week"].isoformat(), int(r["count"]), mom))
    cur = con.cursor()
    cur.execute("DELETE FROM topic_timeseries")
    cur.executemany("INSERT INTO topic_timeseries(topic_id,week,count,momentum) VALUES (?,?,?,?)", out)
    con.commit()
    con.close()
    print("Momentum updated for", df["topic_id"].nunique(), "topics.")

if __name__ == "__main__":
    main()
