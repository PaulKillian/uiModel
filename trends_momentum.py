#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute rolling momentum (slope over last K weeks) for EVERY (topic_id, week)
and write it into topic_timeseries. Also print a Top Rising Trends preview
for the latest week.

Usage:
    python trends_momentum.py --K 8 --print_top 10
"""

import os
import sys
import argparse
import sqlite3
from datetime import datetime
import numpy as np
import pandas as pd

DB = os.path.abspath("trends.db")

def log(msg: str):
    print(msg, flush=True)

def connect_db():
    if not os.path.exists(DB):
        raise FileNotFoundError("DB not found: %s" % DB)
    return sqlite3.connect(DB)

def ensure_tables(con):
    con.execute("""
    CREATE TABLE IF NOT EXISTS topic_timeseries (
        topic_id INTEGER NOT NULL,
        week TIMESTAMP NOT NULL,
        count INTEGER NOT NULL,
        momentum REAL,
        PRIMARY KEY (topic_id, week)
    )
    """)
    con.execute("""
    CREATE TABLE IF NOT EXISTS topics (
        id INTEGER PRIMARY KEY,
        name TEXT,
        keywords TEXT,
        category TEXT
    )
    """)
    con.commit()

def fetch_series(con) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT topic_id, week, count, momentum FROM topic_timeseries ORDER BY week ASC",
        con,
        parse_dates=["week"],
    )
    return df

def fetch_topic_names(con) -> pd.DataFrame:
    return pd.read_sql_query("SELECT id AS topic_id, name FROM topics", con)

def compute_slope(y_array: np.ndarray) -> float:
    """
    Linear regression slope over indices 0..n-1 for counts.
    Returns 0.0 if fewer than 2 valid points.
    """
    y = np.asarray(y_array, dtype=float)
    if y.size < 2 or np.all(np.isnan(y)):
        return 0.0
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return 0.0
    x = np.arange(y.size, dtype=float)[mask]
    y = y[mask]
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)

def rebuild_and_rank(K: int = 8, print_top: int = 10):
    con = connect_db()
    try:
        ensure_tables(con)
        df = fetch_series(con)
        if df.empty:
            log("No topic_timeseries data to compute momentum.")
            return

        # Compute rolling slope for EVERY week per topic
        updates = []  # list of (momentum, topic_id, week_iso)
        for tid, g in df.groupby("topic_id"):
            g = g.sort_values("week").reset_index(drop=True)
            counts = g["count"].astype(float).to_numpy()

            for i in range(len(g)):
                start = max(0, i - (K - 1))
                window = counts[start:i+1]
                slope = compute_slope(window)
                wk = g.loc[i, "week"]
                updates.append((float(slope), int(tid), wk.isoformat()))

        # Write all momentum values
        cur = con.cursor()
        cur.executemany(
            "UPDATE topic_timeseries SET momentum = ? WHERE topic_id = ? AND week = ?",
            updates
        )
        con.commit()
        log("Updated momentum for %d (topic, week) rows (rolling K=%d)." % (len(updates), K))

        # Preview: top rising trends at the latest week
        latest_week = df["week"].max()
        names = fetch_topic_names(con)
        latest = pd.read_sql_query(
            "SELECT topic_id, week, count, momentum FROM topic_timeseries WHERE week = ?",
            con,
            params=(latest_week.isoformat(),),
            parse_dates=["week"]
        )
        if latest.empty:
            log("\nTop Rising Trends:\n(none)")
            return

        latest = latest.merge(names, on="topic_id", how="left")

        def safe_name(row):
            base = row.get("name")
            if base is None or str(base).strip() == "":
                return "Topic #%d" % int(row["topic_id"])
            return str(base)

        latest["readable"] = latest.apply(safe_name, axis=1)
        latest["momentum"] = latest["momentum"].astype(float).fillna(0.0)
        latest["count"] = latest["count"].astype(float).fillna(0.0).astype(int)
        latest = latest.sort_values(["momentum", "count"], ascending=[False, False])

        log("\nTop Rising Trends:\n")
        top = latest.head(int(print_top))
        for _, r in top.iterrows():
            readable = str(r["readable"])
            week_str = r["week"].date().isoformat() if hasattr(r["week"], "date") else str(r["week"])[:10]
            count = int(r["count"])
            mom = float(r["momentum"])
            name30 = (readable[:30] + "…") if len(readable) > 31 else readable
            log("  %-31s  week=%s  count=%3d  momentum=%+0.2f" % (name30, week_str, count, mom))

    finally:
        con.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=8, help="Lookback window (weeks) for rolling slope")
    ap.add_argument("--print_top", type=int, default=10, help="How many rows to print in the preview")
    args = ap.parse_args()

    rebuild_and_rank(K=int(args.K), print_top=int(args.print_top))

if __name__ == "__main__":
    main()
