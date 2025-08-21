#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scan posts for AI art (generative image) trends and write weekly counts into separate tables:

- ai_art_model_trends(model, week, count)
- ai_art_tool_trends(tool, week, count)
- ai_art_technique_trends(technique, week, count)
- ai_art_style_trends(style, week, count)

Usage:
  python tag_ai_art.py
"""
import os
import re
import sqlite3
import pandas as pd

DB_PATH = os.path.abspath("trends.db")

# -------- Dictionaries of patterns (non-capturing groups, case-insensitive) --------
MODELS = {
    "Stable Diffusion": r"\bstable\s+diffusion\b|\bsd(?:\s*1\.5|xl|xlt|xl\s*turbo)\b",
    "Midjourney": r"\bmid\s*journey\b|\bmj\s*v?\d+\b",
    "FLUX": r"\bflux\b",
    "DALL·E": r"\bdall[\.\- ]?e\b",
    "Kandinsky": r"\bkandinsky\b",
    "Imagen": r"\bimagen\b",
    "DeepFloyd IF": r"\bdeepfloyd\b|\bif\s*model\b",
    "Playground": r"\bplayground\s*ai\b",
    "Firefly": r"\b(adobe\s*)?firefly\b",
}

TOOLS = {
    "ComfyUI": r"\bcomfy\s*ui\b",
    "Automatic1111": r"\bautomatic\s*1111\b|\ba1111\b",
    "InvokeAI": r"\binvoke\s*ai\b",
    "Fooocus": r"\bfooocus\b",
    "Runway": r"\brunway\s*ml?\b|\brunway\b",
    "Leonardo AI": r"\bleonardo\s*ai\b",
    "NightCafe": r"\bnight\s*cafe\b",
    "Clipdrop": r"\bclipdrop\b",
    "Replicate": r"\breplicate\.com\b|\breplicate\b",
}

TECHNIQUES = {
    "ControlNet": r"\bcontrol\s*net\b",
    "LoRA": r"\blora\b|\bloc(?:on)?\b|\btextual\s*inversion\b",
    "Inpainting": r"\bin-?painting\b",
    "Outpainting": r"\bout-?painting\b",
    "Img2Img": r"\bimg2img\b|\bimage\s*to\s*image\b",
    "Text2Img": r"\btext2img\b|\btext\s*to\s*image\b",
    "Upscaling": r"\bupscal(?:e|ing|er)\b|\bsuper\s*resolution\b",
    "IP-Adapter": r"\bip-?adapter\b",
    "Depth/Normal": r"\bdepth\s*(?:to)?\s*img\b|\bnormal\s*map\b",
    "Region Prompting": r"\b(?:region|regional)\s*prompt\w*",  # non-capturing to avoid pandas warning
}

STYLES = {
    "Photorealistic": r"\bphoto-?realistic\b|\bphotoreal\b",
    "Anime/Manga": r"\banime\b|\bmanga\b",
    "3D Render": r"\b3d\s*render\b|\boctane\b|\bblender\b",
    "Vector / Flat": r"\bvector\b|\bflat\s*design\b",
    "Isometric": r"\bisometric\b",
    "Pixel Art": r"\bpixel\s*art\b",
    "Watercolor": r"\bwatercolor\b",
    "Oil Painting": r"\boil\s*painting\b|\bimpasto\b",
    "Line Art": r"\bline\s*art\b",
    "Low Poly": r"\blow\s*poly\b",
    "Cartoon / Comic": r"\bcartoon\b|\bcomic\b",
    "Cyberpunk": r"\bcyberpunk\b",
    "Noir": r"\bnoir\b",
    "Retro": r"\bretro\b|\b80s\b|\b90s\b|\bsynthwave\b|\bvaporwave\b",
}

def ensure_tables(con: sqlite3.Connection):
    con.execute("""CREATE TABLE IF NOT EXISTS ai_art_model_trends(
        model TEXT, week DATE, count INTEGER, UNIQUE(model, week)
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS ai_art_tool_trends(
        tool TEXT, week DATE, count INTEGER, UNIQUE(tool, week)
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS ai_art_technique_trends(
        technique TEXT, week DATE, count INTEGER, UNIQUE(technique, week)
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS ai_art_style_trends(
        style TEXT, week DATE, count INTEGER, UNIQUE(style, week)
    )""")
    con.commit()

def trend_from_dict(df: pd.DataFrame, mapping: dict, col_name: str, table: str, con: sqlite3.Connection):
    rows = []
    for label, pat in mapping.items():
        rx = re.compile(pat, re.IGNORECASE)
        mask = df["title"].str.contains(rx, regex=True, na=False) | df["summary"].str.contains(rx, regex=True, na=False)
        if mask.any():
            agg = df.loc[mask].groupby("week").size().reset_index(name="count")
            for _, r in agg.iterrows():
                rows.append((label, r["week"].isoformat(), int(r["count"])))
    if rows:
        cur = con.cursor()
        cur.execute(f"DELETE FROM {table}")
        cur.executemany(f"INSERT OR REPLACE INTO {table}({col_name},week,count) VALUES (?,?,?)", rows)
        con.commit()

def main():
    con = sqlite3.connect(DB_PATH)
    ensure_tables(con)

    posts = pd.read_sql_query(
        "SELECT id, title, summary, published_at FROM posts ORDER BY published_at DESC",
        con, parse_dates=["published_at"]
    )
    if posts.empty:
        print("No posts. Run ingest_rss.py first.")
        return

    posts["title"] = posts["title"].fillna("")
    posts["summary"] = posts["summary"].fillna("")

    # Timezone-safe weekly floor: convert tz-aware -> UTC -> naive, then to Period
    if pd.api.types.is_datetime64tz_dtype(posts["published_at"]):
        pub_naive = posts["published_at"].dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        pub_naive = posts["published_at"]
    posts["week"] = pub_naive.dt.to_period("W").dt.start_time

    # Build trends
    trend_from_dict(posts, MODELS, "model", "ai_art_model_trends", con)
    trend_from_dict(posts, TOOLS, "tool", "ai_art_tool_trends", con)
    trend_from_dict(posts, TECHNIQUES, "technique", "ai_art_technique_trends", con)
    trend_from_dict(posts, STYLES, "style", "ai_art_style_trends", con)

    con.close()
    print("AI art trends updated: models, tools, techniques, styles.")

if __name__ == "__main__":
    main()
