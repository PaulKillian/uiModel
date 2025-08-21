#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, re, sqlite3, pandas as pd

DB_PATH = os.path.abspath("trends.db")

STYLES = {
    "Minimalist": r"\bminimal(?:ism|ist)?\b",
    "Glassmorphism": r"\b(?:glassmorphism|frosted\s+glass|translucent)\b",
    "Neumorphism": r"\bneumorphism\b",
    "Brutalist": r"\bbrutalist\b",
    "Retro": r"\bretro\b",
    "Vaporwave": r"\bvaporwave\b",
    "Material You": r"\bmaterial\s+(?:you|design\s*3|m3|md3)\b",
}

FRAMEWORKS = {
    "Tailwind": r"\btailwind\b",
    "Radix": r"\bradix\b",
    "Material": r"\bmaterial\s+(?:design|you|m3|md3)\b",
    "Framer Motion": r"\bframer\s+motion\b",
    "Shadcn": r"\bshadcn\b",
    "Chakra UI": r"\bchakra\s+ui\b",
    "MUI": r"\bmui\b",
    "Next.js": r"\bnext\.js\b",
    "React": r"\breact\b",
    "Svelte": r"\bsvelte\b",
    "Vue": r"\bvue(?:\.js)?\b",
    "SwiftUI": r"\bswiftui\b",
    "Flutter": r"\bflutter\b",
    "Three.js": r"\bthree\.js\b",
    "Angular": r"\bangular\b"
}

COMPONENTS = [
    "button","dialog","tooltip","table","tabs","accordion","toast","skeleton","grid","typography","color","token",
    "avatar","breadcrumb","dropdown","modal","carousel","pagination","chip","badge","stepper","progress","spinner",
    "datepicker","timepicker","select","combobox","autocomplete","menubar","timeline","kanban","tree","editor",
    "navbar","sidebar","slider","switch","context menu","breadcrumbs","list"
]

def ensure_tables(con):
    con.execute("CREATE TABLE IF NOT EXISTS ui_component_tags(post_id INTEGER, component TEXT, UNIQUE(post_id, component))")
    con.execute("CREATE TABLE IF NOT EXISTS ui_framework_trends(framework TEXT, week DATE, count INTEGER, UNIQUE(framework, week))")
    con.execute("CREATE TABLE IF NOT EXISTS ui_style_trends(style TEXT, week DATE, count INTEGER, UNIQUE(style, week))")
    con.commit()

def main():
    con = sqlite3.connect(DB_PATH)
    ensure_tables(con)

    posts = pd.read_sql_query(
        "SELECT id, title, summary, published_at FROM posts ORDER BY published_at DESC",
        con, parse_dates=["published_at"]
    )
    if posts.empty:
        print("No posts. Run ingest_rss.py first."); return

    posts["title"] = posts["title"].fillna("")
    posts["summary"] = posts["summary"].fillna("")
    posts["week"] = posts["published_at"].dt.to_period("W").dt.start_time

    cur = con.cursor()
    cur.execute("DELETE FROM ui_style_trends")
    cur.execute("DELETE FROM ui_framework_trends")
    cur.execute("DELETE FROM ui_component_tags")
    con.commit()

    # Styles
    style_rows = []
    for style, pat in STYLES.items():
        rx = re.compile(pat, re.IGNORECASE)
        mask = posts["title"].str.contains(rx, regex=True, na=False) | posts["summary"].str.contains(rx, regex=True, na=False)
        if mask.any():
            agg = posts.loc[mask].groupby("week").size().reset_index(name="count")
            for _, r in agg.iterrows():
                style_rows.append((style, r["week"].isoformat(), int(r["count"])))
    if style_rows:
        cur.executemany("INSERT INTO ui_style_trends(style,week,count) VALUES (?,?,?)", style_rows)

    # Frameworks
    fw_rows = []
    for fw, pat in FRAMEWORKS.items():
        rx = re.compile(pat, re.IGNORECASE)
        mask = posts["title"].str.contains(rx, regex=True, na=False) | posts["summary"].str.contains(rx, regex=True, na=False)
        if mask.any():
            agg = posts.loc[mask].groupby("week").size().reset_index(name="count")
            for _, r in agg.iterrows():
                fw_rows.append((fw, r["week"].isoformat(), int(r["count"])))
    if fw_rows:
        cur.executemany("INSERT INTO ui_framework_trends(framework,week,count) VALUES (?,?,?)", fw_rows)

    # Components
    comp_rows = []
    for _, p in posts.iterrows():
        txt = f"{p['title']} {p['summary']}".lower()
        for comp in COMPONENTS:
            pattern = re.compile(rf"\\b{re.escape(comp.lower())}s?\\b")
            if pattern.search(txt):
                comp_rows.append((int(p["id"]), comp))
    if comp_rows:
        cur.executemany("INSERT INTO ui_component_tags(post_id,component) VALUES (?,?)", comp_rows)

    con.commit(); con.close()
    print("UI tags updated: styles, frameworks, components.")

if __name__ == "__main__":
    main()
