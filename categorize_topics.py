#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, re, sqlite3, pandas as pd

DB_PATH = os.path.abspath("trends.db")

CATS = [
    ("Design Systems", r"\b(design\s*system|tokens?|style\s*guide|material|fluent|tailwind|radix|mui|chakra|shadcn)\b"),
    ("Components", r"\b(button|dialog|modal|tooltip|table|tabs?|accordion|toast|skeleton|avatar|navbar|sidebar|carousel|pagination|chip|badge|stepper|progress|spinner|datepicker|timepicker|select|combobox|autocomplete|menubar|breadcrumb|breadcrumbs|list|grid)\b"),
    ("Layout & Navigation", r"\b(layout|navigation|nav|grid|columns?|breakpoints?|responsive|breadcrumbs?)\b"),
    ("Motion & Animation", r"\b(animation|motion|framer|gsap|lottie|micro-?interaction)\b"),
    ("Accessibility", r"\b(a11y|accessibility|aria|contrast|screen\s*reader|wcag)\b"),
    ("Color & Typography", r"\b(color|palette|contrast|typography|font|variable\s*fonts?)\b"),
    ("Prototyping & Handoff", r"\b(figma|framer|invision|zeplin|handoff|prototype)\b"),
    ("Mobile UI", r"\b(swiftui|flutter|cupertino|material\s+you)\b"),
    ("3D & WebGL", r"\b(three\.js|webgl|babylon)\b"),
    ("Data Visualization", r"\b(chart|graphs?|d3|echarts|plotly|datavis|visualization)\b"),
    ("Forms & Input", r"\b(form|validation|input|fields?|autocomplete)\b"),
    ("Performance", r"\b(performance|bundle|optimi[sz]e|lazy\s*load)\b"),
    ("Internationalization", r"\b(i18n|l10n|rtl|locali[sz]ation)\b"),
]

def main():
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT id, name, COALESCE(keywords,'') AS kw FROM topics", con)
    if df.empty:
        print("No topics. Run clustering first."); return

    updates = []
    for _, r in df.iterrows():
        text = f"{r['name']} {r['kw']}".lower()
        cat = "UI/UX"
        for cname, pat in CATS:
            if re.search(pat, text, flags=re.IGNORECASE):
                cat = cname; break
        updates.append((cat, int(r["id"])))

    cur = con.cursor()
    cur.executemany("UPDATE topics SET category=? WHERE id=?", updates)
    con.commit(); con.close()
    print(f"Updated {len(updates)} topics with UI categories.")

if __name__ == "__main__":
    main()
