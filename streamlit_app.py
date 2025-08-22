#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import math
import shutil
import sqlite3
import hashlib
import subprocess
import re
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import requests
from PIL import Image
from urllib.parse import urlparse

# Optional: load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ----------------------------------
# App config
# ----------------------------------
DB_PATH = os.getenv("TRENDS_DB", os.path.abspath("trends.db"))
BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()
CACHE_DIR = BASE_DIR / "image_cache"
CACHE_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="UI/UX Trends", layout="wide")
st.title("UI/UX Trends Dashboard")
st.caption(f"DB: {DB_PATH}")

def goto(page: str):
    st.query_params["page"] = page
    st.rerun()

# ----------------------------------
# Utilities
# ----------------------------------
def momentum_badge(value: float) -> str:
    arrow = "↑" if value > 0.05 else ("→" if abs(value) <= 0.05 else "↓")
    color = "#16a34a" if value > 0.05 else ("#6b7280" if abs(value) <= 0.05 else "#dc2626")
    return f"<span style='font-weight:600;color:{color}'>{arrow} {value:+.2f}</span>"

def run_status(label, args: List[str]):
    with st.status(f"{label}…", expanded=False) as status:
        try:
            proc = subprocess.run([sys.executable, *args], capture_output=True, text=True)
            out = proc.stdout or "(no stdout)"
            err = proc.stderr or ""
            if out.strip():
                st.code(out[-4000:], language="bash")
            if proc.returncode == 0:
                if err.strip():
                    st.caption("(stderr)")
                    st.code(err[-2000:], language="bash")
                status.update(label=f"{label} done", state="complete")
            else:
                st.error(err.strip() or "Unknown error")
                status.update(label=f"{label} failed", state="error")
                st.stop()
        except Exception as e:
            st.error(f"{e}")
            status.update(label=f"{label} failed", state="error")
            st.stop()

def ensure_db():
    if not os.path.exists(DB_PATH):
        st.error(f"Database not found at {DB_PATH}. Run your ingestion/pipeline first.")
        st.stop()

def sanitize_key(s: str, prefix: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9_\-]+", "_", s)
    return f"{prefix}_{s}"[:60]

def key_for_post(prefix: str, *args, idx: int | None = None) -> str:
    """
    Deterministic unique key for Streamlit widgets.
    Combines prefix + args + optional index + short hash.
    """
    parts = [prefix or ""] + [str(a) for a in args if a is not None and str(a) != ""]
    if idx is not None:
        parts.append(str(idx))
    raw = "|".join(parts)
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{h}"

# ----------------------------------
# DB helpers (cached)
# ----------------------------------
@st.cache_data(ttl=60)
def load_topics_with_latest_momentum():
    con = sqlite3.connect(DB_PATH)
    try:
        q = """
        WITH latest AS (SELECT MAX(week) AS max_week FROM topic_timeseries)
        SELECT t.id AS topic_id,
               t.name,
               COALESCE(t.keywords, '') AS keywords,
               COALESCE(t.category, 'Uncategorized') AS category,
               tt.week,
               tt.count,
               ROUND(tt.momentum, 3) AS momentum
        FROM topic_timeseries tt
        JOIN topics t ON t.id = tt.topic_id
        JOIN latest l ON tt.week = l.max_week
        ORDER BY tt.momentum DESC, tt.count DESC
        """
        return pd.read_sql_query(q, con, parse_dates=["week"])
    finally:
        con.close()

@st.cache_data(ttl=60)
def load_topic_series(topic_id: int):
    con = sqlite3.connect(DB_PATH)
    try:
        q = """
        SELECT week, count, COALESCE(momentum, 0) AS momentum
        FROM topic_timeseries
        WHERE topic_id = ?
        ORDER BY week ASC
        """
        return pd.read_sql_query(q, con, params=(topic_id,), parse_dates=["week"])
    finally:
        con.close()

@st.cache_data(ttl=60)
def load_recent_posts_for_topic(topic_id: int, limit: int = 20):
    con = sqlite3.connect(DB_PATH)
    try:
        q = """
        SELECT p.id, p.title, p.url, p.source, p.published_at
        FROM post_topics pt
        JOIN posts p ON p.id = pt.post_id
        WHERE pt.topic_id = ?
        ORDER BY p.published_at DESC
        LIMIT ?
        """
        return pd.read_sql_query(q, con, params=(topic_id, limit))
    finally:
        con.close()

@st.cache_data(ttl=60)
def load_topics_table(limit: int = 200):
    con = sqlite3.connect(DB_PATH)
    try:
        q = """
        WITH latest AS (SELECT MAX(week) AS max_week FROM topic_timeseries)
        SELECT t.id AS topic_id,
               t.name,
               COALESCE(t.keywords, '') AS keywords,
               COALESCE(t.category, 'Uncategorized') AS category,
               tt.week,
               tt.count,
               ROUND(tt.momentum, 3) AS momentum
        FROM topic_timeseries tt
        JOIN topics t ON t.id = tt.topic_id
        JOIN latest l ON tt.week = l.max_week
        ORDER BY t.category ASC, tt.momentum DESC, tt.count DESC
        LIMIT ?
        """
        return pd.read_sql_query(q, con, params=(limit,), parse_dates=["week"])
    finally:
        con.close()

@st.cache_data(ttl=60)
def load_image_sources_from_posts(max_rows: int = 1000):
    con = sqlite3.connect(DB_PATH)
    try:
        q = """
        SELECT id, title, source, image_url
        FROM posts
        WHERE image_url IS NOT NULL AND TRIM(image_url) <> ''
        LIMIT ?
        """
        return pd.read_sql_query(q, con, params=(max_rows,))
    finally:
        con.close()

@st.cache_data(ttl=60)
def load_style_trends():
    con = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql_query("""
            SELECT style, week, count FROM ui_style_trends ORDER BY week ASC
        """, con, parse_dates=["week"])
    finally:
        con.close()

@st.cache_data(ttl=60)
def load_framework_trends():
    con = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql_query("""
            SELECT framework, week, count FROM ui_framework_trends ORDER BY week ASC
        """, con, parse_dates=["week"])
    finally:
        con.close()

@st.cache_data(ttl=60)
def load_component_tags():
    con = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql_query("""
            SELECT c.component, p.published_at AS week
            FROM ui_component_tags c
            JOIN posts p ON p.id = c.post_id
            ORDER BY p.published_at ASC
        """, con, parse_dates=["week"])
    finally:
        con.close()

# New: Posts browser helpers
@st.cache_data(ttl=60)
def load_posts(limit: int = 5000, search: str = "", sources: list[str] | None = None,
               date_from: str | None = None, date_to: str | None = None) -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    try:
        q = "SELECT title, source, published_at, url, image_url, summary FROM posts WHERE 1=1"
        params = []
        if search:
            q += " AND (LOWER(title) LIKE ? OR LOWER(summary) LIKE ?)"
            like = f"%{search.lower()}%"
            params += [like, like]
        if sources:
            placeholders = ",".join("?" * len(sources))
            q += f" AND source IN ({placeholders})"
            params += sources
        if date_from:
            q += " AND published_at >= ?"; params.append(date_from)
        if date_to:
            q += " AND published_at <= ?"; params.append(date_to)
        q += " ORDER BY datetime(published_at) DESC LIMIT ?"
        params.append(int(limit))
        return pd.read_sql_query(q, con, params=params, parse_dates=["published_at"])
    finally:
        con.close()

@st.cache_data(ttl=60)
def load_post_sources() -> list[str]:
    con = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT DISTINCT source FROM posts WHERE source IS NOT NULL ORDER BY source", con)
        return df["source"].dropna().tolist()
    finally:
        con.close()

# ----------------------------------
# OpenAI integration (AI insights)
# ----------------------------------
def ensure_ai_insights_table():
    con = sqlite3.connect(DB_PATH)
    try:
        con.execute("""
        CREATE TABLE IF NOT EXISTS ai_insights (
            topic_id INTEGER PRIMARY KEY,
            generated_at TIMESTAMP NOT NULL,
            model TEXT,
            summary TEXT,
            key_points TEXT,
            risks TEXT,
            opportunities TEXT,
            prompt_hash TEXT
        )
        """)
        con.commit()
    finally:
        con.close()

# ---------- NEW: Application plans (per-post) ----------
def ensure_post_application_plans_table():
    con = sqlite3.connect(DB_PATH)
    try:
        con.execute("""
        CREATE TABLE IF NOT EXISTS post_application_plans (
            url TEXT PRIMARY KEY,
            generated_at TIMESTAMP NOT NULL,
            model TEXT,
            category TEXT,
            title TEXT,
            objective TEXT,
            why_it_matters TEXT,
            prerequisites TEXT,
            steps TEXT,
            code_starters TEXT,
            ui_patterns TEXT,
            metrics TEXT,
            risks_mitigations TEXT,
            timeline TEXT,
            tasks TEXT,
            notes TEXT,
            raw_json TEXT
        )
        """)
        # In case table exists but is missing columns, add them defensively.
        existing_cols = {r[1] for r in con.execute("PRAGMA table_info(post_application_plans)")}
        wanted = ["objective","why_it_matters","prerequisites","steps","code_starters","ui_patterns",
                  "metrics","risks_mitigations","timeline","tasks","notes","raw_json"]
        for col in wanted:
            if col not in existing_cols:
                con.execute(f"ALTER TABLE post_application_plans ADD COLUMN {col} TEXT")
        con.commit()
    finally:
        con.close()

def save_post_application_plan(url: str, category: str, title: str, model: str, data: dict):
    """Persist normalized markdown fields plus raw JSON."""
    def _as_markdown_list(val):
        if isinstance(val, list):
            return "\n".join([f"- {str(x)}" for x in val])
        return str(val or "")

    def _as_code_block(items):
        if isinstance(items, list):
            out = []
            for it in items:
                if isinstance(it, dict):
                    t = it.get("title") or ""
                    snip = it.get("snippet") or it.get("code") or ""
                    out.append((f"**{t}**\n\n" if t else "") + f"```\n{snip}\n```")
                else:
                    out.append(f"```\n{str(it)}\n```")
            return "\n\n".join(out)
        if isinstance(items, dict):
            t = items.get("title") or ""
            snip = items.get("snippet") or items.get("code") or ""
            return (f"**{t}**\n\n" if t else "") + f"```\n{snip}\n```"
        return str(items or "")

    md_objective        = str(data.get("objective", "")).strip()
    md_why              = _as_markdown_list(data.get("why_it_matters", data.get("why", "")))
    md_prereq           = _as_markdown_list(data.get("prerequisites", []))
    md_steps            = _as_markdown_list(data.get("implementation_steps", data.get("steps", [])))
    md_code             = _as_code_block(data.get("code_starters", []))
    md_patterns         = _as_markdown_list(data.get("ui_patterns", []))
    md_metrics          = _as_markdown_list(data.get("metrics", []))
    md_risks            = _as_markdown_list(data.get("risks_mitigations", data.get("risks", [])))
    md_timeline         = _as_markdown_list(data.get("timeline", []))
    md_tasks            = _as_markdown_list(data.get("tasks", []))
    md_notes            = str(data.get("notes", "")).strip()

    con = sqlite3.connect(DB_PATH)
    try:
        con.execute("""
        INSERT INTO post_application_plans(
            url, generated_at, model, category, title,
            objective, why_it_matters, prerequisites, steps, code_starters,
            ui_patterns, metrics, risks_mitigations, timeline, tasks, notes, raw_json
        ) VALUES (
            ?, datetime('now'), ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?
        )
        ON CONFLICT(url) DO UPDATE SET
            generated_at = excluded.generated_at,
            model = excluded.model,
            category = excluded.category,
            title = excluded.title,
            objective = excluded.objective,
            why_it_matters = excluded.why_it_matters,
            prerequisites = excluded.prerequisites,
            steps = excluded.steps,
            code_starters = excluded.code_starters,
            ui_patterns = excluded.ui_patterns,
            metrics = excluded.metrics,
            risks_mitigations = excluded.risks_mitigations,
            timeline = excluded.timeline,
            tasks = excluded.tasks,
            notes = excluded.notes,
            raw_json = excluded.raw_json
        """, (
            url, model, category, title,
            md_objective, md_why, md_prereq, md_steps, md_code,
            md_patterns, md_metrics, md_risks, md_timeline, md_tasks, md_notes,
            json.dumps(data, ensure_ascii=False)
        ))
        con.commit()
    finally:
        con.close()

@st.cache_data(ttl=30)
def load_post_application_for_url(url: str) -> dict | None:
    con = sqlite3.connect(DB_PATH)
    try:
        row = con.execute("""
            SELECT url, generated_at, model, category, title,
                   objective, why_it_matters, prerequisites, steps, code_starters,
                   ui_patterns, metrics, risks_mitigations, timeline, tasks, notes
            FROM post_application_plans WHERE url = ?
        """, (url,)).fetchone()
        if not row:
            return None
        keys = ["url","generated_at","model","category","title",
                "objective","why_it_matters","prerequisites","steps","code_starters",
                "ui_patterns","metrics","risks_mitigations","timeline","tasks","notes"]
        return dict(zip(keys, row))
    finally:
        con.close()

@st.cache_data(ttl=60)
def load_post_applications_by_category(category: str, limit: int = 500) -> list[dict]:
    con = sqlite3.connect(DB_PATH)
    try:
        rows = con.execute("""
            SELECT url, generated_at, model, category, title,
                   objective, why_it_matters, prerequisites, steps, code_starters,
                   ui_patterns, metrics, risks_mitigations, timeline, tasks, notes
            FROM post_application_plans
            WHERE category = ?
            ORDER BY datetime(generated_at) DESC
            LIMIT ?
        """, (category, int(limit))).fetchall()
        keys = ["url","generated_at","model","category","title",
                "objective","why_it_matters","prerequisites","steps","code_starters",
                "ui_patterns","metrics","risks_mitigations","timeline","tasks","notes"]
        return [dict(zip(keys, r)) for r in rows]
    finally:
        con.close()

# ---------- NEW: Saved Plans page helpers ----------
@st.cache_data(ttl=60)
def load_application_categories() -> list[str]:
    con = sqlite3.connect(DB_PATH)
    try:
        rows = con.execute("SELECT DISTINCT COALESCE(category,'Uncategorized') AS c FROM post_application_plans ORDER BY 1").fetchall()
        return [r[0] for r in rows if r and r[0]]
    finally:
        con.close()

@st.cache_data(ttl=60)
def load_post_applications_all(limit: int = 1000, search: str = "", categories: list[str] | None = None) -> list[dict]:
    con = sqlite3.connect(DB_PATH)
    try:
        q = """
        SELECT url, generated_at, model, COALESCE(category,'Uncategorized') AS category, title,
               objective, why_it_matters, prerequisites, steps, code_starters,
               ui_patterns, metrics, risks_mitigations, timeline, tasks, notes
        FROM post_application_plans
        WHERE 1=1
        """
        params = []
        if search:
            q += " AND (LOWER(title) LIKE ? OR LOWER(objective) LIKE ? OR LOWER(steps) LIKE ?)"
            like = f"%{search.lower()}%"
            params += [like, like, like]
        if categories:
            placeholders = ",".join("?" * len(categories))
            q += f" AND category IN ({placeholders})"
            params += categories
        q += " ORDER BY datetime(generated_at) DESC LIMIT ?"
        params.append(int(limit))
        rows = con.execute(q, params).fetchall()
        keys = ["url","generated_at","model","category","title",
                "objective","why_it_matters","prerequisites","steps","code_starters",
                "ui_patterns","metrics","risks_mitigations","timeline","tasks","notes"]
        return [dict(zip(keys, r)) for r in rows]
    finally:
        con.close()

def save_ai_insight(topic_id: int, model: str, summary: str, key_points: str, risks: str, opportunities: str, prompt_hash: str):
    con = sqlite3.connect(DB_PATH)
    try:
        con.execute("""
        INSERT INTO ai_insights (topic_id, generated_at, model, summary, key_points, risks, opportunities, prompt_hash)
        VALUES (?, datetime('now'), ?, ?, ?, ?, ?, ?)
        ON CONFLICT(topic_id) DO UPDATE SET
          generated_at = excluded.generated_at,
          model = excluded.model,
          summary = excluded.summary,
          key_points = excluded.key_points,
          risks = excluded.risks,
          opportunities = excluded.opportunities,
          prompt_hash = excluded.prompt_hash
        """, (int(topic_id), model, summary, key_points, risks, opportunities, prompt_hash))
        con.commit()
    finally:
        con.close()

@st.cache_data(ttl=300)
def load_ai_insight(topic_id: int) -> dict | None:
    con = sqlite3.connect(DB_PATH)
    try:
        row = con.execute("""
            SELECT topic_id, generated_at, model, summary, key_points, risks, opportunities
            FROM ai_insights WHERE topic_id = ?
        """, (int(topic_id),)).fetchone()
        if not row:
            return None
        keys = ["topic_id","generated_at","model","summary","key_points","risks","opportunities"]
        return dict(zip(keys, row))
    finally:
        con.close()

def _openai_client():
    # Prefer env var, then Streamlit secrets, then session_state from sidebar input
    key = os.getenv("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None)
    key = key or st.session_state.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not provided. Set env var or enter it in the sidebar.")
    from openai import OpenAI
    return OpenAI(api_key=key)

# -------- Temperature-safe wrapper for chat calls --------
def _supports_temperature(model: str) -> bool:
    """Some lightweight models (e.g., gpt-5-nano) only support default temperature."""
    m = (model or "").lower()
    return not any(tag in m for tag in ["gpt-5-nano"])

def _chat_create(client, *, model: str, messages: list, temperature: float | None = None, **extra):
    """Wrapper that only forwards `temperature` if the model supports it."""
    params = {"model": model, "messages": messages}
    if temperature is not None and _supports_temperature(model):
        params["temperature"] = temperature
    params.update(extra)
    return client.chat.completions.create(**params)

# -------- Existing trend prompt helpers --------
def build_trend_prompt(topic_name: str, series_df: pd.DataFrame, posts_df: pd.DataFrame) -> tuple[str, str]:
    ts = series_df.sort_values("week").tail(12).copy()
    if not ts.empty:
        ts["week"] = ts["week"].dt.date.astype(str)
    ts_block = [{"week": w, "count": int(c)} for w, c in zip(ts.get("week", []), ts.get("count", []))]

    posts_df = posts_df.sort_values("published_at", ascending=False).head(20).copy()
    posts_block = []
    for _, r in posts_df.iterrows():
        posts_block.append({
            "title": (r.get("title") or "")[:160],
            "source": (r.get("source") or "")[:80],
            "published_at": str(r.get("published_at") or "")[:32],
            "url": r.get("url") or ""
        })

    payload = {
        "topic": topic_name,
        "series": ts_block,
        "recent_posts": posts_block,
        "instructions": (
            "Analyze the trend concisely for a UI/UX team. "
            "Explain what is driving it, notable sub-themes, risks, and opportunities. "
            "Be specific and actionable."
        )
    }
    prompt = json.dumps(payload, ensure_ascii=False)
    p_hash = hashlib.sha1(prompt.encode("utf-8")).hexdigest()
    return prompt, p_hash

def call_openai_for_trend(prompt_text: str, model: str = "gpt-5-nano") -> dict:
    client = _openai_client()
    system = "You are a senior product analyst. Be concise, specific, and practical."
    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "key_points": {"type": "string"},
            "risks": {"type": "string"},
            "opportunities": {"type": "string"}
        },
        "required": ["summary", "key_points", "risks", "opportunities"]
    }
    resp = _chat_create(
        client,
        model=model,
        messages=[
            {"role":"system", "content": system},
            {"role":"user", "content": f"Analyze this trend JSON and return JSON matching the schema.\nSchema:\n{json.dumps(schema)}\n\nData:\n{prompt_text}"}
        ],
        temperature=0.3,
    )
    content = resp.choices[0].message.content.strip()
    start = content.find("{"); end = content.rfind("}")
    if start >= 0 and end > start:
        content = content[start:end+1]
    try:
        data = json.loads(content)
    except Exception:
        data = {"summary": content, "key_points": "", "risks": "", "opportunities": ""}
    for k in ["summary","key_points","risks","opportunities"]:
        data.setdefault(k, "")
    return data

# ---------- Keyword helpers for posts ----------
STOPWORDS = set("""
a about above after again against all am an and any are as at
be because been before being below between both but by could
did do does doing down during each few for from further had has
have having he he'd he'll he's her here here's hers herself him
himself his how how's i i'd i'll i'm i've if in into is it it's
its itself let's me more most my myself nor of on once only or
other ought our ours  ourselves out over own same she she'd she'll
she's should so some such than that that's the their theirs them
themselves then there there's these they they'd they'll they're
they've this those through to too under until up very was we we'd
we'll we're we've were what what's when when's where where's which
while who who's whom why why's with would you you'd you'll you're
you've your yours yourself yourselves
""".split())

def extract_keywords(texts: List[str], topk: int = 25) -> list[tuple[str,int]]:
    cnt = Counter()
    for t in texts:
        t = (t or "").lower()
        t = re.sub(r"http\S+|www\.\S+", " ", t)
        t = re.sub(r"[^a-z0-9\s\-\+\.#]", " ", t)
        toks = re.findall(r"[a-z0-9\+\#][a-z0-9\-\+\#]{1,}", t)
        for w in toks:
            if w in STOPWORDS: continue
            if w.isnumeric(): continue
            cnt[w] += 1
    return cnt.most_common(topk)

def posts_stats_payload(df: pd.DataFrame, scope_label: str) -> dict:
    if df.empty:
        return {"scope": scope_label, "total_posts": 0, "weekly": [], "top_domains": [], "keywords": [], "samples": []}
    # weekly counts
    pub = df["published_at"]
    if pd.api.types.is_datetime64tz_dtype(pub):
        pub = pub.dt.tz_convert("UTC").dt.tz_localize(None)
    wk = pub.dt.to_period("W").dt.start_time
    weekly = (wk.value_counts().rename_axis("week").reset_index(name="count").sort_values("week")
              .assign(week=lambda d: d["week"].dt.strftime("%Y-%m-%d")))
    # domains
    domains = df["url"].dropna().apply(lambda u: urlparse(u).netloc).replace("", np.nan).dropna()
    top_domains = domains.value_counts().head(10).reset_index()
    top_domains.columns = ["domain", "count"]
    # keywords from title + summary
    texts = (df["title"].fillna("") + " " + df["summary"].fillna("")).tolist()
    keywords = extract_keywords(texts, topk=30)
    # samples
    samples = df.head(12)[["title","url","source"]].to_dict("records")
    return {
        "scope": scope_label,
        "total_posts": int(df.shape[0]),
        "weekly": weekly.to_dict(orient="records"),
        "top_domains": top_domains.to_dict(orient="records"),
        "keywords": [{"text": k, "count": int(c)} for k, c in keywords],
        "samples": samples,
    }

def call_openai_for_posts_analysis(payload: dict, model: str = "gpt-5-nano") -> str:
    client = _openai_client()
    system = (
        "You are a senior UX/product analyst. Analyze the provided post stats. "
        "Explain key trends, themes, why they matter, and give clear, actionable takeaways "
        "for a UI/UX design & front-end engineering audience. Be concise and specific."
    )
    user = (
        "Here is JSON describing the posts in this scope. "
        "Please summarize: (1) what's trending and why, (2) notable themes & sub-themes, "
        "(3) risks/pitfalls or hype to watch, (4) opportunities & next steps for a UI team.\n\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )
    resp = _chat_create(
        client,
        model=model,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

# ---------- NEW: per-post Application Plan (persisted) ----------
def call_openai_for_post_application(title: str, summary: str, url: str, model: str = "gpt-5-nano") -> dict:
    client = _openai_client()
    schema = {
        "type": "object",
        "properties": {
            "objective": {"type": "string", "description": "One-line goal for applying this post in a real project."},
            "why_it_matters": {"type": "array", "items": {"type":"string"}, "description": "2–4 bullets on UX/product impact."},
            "prerequisites": {"type": "array", "items": {"type":"string"}, "description": "Dependencies, APIs, assets, or design tokens needed."},
            "implementation_steps": {"type": "array", "items": {"type":"string"}, "description": "5–10 concrete steps to implement."},
            "ui_patterns": {"type": "array", "items": {"type":"string"}, "description": "Relevant components/patterns (e.g., modal, toast, token sync)."},
            "code_starters": {
                "type": "array",
                "items": {
                    "type":"object",
                    "properties": {
                        "title": {"type":"string"},
                        "snippet": {"type":"string"}
                    },
                    "required": ["snippet"]
                },
                "description": "Small code snippets or starter scaffolds (React/Next/Tailwind)."
            },
            "metrics": {"type": "array", "items": {"type":"string"}, "description": "KPIs to track with definitions."},
            "risks_mitigations": {"type": "array", "items": {"type":"string"}, "description": "Risk → mitigation bullets."},
            "timeline": {"type": "array", "items": {"type":"string"}, "description": "Phased timeline (e.g., Day 1, Week 1)."},
            "tasks": {"type": "array", "items": {"type":"string"}, "description": "Checklist tasks for dev/design."},
            "notes": {"type": "string"}
        },
        "required": ["objective","implementation_steps"]
    }
    user = {
        "title": title or "",
        "summary": (summary or "")[:2000],
        "url": url or "",
        "context": "Audience: frontend devs using Next.js + Tailwind + React. Keep it practical and specific."
    }
    resp = _chat_create(
        client,
        model=model,
        messages=[
            {"role":"system","content": "Return ONLY a JSON object matching the schema. Prefer short, concrete bullets and include at least one React/Tailwind snippet in code_starters."},
            {"role":"user","content": f"Schema: {json.dumps(schema)}\n\nData: {json.dumps(user, ensure_ascii=False)}"}
        ],
        temperature=0.2,
    )
    raw = resp.choices[0].message.content or ""
    s = raw.find("{"); e = raw.rfind("}")
    try:
        data = json.loads(raw[s:e+1]) if s >= 0 and e > s else {}
    except Exception:
        data = {"objective": raw, "implementation_steps": []}
    # normalize presence
    for key, default in [
        ("objective",""),("why_it_matters",[]),("prerequisites",[]),("implementation_steps",[]),
        ("ui_patterns",[]),("code_starters",[]),("metrics",[]),("risks_mitigations",[]),
        ("timeline",[]),("tasks",[]),("notes","")
    ]:
        data.setdefault(key, default)
    return data

ensure_ai_insights_table()
ensure_post_application_plans_table()

# ----------------------------------
# UI Focus (AI) section
# ----------------------------------
def render_ui_focus_section():
    st.subheader("🎨 AI UI/UX Focused Trends")
    cols = st.columns([1,1,1,1])
    with cols[0]:
        model = st.text_input("UI Focus model", value="gpt-5-nano", key=key_for_post("ui_focus_model"))
    with cols[1]:
        n_topics = st.number_input("Topics", min_value=3, max_value=8, value=5, step=1, key=key_for_post("ui_focus_ntopics"))
    with cols[2]:
        include_images = st.toggle("Include images", value=True, key=key_for_post("ui_focus_images"))
    with cols[3]:
        run = st.button("Analyze UI Focus", type="primary", key=key_for_post("ui_focus_run"))

    ui_topics = []
    if run:
        try:
            client = _openai_client()
            resp = _chat_create(
                client,
                model=model,
                messages=[
                    {"role":"system","content":
                     "Return a JSON object with a 'topics' array. Each topic has: name, description, and 'articles' (array of {title,url,image})."},
                    {"role":"user","content": f"Give me {n_topics} current UI/UX design trend topics with 2–4 relevant articles each. Include images if possible."}
                ],
                temperature=0.2,
            )
            raw = resp.choices[0].message.content or ""
            s = raw.find("{"); e = raw.rfind("}")
            obj = json.loads(raw[s:e+1]) if s >= 0 and e > s else {}
            ui_topics = obj.get("topics", []) if isinstance(obj.get("topics", []), list) else []
            if not ui_topics:
                st.warning("OpenAI returned no 'topics'. Showing fallbacks.")
        except Exception as e:
            st.error(f"OpenAI UI Focus call failed: {e}")
            ui_topics = []

    if not ui_topics:
        ui_topics = [
            {
                "name": "Minimalist Dashboards",
                "description": "Cleaner layouts with focus on essential metrics.",
                "articles": [
                    {"title": "5 Principles of Minimalist Dashboard Design","url": "https://www.smashingmagazine.com/","image": "https://placehold.co/600x400?text=Minimalist"},
                    {"title": "Reducing Cognitive Load","url": "https://www.nngroup.com/articles/","image": "https://placehold.co/600x400?text=Cognitive+Load"}
                ]
            },
            {
                "name": "Glassmorphism",
                "description": "Frosted glass UI trend with translucency and blur.",
                "articles": [
                    {"title": "Why Glassmorphism Still Works in 2025","url": "https://css-tricks.com/","image": "https://placehold.co/600x400?text=Glassmorphism"}
                ]
            }
        ]

    for i, topic in enumerate(ui_topics, 1):
        with st.container(border=True):
            st.subheader(topic.get("name", f"Topic {i}"))
            st.write(topic.get("description", ""))
            arts = topic.get("articles", []) or []
            for j, art in enumerate(arts):
                c1, c2 = st.columns([1, 3])
                with c1:
                    if include_images:
                        img = art.get("image") or f"https://placehold.co/150x100?text={i}"
                        try:
                            st.image(img, width=120)
                        except Exception:
                            st.caption("(image unavailable)")
                with c2:
                    title = art.get("title") or "(untitled)"
                    url = art.get("url") or "#"
                    st.markdown(f"**[{title}]({url})**")
        st.markdown("---")

# ----------------------------------
# Image cache + embedding (inside app)
# ----------------------------------
def _url_to_fname(url: str) -> str:
    stem = hashlib.sha1(url.encode("utf-8")).hexdigest()
    ext = Path(url.split("?")[0]).suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        ext = ".jpg"
    return stem + ext

def cache_image(url: str, ttl_hours: int = 48, force: bool = False, max_bytes: int = 8*1024*1024) -> str | None:
    path = CACHE_DIR / _url_to_fname(url)
    try:
        if (not force) and path.exists() and (time.time() - path.stat().st_mtime) < ttl_hours * 3600:
            return str(path)
        with requests.get(url, stream=True, timeout=15, headers={"User-Agent": "Mozilla/5.0"}) as r:
            r.raise_for_status()
            total = 0
            tmp = path.with_suffix(path.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(8192):
                    if not chunk: continue
                    total += len(chunk)
                    if total > max_bytes: break
                    f.write(chunk)
            if total > 0:
                os.replace(tmp, path)
        return str(path) if path.exists() else None
    except Exception:
        return str(path) if path.exists() else None

@st.cache_resource
def load_openclip():
    try:
        import open_clip, torch
    except Exception as e:
        st.warning(f"OpenCLIP not available: {e}")
        raise
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device="cpu")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, preprocess, tokenizer

def embed_image_path(img_path: str) -> np.ndarray | None:
    try:
        from PIL import Image as PILImage
        model, preprocess, _ = load_openclip()
        img = PILImage.open(img_path)
        if img.mode in ("P","LA"):
            img = img.convert("RGBA")
        if img.mode == "RGBA":
            bg = PILImage.new("RGBA", img.size, (255,255,255,255))
            img = PILImage.alpha_composite(bg, img).convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")
        import torch
        with torch.no_grad():
            t = preprocess(img).unsqueeze(0)
            feats = model.encode_image(t)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype("float32").squeeze()
    except Exception:
        return None

@st.cache_data(ttl=300)
def embed_style_prompts(prompts: List[str]) -> np.ndarray:
    model, _, tokenizer = load_openclip()
    import torch
    with torch.no_grad():
        toks = tokenizer(prompts)
        feats = model.encode_text(toks)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype("float32")

def label_centroids_with_prompts(centroids: np.ndarray, prompts: List[str]) -> Tuple[List[str], np.ndarray]:
    T = embed_style_prompts(prompts)  # (P, 512)
    sims = centroids @ T.T            # (K, P)
    labels = [prompts[int(i)] for i in sims.argmax(axis=1)]
    return labels, sims

def normalize_rows(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

# ----------------------------------
# Sidebar
# ----------------------------------
with st.sidebar:
    st.header("API")
    api_input = st.text_input("OpenAI API key", type="password", key=key_for_post("api_key_input"))
    if api_input:
        st.session_state["OPENAI_API_KEY"] = api_input

    st.header("Text Pipeline (with AI)")
    analyze_top_n = st.number_input("Analyze top N topics with OpenAI", min_value=1, max_value=100, value=10, step=1, key=key_for_post("ai_analyze_topn"))
    openai_model = st.text_input("OpenAI model", "gpt-5-nano", key=key_for_post("ai_model_input"))

    if st.button("Run Full Text Pipeline", type="primary", key=key_for_post("btn_full_pipeline")):
        ensure_db()
        needed = ["ingest_sample.py", "cluster_topics.py", "trends_momentum.py"]
        missing = [p for p in needed if not os.path.exists(p)]
        if missing:
            st.error("Missing scripts: " + ", ".join(missing))
            st.stop()

        run_status("Ingest posts", ["ingest_sample.py"])
        run_status("Cluster + merge topics", ["cluster_topics.py", "--min_topic_size", "8", "--similarity", "0.95"])
        run_status("Compute momentum", ["trends_momentum.py", "--K", "8"])

        try:
            if not (os.getenv("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY") or (hasattr(st, "secrets") and st.secrets.get("OPENAI_API_KEY"))):
                st.warning("OPENAI_API_KEY not set; skipping AI analysis.")
            else:
                with st.status("Running OpenAI trend analysis…", expanded=False) as status:
                    topics = load_topics_with_latest_momentum()
                    topics = topics.drop_duplicates(subset=["topic_id"], keep="first")
                    topics = topics.sort_values(["momentum","count"], ascending=[False, False]).head(int(analyze_top_n))

                    analyzed = 0
                    for _, r in topics.iterrows():
                        tid = int(r["topic_id"]); name = r["name"]
                        ts = load_topic_series(tid)
                        posts = load_recent_posts_for_topic(tid, 20)
                        prompt_text, p_hash = build_trend_prompt(name, ts, posts)
                        data = call_openai_for_trend(prompt_text, model=openai_model)
                        save_ai_insight(tid, openai_model,
                                        data.get("summary",""),
                                        data.get("key_points",""),
                                        data.get("risks",""),
                                        data.get("opportunities",""),
                                        p_hash)
                        analyzed += 1
                        if analyzed % 3 == 0:
                            st.write(f"Analyzed {analyzed}/{len(topics)} topics…")

                    status.update(label="OpenAI analysis complete", state="complete")
        except Exception as e:
            st.error(f"AI analysis failed: {e}")

        st.success("Text pipeline complete")
        st.cache_data.clear(); st.rerun()

    st.markdown("---")
    st.header("Navigate")
    if st.button("Trends Dashboard", key=key_for_post("nav_dashboard")):
        goto("dashboard")
    if st.button("Image Styles", key=key_for_post("nav_images")):
        goto("images")
    if st.button("Saved Plans", key=key_for_post("nav_plans")):
        goto("plans")

# ----------------------------------
# Routing
# ----------------------------------
ensure_db()
page = st.query_params.get("page", "dashboard")

# ----------------------------------
# Saved Plans page
# ----------------------------------
if page == "plans":
    st.header("📚 Saved Application Plans")

    # Top filters
    col_a, col_b, col_c = st.columns([2, 2, 1])
    with col_a:
        plans_search = st.text_input("Search (title/objective/steps)", "", key=key_for_post("plans_search"))
    with col_b:
        plan_categories = load_application_categories()
        sel_cats = st.multiselect("Filter by category", plan_categories, key=key_for_post("plans_categories"))
    with col_c:
        plans_limit = st.number_input("Limit", min_value=50, max_value=5000, value=1000, step=50, key=key_for_post("plans_limit"))

    # Tabs per category (plus 'All')
    tabs = st.tabs(["All"] + plan_categories)

    # All tab (respects the search + multi-category filter)
    with tabs[0]:
        plans = load_post_applications_all(limit=int(plans_limit),
                                           search=plans_search,
                                           categories=sel_cats if sel_cats else None)
        st.caption(f"Showing {len(plans)} saved plans")
        if not plans:
            st.info("No saved application plans yet.")
        else:
            for rec in plans:
                label = f"{rec.get('title') or '(untitled)'} · {rec.get('category','Uncategorized')} · {rec.get('generated_at')}"
                with st.expander(label, expanded=False):
                    if rec.get("url"):
                        st.markdown(f"[Open article]({rec['url']})  \nModel: `{rec.get('model','')}` • Saved: `{rec.get('generated_at','')}`")
                    if rec.get("objective"):
                        st.markdown("**Objective**")
                        st.markdown(rec["objective"])
                    cols_top = st.columns(2)
                    with cols_top[0]:
                        st.markdown("**Why it matters**")
                        st.markdown(rec.get("why_it_matters","") or "_—_")
                    with cols_top[1]:
                        st.markdown("**Prerequisites**")
                        st.markdown(rec.get("prerequisites","") or "_—_")
                    st.markdown("**Implementation steps**")
                    st.markdown(rec.get("steps","") or "_—_")
                    cols_mid = st.columns(2)
                    with cols_mid[0]:
                        st.markdown("**UI patterns**")
                        st.markdown(rec.get("ui_patterns","") or "_—_")
                    with cols_mid[1]:
                        st.markdown("**Metrics**")
                        st.markdown(rec.get("metrics","") or "_—_")
                    st.markdown("**Code starters**")
                    st.markdown(rec.get("code_starters","") or "_—_")
                    cols_bot = st.columns(2)
                    with cols_bot[0]:
                        st.markdown("**Risks & mitigations**")
                        st.markdown(rec.get("risks_mitigations","") or "_—_")
                    with cols_bot[1]:
                        st.markdown("**Timeline**")
                        st.markdown(rec.get("timeline","") or "_—_")
                    st.markdown("**Tasks**")
                    st.markdown(rec.get("tasks","") or "_—_")
                    if rec.get("notes"):
                        st.markdown("**Notes**")
                        st.markdown(rec["notes"])

    # One tab per category (shows that category only)
    for t, cat in zip(tabs[1:], plan_categories):
        with t:
            plans = load_post_applications_by_category(cat, limit=int(plans_limit))
            st.caption(f"{cat}: {len(plans)} saved plans")
            if not plans:
                st.info("No saved application plans in this category.")
            else:
                for rec in plans:
                    label = f"{rec.get('title') or '(untitled)'} · {rec.get('generated_at')}"
                    with st.expander(label, expanded=False):
                        if rec.get("url"):
                            st.markdown(f"[Open article]({rec['url']})  \nModel: `{rec.get('model','')}` • Saved: `{rec.get('generated_at','')}`")
                        if rec.get("objective"):
                            st.markdown("**Objective**")
                            st.markdown(rec["objective"])
                        cols_top = st.columns(2)
                        with cols_top[0]:
                            st.markdown("**Why it matters**")
                            st.markdown(rec.get("why_it_matters","") or "_—_")
                        with cols_top[1]:
                            st.markdown("**Prerequisites**")
                            st.markdown(rec.get("prerequisites","") or "_—_")
                        st.markdown("**Implementation steps**")
                        st.markdown(rec.get("steps","") or "_—_")
                        cols_mid = st.columns(2)
                        with cols_mid[0]:
                            st.markdown("**UI patterns**")
                            st.markdown(rec.get("ui_patterns","") or "_—_")
                        with cols_mid[1]:
                            st.markdown("**Metrics**")
                            st.markdown(rec.get("metrics","") or "_—_")
                        st.markdown("**Code starters**")
                        st.markdown(rec.get("code_starters","") or "_—_")
                        cols_bot = st.columns(2)
                        with cols_bot[0]:
                            st.markdown("**Risks & mitigations**")
                            st.markdown(rec.get("risks_mitigations","") or "_—_")
                        with cols_bot[1]:
                            st.markdown("**Timeline**")
                            st.markdown(rec.get("timeline","") or "_—_")
                        st.markdown("**Tasks**")
                        st.markdown(rec.get("tasks","") or "_—_")
                        if rec.get("notes"):
                            st.markdown("**Notes**")
                            st.markdown(rec["notes"])
    st.stop()  # prevent the dashboard page from also rendering

# ----------------------------------
# Trends page
# ----------------------------------
if page == "dashboard":
    with st.sidebar:
        st.header("Filters")
        ui_focus_flag = st.toggle("UI Focus", value=True, key=key_for_post("flt_ui_focus"))
        min_momentum = st.slider("Min momentum", -2.0, 2.0, 0.0, 0.05, key=key_for_post("flt_min_mom"))
        search = st.text_input("Search topic/keywords", "", key=key_for_post("flt_search")).strip().lower()
        sort_by = st.selectbox("Sort by", ["Momentum (desc)", "Count (desc)", "Name (A→Z)"], key=key_for_post("flt_sort"))
        st.session_state["flt_max_cards"] = st.slider("Max cards", 5, 100, 20, 1, key=key_for_post("flt_max_cards"))

    topics_df = load_topics_with_latest_momentum()
    if topics_df.empty:
        st.info("No topic_timeseries yet. Run the text pipeline.")
        st.stop()

    if ui_focus_flag:
        keep_mask = (
            topics_df["keywords"].str.contains("button|table|modal|dialog|tooltip|tabs|accordion|toast|skeleton|grid|typography|color|token|animation|motion|material|fluent|tailwind|radix|shadcn", case=False, regex=True)
            | topics_df["name"].str.contains("button|table|dialog|navigation|tokens|typography|color|animation|motion|material|fluent|tailwind|radix|shadcn", case=False, regex=True)
        )
        topics_df = topics_df[keep_mask]

    f = topics_df.copy().drop_duplicates(subset=["topic_id"], keep="first")
    if search:
        mask = f["name"].str.lower().str.contains(search) | f["keywords"].str.lower().str.contains(search)
        f = f[mask]
    f = f[f["momentum"] >= min_momentum]
    if sort_by == "Momentum (desc)":
        f = f.sort_values(["momentum","count"], ascending=[False, False])
    elif sort_by == "Count (desc)":
        f = f.sort_values(["count","momentum"], ascending=[False, False])
    else:
        f = f.sort_values("name", ascending=True)

    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("Topics", f["topic_id"].nunique())
    c2.metric("Median momentum", float(f["momentum"].median()) if not f.empty else 0.0)
    c3.metric("Latest week", f["week"].max().date().isoformat())

    st.markdown("---")

    # UI Focus (AI)
    render_ui_focus_section()

    # -------------------- POSTS: categorized in tabs with per-post Application Plans --------------------
    st.subheader("📰 Posts (categorized)")
    left, mid, right = st.columns([2, 2, 1])
    with left:
        posts_search = st.text_input("Search title/summary", "", key=key_for_post("posts_search"))
    with mid:
        src_options = load_post_sources()
        sel_sources = st.multiselect("Limit to sources (optional)", src_options, key=key_for_post("posts_sources"))
    with right:
        limit = st.number_input("Limit", min_value=100, max_value=20000, value=2000, step=100, key=key_for_post("posts_limit"))

    inter_cols = st.columns([1,1,2,2])
    with inter_cols[0]:
        per_cat_interactive = st.number_input("Max interactive per category", min_value=10, max_value=300, value=50, step=10, key=key_for_post("per_cat"))
    with inter_cols[1]:
        posts_ai_model = st.text_input("Post model", "gpt-5-nano", key=key_for_post("posts_ai_model"))

    df_posts_all = load_posts(limit=limit, search=posts_search, sources=sel_sources)
    if df_posts_all.empty:
        st.info("No posts matched. Try clearing filters or run the ingester.")
    else:
        # Analyze ALL (current filter) – aggregate analyzer
        a_all_cols = st.columns([1,1,6])
        with a_all_cols[0]:
            if st.button("Analyze current filter (AI)", key=key_for_post("btn_ai_posts_all"), type="primary"):
                payload = posts_stats_payload(df_posts_all, scope_label="All filtered posts")
                try:
                    analysis = call_openai_for_posts_analysis(payload, model=posts_ai_model)
                    with st.expander("AI Analysis — All filtered posts", expanded=True):
                        st.write(analysis)
                except Exception as e:
                    st.error(f"Posts AI analysis failed: {e}")

        # Build dynamic tabs by category
        cats = sorted([c for c in df_posts_all["source"].dropna().unique().tolist()])
        if not cats:
            st.info("No categories in current filter.")
        else:
            tabs = st.tabs(cats)
            for tab, cat in zip(tabs, cats):
                with tab:
                    st.caption(f"Category: **{cat}** • Showing up to {limit} rows")

                    # Saved application plans quick access (link to dedicated page)
                    st.markdown("### 📌 Saved Application Plans")
                    st.caption("Saved plans are shown on a dedicated page with tabs and expanders.")
                    if st.button("Open Saved Plans", key=key_for_post("open_saved", cat)):
                        goto("plans")

                    # Interactive list with per-post Application Plan buttons
                    df_cat = df_posts_all[df_posts_all["source"] == cat].copy()
                    if df_cat.empty:
                        st.info("No posts in this category with current filter.")
                        continue

                    st.markdown("### Articles")
                    for i, row in enumerate(df_cat.head(int(per_cat_interactive)).itertuples(index=False)):
                        title = getattr(row, "title")
                        url = getattr(row, "url")
                        summary = getattr(row, "summary") or ""
                        pub = getattr(row, "published_at")
                        img = getattr(row, "image_url")

                        with st.container(border=True):
                            c1, c2 = st.columns([0.15, 0.85])
                            with c1:
                                if isinstance(img, str) and img.strip():
                                    try:
                                        st.image(img, width=120)
                                    except Exception:
                                        st.caption("(image)")
                            with c2:
                                st.markdown(f"**[{title or '(untitled)'}]({url or '#'})**")
                                if pub is not None:
                                    try:
                                        st.caption(str(pub.date()))
                                    except Exception:
                                        st.caption(str(pub))
                                if summary:
                                    st.write(summary[:400] + ("…" if len(summary) > 400 else ""))

                                # Existing plan (if any)
                                existing = load_post_application_for_url(url or "")
                                if existing:
                                    st.success("Application plan saved")
                                    with st.expander("View saved plan"):
                                        if existing.get("objective"):
                                            st.markdown("**Objective**")
                                            st.markdown(existing["objective"])
                                        st.markdown("**Implementation steps**")
                                        st.markdown(existing.get("steps",""))
                                        cols_sh = st.columns(2)
                                        with cols_sh[0]:
                                            st.markdown("**Prerequisites**")
                                            st.markdown(existing.get("prerequisites",""))
                                        with cols_sh[1]:
                                            st.markdown("**Metrics**")
                                            st.markdown(existing.get("metrics",""))

                                # Generate Application Plan button (hash key + idx)
                                if st.button("Generate application plan", key=key_for_post("apply", cat, url, title, pub, idx=i)):
                                    try:
                                        with st.spinner("Generating plan…"):
                                            data = call_openai_for_post_application(
                                                title=title, summary=summary, url=url, model=posts_ai_model
                                            )
                                            save_post_application_plan(
                                                url=url or "", category=cat, title=title or "", model=posts_ai_model, data=data
                                            )
                                            st.success("Saved application plan")
                                            st.cache_data.clear()
                                            st.rerun()
                                    except Exception as e:
                                        st.error(f"Application plan failed: {e}")

                    # Compact table (non-interactive remainder)
                    remainder = df_cat.iloc[int(per_cat_interactive):]
                    if not remainder.empty:
                        st.caption(f"…and {len(remainder)} more (not shown interactively).")
                        tbl = remainder.assign(
                            published_at=remainder["published_at"].dt.strftime("%Y-%m-%d")
                        )[["published_at","title","url","image_url"]].rename(
                            columns={"published_at":"Date","title":"Title","url":"Link","image_url":"Image"}
                        )
                        st.dataframe(tbl, use_container_width=True, hide_index=True)

    st.markdown("---")

    # -------------------- Trend tabs --------------------
    tab_styles, tab_components, tab_frameworks = st.tabs(["Styles", "Components", "Frameworks"])

    with tab_styles:
        df = load_style_trends()
        if df.empty:
            st.info("Run: python tag_ui_components.py to populate style trends.")
        else:
            pick = st.multiselect("Styles", sorted(df["style"].unique().tolist()), key=key_for_post("pick_styles"))
            plot_df = df if not pick else df[df["style"].isin(pick)]
            fig = px.line(plot_df, x="week", y="count", color="style", title="UI Styles momentum")
            st.plotly_chart(fig, use_container_width=True, key=key_for_post("ui_styles_line"))

    with tab_components:
        df = load_component_tags()
        if df.empty:
            st.info("Run: python tag_ui_components.py to populate component tags.")
        else:
            df["week"] = df["week"].dt.to_period("W").dt.start_time
            agg = df.groupby(["component","week"]).size().reset_index(name="count")
            pick = st.multiselect("Components", sorted(agg["component"].unique().tolist()), key=key_for_post("pick_components"))
            plot_df = agg if not pick else agg[agg["component"].isin(pick)]
            fig = px.line(plot_df, x="week", y="count", color="component", title="Component mentions per week")
            st.plotly_chart(fig, use_container_width=True, key=key_for_post("ui_components_line"))

    with tab_frameworks:
        df = load_framework_trends()
        if df.empty:
            st.info("Run: python tag_ui_components.py to populate framework trends.")
        else:
            pick = st.multiselect("Frameworks", sorted(df["framework"].unique().tolist()), key=key_for_post("pick_frameworks"))
            plot_df = df if not pick else df[df["framework"].isin(pick)]
            fig = px.line(plot_df, x="week", y="count", color="framework", title="Framework mentions per week")
            st.plotly_chart(fig, use_container_width=True, key=key_for_post("ui_fw_line"))

    # -------------------- Topic cards (below) --------------------
    n = 0
    for _, row in f.head(int(st.session_state.get("flt_max_cards", 20))).iterrows():
        topic_id = int(row["topic_id"])
        name = row["name"]
        keywords = row["keywords"]
        week = row["week"].date().isoformat()
        count = int(row["count"]) if not pd.isna(row["count"]) else 0
        mom = float(row["momentum"]) if not pd.isna(row["momentum"]) else 0.0

        with st.container(border=True):
            c1a, c2a = st.columns([0.70, 0.30])
            with c1a:
                st.subheader(f"Topic #{topic_id} (Posts: {count}) — {name}")
                st.markdown(f"Keywords: _{keywords}_")
            with c2a:
                st.markdown(momentum_badge(mom), unsafe_allow_html=True)
                st.caption(f"Week: {week}")
                st.caption(f"Posts: {count}")

            with st.expander("Details: weekly history & latest posts"):
                ts = load_topic_series(topic_id)
                if not ts.empty:
                    fig = px.line(ts, x="week", y="count", markers=True, title="Weekly Posts")
                    fig.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig, use_container_width=True, key=key_for_post("ts_line", topic_id))

                    if "momentum" in ts.columns:
                        fig2 = px.line(ts.tail(12), x="week", y="momentum", markers=True, title="Momentum (rolling slope)")
                        fig2.update_layout(height=160, margin=dict(l=10, r=10, t=30, b=10))
                        st.plotly_chart(fig2, use_container_width=True, key=key_for_post("mom_line", topic_id))
                else:
                    st.info("No time-series yet for this topic.")

                posts = load_recent_posts_for_topic(topic_id, limit=20)
                if posts.empty:
                    st.caption("No recent posts linked.")
                else:
                    for _, pr in posts.iterrows():
                        link = pr["url"] or "#"
                        title = pr["title"] or "(untitled)"
                        src = pr.get("source") or ""
                        pub = pr.get("published_at") or ""
                        st.markdown(f"- [{title}]({link})")
                        st.caption(f"{src} • {pub}")

            cached = load_ai_insight(topic_id)
            if cached:
                with st.container(border=True):
                    st.markdown("**AI Summary**")
                    st.write(cached["summary"] or "")
                    cols_ai = st.columns(3)
                    with cols_ai[0]:
                        st.markdown("**Key Points**")
                        st.write(cached["key_points"] or "")
                    with cols_ai[1]:
                        st.markdown("**Risks**")
                        st.write(cached["risks"] or "")
                    with cols_ai[2]:
                        st.markdown("**Opportunities**")
                        st.write(cached["opportunities"] or "")
                    st.caption(f"Analyzed: {cached['generated_at']} • Model: {cached['model']}")
        n += 1

    if n == 0:
        st.warning("No topics matched your filters.")

# ----------------------------------
# Images page: one-button pipeline
# ----------------------------------
if page == "images":
    st.subheader("Image Styles — one-click pipeline")

    left, right = st.columns([0.6, 0.4])
    with left:
        K = st.slider("Number of clusters (styles)", min_value=4, max_value=24, value=10, step=1, key=key_for_post("img_K"))
        ttl_hours = st.number_input("Cache TTL (hours)", min_value=1, max_value=240, value=48, step=1, key=key_for_post("img_ttl"))
        force_cache = st.checkbox("Force re-download", value=False, key=key_for_post("img_force"))
    with right:
        st.caption("Style prompts used to name clusters (editable):")
        default_styles = [
            "Minimalist", "Futuristic", "3D Render", "Photorealistic",
            "Vector / Flat", "Isometric", "Anime / Manga", "Pixel Art",
            "Watercolor", "Oil Painting", "Line Art", "Low Poly",
            "Cartoon / Comic", "Sketch", "Cyberpunk", "Noir", "Retro", "Vaporwave"
        ]
        styles_json = st.text_area("Style prompts (one per line)", "\n".join(default_styles), height=150, key=key_for_post("img_styles_txt"))
        style_prompts = [s.strip() for s in styles_json.splitlines() if s.strip()]

    if st.button("Run Image Pipeline", key=key_for_post("img_run")):
        with st.status("Running image pipeline…", expanded=False) as status:
            posts_df = load_image_sources_from_posts(max_rows=2000)
            urls = [u for u in posts_df["image_url"].astype(str).tolist() if u.strip()]
            urls = list(dict.fromkeys(urls))
            st.write(f"Found {len(urls)} image URLs in posts.")

            local_paths = []
            for i, url in enumerate(urls, 1):
                p = cache_image(url, ttl_hours=int(ttl_hours), force=force_cache)
                if p: local_paths.append(p)
                if i % 25 == 0:
                    st.write(f"Cached {i}/{len(urls)} images…")
            if not local_paths:
                st.error("No images could be cached. Check that posts.image_url exists.")
                status.update(label="Image pipeline failed", state="error")
                st.stop()

            embs = []
            for i, p in enumerate(local_paths, 1):
                v = embed_image_path(p)
                if v is not None:
                    embs.append(v)
                else:
                    embs.append(np.zeros(512, dtype=np.float32))
                if i % 25 == 0:
                    st.write(f"Embedded {i}/{len(local_paths)} images…")

            X = np.vstack(embs).astype("float32")
            X = normalize_rows(X)

            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=int(K), n_init=10, random_state=42)
            labels = km.fit_predict(X)
            centroids = normalize_rows(km.cluster_centers_)

            style_names, sim = label_centroids_with_prompts(centroids, style_prompts)

            df = pd.DataFrame({
                "path": local_paths,
                "cluster": labels,
            })
            df["style"] = df["cluster"].map({i: style_names[i] for i in range(int(K))})

            st.session_state["image_styles_df"] = df
            st.session_state["image_styles_prompts"] = style_prompts

            status.update(label="Image pipeline complete", state="complete")

    df = st.session_state.get("image_styles_df")
    if df is None or df.empty:
        st.info("Run the pipeline to generate image styles.")
    else:
        styles = ["All"] + sorted(df["style"].unique().tolist())
        sel = st.selectbox("Filter by style", styles, index=0, key=key_for_post("img_style_filter"))
        cols_per_row = st.slider("Columns", 2, 8, 5, key=key_for_post("img_cols"))

        view = df if sel == "All" else df[df["style"] == sel]
        st.caption(f"Showing {len(view)} / {len(df)} images")

        for style, g in view.groupby("style"):
            st.markdown(f"### {style}")
            paths = g["path"].tolist()
            for i in range(0, len(paths), cols_per_row):
                cols = st.columns(cols_per_row, gap="small")
                for img_path, col in zip(paths[i:i+cols_per_row], cols):
                    with col:
                        try:
                            st.image(img_path, use_column_width=True)
                        except Exception:
                            st.caption("(failed to display)")

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download cluster assignments (CSV)", data=csv_bytes, file_name="image_styles.csv", mime="text/csv", key=key_for_post("img_dl"))

        if st.button("Clear image cache", key=key_for_post("img_clear")):
            shutil.rmtree(CACHE_DIR, ignore_errors=True)
            CACHE_DIR.mkdir(exist_ok=True)
            st.success("Image cache cleared")
