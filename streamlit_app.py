#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import io
import sys
import time
import json
import math
import shutil
import sqlite3
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import requests
from PIL import Image
from urllib.parse import urlparse

# ----------------------------------
# App config
# ----------------------------------
DB_PATH = os.path.abspath("trends.db")
BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()
CACHE_DIR = BASE_DIR / "image_cache"
CACHE_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="UI/UX Trends", layout="wide")
st.title("UI/UX Trends Dashboard")

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
        st.error(f"Database not found at {DB_PATH}. Run your setup first.")
        st.stop()

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
    """Pull image_url from posts for image pipeline."""
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
    import open_clip, torch
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
    """Return best prompt label per centroid + similarity matrix."""
    T = embed_style_prompts(prompts)  # (P, 512)
    sims = centroids @ T.T            # (K, P) assuming centroids L2-normalized
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
    st.header("Pipeline")
    if st.button("Run Full Text Pipeline", type="primary"):
        ensure_db()
        needed = ["ingest_sample.py", "cluster_topics.py", "trends_momentum.py"]
        missing = [p for p in needed if not os.path.exists(p)]
        if missing:
            st.error("Missing scripts: " + ", ".join(missing))
            st.stop()
        run_status("Ingest posts", ["ingest_sample.py"])
        run_status("Cluster + merge topics", ["cluster_topics.py", "--min_topic_size", "8", "--similarity", "0.95"])
        run_status("Compute momentum", ["trends_momentum.py", "--K", "8"])
        st.success("Text pipeline complete")
        st.cache_data.clear(); st.rerun()

    st.markdown("---")
    st.header("Navigate")
    if st.button("Trends Dashboard"):
        goto("dashboard")
    if st.button("Image Styles"):
        goto("images")

# ----------------------------------
# Routing
# ----------------------------------
ensure_db()
page = st.query_params.get("page", "dashboard")

# ----------------------------------
# Trends page
# ----------------------------------
if page == "dashboard":
    with st.sidebar:
        st.header("Filters")
        min_momentum = st.slider("Min momentum", -2.0, 2.0, 0.0, 0.05)
        search = st.text_input("Search topic/keywords", "").strip().lower()
        sort_by = st.selectbox("Sort by", ["Momentum (desc)", "Count (desc)", "Name (A→Z)"])
        max_cards = st.slider("Max cards", 5, 100, 20, 1)

    topics_df = load_topics_with_latest_momentum()
    if topics_df.empty:
        st.info("No topic_timeseries yet. Run the text pipeline.")
        st.stop()

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
    c2.metric("Median momentum", f["momentum"].median() if not f.empty else 0.0)
    c3.metric("Latest week", f["week"].max().date().isoformat())

    st.markdown("---")

    # Cards
    n = 0
    for _, row in f.head(max_cards).iterrows():
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
                    st.plotly_chart(fig, use_container_width=True, key=f"ts-{topic_id}")

                    if "momentum" in ts.columns:
                        fig2 = px.line(ts.tail(12), x="week", y="momentum", markers=True, title="Momentum (rolling slope)")
                        fig2.update_layout(height=160, margin=dict(l=10, r=10, t=30, b=10))
                        st.plotly_chart(fig2, use_container_width=True, key=f"mom-{topic_id}")
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
        n += 1

    if n == 0:
        st.warning("No topics matched your filters.")

# ----------------------------------
# Images page: one-button pipeline
# ----------------------------------
if page == "images":
    st.subheader("Image Styles — one-click pipeline")

    # Controls
    left, right = st.columns([0.6, 0.4])
    with left:
        K = st.slider("Number of clusters (styles)", min_value=4, max_value=24, value=10, step=1)
        ttl_hours = st.number_input("Cache TTL (hours)", min_value=1, max_value=240, value=48, step=1)
        force_cache = st.checkbox("Force re-download", value=False)
    with right:
        st.caption("Style prompts used to name clusters (editable):")
        default_styles = [
            "Minimalist", "Futuristic", "3D Render", "Photorealistic",
            "Vector / Flat", "Isometric", "Anime / Manga", "Pixel Art",
            "Watercolor", "Oil Painting", "Line Art", "Low Poly",
            "Cartoon / Comic", "Sketch", "Cyberpunk", "Noir", "Retro", "Vaporwave"
        ]
        styles_json = st.text_area("Style prompts (one per line)", "\n".join(default_styles), height=150)
        style_prompts = [s.strip() for s in styles_json.splitlines() if s.strip()]

    # Button: run everything
    if st.button("Run Image Pipeline"):
        with st.status("Running image pipeline…", expanded=False) as status:
            # 1) Gather URLs from DB
            posts_df = load_image_sources_from_posts(max_rows=2000)
            urls = [u for u in posts_df["image_url"].astype(str).tolist() if u.strip()]
            urls = list(dict.fromkeys(urls))  # dedupe order-preserving
            st.write(f"Found {len(urls)} image URLs in posts.")

            # 2) Cache images
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

            # 3) Embed images (OpenCLIP on CPU)
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

            # 4) Cluster (K-Means)
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=int(K), n_init=10, random_state=42)
            labels = km.fit_predict(X)
            centroids = normalize_rows(km.cluster_centers_)

            # 5) Name clusters via CLIP text prompts
            style_names, sim = label_centroids_with_prompts(centroids, style_prompts)

            # 6) Build result frame (path, label)
            df = pd.DataFrame({
                "path": local_paths,
                "cluster": labels,
            })
            df["style"] = df["cluster"].map({i: style_names[i] for i in range(int(K))})

            # Cache for display (session)
            st.session_state["image_styles_df"] = df
            st.session_state["image_styles_prompts"] = style_prompts

            status.update(label="Image pipeline complete", state="complete")

    # Gallery
    df = st.session_state.get("image_styles_df")
    if df is None or df.empty:
        st.info("Run the pipeline to generate image styles.")
    else:
        styles = ["All"] + sorted(df["style"].unique().tolist())
        sel = st.selectbox("Filter by style", styles, index=0)
        cols_per_row = st.slider("Columns", 2, 8, 5)

        view = df if sel == "All" else df[df["style"] == sel]
        st.caption(f"Showing {len(view)} / {len(df)} images")

        # Show grouped by style
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

        # Optional: download CSV of assignments
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download cluster assignments (CSV)", data=csv_bytes, file_name="image_styles.csv", mime="text/csv")

        # Clear cache button
        if st.button("Clear image cache"):
            shutil.rmtree(CACHE_DIR, ignore_errors=True)
            CACHE_DIR.mkdir(exist_ok=True)
            st.success("Image cache cleared")

