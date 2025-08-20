#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Embed images into trends.db: posts.image_embedding

- Finds posts with image_url and NULL image_embedding
- Downloads & preprocesses images for CLIP
- Stores 512-d float32 embeddings (OpenCLIP ViT-B/32) as BLOB
- ASCII-only logging (Windows-safe)
"""

import os
import io
import sqlite3
import requests
import numpy as np
from PIL import Image
import torch

# OpenCLIP
import open_clip

DB = os.path.abspath("trends.db")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ImageEmbedder/1.0"}

def log(msg: str):
    print(str(msg), flush=True)

def connect_db():
    if not os.path.exists(DB):
        raise FileNotFoundError("DB not found: %s" % DB)
    return sqlite3.connect(DB)

def fetch_pending(con):
    q = """
    SELECT id, image_url FROM posts
    WHERE image_url IS NOT NULL AND image_url <> ''
      AND (image_embedding IS NULL OR length(image_embedding) = 0)
    """
    return con.execute(q).fetchall()

def download_image(url: str) -> Image.Image | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content))

        # Normalize modes:
        # - Convert palette or LA/LA;RGBA to RGBA then to RGB (drop alpha) to avoid warnings
        if img.mode in ("P", "LA"):
            img = img.convert("RGBA")
        if img.mode == "RGBA":
            # Drop alpha against white bg
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img).convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception:
        return None

def main():
    device = "cpu"
    log("Using device: %s" % device)

    # Load model + preprocess
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai",
        device=device
    )
    model.eval()

    con = connect_db()
    cur = con.cursor()

    rows = fetch_pending(con)
    if not rows:
        log("No images to embed. (image_embedding already populated or no image_url)")
        return

    log("Pending images: %d" % len(rows))
    done = 0

    with torch.no_grad():
        for pid, url in rows:
            img = download_image(url)
            if img is None:
                continue
            try:
                im_t = preprocess(img).unsqueeze(0).to(device)
                feats = model.encode_image(im_t)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                vec = feats.cpu().numpy().astype("float32").squeeze()  # (512,)
                blob = vec.tobytes()

                cur.execute(
                    "UPDATE posts SET image_embedding = ? WHERE id = ?",
                    (blob, pid)
                )
                done += 1
                if done % 5 == 0:
                    con.commit()
                    log("Processed %d/%d images..." % (done, len(rows)))
            except Exception:
                # Skip problematic image
                continue

    con.commit()
    con.close()
    log("Processed %d/%d images." % (done, len(rows)))
    log("Done.")
    return

if __name__ == "__main__":
    main()
