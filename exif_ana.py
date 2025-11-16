# exif_ana.py
#
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
exif_ana.py
Analyze JPGs referenced in a DuckDB EXIF table, store per-photo analysis and monthly summary,
and generate graphs for camera model, aperture, shutter speed, and ISO (normalized stops).

Usage:
    python exif_ana.py --db exif_csv.db --exif-table exif --image-root /path/to/photos --overwrite

Author: Copilot
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
exif_ana.py
Analyze JPGs referenced in a DuckDB EXIF table, store per-photo analysis and monthly summary,
and generate graphs for camera model, aperture, shutter speed, and ISO (raw + normalized stops).

Usage:
    python exif_ana.py --db exif_csv.db --exif-table exif --image-root /path/to/photos --overwrite
"""

import argparse
import os
import sys
import json
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import duckdb
import pandas as pd
from tqdm import tqdm

import torch
from torchvision import models
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze JPGs in DuckDB EXIF table.")
    parser.add_argument("--db", default="exif_csv.db", help="DuckDB database path.")
    parser.add_argument("--exif-table", default="exif", help="EXIF table name.")
    parser.add_argument("--analysis-table", default="analysis", help="Output analysis table.")
    parser.add_argument("--summary-table", default="summnary", help="Output summary table (requested name).")
    parser.add_argument("--filename-col", default="FileName", help="Column with file name/path.")
    parser.add_argument("--datetime-col", default="ModifyDate", help="Column with date/time.")
    parser.add_argument("--aperture-col", default="Aperture", help="Aperture column (F-number).")
    parser.add_argument("--shutter-col", default="ShutterSpeed", help="Shutter speed column (seconds or fraction).")
    parser.add_argument("--iso-col", default="ISOSetting", help="ISO sensitivity column.")
    parser.add_argument("--model-col", default="Model", help="Camera model column.")
    parser.add_argument("--image-root", default="", help="Root folder to prepend to FileName if relative.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to analyze.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output tables.")
    parser.add_argument("--graphs-dir", default="graphs", help="Directory to save graphs.")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Force device; default auto.")
    parser.add_argument("--topk", type=int, default=5, help="Top-K labels for description/categories.")
    return parser.parse_args()


# ---------------------------
# Utilities
# ---------------------------

def setup_device(choice: Optional[str]) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize_path(name: str, root: str) -> str:
    name = str(name).strip()
    full = name if os.path.isabs(name) or not root else os.path.join(root, name)
    return os.path.normpath(os.path.expanduser(full))

def is_jpg(path: str) -> bool:
    return str(path).lower().endswith((".jpg", ".jpeg"))

def safe_parse_datetime(val: object) -> Optional[datetime]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    fmts = ["%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y:%m:%d"]
    for f in fmts:
        try:
            return datetime.strptime(s, f)
        except Exception:
            continue
    return None

def extract_aperture(val: object) -> Optional[float]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)
    except Exception:
        m = re.search(r"(\d+(\.\d+)?)", str(val))
        return float(m.group(1)) if m else None

def extract_shutter(val: object) -> Optional[float]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip().lower()
    frac = re.match(r"(\d+)\s*/\s*(\d+)", s)
    if frac:
        num, den = float(frac.group(1)), float(frac.group(2))
        return num / den if den else None
    try:
        return float(s)
    except Exception:
        return None

def shutter_display(v: Optional[float]) -> Optional[str]:
    if v is None:
        return None
    if v >= 1:
        return f"{round(v, 3)} s"
    denom = int(round(1.0 / v))
    denom = max(1, denom)
    return f"1/{denom} s"

def normalize_iso(val: object) -> Optional[int]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        iso_val = int(val)
    except Exception:
        try:
            iso_val = int(float(val))
        except Exception:
            return None
    stops = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]
    return min(stops, key=lambda s: abs(s - iso_val))

def build_label_to_category() -> Dict[str, str]:
    return {
        # animals
        "dog": "animals", "cat": "animals", "bird": "animals", "horse": "animals",
        "cow": "animals", "sheep": "animals", "lion": "animals", "tiger": "animals",
        # vehicles
        "car": "vehicles", "sports car": "vehicles", "convertible": "vehicles",
        "jeep": "vehicles", "ship": "vehicles", "airliner": "vehicles", "bicycle": "vehicles",
        "motorcycle": "vehicles", "bus": "vehicles", "train": "vehicles",
        # architecture / urban
        "building": "architecture", "church": "architecture", "castle": "architecture",
        "monastery": "architecture", "mosque": "architecture", "tower": "architecture",
        "bridge": "architecture", "street": "urban", "city": "urban",
        # nature
        "park": "nature", "forest": "nature", "valley": "nature", "mountain": "nature",
        "lake": "nature", "sea": "nature", "beach": "nature",
        # people
        "person": "people", "woman": "people", "man": "people", "child": "people",
        # objects
        "camera": "objects", "laptop": "objects", "cellular telephone": "objects", "smartphone": "objects",
        # food
        "food": "food", "pizza": "food", "hamburger": "food", "coffee": "food",
    }

def prepare_model(device: torch.device):
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.eval()
    model.to(device)
    preprocess = weights.transforms()
    idx_to_label = weights.meta["categories"]
    return model, preprocess, idx_to_label

def load_image_safely(path: str) -> Optional[Image.Image]:
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError, OSError):
        return None

def describe_from_labels(labels: List[str]) -> str:
    if not labels:
        return "No recognizable content."
    if len(labels) == 1:
        return f"Photo likely showing {labels[0]}."
    return f"Photo likely showing {', '.join(labels[:-1])}, and {labels[-1]}."

def infer_categories(labels: List[str], category_map: Dict[str, str]) -> List[str]:
    cats = []
    for l in labels:
        low = l.lower()
        if low in category_map:
            cats.append(category_map[low])
            continue
        for k, v in category_map.items():
            if k in low:
                cats.append(v)
                break
    # dedupe
    out, seen = [], set()
    for c in cats:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out or ["uncategorized"]


# ---------------------------
# Main
# ---------------------------

def main():
    args = parse_args()
    device = setup_device(args.device)
    os.makedirs(args.graphs_dir, exist_ok=True)

    # Connect DuckDB and load data
    conn = duckdb.connect(args.db)
    exif_df = conn.execute(f"SELECT * FROM {args.exif_table}").fetch_df()

    # Build full paths; filter JPGs
    full_paths = exif_df[args.filename_col].astype(str).map(lambda x: normalize_path(x, args.image_root))
    mask = full_paths.map(is_jpg)
    work_df = exif_df.loc[mask].copy()
    work_df["__path__"] = full_paths[mask]

    if args.limit is not None:
        work_df = work_df.iloc[:args.limit].copy()

    if work_df.empty:
        print("No JPG files found to analyze.")
        sys.exit(0)

    # Prepare model
    print(f"Loading model on device: {device}")
    model, preprocess, idx_to_label = prepare_model(device)
    category_map = build_label_to_category()

    # Inference and analysis rows
    records = []
    missing_files = 0

    for _, row in tqdm(work_df.iterrows(), total=len(work_df)):
        filepath = row["__path__"]
        exists = os.path.exists(filepath)
        if not exists:
            missing_files += 1

        labels = []
        desc = "No recognizable content."
        if exists:
            img = load_image_safely(filepath)
            if img is not None:
                tensor = preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(tensor)
                    probs = torch.softmax(logits, dim=1)
                    vals, inds = probs.topk(args.topk, dim=1)
                    labels = [idx_to_label[i.item()] for i in inds[0]]
                    desc = describe_from_labels(labels)

        cats = infer_categories(labels, category_map)

        dt = safe_parse_datetime(row.get(args.datetime_col))
        year = dt.year if dt else None
        month = dt.month if dt else None

        aperture = extract_aperture(row.get(args.aperture_col))
        shutter_seconds = extract_shutter(row.get(args.shutter_col))
        shutter_disp = shutter_display(shutter_seconds)

        # Clean ISO and normalize stops
        iso_raw_val = row.get(args.iso_col)
        iso_clean = pd.to_numeric([iso_raw_val], errors="coerce")[0]
        iso_norm = normalize_iso(iso_clean)

        model_name = row.get(args.model_col)

        records.append({
            "filename": filepath,
            "exists": exists,
            "labels_topk": json.dumps(labels),
            "categories": json.dumps(cats),
            "description": desc,
            "year": year,
            "month": month,
            "camera_model": model_name,
            "aperture": aperture,
            "shutter_seconds": shutter_seconds,
            "shutter_display": shutter_disp,
            "iso": iso_clean,
            "iso_normalized": iso_norm,
        })

    analysis_df = pd.DataFrame.from_records(records)

    # Write analysis table
    if args.overwrite:
        conn.execute(f"DROP TABLE IF EXISTS {args.analysis_table}")
    conn.register("analysis_df", analysis_df)
    conn.execute(f"CREATE TABLE IF NOT EXISTS {args.analysis_table} AS SELECT * FROM analysis_df LIMIT 0")
    conn.execute(f"DELETE FROM {args.analysis_table}")
    conn.execute(f"INSERT INTO {args.analysis_table} SELECT * FROM analysis_df")
    print(f"Wrote {len(analysis_df)} rows to '{args.analysis_table}'.")

    # Build summary table (year, month, count)
    summary = (
        analysis_df.dropna(subset=["year", "month"])
        .groupby(["year", "month"], as_index=False)["filename"]
        .count()
        .rename(columns={"filename": "count"})
        .sort_values(["year", "month"])
    )
    if args.overwrite:
        conn.execute(f"DROP TABLE IF EXISTS {args.summary_table}")
    conn.register("summary", summary)
    conn.execute(f"CREATE TABLE IF NOT EXISTS {args.summary_table} AS SELECT * FROM summary LIMIT 0")
    conn.execute(f"DELETE FROM {args.summary_table}")
    conn.execute(f"INSERT INTO {args.summary_table} SELECT * FROM summary")
    print(f"Wrote {len(summary)} rows to '{args.summary_table}'.")

    # Graphs
    saved_graphs = []

    # Camera model counts
    cm = analysis_df.dropna(subset=["camera_model"])
    if not cm.empty:
        plt.figure(figsize=(10, 6))
        cm["camera_model"].value_counts().sort_values(ascending=False).plot(kind="bar")
        plt.title("Photos by camera model")
        plt.xlabel("Camera model")
        plt.ylabel("Count")
        plt.tight_layout()
        cm_path = os.path.join(args.graphs_dir, "camera_model_counts.png")
        plt.savefig(cm_path)
        plt.close()
        saved_graphs.append(cm_path)

    # Aperture distribution (F-number)
    ap = analysis_df.dropna(subset=["aperture"])
    if not ap.empty:
        plt.figure(figsize=(10, 6))
        bins = [1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.0, 16.0, 22.0, 32.0]
        plt.hist(ap["aperture"], bins=bins, edgecolor="black")
        plt.title("Aperture distribution (F-number)")
        plt.xlabel("F-number")
        plt.ylabel("Count")
        plt.tight_layout()
        ap_path = os.path.join(args.graphs_dir, "aperture_distribution.png")
        plt.savefig(ap_path)
        plt.close()
        saved_graphs.append(ap_path)

    # Shutter speed distribution (seconds, log scale)
    ss = analysis_df.dropna(subset=["shutter_seconds"])
    if not ss.empty:
        plt.figure(figsize=(10, 6))
        plt.hist(ss["shutter_seconds"], bins=50, edgecolor="black")
        plt.xscale("log")
        plt.title("Shutter speed distribution (seconds, log scale)")
        plt.xlabel("Seconds (log scale)")
        plt.ylabel("Count")
        plt.tight_layout()
        ss_path = os.path.join(args.graphs_dir, "shutter_speed_distribution.png")
        plt.savefig(ss_path)
        plt.close()
        saved_graphs.append(ss_path)

    # ISO raw distribution (safe conversion)
    iso_raw = analysis_df.dropna(subset=["iso"])
    if not iso_raw.empty:
        plt.figure(figsize=(10, 6))
        iso_vals = pd.to_numeric(iso_raw["iso"], errors="coerce").dropna()
        plt.hist(iso_vals.astype(float), bins=30, edgecolor="black")
        plt.title("ISO setting distribution (raw)")
        plt.xlabel("ISO value")
        plt.ylabel("Count")
        plt.tight_layout()
        iso_raw_path = os.path.join(args.graphs_dir, "iso_distribution_raw.png")
        plt.savefig(iso_raw_path)
        plt.close()
        saved_graphs.append(iso_raw_path)

    # ISO normalized stops
    iso_norm_df = analysis_df.dropna(subset=["iso_normalized"])
    if not iso_norm_df.empty:
        plt.figure(figsize=(10, 6))
        iso_norm_df["iso_normalized"].value_counts().sort_index().plot(kind="bar")
        plt.title("ISO setting distribution (normalized stops)")
        plt.xlabel("ISO value (standard stops)")
        plt.ylabel("Count")
        plt.tight_layout()
        iso_norm_path = os.path.join(args.graphs_dir, "iso_distribution_normalized.png")
        plt.savefig(iso_norm_path)
        plt.close()
        saved_graphs.append(iso_norm_path)

    if saved_graphs:
        print("Saved graphs:")
        for p in saved_graphs:
            print(f" - {p}")
    else:
        print("No graphs generated (insufficient data).")

    # Final notes
    missing_msg = "some files were missing" if missing_files > 0 else "all files found"
    print(f"Completed analysis; {missing_msg}.")
    print("Done.")


if __name__ == "__main__":
    main()
