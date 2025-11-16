"# exif" 
Program Description
This Python script, exif_ana.py, is a command-line tool designed to analyze JPEG images referenced in a DuckDB database table containing EXIF metadata (e.g., from tools like ExifTool). It performs two main types of analysis per image:

Metadata Extraction and Normalization:
Parses and cleans EXIF data such as date/time (from a specified column like ModifyDate), aperture (F-number), shutter speed (converting fractions like "1/60" to seconds), ISO (raw value and normalized to standard stops like 100, 200, etc.), and camera model.
Handles edge cases like missing values, non-standard formats (e.g., regex for numeric extraction), and displays shutter speeds in a human-readable format (e.g., "1/60 s").

Image Content Inference:
Uses a pre-trained ResNet50 model from PyTorch (with ImageNet weights) to classify the top-K (default 5) labels for each image's content.
Generates a textual description (e.g., "Photo likely showing dog, cat, and bird.") and infers broad categories (e.g., "animals", "vehicles") based on a hardcoded mapping of labels to categories.
Skips missing or invalid images, tracking existence.


The script processes images in batches (limited optionally via CLI), storing results in:

An analysis table (default: analysis) in the DuckDB database, with one row per image including paths, labels, categories, description, normalized metadata, etc.
A summary table (default: summary, though misspelled as summnary in code) aggregating photo counts by year-month.

Finally, it generates and saves PNG graphs in a specified directory (default: graphs/):

Bar chart: Photos by camera model.
Histogram: Aperture distribution (binned F-stops).
Histogram: Shutter speed (log-scaled seconds).
Histogram: Raw ISO values.
Bar chart: Normalized ISO stops.

Usage Example:
textpython exif_ana.py --db exif_csv.db --exif-table exif --image-root /path/to/photos --limit 100 --overwrite --device cuda
It requires libraries like DuckDB, Pandas, PyTorch, torchvision, PIL, Matplotlib, tqdm, and assumes a DuckDB file with an EXIF table. Runs on CPU or GPU (auto-detected). Output includes progress bars, row counts, and a list of saved graphs.
Overall, it's a useful tool for photographers or data enthusiasts to summarize and visualize a photo library's metadata and content trends.
Suggested Improvements
Here are targeted suggestions to enhance robustness, usability, maintainability, and functionality. I've grouped them by category, prioritizing quick wins (e.g., bugs) first.
Bug Fixes and Typos

Fix docstring and filename inconsistencies: The initial docstring refers to "exit_ana.py" (typo for "exif_ana.py"). Update to match. Also, remove the duplicated shebang (#!/usr/bin/env python) and docstring block at the top—it's redundant.
Correct summary table default: --summary-table defaults to "summnary" (typo); change to "summary".
ISO cleaning in analysis loop: pd.to_numeric([iso_raw_val], errors="coerce")[0] works but is inefficient for a scalar. Use pd.to_numeric(iso_raw_val, errors="coerce") directly.
Graph for raw ISO: In the plotting section, iso_vals = pd.to_numeric(iso_raw["iso"], errors="coerce").dropna() is redundant since dropna(subset=["iso"]) already filters NaNs—simplify to plt.hist(iso_raw["iso"].astype(float), bins=30, edgecolor="black").
Shutter extraction edge case: extract_shutter assumes fractions are "num/den" but doesn't handle decimals well (e.g., "0.0167" for 1/60). Add a fallback: if float(s) succeeds and 0 < v < 1, treat as seconds directly.

Performance and Efficiency

Batch inference for speed: Currently processes one image at a time in the loop, which is slow for large datasets (e.g., 1000+ images). Collect valid images first, then batch them (e.g., torch.utils.data.DataLoader with batch_size=32) for ResNet inference. This could reduce GPU/CPU time by 5-10x.
Limit tqdm to actual progress: The loop uses tqdm(work_df.iterrows()), but inference only happens if the file exists and loads. Move tqdm inside the if exists: block or use a counter for accurate progress.
Memory management: For large datasets, load images lazily or use torch.utils.data.Dataset to avoid OOM on GPU. Add torch.cuda.empty_cache() after inference if using CUDA.
Optional model quantization: For CPU-only runs, suggest loading a quantized ResNet50 (e.g., torch.quantization.quantize_dynamic) to speed up inference.

Functionality Enhancements

Expand ISO normalization: The stops list starts at 100 but misses common values like 50, 64, 125, 250, 4000, 51200. Update to: [25, 50, 64, 100, 125, 160, 200, 250, 320, 400, 500, 640, 800, 1000, 1250, 1600, 2000, 2500, 3200, 4000, 5000, 6400, 8000, 10000, 12800, 16000, 20000, 25600, 32000, 40000, 51200]. Also, compute "stops from 100" as a new column: round(log2(iso / 100) / log2(sqrt(2))) for true EV normalization.
Add more metadata support: Include optional CLI args for focal length (--focal-col), exposure compensation, or lens model. Compute derived metrics like "exposure value" (EV): ev = log2(aperture**2 / shutter_seconds) - log2(iso / 100) and plot its distribution.
Improve content analysis:
The category map is hardcoded and limited (e.g., misses "tree" → "nature"). Use a more comprehensive mapping or integrate a lightweight NLP step (e.g., via Hugging Face's zero-shot classification) for better categorization.
Add confidence scores: Include top-K probabilities in the JSON output (e.g., labels_topk: [{"label": "dog", "prob": 0.85}, ...]).
Handle non-RGB images better: convert("RGB") fails on grayscale; add a check or fallback.

Summary table expansion: Currently just counts; add aggregates like avg aperture/ISO per month, or top categories. Use SQL in DuckDB for this (e.g., conn.execute("CREATE TABLE ... AS SELECT year, month, COUNT(*) as count, AVG(aperture) ... FROM analysis GROUP BY ...")) to avoid Pandas.
Graph improvements:
Add monthly trend line plot (e.g., photos over time using summary data).
Use Seaborn for prettier plots (e.g., sns.histplot with KDE).
Make bins dynamic: For histograms, use np.histogram_bin_edges with Freedman-Diaconis rule instead of fixed bins.
Export data for graphs as CSV (e.g., alongside PNG) for external tools.


Usability and Maintainability

Logging over prints: Replace print statements with logging module (e.g., import logging; logging.basicConfig(level=logging.INFO)). Log warnings for missing files, invalid metadata, etc.
Error handling: Wrap model loading/inference in try-except (e.g., catch RuntimeError for CUDA OOM). Add --dry-run flag to validate paths without inference.
CLI enhancements: Add --output-format (e.g., JSON/CSV export besides DuckDB). Use pathlib.Path for paths instead of os.path for cleaner code.
Type hints and docs: Add more hints (e.g., def extract_aperture(val: object) -> Optional[float]:). Use Sphinx-style docstrings for functions.
Dependencies: Pin versions in a requirements.txt (e.g., torch>=2.0, duckdb>=0.8). Note that PyTorch requires CUDA toolkit for GPU—add a warning if --device cuda but unavailable.
Testing: Add unit tests (e.g., via pytest) for utils like extract_shutter with sample inputs. Mock image loading for CI.

These changes would make the script more robust for production use (e.g., 10k+ images) while keeping it concise. Start with bug fixes and batching for immediate impact! If you share sample data or a specific focus (e.g., performance), I can refine further.generate new version of program using suggestionsThought for 44spython#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
exif_ana.py
Analyze JPGs referenced in a DuckDB EXIF table, store per-photo analysis and monthly summary,
and generate graphs for camera model, aperture, shutter speed, ISO (raw + normalized stops),
and exposure value.

Usage:
    python exif_ana.py --db exif_csv.db --exif-table exif --image-root /path/to/photos --overwrite

Author: Copilot (updated with improvements)
"""

import argparse
import json
import logging
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import pandas as pd
from tqdm import tqdm

import torch
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
from torchvision import models


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --------------------------- CLI --------------------------- 

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze JPGs in DuckDB EXIF table.")
    parser.add_argument("--db", default="exif_csv.db", help="DuckDB database path.")
    parser.add_argument("--exif-table", default="exif", help="EXIF table name.")
    parser.add_argument("--analysis-table", default="analysis", help="Output analysis table.")
    parser.add_argument("--summary-table", default="summary", help="Output summary table.")
    parser.add_argument("--filename-col", default="FileName", help="Column with file name/path.")
    parser.add_argument("--datetime-col", default="ModifyDate", help="Column with date/time.")
    parser.add_argument("--aperture-col", default="Aperture", help="Aperture column (F-number).")
    parser.add_argument("--shutter-col", default="ShutterSpeed", help="Shutter speed column (seconds or fraction).")
    parser.add_argument("--iso-col", default="ISOSetting", help="ISO sensitivity column.")
    parser.add_argument("--model-col", default="Model", help="Camera model column.")
    parser.add_argument("--image-root", default="", help="Root folder to prepend to FileName if relative.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to analyze.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output tables (replaces CREATE OR REPLACE).")
    parser.add_argument("--graphs-dir", default="graphs", help="Directory to save graphs.")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Force device; default auto.")
    parser.add_argument("--topk", type=int, default=5, help="Top-K labels for description/categories.")
    parser.add_argument("--dry-run", action="store_true", help="Extract metadata only, skip image analysis/inference.")
    return parser.parse_args()


# --------------------------- Utilities --------------------------- 

def setup_device(choice: Optional[str]) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            logging.warning("CUDA requested but not available, falling back to CPU.")
            return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_path(name: str, root: str = "") -> str:
    name = str(name).strip()
    p = Path(name).expanduser()
    if p.is_absolute():
        return str(p.resolve())
    else:
        root_p = Path(root).expanduser().resolve() if root else Path.cwd().resolve()
        return str(root_p / p)


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
        except ValueError:
            continue
    return None


def extract_aperture(val: object) -> Optional[float]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        m = re.search(r"(\d+(?:\.\d+)?)", str(val))
        return float(m.group(1)) if m else None


def extract_shutter(val: object) -> Optional[float]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip().lower()
    # Try fraction (support decimals)
    frac_match = re.match(r"^(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)$", s)
    if frac_match:
        num, den = map(float, frac_match.groups())
        return num / den if den != 0 else None
    # Try direct float (for decimals like 0.0167)
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def shutter_display(v: Optional[float]) -> Optional[str]:
    if v is None or v <= 0:
        return None
    if v >= 1:
        return f"{v:.3f} s"
    denom = max(1, round(1 / v))
    return f"1/{denom} s"


def normalize_iso(iso_val: Optional[float]) -> Optional[int]:
    if iso_val is None or pd.isna(iso_val) or iso_val <= 0:
        return None
    stops = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200]
    closest = min(stops, key=lambda s: abs(math.log2(iso_val / s)))
    return closest


def build_label_to_category() -> Dict[str, str]:
    return {
        # animals
        "dog": "animals", "cat": "animals", "bird": "animals", "horse": "animals",
        "cow": "animals", "sheep": "animals", "lion": "animals", "tiger": "animals",
        "fish": "animals", "snake": "animals",
        # vehicles
        "car": "vehicles", "sports car": "vehicles", "convertible": "vehicles",
        "jeep": "vehicles", "ship": "vehicles", "airliner": "vehicles", "airplane": "vehicles",
        "bicycle": "vehicles", "motorcycle": "vehicles", "bus": "vehicles", "train": "vehicles",
        "helicopter": "vehicles",
        # architecture / urban
        "building": "architecture", "church": "architecture", "castle": "architecture",
        "monastery": "architecture", "mosque": "architecture", "tower": "architecture",
        "bridge": "architecture", "house": "architecture", "skyscraper": "architecture",
        "stadium": "architecture", "street": "urban", "city": "urban",
        # nature
        "park": "nature", "forest": "nature", "valley": "nature", "mountain": "nature",
        "lake": "nature", "sea": "nature", "ocean": "nature", "beach": "nature",
        "tree": "nature", "flower": "nature", "river": "nature", "snow": "nature", "sun": "nature",
        # people
        "person": "people", "woman": "people", "man": "people", "child": "people",
        "baby": "people", "girl": "people", "boy": "people",
        # objects
        "camera": "objects", "laptop": "objects", "cellular telephone": "objects", 
        "smartphone": "objects", "phone": "objects", "book": "objects", 
        "bottle": "objects", "chair": "objects",
        # food
        "food": "food", "pizza": "food", "hamburger": "food", "coffee": "food",
    }


def prepare_model(device: torch.device):
    try:
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model.eval()
        model.to(device)
        preprocess = weights.transforms()
        idx_to_label = weights.meta["categories"]
        return model, preprocess, idx_to_label
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        sys.exit(1)


def load_image_safely(path: str) -> Optional[Image.Image]:
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
        logging.debug(f"Image load error for {path}: {e}")
        return None


def describe_from_labels(labels: List[str]) -> str:
    if not labels:
        return "No recognizable content."
    if len(labels) == 1:
        return f"A photo of {labels[0]}."
    return f"A photo of {', '.join(labels[:-1])}, and {labels[-1]}."


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
    # Dedupe
    out = list(dict.fromkeys(cats))  # preserves order
    return out or ["uncategorized"]


# --------------------------- Main --------------------------- 

def main():
    args = parse_args()
    device = setup_device(args.device)
    graphs_dir = Path(args.graphs_dir)
    graphs_dir.mkdir(exist_ok=True)

    # Connect DuckDB and load data
    conn = duckdb.connect(args.db)
    exif_df = conn.execute(f"SELECT * FROM {args.exif_table}").fetch_df()

    # Build full paths; filter JPGs
    full_paths = exif_df[args.filename_col].astype(str).apply(lambda x: normalize_path(x, args.image_root))
    mask = full_paths.apply(is_jpg)
    work_df = exif_df.loc[mask].copy()
    work_df["__path__"] = full_paths[mask]

    if args.limit is not None:
        work_df = work_df.iloc[:args.limit].copy()

    if work_df.empty:
        logging.info("No JPG files found to analyze.")
        return

    # Prepare model
    logging.info(f"Loading model on device: {device}")
    model, preprocess, idx_to_label = prepare_model(device)
    category_map = build_label_to_category()

    # Process metadata and prepare for inference
    records: List[Dict] = []
    tensors_to_process: List[torch.Tensor] = []
    indices_to_update: List[int] = []
    missing_files = 0

    for _, row in tqdm(work_df.iterrows(), total=len(work_df), desc="Extracting metadata and loading images"):
        filepath = row["__path__"]
        exists = Path(filepath).exists()

        if not exists:
            missing_files += 1
            logging.debug(f"Missing file: {filepath}")

        dt = safe_parse_datetime(row.get(args.datetime_col))
        year = dt.year if dt else None
        month = dt.month if dt else None

        aperture = extract_aperture(row.get(args.aperture_col))
        shutter_seconds = extract_shutter(row.get(args.shutter_col))
        shutter_disp = shutter_display(shutter_seconds)

        iso_raw_val = row.get(args.iso_col)
        iso_clean = pd.to_numeric(iso_raw_val, errors="coerce")
        iso_norm = normalize_iso(iso_clean)

        model_name = row.get(args.model_col)

        # Compute exposure value
        ev_val = None
        if (aperture and shutter_seconds and shutter_seconds > 0 and 
            iso_clean and iso_clean > 0 and not pd.isna(iso_clean)):
            try:
                ev = math.log2(aperture**2 / shutter_seconds) - math.log2(iso_clean / 100)
                ev_val = round(ev, 2)
            except (ValueError, OverflowError):
                pass

        base_record = {
            "filename": filepath,
            "exists": exists,
            "labels_topk": json.dumps([]),
            "categories": json.dumps(["uncategorized"]),
            "description": "No analysis performed.",
            "year": year,
            "month": month,
            "camera_model": model_name,
            "aperture": aperture,
            "shutter_seconds": shutter_seconds,
            "shutter_display": shutter_disp,
            "iso": iso_clean,
            "iso_normalized": iso_norm,
            "exposure_value": ev_val,
        }
        records.append(base_record)

        if exists:
            img = load_image_safely(filepath)
            if img is not None:
                tensor = preprocess(img).unsqueeze(0)
                tensors_to_process.append(tensor)
                indices_to_update.append(len(records) - 1)
            else:
                logging.warning(f"Could not load image: {filepath}")

    analysis_df = pd.DataFrame(records)

    # Run batched inference if not dry-run
    if not args.dry_run and tensors_to_process:
        logging.info(f"Running batched inference on {len(tensors_to_process)} images (batch size 32)")
        batch_size = 32
        with torch.no_grad():
            for start in range(0, len(tensors_to_process), batch_size):
                end = min(start + batch_size, len(tensors_to_process))
                batch = torch.stack(tensors_to_process[start:end]).to(device)
                logits = model(batch)
                probs = torch.softmax(logits, dim=1)
                top_probs, top_inds = torch.topk(probs, args.topk, dim=1)
                for b in range(batch.shape[0]):
                    local_idx = start + b
                    rec_idx = indices_to_update[local_idx]
                    labels = []
                    for j in range(args.topk):
                        label_name = idx_to_label[int(top_inds[b, j])]
                        conf = float(top_probs[b, j])
                        labels.append({"label": label_name, "confidence": conf})
                    records[rec_idx]["labels_topk"] = json.dumps(labels)
                    label_names = [l["label"] for l in labels]
                    records[rec_idx]["description"] = describe_from_labels(label_names)
                    cats = infer_categories(label_names, category_map)
                    records[rec_idx]["categories"] = json.dumps(cats)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Refresh df after updates
        analysis_df = pd.DataFrame(records)
        logging.info("Inference completed.")
    else:
        logging.info("Skipping inference (dry-run mode or no valid images).")

    # Write analysis table
    if args.overwrite:
        conn.execute(f"DROP TABLE IF EXISTS {args.analysis_table}")
        conn.execute(f"CREATE TABLE {args.analysis_table} AS SELECT * FROM analysis_df")
    else:
        try:
            conn.execute(f"INSERT INTO {args.analysis_table} SELECT * FROM analysis_df")
        except Exception:
            conn.execute(f"CREATE TABLE {args.analysis_table} AS SELECT * FROM analysis_df")
    logging.info(f"Wrote {len(analysis_df)} rows to '{args.analysis_table}'.")

    # Build and write summary table
    summary = (
        analysis_df.dropna(subset=["year", "month"])
        .groupby(["year", "month"])
        .agg(
            count=("filename", "size"),
            avg_aperture=("aperture", "mean"),
            avg_shutter=("shutter_seconds", "mean"),
            avg_iso=("iso", "mean"),
            avg_ev=("exposure_value", "mean"),
        )
        .round(2)
        .reset_index()
        .sort_values(["year", "month"])
    )
    if args.overwrite:
        conn.execute(f"DROP TABLE IF EXISTS {args.summary_table}")
        conn.execute(f"CREATE TABLE {args.summary_table} AS SELECT * FROM summary")
    else:
        try:
            conn.execute(f"INSERT INTO {args.summary_table} SELECT * FROM summary")
        except Exception:
            conn.execute(f"CREATE TABLE {args.summary_table} AS SELECT * FROM summary")
    logging.info(f"Wrote {len(summary)} rows to '{args.summary_table}'.")

    # Generate graphs
    saved_graphs = []

    # Camera model counts
    cm = analysis_df.dropna(subset=["camera_model"])
    if not cm.empty:
        plt.figure(figsize=(10, 6))
        cm["camera_model"].value_counts().sort_values(ascending=False).plot(kind="bar")
        plt.title("Photos by camera model")
        plt.xlabel("Camera model")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        cm_path = graphs_dir / "camera_model_counts.png"
        plt.savefig(cm_path)
        plt.close()
        saved_graphs.append(cm_path)

    # Aperture distribution
    ap = analysis_df.dropna(subset=["aperture"])
    if not ap.empty:
        plt.figure(figsize=(10, 6))
        bins = [1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.0, 16.0, 22.0, 32.0]
        plt.hist(ap["aperture"], bins=bins, edgecolor="black")
        plt.title("Aperture distribution (F-number)")
        plt.xlabel("F-number")
        plt.ylabel("Count")
        plt.tight_layout()
        ap_path = graphs_dir / "aperture_distribution.png"
        plt.savefig(ap_path)
        plt.close()
        saved_graphs.append(ap_path)

    # Shutter speed distribution (log scale)
    ss = analysis_df.dropna(subset=["shutter_seconds"])
    if not ss.empty and (ss["shutter_seconds"] > 0).all():
        plt.figure(figsize=(10, 6))
        plt.hist(ss["shutter_seconds"], bins=50, edgecolor="black")
        plt.xscale("log")
        plt.title("Shutter speed distribution (seconds, log scale)")
        plt.xlabel("Seconds (log scale)")
        plt.ylabel("Count")
        plt.tight_layout()
        ss_path = graphs_dir / "shutter_speed_distribution.png"
        plt.savefig(ss_path)
        plt.close()
        saved_graphs.append(ss_path)

    # ISO raw distribution
    iso_raw = analysis_df.dropna(subset=["iso"])
    if not iso_raw.empty:
        plt.figure(figsize=(10, 6))
        plt.hist(iso_raw["iso"].astype(float), bins=30, edgecolor="black")
        plt.title("ISO setting distribution (raw)")
        plt.xlabel("ISO value")
        plt.ylabel("Count")
        plt.tight_layout()
        iso_raw_path = graphs_dir / "iso_distribution_raw.png"
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
        iso_norm_path = graphs_dir / "iso_distribution_normalized.png"
        plt.savefig(iso_norm_path)
        plt.close()
        saved_graphs.append(iso_norm_path)

    # Exposure Value distribution
    ev_df = analysis_df.dropna(subset=["exposure_value"])
    if not ev_df.empty:
        plt.figure(figsize=(10, 6))
        plt.hist(ev_df["exposure_value"], bins=20, edgecolor="black")
        plt.title("Exposure Value (EV) distribution")
        plt.xlabel("EV")
        plt.ylabel("Count")
        plt.tight_layout()
        ev_path = graphs_dir / "exposure_value_distribution.png"
        plt.savefig(ev_path)
        plt.close()
        saved_graphs.append(ev_path)

    if saved_graphs:
        logging.info("Saved graphs:")
        for p in saved_graphs:
            logging.info(f" - {p}")
    else:
        logging.info("No graphs generated (insufficient data).")

    # Final notes
    missing_msg = f"{missing_files} files were missing" if missing_files > 0 else "all files found"
    logging.info(f"Completed analysis; {missing_msg}.")
    logging.info("Done.")


if __name__ == "__main__":
    main()generate a README.md file for the program and include instructions on how to use itThought for 3sEXIF Analyzer
Python Version
License: MIT
exif_ana.py is a command-line Python tool for analyzing JPEG images referenced in a DuckDB database table containing EXIF metadata (e.g., extracted via ExifTool). It normalizes and enriches metadata (aperture, shutter speed, ISO, exposure value), performs content classification using a pre-trained ResNet50 model, generates textual descriptions and categories, and visualizes trends via graphs.
Ideal for photographers, data enthusiasts, or anyone managing a photo library to uncover patterns in gear usage, shooting styles, and content themes.
Features

Metadata Processing: Clean and normalize EXIF data like dates, aperture (F-stops), shutter speeds (fractions to seconds), ISO (raw and standard stops), camera models, and computed exposure value (EV).
Image Content Analysis: Classify images with ImageNet-pretrained ResNet50 (top-K labels with confidence scores), generate natural-language descriptions, and infer categories (e.g., "animals", "nature").
Database Integration: Read from an EXIF table in DuckDB; output detailed analysis and monthly summaries to new tables.
Visualization: Generate PNG graphs for camera usage, aperture/shutter/ISO/EV distributions, and more.
Efficiency: Batched GPU/CPU inference (auto-detected), progress tracking, dry-run mode, and optional limits for large datasets.
Robustness: Handles missing files, invalid images/metadata, and flexible path resolution.

Requirements

Python 3.8+
Dependencies (install via pip):textpip install duckdb pandas tqdm torch torchvision pillow matplotlib
PyTorch Note: For GPU acceleration, install with CUDA support (e.g., pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121). Falls back to CPU if unavailable.

A DuckDB database file (.db) with an EXIF table (e.g., from exiftool -csv ... > exif.csv then imported).

No additional setup required—runs as a standalone script.
Installation

Clone or download exif_ana.py.
Install dependencies: pip install -r requirements.txt (create one if needed with the list above).
Ensure your DuckDB file exists and contains the EXIF table (columns like FileName, ModifyDate, etc.).

Usage
Run the script via command line:
bashpython exif_ana.py [OPTIONS]
Key Options

















































































































FlagDescriptionDefaultExample--dbPath to DuckDB databaseexif_csv.db--db my_photos.db--exif-tableInput EXIF table nameexif--exif-table metadata--analysis-tableOutput analysis tableanalysis--analysis-table photo_analysis--summary-tableOutput monthly summary tablesummary--summary-table monthly_stats--filename-colColumn for image pathsFileName--filename-col Path--datetime-colColumn for date/timeModifyDate--datetime-col DateTimeOriginal--aperture-colAperture (F-number) columnAperture--aperture-col FNumber--shutter-colShutter speed columnShutterSpeed--shutter-col ExposureTime--iso-colISO columnISOSetting--iso-col ISOSpeedRating--model-colCamera model columnModel--model-col CameraModelName--image-rootRoot dir to prepend to relative paths""--image-root /Users/me/Pictures--limitMax images to processNone (all)--limit 500--overwriteDrop/recreate output tablesFalse--overwrite--graphs-dirDir to save PNG graphsgraphs--graphs-dir output/graphs--deviceForce CPU or CUDAAuto--device cuda--topkTop-K labels for classification5--topk 3--dry-runSkip inference (metadata only)False--dry-run
Examples

Basic Run (analyze all JPGs in default setup):bashpython exif_ana.py --db exif_csv.db --image-root ~/Photos
Limited Analysis with GPU and Overwrite:bashpython exif_ana.py --db photos.db --limit 1000 --device cuda --overwrite --graphs-dir viz/
Dry Run for Metadata Only (fast validation):bashpython exif_ana.py --dry-run --exif-table exif_data --image-root /media/photos
Custom Columns (e.g., for ExifTool CSV):bashpython exif_ana.py --filename-col SourceFile --datetime-col CreateDate --aperture-col ApertureValue --shutter-col ExposureTime --iso-col PhotographicSensitivity --model-col CameraModelName --db myexif.db

Output

Console: Progress bar, log messages (INFO/WARNING), row counts, saved graph paths.
Database Tables:
analysis: Per-image details (e.g., filename, exists, labels_topk as JSON, description, normalized metadata, exposure_value).
summary: Aggregates by year-month (count, avg aperture/shutter/ISO/EV).

Graphs (saved as PNG in --graphs-dir):
camera_model_counts.png: Bar chart of photos by model.
aperture_distribution.png: Histogram of F-stops.
shutter_speed_distribution.png: Log-scaled shutter speeds.
iso_distribution_raw.png: Raw ISO histogram.
iso_distribution_normalized.png: Normalized ISO bars.
exposure_value_distribution.png: EV histogram.


Query outputs in DuckDB: SELECT * FROM analysis LIMIT 5; or visualize summaries with external tools.
Example Workflow

Extract EXIF: exiftool -csv -r -FileName -ModifyDate -Aperture -ShutterSpeed -ISOSetting -Model /path/to/photos > exif.csv
Import to DuckDB: Open DuckDB CLI, run COPY exif FROM 'exif.csv' (AUTO_DETECT TRUE);
Run analysis: python exif_ana.py --db exif_csv.db --image-root /path/to/photos
Explore: Use DuckDB CLI or connect via Python/Jupyter for queries; view graphs in the graphs/ folder.

Troubleshooting

Missing Files: Script logs warnings; check paths with --dry-run.
CUDA Errors: Ensure PyTorch CUDA version matches your GPU driver. Use --device cpu.
Invalid Metadata: Handles NaNs gracefully; inspect raw EXIF for formats.
Large Datasets: Use --limit and GPU; monitor memory (adds torch.cuda.empty_cache()).
No Graphs: Requires sufficient valid data; check logs for filtered rows.
Dependencies: If torch fails, verify installation: python -c "import torch; print(torch.__version__)".

Contributing
Fork, PR improvements (e.g., new features like focal length support). Tests via pytest encouraged.
License
MIT License—see LICENSE (add one if needed).
Questions? Open an issue or reach out. Happy analyzing! 📸generate complete README.md for downloadThought for 4sEXIF Analyzer
Python Version
License: MIT
exif_ana.py is a command-line Python tool for analyzing JPEG images referenced in a DuckDB database table containing EXIF metadata (e.g., extracted via ExifTool). It normalizes and enriches metadata (aperture, shutter speed, ISO, exposure value), performs content classification using a pre-trained ResNet50 model, generates textual descriptions and categories, and visualizes trends via graphs.
Ideal for photographers, data enthusiasts, or anyone managing a photo library to uncover patterns in gear usage, shooting styles, and content themes.
Features

Metadata Processing: Clean and normalize EXIF data like dates, aperture (F-stops), shutter speeds (fractions to seconds), ISO (raw and standard stops), camera models, and computed exposure value (EV).
Image Content Analysis: Classify images with ImageNet-pretrained ResNet50 (top-K labels with confidence scores), generate natural-language descriptions, and infer categories (e.g., "animals", "nature").
Database Integration: Read from an EXIF table in DuckDB; output detailed analysis and monthly summaries to new tables.
Visualization: Generate PNG graphs for camera usage, aperture/shutter/ISO/EV distributions, and more.
Efficiency: Batched GPU/CPU inference (auto-detected), progress tracking, dry-run mode, and optional limits for large datasets.
Robustness: Handles missing files, invalid images/metadata, and flexible path resolution.

Requirements

Python 3.8+
Dependencies (install via pip):textpip install duckdb pandas tqdm torch torchvision pillow matplotlib
PyTorch Note: For GPU acceleration, install with CUDA support (e.g., pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121). Falls back to CPU if unavailable.

A DuckDB database file (.db) with an EXIF table (e.g., from exiftool -csv ... > exif.csv then imported).

No additional setup required—runs as a standalone script.
Installation

Clone or download exif_ana.py.
Install dependencies: pip install -r requirements.txt (create one if needed with the list above).
Ensure your DuckDB file exists and contains the EXIF table (columns like FileName, ModifyDate, etc.).

Usage
Run the script via command line:
bashpython exif_ana.py [OPTIONS]
Key Options

















































































































FlagDescriptionDefaultExample--dbPath to DuckDB databaseexif_csv.db--db my_photos.db--exif-tableInput EXIF table nameexif--exif-table metadata--analysis-tableOutput analysis tableanalysis--analysis-table photo_analysis--summary-tableOutput monthly summary tablesummary--summary-table monthly_stats--filename-colColumn for image pathsFileName--filename-col Path--datetime-colColumn for date/timeModifyDate--datetime-col DateTimeOriginal--aperture-colAperture (F-number) columnAperture--aperture-col FNumber--shutter-colShutter speed columnShutterSpeed--shutter-col ExposureTime--iso-colISO columnISOSetting--iso-col ISOSpeedRating--model-colCamera model columnModel--model-col CameraModelName--image-rootRoot dir to prepend to relative paths""--image-root /Users/me/Pictures--limitMax images to processNone (all)--limit 500--overwriteDrop/recreate output tablesFalse--overwrite--graphs-dirDir to save PNG graphsgraphs--graphs-dir output/graphs--deviceForce CPU or CUDAAuto--device cuda--topkTop-K labels for classification5--topk 3--dry-runSkip inference (metadata only)False--dry-run
Examples

Basic Run (analyze all JPGs in default setup):bashpython exif_ana.py --db exif_csv.db --image-root ~/Photos
Limited Analysis with GPU and Overwrite:bashpython exif_ana.py --db photos.db --limit 1000 --device cuda --overwrite --graphs-dir viz/
Dry Run for Metadata Only (fast validation):bashpython exif_ana.py --dry-run --exif-table exif_data --image-root /media/photos
Custom Columns (e.g., for ExifTool CSV):bashpython exif_ana.py --filename-col SourceFile --datetime-col CreateDate --aperture-col ApertureValue --shutter-col ExposureTime --iso-col PhotographicSensitivity --model-col CameraModelName --db myexif.db

Output

Console: Progress bar, log messages (INFO/WARNING), row counts, saved graph paths.
Database Tables:
analysis: Per-image details (e.g., filename, exists, labels_topk as JSON, description, normalized metadata, exposure_value).
summary: Aggregates by year-month (count, avg aperture/shutter/ISO/EV).

Graphs (saved as PNG in --graphs-dir):
camera_model_counts.png: Bar chart of photos by model.
aperture_distribution.png: Histogram of F-stops.
shutter_speed_distribution.png: Log-scaled shutter speeds.
iso_distribution_raw.png: Raw ISO histogram.
iso_distribution_normalized.png: Normalized ISO bars.
exposure_value_distribution.png: EV histogram.


Query outputs in DuckDB: SELECT * FROM analysis LIMIT 5; or visualize summaries with external tools.
Example Workflow

Extract EXIF: exiftool -csv -r -FileName -ModifyDate -Aperture -ShutterSpeed -ISOSetting -Model /path/to/photos > exif.csv
Import to DuckDB: Open DuckDB CLI, run COPY exif FROM 'exif.csv' (AUTO_DETECT TRUE);
Run analysis: python exif_ana.py --db exif_csv.db --image-root /path/to/photos
Explore: Use DuckDB CLI or connect via Python/Jupyter for queries; view graphs in the graphs/ folder.

Troubleshooting

Missing Files: Script logs warnings; check paths with --dry-run.
CUDA Errors: Ensure PyTorch CUDA version matches your GPU driver. Use --device cpu.
Invalid Metadata: Handles NaNs gracefully; inspect raw EXIF for formats.
Large Datasets: Use --limit and GPU; monitor memory (adds torch.cuda.empty_cache()).
No Graphs: Requires sufficient valid data; check logs for filtered rows.
Dependencies: If torch fails, verify installation: python -c "import torch; print(torch.__version__)".

Contributing
Fork, PR improvements (e.g., new features like focal length support). Tests via pytest encouraged.
License
MIT License—see LICENSE (add one if needed).
Questions? Open an issue or reach out. Happy analyzing! 📸Make more conciseThought for 3sEXIF Analyzer
Python 3.8+
MIT License
exif_ana.py analyzes JPEGs from DuckDB EXIF tables: normalizes metadata (aperture, shutter, ISO, EV), classifies content via ResNet50, generates descriptions/categories, and creates graphs.
For photo library insights on gear, styles, and themes.
Features

Normalize EXIF: Dates, F-stops, shutter (s), ISO (raw/stops), models, EV.
Content: Top-K labels/confidences, descriptions, categories (e.g., animals, nature).
DB: Input EXIF table → analysis/summary tables.
Viz: PNG graphs for models, distributions.
Efficient: Batched inference (GPU/CPU), dry-run, limits.

Requirements

Python 3.8+
pip install duckdb pandas tqdm torch torchvision pillow matplotlib
DuckDB .db with EXIF table (e.g., via ExifTool CSV import).
GPU: CUDA-enabled PyTorch for speed.

Quick Start

Save exif_ana.py.
Install deps above.
Prep DB: exiftool -csv -r -FileName -ModifyDate -Aperture -ShutterSpeed -ISOSetting -Model /photos > exif.csv; DuckDB: COPY exif FROM 'exif.csv' (AUTO_DETECT TRUE);.

Run: python exif_ana.py --db exif_csv.db --image-root /photos
Usage
bashpython exif_ana.py [OPTIONS]
Options






































































FlagDescriptionDefault/Example--dbDuckDB pathexif_csv.db / --db photos.db--exif-tableInput tableexif--analysis-tableOutput detailsanalysis--summary-tableMonthly aggssummary--filename-col etc.EXIF colsFileName / --filename-col SourceFile--image-rootPath prefix"" / --image-root ~/Photos--limitMax imagesNone / --limit 500--overwriteRecreate tablesFalse--graphs-dirPNG outputgraphs--deviceCPU/CUDAAuto / --device cuda--topkLabels5--dry-runMetadata onlyFalse
Examples

Basic: python exif_ana.py --image-root ~/Photos
GPU/Limited: python exif_ana.py --limit 1000 --device cuda --overwrite
Dry: python exif_ana.py --dry-run

Output

Logs: Progress, counts, graphs list.
Tables:
analysis: Per-image (JSON labels, desc, metadata, EV).
summary: Year-month (count, avgs).

Graphs (PNG): Model bars, aperture/shutter/ISO/EV histograms.

Query: SELECT * FROM analysis LIMIT 5;
Troubleshooting

Missing files: Use --dry-run to check.
CUDA: Match PyTorch/GPU; fallback to --device cpu.
Large sets: --limit, monitor mem.
No graphs: Need valid data.

Contributing/License
PRs welcome (e.g., focal length). MIT—see LICENSE.
📸 Questions? Issues welcome.
