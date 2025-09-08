#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Video Illumination Quality Comparison
-------------------------------------
This script processes videos listed in a CSV file and compares classifications
using three different thresholds:
1. Fixed threshold (default = 50).
2. Dataset median brightness.
3. Dataset percentile brightness (e.g., 25th percentile).
"""

import csv
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def ensure_dir(p: Path | str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def is_video_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in VIDEO_EXTS


# ------------------------------------------------------------
# STEP 1: Collect dataset brightness values
# ------------------------------------------------------------
def collect_dataset_brightness(csv_file: str, base_dir: str,
                               video_col: str = "Video Name") -> List[float]:
    all_vals: List[float] = []
    csv_path = Path(csv_file).expanduser().resolve()
    base = Path(base_dir).expanduser().resolve()

    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rel = str(row.get(video_col, "")).strip()
            if not rel:
                continue
            vid_path = base / rel
            if not vid_path.exists() or not is_video_file(vid_path):
                continue

            cap = cv2.VideoCapture(str(vid_path))
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                all_vals.append(float(np.mean(gray)))
            cap.release()

    return all_vals


# ------------------------------------------------------------
# STEP 2: Quality classification function
# ------------------------------------------------------------
def illumination_quality(vals: List[float], threshold: float,
                         poor_ratio: float = 0.3, good_ratio: float = 0.6) -> Tuple[float, str]:
    if not vals:
        return 0.0, "Poor"

    good_frames = sum(1 for v in vals if v >= threshold)
    ratio_good = good_frames / len(vals)

    if ratio_good < poor_ratio:
        label = "Poor"
    elif ratio_good < good_ratio:
        label = "Uncertain"
    else:
        label = "Good"

    return ratio_good, label


# ------------------------------------------------------------
# STEP 3: Process one video and compare thresholds
# ------------------------------------------------------------
def process_video(video_path: str, thresholds: dict, output_dir: str,
                  original_label: Optional[str] = None) -> dict:
    vp = Path(video_path)
    video_name = vp.stem
    folder_name = vp.parent.name

    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        print(f"[ERROR] Couldn't open {video_name}")
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    vals: List[float] = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vals.append(float(np.mean(gray)))
    cap.release()

    if not vals:
        print(f"[WARN] {video_name}: no frames decoded.")
        return {}

    overall = float(np.mean(vals))

    # Apply all thresholds
    results = {}
    for name, th in thresholds.items():
        ratio_good, quality = illumination_quality(vals, th)
        results[name] = {
            "Threshold": th,
            "Bright Ratio": ratio_good,
            "Quality": quality
        }

    # Print summary to console
    print(f"[RESULT] {video_name} (Label={original_label}, Avg={overall:.2f}):")
    for name, res in results.items():
        print(f"   - {name}: th={res['Threshold']:.1f}, "
              f"ratio={res['Bright Ratio']:.2f}, quality={res['Quality']}")

    # Save histogram with threshold lines
    out_dir = Path(output_dir) / folder_name
    ensure_dir(out_dir)
    plt.figure(figsize=(10, 6))
    plt.hist(vals, bins=50, alpha=0.7, label="Frame Brightness")
    for name, res in results.items():
        plt.axvline(res["Threshold"], linestyle="--", label=f"{name} th={res['Threshold']:.1f}")
    plt.title(f"{video_name} Illumination (Avg={overall:.2f})")
    plt.xlabel("Average frame brightness (0–255)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(out_dir / f"{video_name}_illumination_comparison.png", bbox_inches="tight")
    plt.close()

    # Return structured results
    return {
        "Video Name": video_name,
        "Original Label": original_label,
        "FPS": fps,
        "Total Frames": frame_count,
        "Average Illumination": overall,
        **{f"{name} Quality": res["Quality"] for name, res in results.items()},
        **{f"{name} Ratio": res["Bright Ratio"] for name, res in results.items()}
    }


# ------------------------------------------------------------
# STEP 4: Orchestrator
# ------------------------------------------------------------
def process_videos_from_csv(csv_file: str,
                            base_dir: str,
                            output_dir: str,
                            percentile: float = 25.0,
                            fixed_threshold: float = 50.0,
                            summary_csv: Optional[str] = None) -> None:
    # First collect dataset brightness for adaptive thresholds
    print("[INFO] Collecting dataset brightness values...")
    all_vals = collect_dataset_brightness(csv_file, base_dir)
    if not all_vals:
        print("[ERROR] No frames collected.")
        return

    median_th = float(np.median(all_vals))
    perc_th = float(np.percentile(all_vals, percentile))

    thresholds = {
        "Fixed": fixed_threshold,
        "Median": median_th,
        f"Percentile_{percentile}": perc_th
    }

    print(f"[INFO] Thresholds: Fixed={fixed_threshold}, Median={median_th:.2f}, "
          f"{percentile}th Percentile={perc_th:.2f}")

    # Prepare CSV writer
    csv_writer = None
    if summary_csv:
        ensure_dir(Path(summary_csv).parent)
        csv_fh = open(summary_csv, "w", newline="", encoding="utf-8")
        fieldnames = ["Video Name", "Original Label", "FPS", "Total Frames",
                      "Average Illumination"] + \
                     [f"{name} Quality" for name in thresholds.keys()] + \
                     [f"{name} Ratio" for name in thresholds.keys()]
        csv_writer = csv.DictWriter(csv_fh, fieldnames=fieldnames)
        csv_writer.writeheader()
    else:
        csv_fh = None

    # Process videos
    csv_path = Path(csv_file).expanduser().resolve()
    base = Path(base_dir).expanduser().resolve()
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rel = str(row.get("Video Name", "")).strip()
            label = str(row.get("Label", "")).strip()
            if not rel:
                continue
            vid_path = base / rel
            if not vid_path.exists() or not is_video_file(vid_path):
                continue

            print(f"[PROCESS] {vid_path}")
            res = process_video(str(vid_path), thresholds, output_dir, original_label=label)

            if res and csv_writer:
                csv_writer.writerow(res)

    if summary_csv and csv_fh:
        csv_fh.close()


# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------
if __name__ == "__main__":
    process_videos_from_csv(
        csv_file="/mnt/sunlab-nas-1/CVAT/merged_labels_2update.csv",
        base_dir="/mnt/sunlab-nas-1/CVAT",
        output_dir="/mnt/sunlab-nas-1/CVAT/Patrick/test_compare",
        percentile=25.0,
        fixed_threshold=50.0,
        summary_csv="/mnt/sunlab-nas-1/CVAT/Patrick/test_compare/illumination_comparison.csv"
    )

