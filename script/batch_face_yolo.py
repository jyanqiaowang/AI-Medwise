#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch YOLOv8-face detection over a CSV list of videos.

Input CSV must have columns: ["Video Name", "Label"]
"Video Name" is a path relative to BASE_DIR (e.g., first_label_project/40014/40008_8.mp4)

This script will:
- Run YOLOv8-face detection on sampled frames for each video
- Save up to N annotated frames to out_frames/<video_stem>/
- Append "FaceDetectRatio" and "Decision" columns after "Label"
- Write an output CSV next to the input (default: <input>_with_face_yolo.csv)

Usage example:
  python batch_face_check_yolov8.py /mnt/data/merged_labels.csv \
      --base /mnt/sunlab-nas-1/CVAT \
      --weights /mnt/data/yolov8n-face.pt \
      --sample-rate 10 \
      --conf 0.5 \
      --iou 0.5 \
      --max-save 20 \
      --out-dir out_frames \
      --poor-threshold 0.50 for binary decision
"""

import os
import sys
import csv
import argparse
from pathlib import Path
import cv2
import pandas as pd

try:
    from ultralytics import YOLO
except Exception as e:
    print("[ERR] Failed to import Ultralytics. Install with: pip install ultralytics")
    raise


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def face_detect_ratio_yolov8(video_path: Path, sample_rate: int, model, conf: float, iou: float,
                             save_dir: Path = None, max_save: int = 20) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return 0.0

    checked = detected = saved = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % sample_rate != 0:
            continue

        checked += 1
        results = model.predict(frame, conf=conf, iou=iou, device="cpu", verbose=False)
        faces = []
        if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            for b in results[0].boxes:
                xyxy = b.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, xyxy[:4])
                faces.append((x1, y1, x2, y2))

        if faces:
            detected += 1
            if save_dir is not None and saved < max_save:
                draw = frame.copy()
                for (x1, y1, x2, y2) in faces:
                    cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                out_path = save_dir / f"frame_{frame_idx}.jpg"
                cv2.imwrite(str(out_path), draw)
                saved += 1

    cap.release()
    return detected / max(1, checked)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str, help="Input CSV with columns: Video Name, Label")
    # parser.add_argument("--base", type=str, default=None,
    #                     help="Base directory to prefix to 'Video Name'")
    parser.add_argument("--weights", type=str, required=True, help="YOLOv8-face weights path (.pt)")
    parser.add_argument("--sample-rate", type=int, default=10, help="Detect every N frames")
    parser.add_argument("--conf", type=float, default=0.5, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="YOLO IoU threshold")
    parser.add_argument("--max-save", type=int, default=20, help="Max annotated frames to save per video")
    parser.add_argument("--out-dir", type=str, default="out_frames", help="Root folder to save annotated frames")
    parser.add_argument("--t1", type=float, default=0.35, help="Lower FaceDetectRatio threshold for auto poor")
    parser.add_argument("--t2", type=float, default=0.55, help="Upper FaceDetectRatio threshold for auto not_poor")

    parser.add_argument("--output-csv", type=str, default=None,
                        help="Output CSV path (defaults to <input>_with_face_yolo.csv)")
    args = parser.parse_args()

    in_csv = Path(args.csv_path)
    # base_dir = Path(args.base)
    out_csv = Path(args.output_csv) if args.output_csv else in_csv.with_name(in_csv.stem + "_with_face_yolo.csv")
    out_root = Path(args.out_dir)
    ensure_dir(out_root)

    # Load YOLOv8-face model
    model = YOLO(args.weights)

    # Read input CSV
    df = pd.read_csv(in_csv)
    if not {"video_path", "pred_label"}.issubset(df.columns):
        raise ValueError("Input CSV must have columns: 'video_path', 'pred_label'")

    face_ratios, decisions = [], []

    for i, row in df.iterrows():
        rel_path = row["video_path"]
        full_path = rel_path

        safe_stem = Path(rel_path).stem
        video_out_dir = out_root / safe_stem
        ensure_dir(video_out_dir)

        ratio = face_detect_ratio_yolov8(full_path,
                                         sample_rate=args.sample_rate,
                                         model=model,
                                         conf=args.conf,
                                         iou=args.iou,
                                         save_dir=video_out_dir,
                                         max_save=args.max_save)
        if ratio < args.t1:
            decision = "poor"
        elif ratio >= args.t2:
            decision = "not_poor"
        else:
            decision = "borderline"

        print(f"[{i+1}/{len(df)}] {rel_path} -> ratio={ratio:.3f} decision={decision}")
        face_ratios.append(ratio)
        decisions.append(decision)

    # Insert after 'Label'
    insert_pos = list(df.columns).index("pred_label") + 1
    df.insert(insert_pos, "FaceDetectRatio", face_ratios)
    df.insert(insert_pos + 1, "Decision", decisions)

    df.to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")
    print(f"[DONE] Wrote: {out_csv}")
    print(f"[INFO] Annotated frames saved under: {out_root.resolve()}")


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
