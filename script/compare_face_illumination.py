#!/usr/bin/env python3
"""
compare_two_excel.py

Usage:
    python compare_face_illumination.py /path/to/illumination.xlsx /path/to/face.csv -o ./output

Outputs (in output dir):
 - match.csv
 - mismatch.csv
 - summary.json
 - summary.txt
"""
import pandas as pd
import os
import sys
import argparse
import json

def load_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        # try utf-8 first, fallback to latin1 if decoding error
        try:
            return pd.read_csv(path)
        except Exception as e:
            try:
                return pd.read_csv(path, encoding="latin1")
            except Exception as e2:
                raise RuntimeError(f"Failed to read CSV {path}: {e2}") from e2
    elif ext in [".xls", ".xlsx"]:
        try:
            return pd.read_excel(path)
        except Exception as e:
            raise RuntimeError(f"Failed to read Excel {path}: {e}")
    else:
        raise ValueError(f"Unsupported file format: {ext} (path={path})")

def normalize_columns(df):
    # Convert all column names to lowercase (for easier matching), keep spaces/underscores
    lower_map = {c: c.lower().strip() for c in df.columns}
    df = df.rename(columns=lower_map)

    # Candidate names for video_path
    vp_candidates = ["video name", "video_name", "video_path", "video"]
    found = False
    for cand in vp_candidates:
        if cand in df.columns:
            df = df.rename(columns={cand: "video_path"})
            found = True
            break
    if not found:
        raise ValueError("Missing video path column. Expected one of: " + ", ".join(vp_candidates))

    # Candidate names for label
    label_candidates = ["label", "pred_label", "pred label", "prediction"]
    found = False
    for cand in label_candidates:
        if cand in df.columns:
            df = df.rename(columns={cand: "pred_label"})
            found = True
            break
    if not found:
        raise ValueError("Missing label/pred_label column. Expected one of: " + ", ".join(label_candidates))

    # Decision column (case-insensitive)
    if "decision" not in df.columns:
        raise ValueError("Missing decision column (Decision/decision).")
    # Normalize decision text
    df["decision"] = df["decision"].astype(str).str.strip().str.lower()
    df["decision"] = df["decision"].replace({
        "poor quality": "poor quality",
        "poor": "poor quality",
        "borderline": "borderline",
        "not poor quality": "not poor quality",
        "not poor": "not poor quality",
        "not_poor": "not poor quality",
        "not_poor quality": "not poor quality"
    })

    # Find columns containing "ratio"
    ratio_cols = [c for c in df.columns if "ratio" in c]
    if len(ratio_cols) == 0:
        raise ValueError(f"No ratio column found in columns: {list(df.columns)}")
    if len(ratio_cols) > 1:
        # Not fatal, but warn user
        print("Warning: multiple ratio-like columns found, using the first one:", ratio_cols)
    df = df.rename(columns={ratio_cols[0]: "ratio"})

    # Clean up whitespace in video_path
    df["video_path"] = df["video_path"].astype(str).str.strip()

    return df


def map_pred_label(label) -> str:
    # 统一转字符串处理
    s = str(label).strip()
    if s in {"0", "1", "2"}:
        return "not poor quality"
    elif s == "3":
        return "poor quality"
    return s



def process_csv(illumination_path, face_path, output_dir="./output"):
    # Read files
    df_illum = load_file(illumination_path)
    df_face = load_file(face_path)

    # Normalize column names
    df_illum = normalize_columns(df_illum)
    df_face = normalize_columns(df_face)

    # Rename decision/ratio to more meaningful column names
    df_illum = df_illum.rename(columns={"decision": "decision_illumination", "ratio": "underlit_ratio"})
    df_face = df_face.rename(columns={"decision": "decision_face", "ratio": "face_ratio"})

    # Keep only necessary columns
    need_illum_cols = ["video_path", "pred_label", "decision_illumination", "underlit_ratio"]
    need_face_cols = ["video_path", "pred_label", "decision_face", "face_ratio"]

    left = df_illum[need_illum_cols]
    right = df_face[need_face_cols]

    # Inner join
    merged = pd.merge(left, right, on=["video_path", "pred_label"], how="inner")
    total = len(merged)

    # --- NEW: mapped decisions for comparison ---
    merged["pred_label_mapped"] = merged["pred_label"].apply(map_pred_label)
    merged["illum_mapped"] = merged["decision_illumination"]
    merged["face_mapped"] = merged["decision_face"]

    # Match/mismatch based on mapped labels
    match_mask = (merged["illum_mapped"] == merged["face_mapped"]) & (merged["illum_mapped"] == merged["pred_label_mapped"])
    match_df = merged[match_mask]
    mismatch_df = merged[~match_mask]

    match_count = len(match_df)
    mismatch_count = len(mismatch_df)
    match_ratio = match_count / total if total > 0 else 0.0

    # Borderline stats
    border_any_mask = (merged["illum_mapped"] == "borderline") | (merged["face_mapped"] == "borderline")
    border_any_count = int(border_any_mask.sum())
    border_any_ratio = border_any_count / total if total > 0 else 0.0

    border_both_mask = (merged["illum_mapped"] == "borderline") & (merged["face_mapped"] == "borderline")
    border_both_count = int(border_both_mask.sum())
    border_both_ratio = border_both_count / total if total > 0 else 0.0

    # Distribution of original (not mapped) decisions
    dist_illum = merged["decision_illumination"].value_counts().to_dict()
    dist_face = merged["decision_face"].value_counts().to_dict()

    # Save match / mismatch CSVs (keep raw pred_label, no mapped columns)
    os.makedirs(output_dir, exist_ok=True)
    out_cols = ["video_path", "decision_illumination", "decision_face", "pred_label", "underlit_ratio", "face_ratio"]
    match_df[out_cols].to_csv(os.path.join(output_dir, "match.csv"), index=False)
    mismatch_df[out_cols].to_csv(os.path.join(output_dir, "mismatch.csv"), index=False)

    # Save summary
    summary = {
        "total_merged": int(total),
        "match_count": int(match_count),
        "mismatch_count": int(mismatch_count),
        "match_ratio": float(match_ratio),
        "borderline_any_count": int(border_any_count),
        "borderline_any_ratio": float(border_any_ratio),
        "borderline_both_count": int(border_both_count),
        "borderline_both_ratio": float(border_both_ratio),
        "decision_distribution_illumination": dist_illum,
        "decision_distribution_face": dist_face
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("Summary\n=======\n")
        f.write(f"Total merged rows: {total}\n")
        f.write(f"Match count: {match_count}  (match_ratio = {match_ratio:.4f})\n")
        f.write(f"Mismatch count: {mismatch_count}\n\n")
        f.write("Borderline stats:\n")
        f.write(f"  Any borderline count: {border_any_count} (ratio = {border_any_ratio:.4f})\n")
        f.write(f"  Both borderline count: {border_both_count} (ratio = {border_both_ratio:.4f})\n\n")
        f.write("Decision distributions (illumination):\n")
        f.write(json.dumps(dist_illum, indent=2, ensure_ascii=False))
        f.write("\nDecision distributions (face):\n")
        f.write(json.dumps(dist_face, indent=2, ensure_ascii=False))
        f.write("\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved results in {output_dir}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare illumination and face CSV/XLSX and output match/mismatch plus stats.")
    parser.add_argument("illumination", help="illumination file (.csv/.xlsx) (should contain keyword 'ratio' in one column)")
    parser.add_argument("face", help="face file (.csv/.xlsx) (should contain keyword 'ratio' in one column)")
    parser.add_argument("-o", "--output_dir", default="./output", help="output directory")
    args = parser.parse_args()
    process_csv(args.illumination, args.face, args.output_dir)
