#!/usr/bin/env python3
"""
split_mismatch.py

Usage:
    python split_mismatch.py /path/to/mismatch.csv -o ./output

This script splits mismatch cases into:
 A) face_decision == illumination_decision but BOTH differ from pred_label (after mapping 0/1/2->not poor, 3->poor)
 B) face_decision != illumination_decision

Outputs (in output dir):
 - mismatch_same_decisions_diff_predlabel.csv
 - mismatch_diff_decisions.csv
 - summary.json
 - summary.txt
"""

import os
import sys
import argparse
import json
import pandas as pd

# ---------- helpers ----------
def load_csv_any_encoding(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, encoding="latin1")

def normalize_decision_series(s: pd.Series) -> pd.Series:
    """Normalize decision text to: 'poor quality' | 'not poor quality' | 'borderline'."""
    s = s.astype(str).str.strip().str.lower()
    return s.replace({
        "poor quality": "poor quality",
        "poor": "poor quality",
        "not poor quality": "not poor quality",
        "not poor": "not poor quality",
        "not_poor": "not poor quality",
        "not_poor quality": "not poor quality",
        "borderline": "borderline"
    })

def map_pred_label_to_decision(x) -> str:
    """Map pred_label (0/1/2/3 or their strings) to decision space."""
    s = str(x).strip()
    if s in {"0","1","2"}:
        return "not poor quality"
    if s == "3":
        return "poor quality"
    # If already textual, normalize it too (robustness)
    return normalize_decision_series(pd.Series([s])).iloc[0]

# ---------- main ----------
def main(mismatch_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    df = load_csv_any_encoding(mismatch_path)
    # lower columns for robust access, but keep original too
    colmap = {c: c.lower().strip() for c in df.columns}
    df = df.rename(columns=colmap)

    # expected columns (from your previous script outputs)
    needed = {
        "decision_illumination",
        "decision_face",
        "pred_label",
    }
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"mismatch.csv missing required columns: {missing}. "
                         f"Found: {list(df.columns)}")

    # Normalize decisions + map pred_label
    df["illum_norm"] = normalize_decision_series(df["decision_illumination"])
    df["face_norm"]  = normalize_decision_series(df["decision_face"])
    df["pred_mapped"] = df["pred_label"].apply(map_pred_label_to_decision)

    # A 类：face==illum 且 与 pred_mapped 不一致
    same_decisions = (df["face_norm"] == df["illum_norm"])
    diff_with_pred = (df["face_norm"] != df["pred_mapped"]) | (df["illum_norm"] != df["pred_mapped"])
    cat_A_mask = same_decisions & diff_with_pred

    # B 类：face!=illum
    cat_B_mask = (df["face_norm"] != df["illum_norm"])

    # 导出
    a_df = df[cat_A_mask].copy()
    b_df = df[cat_B_mask].copy()

    # 为了排查方便，把规范化后的列也带出去
    out_cols_base = []
    # 保留原来有的常见列，不强制要求
    for c in ["video_path", "decision_illumination", "decision_face", "pred_label",
              "underlit_ratio", "face_ratio"]:
        if c in df.columns:
            out_cols_base.append(c)
    out_cols_debug = ["illum_norm", "face_norm", "pred_mapped"]

    a_out = a_df[out_cols_base + out_cols_debug] if out_cols_base else a_df
    b_out = b_df[out_cols_base + out_cols_debug] if out_cols_base else b_df

    a_path = os.path.join(output_dir, "mismatch_same_decisions_diff_predlabel.csv")
    b_path = os.path.join(output_dir, "mismatch_diff_decisions.csv")
    a_out.to_csv(a_path, index=False)
    b_out.to_csv(b_path, index=False)

    # 统计信息
    total = len(df)
    a_cnt = len(a_df)
    b_cnt = len(b_df)

    # 交集校验：A/B 两类理论上可能有交集吗？
    # A 定义要求 face==illum；B 定义 face!=illum；两者互斥，无交集。
    # 但若存在数据脏/不可达，下面给一个 sanity。
    overlap_cnt = len(df[cat_A_mask & cat_B_mask])

    summary = {
        "total_mismatch_rows": int(total),
        "cat_A_same_decisions_diff_predlabel_count": int(a_cnt),
        "cat_A_ratio": float(a_cnt / total) if total else 0.0,
        "cat_B_diff_decisions_count": int(b_cnt),
        "cat_B_ratio": float(b_cnt / total) if total else 0.0,
        "overlap_count_should_be_zero": int(overlap_cnt),
        "outputs": {
            "A_file": a_path,
            "B_file": b_path
        }
    }

    # 保存 summary
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("Split mismatch summary\n=======================\n")
        f.write(f"Total mismatch rows: {total}\n")
        f.write(f"A) same decisions (face==illum) but differ from pred_label: {a_cnt}\n")
        f.write(f"B) different decisions (face!=illum): {b_cnt}\n")
        f.write(f"Overlap (should be 0): {overlap_cnt}\n")
        f.write(f"A file: {a_path}\nB file: {b_path}\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved A: {a_path}\nSaved B: {b_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split mismatch cases into two categories.")
    parser.add_argument("mismatch_csv", help="Path to mismatch.csv")
    parser.add_argument("-o", "--output_dir", default="./output_split", help="Output directory")
    args = parser.parse_args()
    main(args.mismatch_csv, args.output_dir)
