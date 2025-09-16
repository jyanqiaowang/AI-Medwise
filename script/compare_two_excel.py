#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd
import re

def canon(s: str) -> str:
    """标准化列名：小写 + 非字母数字归一为空格 + 去首尾"""
    return re.sub(r'[^0-9a-z]+', ' ', str(s).lower()).strip()

def load_table(path, sheet=None):
    path = Path(path)
    if path.suffix.lower() in [".xls", ".xlsx"]:
        return pd.read_excel(path, sheet_name=(sheet if sheet else 0))
    return pd.read_csv(path)

def normalize_numeric_label(x) -> str:
    """把0/1/2归到not_poor，3归到poor，其它直接返回字符串"""
    try:
        v = int(x)
        return "poor" if v == 3 else "not_poor"
    except:
        # 如果是字符串
        s = str(x).lower().strip()
        if s in {"3", "poor", "poor quality"}:
            return "poor"
        if s in {"0", "1", "2", "not poor", "not poor quality"}:
            return "not_poor"
        if s == "borderline":
            return "borderline"
        return s

def main():
    ap = argparse.ArgumentParser(description="Compare two CSV/Excel files with numeric labels 0-3 (3=poor, 0/1/2=not_poor).")
    ap.add_argument("--file-a", required=True)
    ap.add_argument("--file-b", required=True)
    ap.add_argument("--key-col", default="video_path")
    ap.add_argument("--decision-col", default="Decision")
    ap.add_argument("--label-col", default="label")  # 或 pred_label
    ap.add_argument("--out-dir", default="compare_numeric_out")
    args = ap.parse_args()

    dfA = load_table(args.file_a)
    dfB = load_table(args.file_b)

    # 统一列名匹配
    colsA = {canon(c): c for c in dfA.columns}
    colsB = {canon(c): c for c in dfB.columns}
    def colA(name): return colsA.get(canon(name), name)
    def colB(name): return colsB.get(canon(name), name)

    # 检查必要列
    if canon(args.key_col) not in colsA or canon(args.key_col) not in colsB:
        raise ValueError("Key column not found in both files.")
    if canon(args.decision_col) not in colsA or canon(args.decision_col) not in colsB:
        raise ValueError("Decision column not found in both files.")

    labelA = colsA.get(canon(args.label_col), None)
    labelB = colsB.get(canon(args.label_col), None)

    needA = [colA(args.key_col), colA(args.decision_col)]
    needB = [colB(args.key_col), colB(args.decision_col)]
    if labelA: needA.append(labelA)
    if labelB: needB.append(labelB)

    A = dfA[needA].copy()
    B = dfB[needB].copy()

    # 合并
    merged = A.merge(
        B,
        left_on=colA(args.key_col),
        right_on=colB(args.key_col),
        how="inner",
        suffixes=("_a", "_b")
    )
    if merged.empty:
        raise SystemExit("No common rows between the two files.")

    # 统一video_path列
    key_candidates = [f"{colA(args.key_col)}_a", f"{colB(args.key_col)}_b",
                      colA(args.key_col), colB(args.key_col)]
    key_found = next((c for c in key_candidates if c in merged.columns), None)
    if key_found is None:
        raise ValueError("Key column not found after merge.")
    merged["video_path"] = merged[key_found]
    for c in key_candidates:
        if c in merged.columns and c != "video_path":
            merged.drop(columns=[c], inplace=True)

    # 归一化 decision 和 label
    merged["decision_a"] = merged[f"{colA(args.decision_col)}_a"].map(normalize_numeric_label)
    merged["decision_b"] = merged[f"{colB(args.decision_col)}_b"].map(normalize_numeric_label)

    if labelA and f"{labelA}_a" in merged.columns:
        merged["label_a"] = merged[f"{labelA}_a"].map(normalize_numeric_label)
    if labelB and f"{labelB}_b" in merged.columns:
        merged["label_b"] = merged[f"{labelB}_b"].map(normalize_numeric_label)

    # 选择参考label
    label_col = "label_a" if "label_a" in merged.columns else ("label_b" if "label_b" in merged.columns else None)

    # 计算
    merged["decision_equal"] = merged["decision_a"] == merged["decision_b"]
    merged["has_borderline"] = merged["decision_a"].eq("borderline") | merged["decision_b"].eq("borderline")

    if label_col:
        p_a_match = (merged["decision_a"] == merged[label_col]).mean()
        p_b_match = (merged["decision_b"] == merged[label_col]).mean()
    else:
        p_a_match = p_b_match = None

    # 输出
    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)
    merged[merged["decision_equal"]].to_csv(outdir / "matches.csv", index=False)
    merged[~merged["decision_equal"]].to_csv(outdir / "mismatches.csv", index=False)

    print(f"Total common: {len(merged)}")
    print(f"Matches: {merged['decision_equal'].sum()} ({merged['decision_equal'].mean()*100:.2f}%)")
    print(f"Borderline prevalence (either is 'borderline'): {merged['has_borderline'].mean()*100:.2f}%")
    if label_col:
        print(f"Decision_a vs label match rate: {p_a_match*100:.2f}%")
        print(f"Decision_b vs label match rate: {p_b_match*100:.2f}%")
    print(f"Results saved to: {outdir}")

if __name__ == "__main__":
    main()
