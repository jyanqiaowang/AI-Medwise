#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import re

# ---------- utils ----------

def canon(s: str) -> str:
    """标准化列名：小写 + 非字母数字归一为空格 + 去首尾"""
    return re.sub(r'[^0-9a-z]+', ' ', str(s).lower()).strip()

def load_table(path, sheet=None):
    """读表：Excel 强制读取第一个 sheet（或指定 sheet），CSV 直接读"""
    path = Path(path)
    if path.suffix.lower() in [".xls", ".xlsx"]:
        return pd.read_excel(path, sheet_name=(sheet if sheet is not None else 0))
    return pd.read_csv(path)

def pick_label_column(df, prefer="label", fallback="pred_label"):
    cols = {canon(c): c for c in df.columns}
    if prefer and canon(prefer) in cols:
        return cols[canon(prefer)]
    if fallback and canon(fallback) in cols:
        return cols[canon(fallback)]
    return None

def find_ratio_column(df, override=None):
    """
    自动找 ratio 列：
    1) override 存在且在 df 中，直接用
    2) 否则列名包含 'ratio' 的候选里优先：
       - 精确 'FaceDetectRatio'
       - 以 'face' 开头且含 'ratio'
       - 数值型的 ratio 列
       - 兜底：第一个包含 ratio 的列
    """
    if override and override in df.columns:
        return override

    names = list(df.columns)
    name_lc = [n.lower() for n in names]
    cand_idx = [i for i, n in enumerate(name_lc) if "ratio" in n]
    if not cand_idx:
        return None

    # 1) FaceDetectRatio
    for i in cand_idx:
        if name_lc[i] == "facedetectratio":
            return names[i]
    # 2) 以 face 开头
    for i in cand_idx:
        if name_lc[i].startswith("face"):
            return names[i]
    # 3) 数值型
    for i in cand_idx:
        if pd.api.types.is_numeric_dtype(df[names[i]]):
            return names[i]
    # 4) 兜底
    return names[cand_idx[0]]

def normalize_decision(s: str) -> str:
    """
    统一 decision 同义词：
      - poor / poor quality  -> 'poor'
      - not poor / not_poor / not poor quality -> 'not_poor'
      - borderline -> 'borderline'
    其他：清洗后直接返回清洗值（以免丢信息）
    """
    if s is None:
        return ""
    raw = str(s).strip().lower()
    # 去掉多余空白和标点（把非字母数字归一为空格，然后压缩空格）
    cleaned = re.sub(r'[^0-9a-z]+', ' ', raw).strip()

    # 明确映射
    if cleaned in {"poor", "poor quality"}:
        return "poor"
    if cleaned in {"not poor", "not poor quality", "not_poor"}:
        return "not_poor"
    if cleaned == "borderline":
        return "borderline"

    # 常见写法兜底（如 notpoor / poorquality）
    if cleaned.replace(" ", "") in {"notpoor", "notpoorquality"}:
        return "not_poor"
    if cleaned.replace(" ", "") in {"poorquality"}:
        return "poor"

    return cleaned  # 兜底返回清洗后的字符串

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Compare two result tables and split matches vs mismatches.")
    ap.add_argument("--file-a", required=True, help="First Excel/CSV file (method A)")
    ap.add_argument("--file-b", required=True, help="Second Excel/CSV file (method B)")
    ap.add_argument("--sheet-a", default=None, help="Sheet for file A (omit if single sheet)")
    ap.add_argument("--sheet-b", default=None, help="Sheet for file B (omit if single sheet)")

    ap.add_argument("--key-col", default="video_path", help="Join key column (default: video_path)")
    ap.add_argument("--decision-col", default="Decision", help="Decision column name in both files (default: Decision)")
    ap.add_argument("--label-col", default="label",
                    help="Label column (default: label; falls back to pred_label if missing)")

    # 各文件可单独指定 ratio 列名（可选）
    ap.add_argument("--ratio-col-a", default=None, help="Override ratio column name for file A")
    ap.add_argument("--ratio-col-b", default=None, help="Override ratio column name for file B")

    ap.add_argument("--out-dir", default="compare_out", help="Output directory")
    args = ap.parse_args()

    # 读入
    dfA = load_table(args.file_a, sheet=args.sheet_a)
    dfB = load_table(args.file_b, sheet=args.sheet_b)

    # 规范名映射（支持 Video_Name ≈ Video Name）
    colsA = {canon(c): c for c in dfA.columns}
    colsB = {canon(c): c for c in dfB.columns}
    def colA(name: str) -> str: return colsA.get(canon(name), name)
    def colB(name: str) -> str: return colsB.get(canon(name), name)

    # 检查关键列存在
    if canon(args.key_col) not in colsA or canon(args.key_col) not in colsB:
        raise ValueError(f"Key column '{args.key_col}' not found in both files.\n"
                         f"File A cols: {list(dfA.columns)}\nFile B cols: {list(dfB.columns)}")
    if canon(args.decision_col) not in colsA or canon(args.decision_col) not in colsB:
        raise ValueError(f"Decision column '{args.decision_col}' not found in both files.\n"
                         f"File A cols: {list(dfA.columns)}\nFile B cols: {list[dfB.columns]}")

    # label 列
    labelA = pick_label_column(dfA, prefer=args.label_col, fallback="pred_label")
    labelB = pick_label_column(dfB, prefer=args.label_col, fallback="pred_label")

    # ratio 列
    ratioA = find_ratio_column(dfA, override=args.ratio_col_a)
    ratioB = find_ratio_column(dfB, override=args.ratio_col_b)
    if ratioA: print(f"[INFO] File A ratio column: {ratioA}")
    else:      print("[WARN] No ratio-like column found in file A; ratios_A will be empty.")
    if ratioB: print(f"[INFO] File B ratio column: {ratioB}")
    else:      print("[WARN] No ratio-like column found in file B; ratios_B will be empty.")

    # 取必要列
    needA = [colA(args.key_col), colA(args.decision_col)]
    needB = [colB(args.key_col), colB(args.decision_col)]
    if ratioA: needA.append(ratioA)
    if ratioB: needB.append(ratioB)
    if labelA: needA.append(labelA)
    if labelB: needB.append(labelB)

    A = dfA[needA].copy()
    B = dfB[needB].copy()

    # 以 key 去重
    dupA = A.duplicated(subset=colA(args.key_col)).sum()
    dupB = B.duplicated(subset=colB(args.key_col)).sum()
    if dupA > 0:
        print(f"[WARN] File A: {dupA} duplicate keys; keeping first.")
        A = A.drop_duplicates(subset=colA(args.key_col), keep="first")
    if dupB > 0:
        print(f"[WARN] File B: {dupB} duplicate keys; keeping first.")
        B = B.drop_duplicates(subset=colB(args.key_col), keep="first")

    # 合并（只比较两边都存在的视频）
    merged = A.merge(
        B, left_on=colA(args.key_col), right_on=colB(args.key_col),
        how="inner", suffixes=("_a", "_b")
    )
    if merged.empty:
        raise SystemExit("No common videos found between A and B on the given key.")

    # 标准化 decision（加入同义词映射）
    decA = merged[f"{colA(args.decision_col)}_a"].astype(str).map(normalize_decision)
    decB = merged[f"{colB(args.decision_col)}_b"].astype(str).map(normalize_decision)
    merged["decision_equal"] = (decA == decB)
    merged["has_borderline"] = decA.eq("borderline") | decB.eq("borderline")

    # ---- 合并后统一 key：可能是 _a / _b / 无后缀；逐一判断 ----
    out = merged.copy()
    left_key_base  = colA(args.key_col)
    right_key_base = colB(args.key_col)
    candidates = [
        f"{left_key_base}_a",
        f"{right_key_base}_b",
        left_key_base,
        right_key_base
    ]
    key_found = None
    for c in candidates:
        if c in out.columns:
            key_found = c
            break
    if key_found is None:
        raise ValueError(
            f"Key column not found after merge. Tried: {candidates}. "
            f"Merged columns: {list(out.columns)}"
        )

    out["video_path"] = out[key_found]
    # 清理 key 列（存在才删）
    for c in set(candidates):
        if c in out.columns and c != "video_path":
            out.drop(columns=[c], inplace=True)

    # 统一命名
    rename_map = {
        f"{colA(args.decision_col)}_a": "decision_a",
        f"{colB(args.decision_col)}_b": "decision_b",
    }
    if ratioA and (ratioA + "_a") in out.columns:
        rename_map[ratioA + "_a"] = "ratio_a"
    if ratioB and (ratioB + "_b") in out.columns:
        rename_map[ratioB + "_b"] = "ratio_b"
    if labelA and (labelA + "_a") in out.columns:
        rename_map[labelA + "_a"] = "label_a"
    if labelB and (labelB + "_b") in out.columns:
        rename_map[labelB + "_b"] = "label_b"

    out = out.rename(columns=rename_map)

    # 只保留存在的列，避免 KeyError
    candidate_cols = [
        "video_path", "decision_a", "decision_b",
        "ratio_a", "ratio_b",
        "label_a", "label_b",
        "has_borderline", "decision_equal"
    ]
    keep_cols = [c for c in candidate_cols if c in out.columns]
    out = out[keep_cols]

    # 拆分
    matches = out[out["decision_equal"]].copy()
    mismatches = out[~out["decision_equal"]].copy()

    # 统计
    total = len(out)
    match_ratio = len(matches) / total if total else 0.0
    borderline_ratio = out["has_borderline"].mean() if total else 0.0

    # 导出
    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)
    matches_path = outdir / "matches.csv"
    mismatches_path = outdir / "mismatches.csv"
    matches.to_csv(matches_path, index=False)
    mismatches.to_csv(mismatches_path, index=False)

    print(f"[OK] Common videos: {total}")
    print(f"[OK] Matches: {len(matches)}  ({match_ratio*100:.2f}%)")
    print(f"[OK] Mismatches: {len(mismatches)}  ({(1-match_ratio)*100:.2f}%)")
    print(f"[OK] Borderline prevalence (either is 'borderline'): {borderline_ratio*100:.2f}%")
    print(f"[SAVE] {matches_path}")
    print(f"[SAVE] {mismatches_path}")

if __name__ == "__main__":
    main()
