import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score

def parse_thresholds(spec: str):
    """
    Parse thresholds in form "start:end:step" or comma list "0.3,0.35,0.4".
    Returns a sorted numpy array of floats.
    """
    spec = spec.strip()
    if ":" in spec:
        start, end, step = map(float, spec.split(":"))
        vals = np.arange(start, end + 1e-9, step)
        return np.round(vals, 6)
    else:
        vals = [float(x) for x in spec.split(",") if x.strip()]
        return np.array(sorted(vals))

def load_df(path: Path, label_col: str, ratio_col: str):
    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found in {path}")
    if ratio_col not in df.columns:
        raise ValueError(f"Column '{ratio_col}' not found in {path}")
    return df

def compute_pr_for_df(df: pd.DataFrame, label_col: str, ratio_col: str,
                      positive_str: str, ratio_thresholds: np.ndarray):
    # y_true: 1 = poor quality, 0 = not poor
    y_true = df[label_col].astype(str).str.strip().str.lower().eq(
        positive_str.strip().lower()).astype(int).values
    ratios = df[ratio_col].astype(float).values

    rows = []
    for rt in ratio_thresholds:
        # decision: predict poor if FaceDetectRatio < rt
        y_pred = (ratios < rt).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        rows.append((float(rt), float(prec), float(rec)))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default=".", help="Directory containing CSV files")
    ap.add_argument("--confs", type=str, required=True,
                    help="Comma-separated list of conf values, e.g., 0.4,0.5,0.6,0.7,0.8")
    ap.add_argument("--pattern", type=str, default="{conf}conf_1update_yolo.csv",
                    help="Filename pattern, must contain '{conf}' placeholder")
    ap.add_argument("--ratio-col", type=str, default="FaceDetectRatio")
    ap.add_argument("--label-col", type=str, default="Label")
    ap.add_argument("--positive-str", type=str, default="poor quality")
    ap.add_argument("--thresholds", type=str, default="0.20:0.70:0.05",
                    help="Thresholds for FaceDetectRatio: 'start:end:step' or 'v1,v2,...'")
    ap.add_argument("--out-png", type=str, default="pr_curves.png")
    ap.add_argument("--out-csv", type=str, default="pr_table.csv")
    args = ap.parse_args()

    base = Path(args.dir)
    conf_vals = [c.strip() for c in args.confs.split(",") if c.strip()]
    ratio_thresholds = parse_thresholds(args.thresholds)

    all_rows = []  # (conf, ratio_th, precision, recall)
    curves = {}    # conf -> list of (recall, precision, ratio_th)

    for conf in conf_vals:
        fname = args.pattern.format(conf=conf)
        fpath = base / fname
        if not fpath.exists():
            print(f"[WARN] File not found for conf={conf}: {fpath}")
            continue
        df = load_df(fpath, args.label_col, args.ratio_col)
        rows = compute_pr_for_df(df, args.label_col, args.ratio_col,
                                 args.positive_str, ratio_thresholds)

        for rt, prec, rec in rows:
            all_rows.append((conf, rt, prec, rec))
        curves[conf] = [(rec, prec, rt) for (rt, prec, rec) in rows]

    if not all_rows:
        raise SystemExit("No results computed. Check --dir, --confs, and --pattern.")

    # Save results table
    out_csv = Path(args.out_csv)
    pd.DataFrame(all_rows, columns=["conf","ratio_th","precision","recall"]).to_csv(out_csv, index=False)
    print(f"[OK] Wrote table: {out_csv.resolve()}")

    # Plot PR curves
    plt.figure(figsize=(8,6))
    for conf, points in curves.items():
        points = sorted(points, key=lambda x: x[0])  # sort by recall
        recalls = [p[0] for p in points]
        precisions = [p[1] for p in points]
        labels = [f"{p[2]:.2f}" for p in points]

        plt.plot(recalls, precisions, marker="o", label=f"conf={conf}")
        # Annotate every 2nd point to avoid clutter
        for i, (x, y, lab) in enumerate(zip(recalls, precisions, labels)):
            if i % 2 == 0:
                plt.text(x + 0.002, y + 0.002, lab, fontsize=7)

    plt.xlabel("Recall (Poor)")
    plt.ylabel("Precision (Poor)")
    plt.title("Precision–Recall Curves across FaceDetectRatio thresholds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out_png = Path(args.out_png)
    plt.savefig(out_png, dpi=180)
    print(f"[OK] Wrote PR plot: {out_png.resolve()}")

if __name__ == "__main__":
    main()
