#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
# — keep your helpers —
SCORE_COLS = ("rerank_score", "cos_score", "score")

def get_score_col(df: pd.DataFrame) -> str:
    for c in SCORE_COLS:
        if c in df.columns:
            return c
    raise ValueError(f"Missing score column (one of {SCORE_COLS})")

def prepare_labels_scores(df: pd.DataFrame, qcol="Question", lcol="Label"):
    if qcol not in df.columns or lcol not in df.columns:
        raise ValueError(f"Expected columns '{qcol}' and '{lcol}'")
    scol = get_score_col(df)
    df = df[[qcol, lcol, scol]].dropna().copy()
    df[qcol] = df[qcol].astype(str).str.strip()
    df[lcol] = pd.to_numeric(df[lcol], errors="coerce").fillna(0).astype(int)
    df[scol] = pd.to_numeric(df[scol], errors="coerce")
    y = df[lcol].to_numpy(dtype=int)
    s = df[scol].to_numpy(dtype=float)
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    if y.size == 0 or y.sum() == 0 or y.sum() == y.size:
        raise ValueError("Need mixed 0/1 labels after filtering.")
    return y, s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    help="name=path.(xlsx|csv|csv.gz) — must include Question, Label, and one of {rerank_score, cos_score, score}")
    args = ap.parse_args()

    f1_best = {}     # name -> (best_f1, best_thr)
    curves = {}      # name -> (thr, f1_at_thr)

    for spec in args.model:
        name, path = spec.split("=", 1)
        df = pd.read_csv(path, compression="gzip", low_memory=False)
        y, s = prepare_labels_scores(df, qcol="Question", lcol="Label")

        # PR points (sklearn returns thresholds length = len(prec)-1)
        prec, rec, thr = precision_recall_curve(y, s)

        # compute F1 for each threshold-aligned point
        # (use prec[:-1], rec[:-1] to match thr length)
        p = prec[:-1]
        r = rec[:-1]
        f1 = (2 * p * r) / (p + r + 1e-12)

        # keep curve and best point
        curves[name] = (thr, f1)
        idx = int(np.nanargmax(f1))
        f1_best[name] = (float(f1[idx]), float(thr[idx]))
        print(f"{name:20s} best F1={f1_best[name][0]:.4f} @ threshold={f1_best[name][1]:.6f}")

    # plot
    palette = plt.get_cmap("tab10").colors
    fig, ax = plt.subplots(figsize=(9,6))

    # order by best F1 (desc) so legend is tidy
    names_sorted = sorted(f1_best.keys(), key=lambda n: f1_best[n][0], reverse=True)

    for i, name in enumerate(names_sorted):
        color = palette[i % 10]
        thr, f1 = curves[name]
        best_f1, best_thr = f1_best[name]
        ax.plot(thr, f1, lw=1.8, color=color,
                label=f"{name} (max F1={best_f1:.3f} @ t={best_thr:.3f})")

        # mark best point
        ax.plot([best_thr], [best_f1], marker="o", ms=4, color=color)

    ax.set_title("F1 score vs threshold")
    ax.set_xlabel("Threshold on model score")
    ax.set_ylabel("F1 score")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=True, framealpha=0.85)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()