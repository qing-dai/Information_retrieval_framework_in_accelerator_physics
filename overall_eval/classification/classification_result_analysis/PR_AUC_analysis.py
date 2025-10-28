#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

SCORE_COLS = ("rerank_score", "cos_score", "score")

def read_table(p: str) -> pd.DataFrame:
    p = Path(p)
    if p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p, dtype=str, keep_default_na=False)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Unsupported file: {p}")

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
        raise ValueError("Need mixed 0/1 labels after filtering to compute AUCs.")
    return y, s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    help="name=path.(xlsx|csv) — must include Question, Label, and one of {rerank_score, cos_score, score}")
    args = ap.parse_args()

    curves, praucs = {}, {}
    for spec in args.model:
        name, path = spec.split("=", 1)
        df = pd.read_csv(path, compression="gzip", low_memory=False)
        y, s = prepare_labels_scores(df, qcol="Question", lcol="Label")
        pos_rate = y.mean()
        precision, recall, _ = precision_recall_curve(y, s)
        prauc = average_precision_score(y, s)
        curves[name] = (recall, precision)
        praucs[name] = prauc
        print(f"{name:20s} PR-AUC={prauc:.4f}")

    pairs = [
        ("mpnet-base","mpnet-ft"),
        ("phy-base","phy-ft"),
        ("accphy-base","accphy-ft"),
         ("bge-base","bge-ft"),
        ("qwen3-0.6b-base","qwen3-0.6b-ft"),
    ]
    pairs = sorted(pairs, key=lambda p: praucs.get(p[1], -1.0), reverse=True)

    palette = plt.get_cmap("tab10").colors
    fig, ax = plt.subplots(figsize=(9, 6))

    for i, (base, ft) in enumerate(pairs):
        color = palette[i % 10]
        if base in curves:
            recall_b, precision_b = curves[base]
            ax.plot(recall_b, precision_b, ls="--", drawstyle="steps-post", lw=1.2, alpha=0.7, color=color,
                    label=f"{base} ({praucs.get(base, float('nan')):.3f})")
        if ft in curves:
            recall_f, precision_f = curves[ft]
            delta = praucs.get(ft, 0.0) - praucs.get(base, 0.0)
            ax.plot(recall_f, precision_f, ls="-", drawstyle="steps-post", lw=2.0, color=color,
                    label=f"{ft} ({praucs.get(ft, float('nan')):.3f}, Δ{delta:+.3f})")

    # chance
    ax.axhline(y=pos_rate, ls="--", lw=0.8, color="gray", alpha=0.6, label=f"Baseline (prevalence={pos_rate:.3f})")
    ax.set_title("PR Full-corpus— all models (base dashed, ft solid)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(alpha=0.25)

    # legend  
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=True, framealpha=0.85, facecolor="white", edgecolor="0.8",
        fontsize=9, ncol=1, handlelength=2.8, labelspacing=0.4)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()