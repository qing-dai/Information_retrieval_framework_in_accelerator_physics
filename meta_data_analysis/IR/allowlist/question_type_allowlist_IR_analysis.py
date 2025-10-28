#!/usr/bin/env python3
import argparse, math
from pathlib import Path
import pandas as pd
from ranx import Qrels, Run, evaluate

SCORE_COLS = ("rerank_score", "cos", "score")

def read_table(p: str) -> pd.DataFrame:
    p = Path(p)
    if p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p, dtype=str, keep_default_na=False)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Unsupported file: {p}")

def build_qrels_run_from_df(df: pd.DataFrame,
                            qcol="Question", dcol="chunk_text", lcol="relevance_label"):
    df = df.dropna(subset=[qcol, dcol, lcol]).copy()
    df[qcol] = df[qcol].astype(str).str.strip()
    df[dcol] = df[dcol].astype(str).str.strip()
    df[lcol] = pd.to_numeric(df[lcol], errors="coerce").fillna(0).astype(int)

    queries = list(dict.fromkeys(df[qcol].tolist()))
    docs    = list(dict.fromkeys(df[dcol].tolist()))
    q2id = {q: f"Q{i}" for i, q in enumerate(queries)}
    d2id = {d: f"D{i}" for i, d in enumerate(docs)}

    # qrels (positives only)
    qrels_dict = {}
    for q, g in df[df[lcol] == 1].groupby(qcol):
        qrels_dict[q2id[q]] = {d2id[d]: 1 for d in g[dcol].tolist()}
    qrels = Qrels(qrels_dict)

    # score column
    score_col = next((c for c in SCORE_COLS if c in df.columns), None)
    if not score_col:
        raise ValueError(f"Missing score column (one of {SCORE_COLS})")

    # run (allowlist in this df)
    run_dict = {}
    for q, g in df.groupby(qcol):
        qid = q2id[q]
        dd = {}
        for _, r in g.iterrows():
            did = d2id[r[dcol]]
            try:
                s = float(r[score_col])
                if math.isfinite(s):
                    dd[did] = s
            except Exception:
                pass
        if dd:
            run_dict[qid] = dd

    return Qrels(qrels_dict), Run(run_dict)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", action="append", required=True,
        help="name=allowlist_scores.(xlsx|csv) — must include Question, chunk_text, relevance_label, Question_type, and cos/rerank_score"
    )
    ap.add_argument("--ks", default="3,5,10,15,20")
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    metrics = [f"ndcg@{k}" for k in ks] + [f"map@{k}" for k in ks]

    for spec in args.model:
        name, path = spec.split("=", 1)
        df = read_table(path)

        if "Question_type" not in df.columns:
            raise ValueError("Column 'Question_type' missing in allowlist file.")

        # case-insensitive normalization
        df["Question_type"] = (
            df["Question_type"].astype(str).str.strip().str.casefold()
        )

        rows = []
        for qtype, g in df.groupby("Question_type"):
            qrels_sub, run_sub = build_qrels_run_from_df(g)
            if len(qrels_sub.qrels) == 0:
                continue
            rep = evaluate(qrels_sub, run_sub, metrics)
            row = {"Question_type": qtype.title(), "model": name}
            row.update({m: float(rep[m]) for m in metrics})
            rows.append(row)

        out = pd.DataFrame(rows)
        if out.empty:
            print(f"\n=== {name} ===\n(no queries with positives per Question_type)")
            continue

        print(f"\n=== {name} — nDCG/MAP by Question_type (allowlist, case-insensitive) ===")
        print(out.sort_values("Question_type")
                 .set_index(["Question_type","model"])
                 .round(4))

if __name__ == "__main__":
    main()