#!/usr/bin/env python3
import argparse, math
import pandas as pd
from ranx import Qrels, Run, evaluate

def build_qrels_run_from_df(df,
                            qid_col="qid",
                            did_col="did",
                            rel_col="relevance_label",
                            score_col="cos_full"):
    df = df.dropna(subset=[qid_col, did_col, rel_col]).copy()
    df[rel_col]   = pd.to_numeric(df[rel_col], errors="coerce").fillna(0).astype(int)
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce").fillna(0).astype(float)

    # qrels: positives only
    qrels_dict = {}
    for qid, g in df[df[rel_col] == 1].groupby(qid_col):
        qrels_dict[qid] = {did: 1 for did in g[did_col].tolist()}
    qrels = Qrels(qrels_dict)

    # run: all docs with scores (full corpus)
    run_dict = {}
    for qid, g in df.groupby(qid_col):
        dd = {}
        for _, r in g.iterrows():
            s = float(r[score_col])
            if math.isfinite(s):
                dd[r[did_col]] = s
        if dd:
            run_dict[qid] = dd
    return qrels, Run(run_dict)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    help="name=full_corpus_with_scores.csv.gz")
    ap.add_argument("--ks", default="3,5,10,15,20")
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    metrics = [f"ndcg@{k}" for k in ks] + [f"map@{k}" for k in ks]

    for spec in args.model:
        name, path = spec.split("=", 1)
        print(f"\nReading {path} ...")
        df = pd.read_csv(path, compression="gzip", low_memory=False)

        if "specific_to_paper" not in df.columns:
            raise ValueError("Column 'specific_to_paper' missing in full corpus file.")

        # normalize flag → "yes"/"no"
        df["specific_to_paper"] = (
            df["specific_to_paper"].astype(str).str.strip().str.lower()
              .replace({"true": "yes", "false": "no", "1": "yes", "0": "no"})
        )

        rows = []
        for tag, g in df.groupby("specific_to_paper"):
            qrels_sub, run_sub = build_qrels_run_from_df(g)
            if len(qrels_sub.qrels) == 0:
                continue
            rep = evaluate(qrels_sub, run_sub, metrics)
            row = {"specific_to_paper": tag, "model": name}
            row.update({m: float(rep[m]) for m in metrics})
            rows.append(row)

        out = pd.DataFrame(rows)
        if out.empty:
            print(f"\n=== {name} ===\n(no queries with positives per specific_to_paper)")
            continue

        print(f"\n=== {name} — nDCG/MAP by specific_to_paper (full corpus) ===")
        print(out.sort_values("specific_to_paper")
                 .set_index(["specific_to_paper", "model"])
                 .round(4))

if __name__ == "__main__":
    main()