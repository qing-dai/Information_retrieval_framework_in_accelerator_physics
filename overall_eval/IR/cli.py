# cli.py
import os
#cap threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"   # mac only, harmless elsewhere
os.environ["ARROW_NUM_THREADS"] = "1"        
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MKL_THREADING_LAYER"] = "GNU"    # avoids Intel/iomp weirdness
os.environ["MALLOC_ARENA_MAX"] = "1"         # reduces TLS/arena pressure


import argparse, pandas as pd
import math
from pathlib import Path
from data import robust_read_data, build_ir_index
from models import load_model
from scorer import encode_pairs, make_ranx_run, compute_sims, make_allow_run_with_reranker
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from bge_utils import is_bge_model, BGE_QUERY_PROMPT
from ranx import Qrels, Run, evaluate
from save_allowlist_scores import save_allowlist_scores, save_allowlist_scores_reranker
from save_full_corpus_scores import save_full_corpus_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="Filtered_Positive_Questions.xlsx")
    ap.add_argument("--model", required=True)
    ap.add_argument("--kind", choices=["st","qwen","st-last-token","qwen-reranker"], default="st")
    ap.add_argument("--device", default="cuda:5")
    ap.add_argument("--add-instruction", action="store_true")
    ap.add_argument("--ks", default="3,5,10,15,20")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    metrics = [f"ndcg@{k}" for k in ks] + [f"map@{k}" for k in ks]

    print("Reading input data...")
    df = robust_read_data(args.input)
    print("Building queries, corpus, relevant, candidates...")
    queries, corpus, qids, dids, qrels, allowlist = build_ir_index(df)

    print(f"Loading model {args.model}")
    model = load_model(args.model, args.kind, args.device)

    # ===== Path A: embedding encoders =====
    # ---- Decide instruction policy ----
    if args.kind != "qwen-reranker":
        is_bge = is_bge_model(args.model)
        # default: auto-enable instruction for BGE unless user forced --no-add-instruction
        auto_add = is_bge
        add_instruction = args.add_instruction or auto_add
        query_instruction = None
        if add_instruction and is_bge:
            query_instruction = BGE_QUERY_PROMPT
        
        print("Encoding embeddings...")
        with torch.no_grad():
            q_emb, d_emb = encode_pairs(model, queries, corpus, add_instruction=add_instruction, bs=16,
                                        query_instruction=query_instruction,
                                        prefer_prompt_name=True)
            
        sims = compute_sims(q_emb, d_emb)
        print("Building ranx run...")
        run_allow, run_full = make_ranx_run(sims, qids, dids, allowlist)

        print("Evaluating...")
        print("🪭 per-query, allowlist...")
        rep_allow = evaluate(qrels, run_allow, metrics)

        print("🏮 per-query, full corpus ...")
        rep_full = evaluate(qrels, run_full, metrics)

        print("start to save scores to excels...")
        allow_df = save_allowlist_scores(df, queries, corpus, sims)
        full_df = save_full_corpus_scores(df, queries,corpus, qids, dids, sims)

        model_name = Path(args.output)

        # 1) Small sheet → Excel
        xl_path = Path(f"{model_name}_allowlist_with_scores.xlsx")
        allow_df.to_excel(xl_path, index=False)
        print(f"[saved] → {xl_path}")

        # 2) Big table → Parquet (and optional CSV)
        pq_path = Path(f"{model_name}_full_corpus_with_scores.csv.gz")
        full_df.to_csv(pq_path, index=False, compression="gzip")
        print(f"[saved] → {pq_path}")

        rows = []
        rows.append({
            "model": f"model: {args.output}",
            "task": "🪭 per-query, allowlist...",
            **{m: float(rep_allow[m]) for m in metrics},
        })
        rows.append({
            "model": f"model: {args.output}",
            "task": "🏮 per-query, full corpus...",
            **{m: float(rep_full[m]) for m in metrics},
        })
        print(pd.DataFrame([rows]).to_string(index=False))
        return
    
    # ===== Path B: Qwen reranker (allowlist only) =====
    print("Scoring with Qwen reranker over allowlist only...")
    allow_rerank_dict = make_allow_run_with_reranker(
        qids=qids, dids=dids, queries=queries, corpus=corpus,
        allowlist=allowlist, reranker=model,
    )


    print("Evaluating (allowlist only)...")
    rep_allow = evaluate(qrels, Run(allow_rerank_dict), metrics)

    print("Saving reranker allowlist scores...")
    allow_df = save_allowlist_scores_reranker(
        df, queries, corpus, qids, dids, allow_rerank_dict
    )
    model_name = Path(args.output)
    xl_path = Path(f"{model_name}_allowlist_scores.xlsx")
    allow_df.to_excel(xl_path, index=False)
    print(f"[saved] → {xl_path}")

    rows = [{"model": f"model: {args.output}", "task": "🪭 per-query, allowlist (reranker)...", **{m: float(rep_allow[m]) for m in metrics}}]
    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == "__main__":
    main()