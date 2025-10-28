#!/usr/bin/env python3
import os
# cap threads (keeps ROCm/CUDA/BLAS sane)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["ARROW_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["MALLOC_ARENA_MAX"] = "1"

import argparse
import gzip
import pandas as pd
import numpy as np
from pathlib import Path
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from models import load_model
from scorer import encode_pairs  
from bge_utils import is_bge_model, BGE_QUERY_PROMPT

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)

def write_header(path):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("Question,chunk_text,Label,score\n")

def append_block(path, rows):
    # rows: list of tuples (q, c, label, score)
    df = pd.DataFrame(rows, columns=["Question","chunk_text","Label","score"])
    with gzip.open(path, "at", encoding="utf-8") as f:
        df.to_csv(f, header=False, index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/home/qdai/IR_system_test/ir_eval/Classification_test_dataset.xlsx")
    ap.add_argument(
        "--model", action="append", required=True,
        help="Format: name=path_to_model. Repeatable. e.g. --model bge=/path/bge-m3 --model qwen=/path/qwen3-embed"
    )
    ap.add_argument("--label-col", default="Label", help="Original label column name (default: relevance_label)")
    ap.add_argument("--kind", choices=["st","qwen","st-last-token","qwen-reranker"], default="st")
    ap.add_argument("--device", default="cuda:5")
    ap.add_argument("--add-instruction", action="store_true", help="add instruction prompt for bge and qwen")
    ap.add_argument("--q-batch", type=int, default=32, help="Query batch size for scoring (rows per chunk)")
    ap.add_argument("--outdir", default="classification_fullcorpus_output", help="Where to drop the csv.gz files")
    args = ap.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {in_path}")
    df = pd.read_excel(in_path)
    if "Question" not in df.columns or "chunk_text" not in df.columns:
        raise ValueError("Input must contain 'Question' and 'chunk_text'")

    # Prepare uniques
    questions = (
        df["Question"].fillna("").astype(str).drop_duplicates().tolist()
    )
    chunks = (
        df["chunk_text"].fillna("").astype(str).drop_duplicates().tolist()
    )
    print(f"Unique questions: {len(questions)} | Unique chunks: {len(chunks)}")

    # Build label lookup for original pairs
    lbl_col = args.label_col if args.label_col in df.columns else None
    if lbl_col:
        # ensure int labels (0/1 typical)
        labels_series = pd.to_numeric(df[lbl_col], errors="coerce").fillna(0).astype(int)
    else:
        labels_series = pd.Series([1] * len(df))  # originals default to 1 if no label present

    # Use a dict for quick check of original pairs → label
    # (q, c) → label
    orig_labels = {}
    for q, c, lab in zip(df["Question"].astype(str), df["chunk_text"].astype(str), labels_series.tolist()):
        orig_labels[(q, c)] = int(lab)

    for spec in args.model:
        name, model_path = spec.split("=", 1)
        print(f"\n>>> Loading [{name}] from {model_path}")
        model = load_model(model_path, args.kind, args.device)

        # BGE instruction logic
        is_bge = is_bge_model(model_path)
        auto_add = is_bge
        add_instruction = args.add_instruction or auto_add
        query_instruction = BGE_QUERY_PROMPT if (add_instruction and is_bge) else None
        print("encoding embeddings...")
        with torch.no_grad():
            q_emb, d_emb = encode_pairs(model, questions, chunks, add_instruction=add_instruction, bs=64,
                                        query_instruction=query_instruction, prefer_prompt_name=True)
            
        # ensure numpy 
        if hasattr(q_emb, "detach"):
            q_emb = q_emb.detach().cpu().numpy()
        if hasattr(d_emb, "detach"):
            d_emb = d_emb.detach().cpu().numpy()


        # Output path
        out_path = outdir / f"{in_path.stem}_fullcorpus_{name}.csv.gz"
        print(f"Writing: {out_path}")
        write_header(out_path)

        # Score in query-batches to limit memory
        Q = len(questions)
        D = len(chunks)
        bsz = max(1, args.q_batch)
        for start in range(0, Q, bsz):
            end = min(start + bsz, Q)
            q_block = q_emb[start:end]                                  # (b, dim)
            sims = np.matmul(q_block, d_emb.T)                          # (b, D) cosine scores

            rows = []
            for i, q in enumerate(questions[start:end]):
                # fill rows for this query across ALL chunks
                sim_row = sims[i]                                       # (D,)
                for j, c in enumerate(chunks):
                    label = orig_labels.get((q, c), 0)
                    rows.append((q, c, label, float(sim_row[j])))

            append_block(out_path, rows)
            print(f"  wrote queries {start}–{end-1} / {Q} (×{D} chunks)")

        print(f"✅ Finished {name}: {out_path}")

    print("\nAll models done.")

if __name__ == "__main__":
    main()