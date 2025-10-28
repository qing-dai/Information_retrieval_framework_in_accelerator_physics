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

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from models import load_model
from scorer import encode_pairs, make_ranx_run, compute_sims, make_allow_run_with_reranker
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from bge_utils import is_bge_model, BGE_QUERY_PROMPT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/home/qdai/IR_system_test/ir_eval/Classification_test_dataset.xlsx")
    ap.add_argument(
        "--model", action="append", required=True,
        help="Format: name=path_to_model. Can specify multiple, e.g. --model bge=/path/to/bge-m3 --model qwen=/path/to/qwen3"
    )
    ap.add_argument("--kind", choices=["st","qwen","st-last-token","qwen-reranker"], default="st")
    ap.add_argument("--device", default="cuda:5")
    ap.add_argument("--add-instruction", action="store_true")
    args = ap.parse_args()

    input_path = Path(args.input)
    #allowlist dir
    outdir = Path("classification_allowlist_output")
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Reading input: {input_path}")
    df = pd.read_excel(input_path)
    if "Question" not in df.columns or "chunk_text" not in df.columns:
        raise ValueError("Input must contain 'Question' and 'chunk_text' columns")
    
    queries = df["Question"].astype(str).tolist()
    corpus = df["chunk_text"].astype(str).tolist()

    for spec in args.model:
        name, path = spec.split("=", 1)
        print(f"\n>>> Loading model [{name}] from: {path}")
        model = load_model(path, args.kind, args.device)
        is_bge = is_bge_model(path)
        # default: auto-enable instruction for BGE unless user forced --no-add-instruction
        auto_add = is_bge
        add_instruction = args.add_instruction or auto_add
        query_instruction = None
        if add_instruction and is_bge:
            query_instruction = BGE_QUERY_PROMPT
        
        print("Encoding embeddings...")
        with torch.no_grad():
            q_emb, d_emb = encode_pairs(model, queries, corpus, add_instruction=add_instruction, bs=32,
                                        query_instruction=query_instruction,
                                        prefer_prompt_name=True)
            
        # ensure numpy 
        if hasattr(q_emb, "detach"):
            q_emb = q_emb.detach().cpu().numpy()
        if hasattr(d_emb, "detach"):
            d_emb = d_emb.detach().cpu().numpy()
        
        df_copy = df.copy()
        df_copy["cos_score"] = np.sum(q_emb * d_emb, axis=1)
        output_path = outdir/f"{input_path.stem}_{name}.xlsx"
        print(f"Saving to {output_path}")
        df_copy.to_excel(output_path, index=False)

    print("\n✅ All models processed.")

if __name__ == "__main__":
    main()