#!/usr/bin/env python3
import os
import json
import math
import time
from pathlib import Path

import numpy as np
import faiss
from dotenv import load_dotenv

from google import genai
from google.genai.errors import ClientError, ServerError

# Your local helpers
from build_vector_store import Document, normalize, VectorStore
from paragraph_chunk import chunk_paragraphs


# ──────────────────────────────────────────────────────────────────────────────
# Core functions
# ──────────────────────────────────────────────────────────────────────────────
def build_paths(input_json: Path, output_dir: Path):
    """Return all standard file paths for the vector store pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "input_json": input_json,
        "output_dir": output_dir,
        "index_file": output_dir / "paragraph.index",
        "metadata_file": output_dir / "paragraph_metadata.pkl",
        "checkpoint_file": output_dir / "progress.json",
    }


def embed_and_build_store(
    paths: dict,
    model_name: str = "text-embedding-004",
    rate_limit: int = 130,
    max_retries: int = 5,
    base_delay: float = 1.0,
    min_tokens_per_chunk: int = 200,
):
    """
    Build/append a FAISS vector store from a JSON of documents with text chunks.
    Resumes from checkpoint; respects rate limits; retries transient errors.
    """
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY1") or os.getenv("GOOGLE_API_KEY2")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY1 / GOOGLE_API_KEY in environment.")
    client = genai.Client(api_key=api_key)

    input_json   = Path(paths["input_json"])
    index_file   = Path(paths["index_file"])
    metadata_file= Path(paths["metadata_file"])
    checkpoint   = Path(paths["checkpoint_file"])

    sleep_per_call = 60.0 / float(rate_limit)

    # Load dataset
    with open(input_json, "r", encoding="utf-8") as f:
        all_text = json.load(f)

    # Resume or start fresh
    if checkpoint.exists() and index_file.exists() and metadata_file.exists():
        with open(checkpoint, "r", encoding="utf-8") as f:
            done = set(json.load(f).get("processed", []))
        tmp_idx = faiss.read_index(str(index_file))
        dim = tmp_idx.d
        print(f"Resuming: index dim={dim}")
        vs = VectorStore.load(dim, index_file, metadata_file)
        print(f"Loaded {len(vs.documents)} docs; will skip {len(done)} files")
    else:
        done = set()
        vs = None

    # Main loop
    for item in all_text:
        fn = item["file_name"]
        if fn in done:
            continue

        print(f"\n▶ Processing {fn}")
        chunks = chunk_paragraphs(item["text_data"], min_tokens=min_tokens_per_chunk)
        total_chunks = len(chunks)

        if total_chunks == 0:
            print("  No chunks; skipping.")
            done.add(fn)
            _save_checkpoint(checkpoint, done)
            continue

        # Determine per-minute batching
        batch_size = total_chunks if total_chunks <= rate_limit else math.ceil(total_chunks / rate_limit)
        num_calls = math.ceil(total_chunks / batch_size)
        print(f"  total_chunks={total_chunks} → batch_size={batch_size} → calls={num_calls}")

        docs = []
        for start in range(0, total_chunks, batch_size):
            batch = chunks[start:start + batch_size]

            # Retry/backoff for this batch
            resp = _embed_with_retries(
                client=client,
                model_name=model_name,
                texts=batch,
                max_retries=max_retries,
                base_delay=base_delay,
            )

            # Unpack embeddings
            for emb_obj, text in zip(resp.embeddings, batch):
                emb = normalize(np.array(emb_obj.values, dtype="float32"))
                docs.append(Document(text=text, embedding=emb, filename=fn))

            time.sleep(sleep_per_call)

        # Initialize store if first time
        if vs is None:
            vs = VectorStore(len(docs[0].embedding))
        vs.add_documents(docs)

        # Persist
        vs.save(index_file=index_file, metadata_file=metadata_file)

        done.add(fn)
        _save_checkpoint(checkpoint, done)
        print(f"✅ Finished {fn} ({len(docs)} chunks indexed)")

    print("\n🎉 All files processed!")


def _embed_with_retries(client, model_name: str, texts: list[str], max_retries: int, base_delay: float):
    """Call embed API with exponential backoff on transient failures."""
    for attempt in range(1, max_retries + 1):
        try:
            return client.models.embed_content(
                model=model_name,
                contents=texts,
                config={"task_type": "RETRIEVAL_DOCUMENT"},
            )
        except (ClientError, ServerError) as e:
            status = getattr(e, "status_code", None)
            retriable = isinstance(e, ServerError) or status == 429
            if retriable and attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                print(f"    → attempt {attempt} failed ({e}); retrying in {delay:.1f}s")
                time.sleep(delay)
            else:
                raise


def _save_checkpoint(checkpoint_path: Path, done_set: set[str]):
    checkpoint_path.write_text(json.dumps({"processed": sorted(done_set)}), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# Run configs (pick one by uncommenting)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ====== Choose ONE dataset by uncommenting its block ======

    # --- IPAC combined corpus ---
    base = Path("/Users/rosydai/Desktop/Master_thesis/IPAC_paper/Benchmark_dataset/vector_store/IPAC")
    paths = build_paths(
        input_json = base / "IPAC_combined_clean_reference.json",
        output_dir = base,
    )

    # --- Expert corpus (original) ---
    # base = Path("/Users/rosydai/Desktop/Master_thesis/IPAC_paper/Benchmark_dataset/vector_store/expert")
    # paths = build_paths(
    #     input_json = base / "expert_text.json",
    #     output_dir = base / "vector_store" / "retrieval_embedding",
    # )

    # --- Expert corpus (cleaned references removed) ---
    # base = Path("/Users/rosydai/Desktop/Master_thesis/IPAC_paper/Benchmark_dataset/vector_store/expert")
    # paths = build_paths(
    #     input_json = base / "expert_text_clean_reference.json",
    #     output_dir = base ,
    # )

    # ====== Embedding/build params (tweak if needed) ======
    embed_and_build_store(
        paths=paths,
        model_name="text-embedding-004",
        rate_limit=130,            # max calls/min
        max_retries=5,
        base_delay=1.0,
        min_tokens_per_chunk=200, # keep consistent with your pipeline
    )