#!/usr/bin/env python3
"""
Async batch annotation to Excel with caching, retry/backoff, key rotation.

- Retrieves expert pdf chunks per question from VectorStore (FAISS)
- Sends prompts in concurrent batches to Gemini-2.5-Pro via LiteLLM
- Caches based on exact (Question + chunk) hash, preserving special symbols
- Retries on transient errors; rotates through multiple API keys on exhaustion
- Writes results to an Excel file (.xlsx) to preserve all symbols cleanly
"""

import os
import sys
import asyncio
import hashlib
from pathlib import Path


import faiss
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from litellm import acompletion

from build_vector_store import VectorStore  



# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────
def pair_key(question: str, chunk: str) -> str:
    """Stable cache key for exact question+chunk bytes."""
    return hashlib.blake2b((question + "\x1e" + chunk).encode("utf-8"),
                           digest_size=12).hexdigest()


async def send_one(question: str, chunk: str, llm_model: str,
                   api_keys: list[str], max_retries: int) -> tuple[str, str]:
    """
    One async call to the LLM with retry + API key rotation.
    Returns (thought, reply) — strings (may be empty on error).
    """
    system_instruction = (
        "You are a helpful assistant who helps accelerator physicists find useful information in technical documents. "
        "You will be given a question and some text. Decide whether the text helps answer the question. "
        "Always reply in exactly two lines:\n"
        "[ Guess ]: <Yes or No>\n"
        "[ Confidence ]: <0 - 100 >\n"
        "Never add any extra text."
    )

    prompt = (
        f"Question: {question}\n"
        f"Paragraph: {chunk}\n"
        "Is this paragraph helpful for answering the question? "
        "Note that the <paragraph> can be helpful even "
        "if it only addresses part of the question without fully answering it. "
        "Provide your best guess for this question and your confidence that the guess is correct. "
        "Reply in the following format:\n"
        "[ Guess ]: <Your most likely guess, should be one of 'Yes' or 'No'>\n"
        "[ Confidence ]: <Give your honest confidence score "
        "between 0 and 100 about the correctness of your guess. "
        "0 means very likely wrong, and 100 means very confident.>"
    )

    key_idx = 0
    last_exc = None

    for attempt in range(1, max_retries + 1):
        try:
            # ensure LiteLLM uses the current key
            os.environ["GEMINI_API_KEY"] = api_keys[key_idx]

            resp = await acompletion(
                model=llm_model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ],
                thinking={"type": "enabled", "budget_tokens": 8192},
                max_tokens=8192,
            )
            choice = resp.choices[0].message
            thought = getattr(choice, "reasoning_content", "") or ""
            reply = getattr(choice, "content", "") or ""
            return thought, reply

        except Exception as e:
            last_exc = e
            msg = str(e)

            # Rotate key on RESOURCE_EXHAUSTED
            if "RESOURCE_EXHAUSTED" in msg or "exhausted" in msg:
                prev = key_idx
                key_idx = (key_idx + 1) % max(1, len(api_keys))
                os.environ["GEMINI_API_KEY"] = api_keys[key_idx]
                print(f"Resource exhausted on key {prev}; switched to key {key_idx}")
                continue

            # Retry on transient-ish signals
            transient = any(k in msg for k in ("429", "503", "timeout", "Rate limit", "temporarily"))
            if transient and attempt < max_retries:
                backoff = 2 ** (attempt - 1)
                print(f"Transient error (attempt {attempt}); retrying in {backoff}s: {e}")
                await asyncio.sleep(backoff)
                continue

    return "", f"ERROR: {last_exc}" if last_exc else ""


async def run_pipeline(
    excel_path: Path,
    output_xlsx: Path,
    index_file: Path,
    meta_file: Path,
    llm_model: str,
    batch_size: int,
    max_retries: int,
    api_keys: list[str],
):
    """
    Main async pipeline — Excel in/out, preserve symbols, use FAISS VectorStore.
    """
    load_dotenv()
    if not api_keys:
        api_keys = [os.getenv("GOOGLE_API_KEY1")]

    if not api_keys:
        raise RuntimeError("No Gemini API key found. Set GOOGLE_API_KEY* in env.")

    # Initialize FAISS-backed VectorStore
    dim = faiss.read_index(str(index_file)).d
    vs = VectorStore.load(dim, str(index_file), str(meta_file))

    # Read Excel as pure text to preserve symbols
    df = pd.read_excel(excel_path, dtype=str, keep_default_na=False, engine="openpyxl")

    # Build (question, chunk, meta) pairs (NO strip/cleanup)
    pairs = []
    print("Building question-chunk pairs…")
    for _, row in df.iterrows():
        ref_file = row.get("Referenced_file(s)", "")
        question = row.get("Question", "")
        if not question:
            continue

        chunk_ids = vs.file_to_indices.get(ref_file, [])
        if not chunk_ids:
            print(f"No chunks found for {ref_file}, skipping…")
            continue

        for cid in chunk_ids:
            chunk_text = vs.documents[cid].text  # keep exact text
            meta = {
                "Name":               row.get("Name", ""),
                "Question":           question,
                "Answer":             row.get("Answer", ""),
                "Question_type":      row.get("Question_type", ""),
                "Expert_file":        row.get("Referenced_file(s)", ""),
                "Pages":              row.get("Pages", ""),
                "Specific_question?": row.get("Specific_question?", ""),
            }
            pairs.append((question, chunk_text, meta))

    # Load existing Excel (if any) and build cache set
    done = set()
    if output_xlsx.exists():
        old = pd.read_excel(output_xlsx, dtype=str, keep_default_na=False, engine="openpyxl")
        for _, r in old.iterrows():
            q = r.get("Question", "") or ""
            c = r.get("chunk_text", "") or ""
            done.add(pair_key(q, c))
    else:
        old = None

    # Filter out pairs already annotated
    pairs = [(q, c, m) for (q, c, m) in pairs if pair_key(q, c) not in done]
    print(f"Total pairs to process: {len(pairs)}")

    # Run in batches and collect rows
    rows = []
    total_batches = (len(pairs) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(pairs), batch_size), total=total_batches, desc="Batches"):
        batch = pairs[i:i + batch_size]
        coros = [send_one(q, chunk, llm_model, api_keys, max_retries) for q, chunk, _ in batch]
        results = await asyncio.gather(*coros, return_exceptions=True)

        for (q, chunk, meta), res in zip(batch, results):
            if isinstance(res, Exception):
                thought, reply = "", f"ERROR: {res}"
            else:
                thought, reply = res

            rows.append({
                "Name": meta.get("Name", ""),
                "Question": q,
                "Answer": meta.get("Answer", ""),
                "Question_type": meta.get("Question_type", ""),
                "Expert_file": meta.get("Expert_file", ""),
                "Pages": meta.get("Pages", ""),
                "chunk_text": chunk,
                "thought": thought,
                "response": reply,
                "Specific_question?": meta.get("Specific_question?", ""),
            })

    # Append to existing Excel (if present) and save
    df_out = pd.DataFrame(rows, dtype=str)
    if old is not None and not old.empty:
        df_out = pd.concat([old, df_out], ignore_index=True)

    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_excel(output_xlsx, index=False, engine="openpyxl")
    print(f"✅ Done — wrote {len(rows)} new rows → {output_xlsx}")


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint — choose ONE preset by (un)commenting
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ==== Preset A: HABRA server paths ====
    # excel_path = Path("/home/qdai/Questions_for_training.xlsx")
    # output_xlsx = Path("/home/qdai/expert_annotation_integer_confidence_training_1.xlsx")
    # vector_dir  = Path("/home/qdai/expert_pdf_mineru/vector_store/retrieval_embedding")
    # index_file  = vector_dir / "paragraph.index"
    # meta_file   = vector_dir / "paragraph_metadata.pkl"

    # ==== Preset B: Local Mac (recommended) ====
    excel_path = Path.cwd() / "questions_VS_expert_papers_batch4.xlsx"
    output_xlsx = Path.cwd() / "expert_annotation_integer_confidence_training_batch_4.xlsx"
    vector_dir  = Path("/Users/rosydai/Desktop/Master_thesis/IPAC_paper/Benchmark_dataset/vector_store/expert")
    index_file  = vector_dir / "paragraph.index"
    meta_file   = vector_dir / "paragraph_metadata.pkl"

    # ==== Preset C: Another dataset ====
    # excel_path = Path.cwd() / "Initial_QA.xlsx"
    # output_xlsx = Path.cwd() / "lite_expert_annotation_integer_confidence.xlsx"
    # vector_dir  = Path("/path/to/vector_store/retrieval_embedding")
    # index_file  = vector_dir / "paragraph.index"
    # meta_file   = vector_dir / "paragraph_metadata.pkl"

    # knobs
    llm_model   = "gemini/gemini-2.5-pro-preview-06-05"
    batch_size  = 5
    max_retries = 3
    api_keys    = os.getenv("GOOGLE_API_KEY1")

 

    asyncio.run(run_pipeline(
        excel_path=excel_path,
        output_xlsx=output_xlsx,
        index_file=index_file,
        meta_file=meta_file,
        llm_model=llm_model,
        batch_size=batch_size,
        max_retries=max_retries,
        api_keys=api_keys,
    ))