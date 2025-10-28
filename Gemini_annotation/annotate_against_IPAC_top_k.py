#!/usr/bin/env python3
"""
Async top-K retrieval annotation → Excel, with JSON cache + retry (single API key).

- Retrieves top-K chunks per question from VectorStore (FAISS)
- Sends prompts concurrently to Gemini-2.5-Pro via LiteLLM
- Caches prompt→(thought,response) by exact (Question, chunk) to avoid duplicate calls
- Retries transient errors (no key rotation)
- Reads & writes Excel (.xlsx) to preserve special symbols
"""

import os
import sys
import json
import asyncio
import hashlib
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from litellm import acompletion
from google import genai
from google.genai.errors import ClientError, ServerError

from build_vector_store import VectorStore  # noqa: E402


# ────────────────────────────── Config knobs ──────────────────────────────────
LLM_MODEL   = "gemini/gemini-2.5-pro-preview-06-05"
EMBED_MODEL = "text-embedding-004"
BATCH_SIZE  = 5
MAX_RETRIES = 3
TOP_K       = 40  # retrieve this many chunks per question

SYSTEM_INSTRUCTION = (
    "You are a helpful assistant who helps accelerator physicists find useful information in technical documents. "
    "You will be given a question and some text. Decide whether the text helps answer the question. "
    "Always reply in exactly two lines:\n"
    "[ Guess ]: <Yes or No>\n"
    "[ Confidence ]: <0 - 100 >\n"
    "Never add any extra text."
)

# ────────────────────────────── Helpers ───────────────────────────────────────
def pair_key(question: str, chunk: str) -> str:
    """Stable cache key for exact question+chunk bytes (no stripping)."""
    return hashlib.blake2b((question + "\x1e" + chunk).encode("utf-8"),
                           digest_size=12).hexdigest()


def load_or_init_cache(cache_file: Path) -> dict:
    if cache_file.exists():
        with cache_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache_file: Path, cache: dict) -> None:
    cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def make_client() -> genai.Client:
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY4")
    if not api_key:
        raise RuntimeError("Please set GOOGLE_API_KEY4 in your environment/.env")
    os.environ["GEMINI_API_KEY"] = api_key  # for LiteLLM
    return genai.Client(api_key=api_key)


def embed_query_with_retry(client: genai.Client, text: str, max_retries: int = MAX_RETRIES) -> np.ndarray:
    """Embed a query string with simple retries (single key, no rotation)."""
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.models.embed_content(
                model=EMBED_MODEL,
                contents=[text],
                config={"task_type": "RETRIEVAL_QUERY"}
            )
            return np.array(resp.embeddings[0].values, dtype="float32")
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                backoff = 2 ** (attempt - 1)
                print(f"[embed] transient error (attempt {attempt}), retrying in {backoff}s: {e}")
                __import__("time").sleep(backoff)
                continue
            break
    raise last_err


async def send_one(question: str,
                   chunk: str,
                   cache: dict,
                   cache_file: Path,
                   max_retries: int = MAX_RETRIES) -> Tuple[str, str]:
    """
    One async call to the LLM with retry (single key). Returns (thought, reply).
    Uses JSON cache keyed by exact (question, chunk).
    """
    cache_key_json = json.dumps([question, chunk], ensure_ascii=False)
    if cache_key_json in cache:
        thought, reply = cache[cache_key_json]
        return thought, reply

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

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = await acompletion(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": prompt},
                ],
                thinking={"type": "enabled", "budget_tokens": 8192},
                max_tokens=8192,
            )
            choice = resp.choices[0].message
            thought = getattr(choice, "reasoning_content", "") or ""
            reply  = getattr(choice, "content", "") or ""
            cache[cache_key_json] = [thought, reply]
            save_cache(cache_file, cache)
            return thought, reply
        except Exception as e:
            last_exc = e
            if attempt < max_retries:
                backoff = 2 ** (attempt - 1)
                print(f"[llm] transient error (attempt {attempt}), retrying in {backoff}s: {e}")
                await asyncio.sleep(backoff)
                continue
            break
    return "", f"ERROR: {last_exc}" if last_exc else ""


# ────────────────────────────── Main pipeline ────────────────────────────────
async def run_pipeline(
    input_xlsx: Path,
    output_xlsx: Path,
    cache_file: Path,
    index_file: Path,
    meta_file: Path,
):
    # single-key client
    gclient = make_client()

    # cache
    cache = load_or_init_cache(cache_file)

    # load VectorStore
    dim = faiss.read_index(str(index_file)).d
    vs = VectorStore.load(dim, str(index_file), str(meta_file))

    # read Excel (preserve all symbols)
    df = pd.read_excel(input_xlsx, dtype=str, keep_default_na=False, engine="openpyxl")

    # dedup from prior output
    if output_xlsx.exists():
        old = pd.read_excel(output_xlsx, dtype=str, keep_default_na=False, engine="openpyxl")
        old_done = {pair_key(r.get("Question", "") or "", r.get("chunk_text", "") or "") for _, r in old.iterrows()}
    else:
        old, old_done = None, set()

    # build (question, chunk, meta) pairs
    pairs = []
    print("Building question → top-K chunk pairs…")
    for _, row in df.iterrows():
        q = row.get("Question", "")
        if not q:
            continue

        # embed & retrieve
        q_emb = embed_query_with_retry(gclient, q, MAX_RETRIES)
        hits = vs.search(q_emb, top_k=TOP_K)

        for rank, hit in enumerate(hits, start=1):
            chunk = hit["text"]  # do not strip() to keep exact symbols
            meta = {
                "Name":               row.get("Name", ""),
                "Question":           q,
                "Question_type":      row.get("Question_type", ""),
                "Expert_file":        row.get("Referenced_file(s)", ""),
                "Pages":              row.get("Pages", ""),
                "Specific_question?": row.get("Specific_question?", ""),
                "filename":           hit["filename"],
                "rank":               rank,
                "score":              float(hit["score"]),
            }
            if pair_key(q, chunk) in old_done:
                continue
            pairs.append((q, chunk, meta))

    print(f"Total pairs to process (new): {len(pairs)}")

    # annotate in batches
    out_rows = []
    total_batches = (len(pairs) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in tqdm(range(0, len(pairs), BATCH_SIZE), total=total_batches, desc="Batches"):
        batch = pairs[i:i + BATCH_SIZE]
        coros = [send_one(q, chunk, cache, cache_file, MAX_RETRIES) for q, chunk, _ in batch]
        results = await asyncio.gather(*coros, return_exceptions=True)

        for (q, chunk, meta), res in zip(batch, results):
            if isinstance(res, Exception):
                thought, reply = "", f"ERROR: {res}"
            else:
                thought, reply = res
            out_rows.append({
                **meta,
                "chunk_text": chunk,
                "thought":    thought,
                "response":   reply,
            })

    # append to Excel and save
    df_new = pd.DataFrame(out_rows, dtype=str)
    if old is not None and not old.empty:
        df_out = pd.concat([old, df_new], ignore_index=True)
    else:
        df_out = df_new

    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_excel(output_xlsx, index=False, engine="openpyxl")
    print(f"✅ Done — wrote {len(df_new)} new rows → {output_xlsx}")


# ────────────────────────────── Entrypoint (choose a preset) ────────────────
if __name__ == "__main__":

    # ==== Preset B: Local Mac (default) ====
    input_xlsx = Path.cwd() / "general_questions_batch1.xlsx"
    output_xlsx = Path.cwd() / "IPAC_integer_confidence_training1.xlsx"
    cache_file  = Path.cwd() / "vs_prompt_cache_integer_confidence1.json"
    vector_dir  = Path("/Users/rosydai/Desktop/Master_thesis/IPAC_paper/Benchmark_dataset/vector_store/IPAC")
    index_file  = vector_dir / "paragraph.index"
    meta_file   = vector_dir / "paragraph_metadata.pkl"

    asyncio.run(run_pipeline(
        input_xlsx=input_xlsx,
        output_xlsx=output_xlsx,
        cache_file=cache_file,
        index_file=index_file,
        meta_file=meta_file,
    ))