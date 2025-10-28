# scorer.py
import numpy as np
from typing import List, Dict, Optional, Tuple
from metrics import ndcg_at_k, ap_at_k
from ranx import Qrels, Run, evaluate
import math
from qwenreranker import QwenReranker

def compute_sims(q_emb: np.ndarray, d_emb: np.ndarray) -> np.ndarray:
    """The two embeddings must be normalized! """
    return np.matmul(q_emb, d_emb.T)

def encode_pairs(model, queries: List[str], corpus: List[str], add_instruction: bool, bs: int,
                query_instruction: Optional[str] = None, prefer_prompt_name: bool = True,):
    use_prompt = False
    if add_instruction and prefer_prompt_name:
        prompts = getattr(model, "prompts", None)
        use_prompt = isinstance(prompts, dict) and prompts["query"]
        if use_prompt:
            print(f"for {model}, use prompt: {prompts}")

    if add_instruction and (not use_prompt) and query_instruction:
        print(f"encode instruction {query_instruction} for BGE queries")
        queries = [f"{query_instruction}{q}" for q in queries]


    q_emb = model.encode(queries, batch_size=bs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True,
                         prompt_name="query" if (add_instruction and use_prompt) else None)
    d_emb = model.encode(corpus,  batch_size=bs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
    return q_emb, d_emb


def make_ranx_run(sims,
             qids: List[str],
             dids: List[str],
             allowlist: Optional[Dict[str, set[str]]] = None
             ) -> Tuple[Run, Run]:
    # cosine via normalized embeddings, result shape(num_queries, num_docs)
    # sims = np.matmul(q_emb, d_emb.T)
    D = sims.shape[1]
    run_dict_allow = {}
    run_dict_full = {}
    # D: the number of docs, used when we need "all docs" as candidates
    
    # qids: the unique query IDs
    for qi, qid in enumerate(qids):
        # -- full corpus run (always all docs)
        full_scores = sims[qi]
        run_dict_full[qid] = {dids[j]: float(full_scores[j]) for j in range(D)}

        # -- allowlist run (only subset per query)
        if allowlist is not None:
            # fetch all candidate doc IDs for one query, sorted, reserving order, so the result is always reproducible
            cand_ids = sorted(list(allowlist.get(qid, [])))
            if not cand_ids:
                continue
            # check if the type of the first element is a string like "D5",
            # in our case, yes, we have "D0", "D1", ...", we then convert them to integers [0,1,2,...]
            if isinstance(cand_ids[0], str) and cand_ids[0].startswith("D"):
                eligible = np.array([int(cid[1:]) for cid in cand_ids], dtype=np.int64)
            # otherwise, we assume they are already integers
            else:
                eligible = np.array(cand_ids, dtype=np.int64)

            # fetch similarity scores for this query, only for the eligible docs
            # sims shape is (num_queries, num_docs), so sims[qi, eligible] gives us the scores for query qi and the eligible docs
            scores = sims[qi, eligible]
            # create a dict of {doc_id: score} for this query
            # dids[int(j)]: convert the integer index back to the original doc ID string
            # float(s): ensure the score is a float,
            # e.g. {"D0": 0.95, "D3": 0.87}
            hits = {dids[int(j)]: float(s) for j, s in zip(eligible, scores)}
            run_dict_allow[qid] = hits

    # show a quick sanity check
    first_qid = qids[0]
    print(f"{first_qid} | allowlist docs: {len(run_dict_allow.get(first_qid,{}))} | full corpus docs: {len(run_dict_full[first_qid])}")
    #print(f"{first_qid} | allowlist docs: {run_dict_allow[first_qid]} | full corpus docs: {run_dict_full[first_qid]}")

    # add to the run dict, e.g. {"Q0": {"D0": 0.95, "D3": 0.87}, "Q1": {"D2": 0.99}}
    run_allow = Run(run_dict_allow)
    run_full = Run(run_dict_full)
    return run_allow, run_full

def make_allow_run_with_reranker(
    qids: List[str],
    dids: List[str],
    queries: List[str],
    corpus: List[str],
    allowlist: Dict[str, set[str]],
    reranker,                     # models.QwenReranker
    batch_size: int = 4,
) -> dict:
    run_dict = {}
    # Map doc-id string -> integer index
    # {"D1":1, "D2":2, ...}
    d2idx = {dids[i]: i for i in range(len(dids))}
    for qi, qid in enumerate(qids):
        cand = sorted(list(allowlist.get(qid, [])))
        if not cand:
            continue
        # Build pairs
        q = queries[qi]
        pairs = []
        cand_dids = []
        for did in cand:
            j = d2idx.get(did) if isinstance(did, str) else did
            doc = corpus[j]
            pairs.append(reranker.format_instruction(q, doc))
            cand_dids.append(dids[j])
        # Score batch-wise
        # print(f"{qid} pairs: {pairs}")
        scores = reranker.score(pairs, batch_size=batch_size) #from models.py
        # scores = reranker.score_pairs_batched(pairs) #from new reranker class
        # print(scores)
        run_dict[qid] = {cand_dids[k]: float(scores[k]) for k in range(len(cand_dids))}
    n_queries = len(run_dict)
    n_docs = sum(len(v) for v in run_dict.values())
    print(f"[sanity] built run_dict for {n_queries} queries, {n_docs} doc scores total")
    for qid in list(run_dict.keys())[:3]:
        sample = list(run_dict[qid].items())[:3]
        print(f" {qid}: {len(run_dict[qid])} docs | sample {sample}")
        
    # confirm each query has unique doc ids and finite scores
    for qid, dd in run_dict.items():
        assert len(dd) == len(set(dd.keys())), f"duplicate dids in {qid}"
        for did, s in dd.items():
            assert isinstance(s, float) and math.isfinite(s), f"bad score {s} in {qid}->{did}"

    first_qid = qids[0]
    print(f"{first_qid} | allowlist docs (reranker): {len(run_dict.get(first_qid, {}))}")
    print(f"{first_qid} | allowlist docs (reranker): {run_dict[first_qid]}")
    return run_dict

if __name__ == "__main__":
    reranker = QwenReranker(
        "/home/qdai/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots/6e9e69830b95c52b5fd889b7690dda3329508de3",
        "cuda:6"
    )

    qids = ["Q0", "Q1"]
    dids = ["D0", "D1", "D2", "D3"]
    queries = [
        "What is the capital of China?",
        "Explain gravity."
    ]
    corpus = [
        "Beijing is the capital of China.",          # D0
        "Shanghai is a large city in China.",        # D1
        "Gravity pulls objects toward each other.",  # D2
        "The Moon orbits the Earth."                 # D3
    ]

    # FAISS top-k candidates (for example)
    allowlist = {'Q0': [0, 1], 'Q1': [2, 3]}

    run_dict = make_allow_run_with_reranker(qids, dids, queries, corpus, allowlist, reranker)
    print(run_dict)
