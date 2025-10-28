import pandas as pd, numpy as np
from tqdm import tqdm


def save_allowlist_scores(
    df: pd.DataFrame,
    queries, corpus, 
    sims: np.ndarray,
    query_col: str = "Question",
    doc_col: str = "chunk_text",
):
    """
    Minimal, robust mapping: string-strip the two key columns,
    look up (qi, di), assign sims[qi, di] -> df['cos'].
    """
    # maps based on exact strings produced in build_ir_index
    q2idx = {q: i for i, q in enumerate(queries)}
    d2idx = {d: i for i, d in enumerate(corpus)}

    # safe normalization on-the-fly
    def _lookup(row):
        q = str(row[query_col]).strip()
        d = str(row[doc_col]).strip()
        qi = q2idx.get(q, None)
        di = d2idx.get(d, None)
        if qi is None or di is None:
            return np.nan
        return float(sims[qi, di])
    
    tqdm.pandas(des="Attaching cosin scores for allowlist")

    out_df = df.copy()
    out_df["cos"] = out_df.apply(_lookup, axis=1)
    return pd.DataFrame(out_df)

def save_allowlist_scores_reranker(
    df: pd.DataFrame,
    queries, corpus,
    qids, dids,
    run_allow_rerank,                 # ranx.Run from make_allow_run_with_reranker(...)
    query_col: str = "Question",
    doc_col: str = "chunk_text",
    out_col: str = "rerank_score",
):
    """
    Attach reranker P(yes) for allowlist pairs to df[out_col].
    Uses the ranx Run (qid -> {did: score}) as the source of truth.
    """
    # string -> index maps (match build_ir_index behavior)
    q2idx = {q: i for i, q in enumerate(queries)}
    d2idx = {d: i for i, d in enumerate(corpus)}

    # id -> index maps
    qid2i = {qid: i for i, qid in enumerate(qids)}
    did2i = {did: i for i, did in enumerate(dids)}

    # (qi, di) -> score lookups from Run
    score_lookup = {}
    # run_allow_rerank.run is a dict: {qid: {did: score}}
    for qid, did_scores in run_allow_rerank.items():
        qi = qid2i[qid]
        for did, s in did_scores.items():
            di = did2i[did]
            score_lookup[(qi, di)] = float(s)

    def _lookup(row):
        q = str(row[query_col]).strip()
        d = str(row[doc_col]).strip()
        qi = q2idx.get(q)
        di = d2idx.get(d)
        if qi is None or di is None:
            return np.nan
        return score_lookup.get((qi, di), np.nan)

    try:
        tqdm.pandas(desc="Attaching reranker scores for allowlist")
        out = df.copy()
        out[out_col] = out.progress_apply(_lookup, axis=1)
    except Exception:
        out = df.copy()
        out[out_col] = out.apply(_lookup, axis=1)

    return pd.DataFrame(out)