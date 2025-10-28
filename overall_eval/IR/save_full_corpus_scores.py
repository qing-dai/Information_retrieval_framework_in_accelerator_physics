import numpy as np
import pandas as pd
from tqdm import tqdm


def save_full_corpus_scores(
    df: pd.DataFrame,
    queries, corpus, qids, dids, sims: np.ndarray,
    query_col="Question", doc_col="chunk_text",
):
    """
    Output columns:
    [qid, did, Question, chunk_text, cos_full, rank_full,
     Source, Question_type, specific_to_paper, relevance_label]
    """
    # ID maps
    q2id  = {q: qids[i] for i, q in enumerate(queries)}
    d2id  = {d: dids[i] for i, d in enumerate(corpus)}

    # --- fixed per-question fields
    q_fixed = (
        df[[query_col, "Question_type", "specific_to_paper"]]
        .drop_duplicates(subset=[query_col])
        .set_index(query_col)
    )

    # --- pair-level labels (only where annotated)
    pair_meta = (
        df[[query_col, doc_col, "relevance_label"]]
        .drop_duplicates(subset=[query_col, doc_col])
        .set_index([query_col, doc_col])
    )

    # --- chunk-based Source: IPAC if IPAC_filename non-empty, else expert
    if "IPAC_filename" not in df.columns:
        raise KeyError("Expected column 'IPAC_filename' to infer Source for full-corpus.")

    d_ipac = (
        df[[doc_col, "IPAC_filename"]]
        .drop_duplicates(subset=[doc_col])
        .set_index(doc_col)
    )

    def infer_source_for_chunk(d: str):
        if d not in d_ipac.index:
            return np.nan
        val = d_ipac.at[d, "IPAC_filename"]
        if pd.isna(val) or str(val).strip() == "":
            return "expert"
        return "IPAC"

    rows = []
    for qi, q in tqdm(list(enumerate(queries)), total=len(queries), desc="Scoring full corpus (by query)"):
        scores = sims[qi]
        order = np.argsort(-scores)

        qt = q_fixed.loc[q, "Question_type"] if q in q_fixed.index else np.nan
        sp = q_fixed.loc[q, "specific_to_paper"] if q in q_fixed.index else np.nan

        for rank, di in enumerate(order, start=1):
            d = corpus[di]

            # relevance_label: 1/0 if annotated; else 0
            if (q, d) in pair_meta.index:
                lbl = int(pair_meta.loc[(q, d)]["relevance_label"])
            else:
                lbl = 0

            # Source purely from chunk metadata
            src = infer_source_for_chunk(d)


            rows.append({
                "qid": q2id[q],
                "did": d2id[d],
                query_col: q,
                doc_col: d,
                "cos_full": float(scores[di]),
                "rank_full": rank,
                "Source": src,
                "Question_type": qt,
                "specific_to_paper": sp,
                "relevance_label": lbl,
            })

    return pd.DataFrame(rows)