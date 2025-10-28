import pandas as pd, numpy as np
from typing import List, Dict, Tuple, Set
from pathlib import Path
from ranx import Qrels, Run, evaluate


# TEST_CSV = '/home/qdai/selected_20_with_positive.csv'
TEST_CSV = 'Filtered_Positive_Questions.xlsx'

def robust_read_data(path: str) -> pd.DataFrame:
    path = Path(path)  # ✅ convert string to Path
    suffix = path.suffix.lower()

    if suffix == ".csv":
        for enc in ["utf-8", "utf-8-sig", "ISO-8859-1", "cp1252"]:
            try:
                return pd.read_csv(path, encoding=enc)
            except Exception:
                pass
        return pd.read_csv(path, encoding="ISO-8859-1", encoding_errors="replace")

    elif suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=str, keep_default_na=False)

    else:
        raise ValueError(f"Unsupported file extension: {suffix}")


# --------------------
# Build IR index with per-query allowlist (candidates)
# --------------------
def build_ir_index(df: pd.DataFrame,
               query_col="Question",
               doc_col="chunk_text",
               label_col="relevance_label"
               ) -> Tuple[List[str], List[str], List[str], List[str], Qrels, Dict[str, Set[str]]]:
    # remove any rows where query/doc/label is missing (NaN); copy() make a separate dataframe
    df = df.dropna(subset=[query_col, doc_col, label_col]).copy()
    # 1. convert these columns to strings, 2. remove leading/trailing whitespace,
    df[query_col] = df[query_col].astype(str).str.strip()
    df[doc_col]   = df[doc_col].astype(str).str.strip()
    # force the label column to be numeric
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)

    # keep queries that have ≥1 positive, index means the query value, by which we group the df
    pos_q = set(df.groupby(query_col)[label_col].max().loc[lambda s: s == 1].index)
    # we only want queries with at least one positive
    df = df[df[query_col].isin(pos_q)].copy()
    if df.empty:
        raise ValueError("No queries with at least one positive.")

    # unique queries and docs, preserving order
    queries = list(dict.fromkeys(df[query_col].tolist()))
    corpus  = list(dict.fromkeys(df[doc_col].tolist()))
    # dicts of query/doc to IDs
    # {q1: "Q1", q2: "Q2",...}
    q2id = {q: f"Q{i}" for i, q in enumerate(queries)}
    # {d1: "D1", d2: "D2",...}
    d2id = {d: f"D{i}" for i, d in enumerate(corpus)}

    # ground truth relevance mapping
    # queries ID → {positive doc ID → 1}
    # e.g. {"Q0": {"D0": 1, "D1": 1}, "Q1": {"D2": 1}}
    qrels_dict: Dict[str, Dict[str, int]] = {}
    for q, g in df[df[label_col] == 1].groupby(query_col):
        qid = q2id[q]
        qrels_dict[qid] = {d2id[d]: 1 for d in g[doc_col].tolist()}

    # allowlist (per-query candidates)
    # for each query ID, create an empty set
    allowlist: Dict[str, Set[str]] = {q2id[q]: set() for q in queries}
    # fill it with all docs ID (positive and negative) associated with that query
    # e.g. {"Q0": {"D0", "D1", "D3"}, "Q1": {"D2", "D4"}}
    for q, d in df[[query_col, doc_col]].itertuples(index=False, name=None):
        allowlist[q2id[q]].add(d2id[d])

    # final lists of IDs (in same order as queries/corpus)
    # ["Q1","Q2",...]
    qids = [q2id[q] for q in queries]
    # ["D1","D2",...]
    dids = [d2id[d] for d in corpus]
    qrels = Qrels(qrels_dict)

    print(f"Queries={len(queries)} | Corpus={len(corpus)} | "
          f"Q with positives={len(qrels_dict)} | "
          f"Avg candidates/query={sum(map(len, allowlist.values()))/len(allowlist):.1f}")

    return queries, corpus, qids, dids, qrels, allowlist

if __name__ == "__main__":
    df = robust_read_data("Filtered_Positive_Questions.xlsx")
    queries, corpus, qids, dids, qrels, allowlist = build_ir_index(df)
    assert len(queries) == len(qids)
    print(allowlist["Q0"], allowlist["Q1"], allowlist["Q2"])
    print(f"allowlist: {allowlist}")
    print(f"qrels: {qrels}")