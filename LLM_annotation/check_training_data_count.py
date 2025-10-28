#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

REQUIRED_COLS = {
    "ID","Name","Question","Question_type","Expert_file","Pages","Specific_question?",
    "filename","rank","score","chunk_text","thought","response","Source","label"
}

def robust_read_excel(path: Path, sheet: str | None) -> pd.DataFrame:
    if sheet:
        df = pd.read_excel(path, sheet_name=sheet, dtype=str)
    else:
        df = pd.read_excel(path, dtype=str)
    return df.fillna("")

def coerce_label(col: pd.Series) -> pd.Series:
    s = col.astype(str).str.strip().str.lower()
    map_dict = {"1":"1", "0":"0", "true":"1", "false":"0", "yes":"1", "no":"0"}
    s = s.map(lambda x: map_dict.get(x, x))
    out = pd.to_numeric(s, errors="coerce").astype("Int64")
    return out

def main():
    ap = argparse.ArgumentParser(
        description="Check per-question pos/neg coverage + histograms"
    )
    ap.add_argument("--input", default="total_training_data.xlsx", help="Input Excel file")
    ap.add_argument("--sheet", default=None, help="Optional sheet name")
    ap.add_argument("--out-dir", default="analysis_reports", help="Output folder")
    args = ap.parse_args()

    path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = robust_read_excel(path, args.sheet)

    # Schema check (same as before)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    # Labels → 0/1 (same as before)
    df["label_num"] = coerce_label(df["label"])
    before = len(df)
    df = df.dropna(subset=["label_num"]).copy()
    df["label_num"] = df["label_num"].astype(int)
    dropped = before - len(df)
    if dropped:
        print(f"[info] Dropped {dropped} rows with non-parseable labels.")

    # Overall distribution (same)
    counts = df["label_num"].value_counts().sort_index()
    pos, neg = int(counts.get(1, 0)), int(counts.get(0, 0))
    total = pos + neg
    print("\n=== Overall label distribution ===")
    print(f"Total rows: {total}")
    print(f"Positives (1): {pos}  | Negatives (0): {neg}")

    # Per-question coverage (same as before)
    grp = df.groupby("Question")["label_num"]
    per_q = pd.DataFrame({
        "pos_count": grp.apply(lambda s: (s == 1).sum()),
        "neg_count": grp.apply(lambda s: (s == 0).sum()),
        "total": grp.size()
    }).reset_index()
    per_q["has_pos"] = per_q["pos_count"] > 0
    per_q["has_neg"] = per_q["neg_count"] > 0
    per_q["ok_pos_and_neg"] = per_q["has_pos"] & per_q["has_neg"]

    # Save the original reports (unchanged)
    bad = per_q[~per_q["ok_pos_and_neg"]].sort_values(
        ["pos_count","neg_count","total"], ascending=[True, True, False]
    )
    per_q.to_csv(out_dir / "per_question_coverage.csv", index=False)
    bad.to_csv(out_dir / "questions_missing_pos_or_neg.csv", index=False)

    print("\n=== Per-question coverage summary ===")
    print(f"Unique Questions: {len(per_q)}")
    print(f"OK (≥1 pos & ≥1 neg): {(per_q['ok_pos_and_neg']).sum()}")
    print(f"Missing either pos or neg: {len(bad)}")

    # NEW: histograms of counts across questions
    pos_hist = per_q["pos_count"].value_counts().sort_index()
    neg_hist = per_q["neg_count"].value_counts().sort_index()

    print("\n=== Histogram: #positives per question → #questions ===")
    for n, c in pos_hist.items():
        print(f"{n} positives: {c} questions")

    print("\n=== Histogram: #negatives per question → #questions ===")
    for n, c in neg_hist.items():
        print(f"{n} negatives: {c} questions")

    pos_hist_df = pos_hist.rename("num_questions").reset_index().rename(columns={"index":"num_positives"})
    neg_hist_df = neg_hist.rename("num_questions").reset_index().rename(columns={"index":"num_negatives"})
    pos_hist_df.to_csv(out_dir / "positives_per_question_histogram.csv", index=False)
    neg_hist_df.to_csv(out_dir / "negatives_per_question_histogram.csv", index=False)

    # OPTIONAL: joint table (#pos vs #neg)
    joint = (per_q.groupby(["pos_count","neg_count"])
                 .size()
                 .unstack(fill_value=0)
                 .sort_index(axis=0)
                 .sort_index(axis=1))
    joint.to_csv(out_dir / "joint_pos_neg_table.csv")

    print(f"\n[done] Wrote:")
    print(f" - {out_dir/'per_question_coverage.csv'}")
    print(f" - {out_dir/'questions_missing_pos_or_neg.csv'}")
    print(f" - {out_dir/'positives_per_question_histogram.csv'}")
    print(f" - {out_dir/'negatives_per_question_histogram.csv'}")
    print(f" - {out_dir/'joint_pos_neg_table.csv'}")

if __name__ == "__main__":
    main()