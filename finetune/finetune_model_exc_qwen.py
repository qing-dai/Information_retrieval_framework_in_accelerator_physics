#!/usr/bin/env python3
"""Fine-tune a SentenceTransformer model on question → chunk relevance data.

This script loads paired training data (e.g. question + chunk + response) from an
Excel/CSV file, converts it into a Hugging Face dataset, and fine-tunes a
SentenceTransformer model with a contrastive objective. The script is modular and
configurable via command-line flags so different base models and datasets can be
reused.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from dotenv import load_dotenv
load_dotenv()
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from datasets import Dataset
import os

import wandb

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss, ContrastiveLoss
from bge_utils import is_bge_model, BGELoss

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a SentenceTransformer using contrastive loss",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-mpnet-base-v2",
        help="Base SentenceTransformer model to fine-tune",
    )
    parser.add_argument(
        "--train-file",
        default="total_training_data.xlsx",
        help="Path to the Excel/CSV file containing training pairs",
    )
    parser.add_argument(
        "--question-col",
        default="Question",
        help="Column name for the query/question text",
    )
    parser.add_argument(
        "--chunk-col",
        default="chunk_text",
        help="Column name for the retrieved chunk text",
    )
    parser.add_argument(
        "--label-col",
        default="label",
        help="Column containing numeric labels (0/1 or floats)",
    )
    parser.add_argument(
        "--eval-split",
        type=float,
        default=0.1,
        help="Fraction of data held out for evaluation (set 0 to disable eval)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optionally subsample this many examples after shuffling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for shuffling and splitting",
    )

    # Training hyper-parameters
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 training where supported")
    parser.add_argument("--bf16", action="store_true", help="Enable BF16 training where supported")

    parser.add_argument(
        "--output-dir",
        default="/home/qdai/embeddings_new_try",
        help="Directory where checkpoints and final model will be stored",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name (also used for output sub-directory)",
    )

    # wandb logging
    parser.add_argument(
        "--wandb-project",
        default=os.getenv("WANDB_PROJECT"),
        help="Weights & Biases project name(already set up in .env)",
    )
    parser.add_argument(
        "--wandb-entity",
        default=os.getenv("WANDB_ENTITY"),
        help="Weights & Biases entity/account name(already set up in .env)",
    )

    parser.add_argument("--bge-contrastive", dest="bge_contrastive", action="store_true",
                   help="Fine-tune BGE with ContrastiveLoss on 0/1 labels (no positives-only filter).")

    # BGE query instruction controls
    parser.add_argument(
        "--query-instruction",
        default="Represent this sentence for searching relevant passages: ",
        help="Instruction prefix for queries (BGE style).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Training file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path, dtype=str, keep_default_na=False)
    elif suffix == ".csv":
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    else:
        raise ValueError(f"Unsupported file extension '{suffix}' – use CSV or Excel")
    return df


def build_sentence_pairs(
    df: pd.DataFrame,
    question_col: str,
    chunk_col: str,
    label_col: str,
    apply_query_instruction: bool,
    query_instruction: str,
) -> pd.DataFrame:
    records: List[Tuple[str, str, float]] = []
    for row in df.itertuples(index=False):
        question = getattr(row, question_col, None)
        chunk = getattr(row, chunk_col, None)
        raw_label = getattr(row, label_col, None)

        if raw_label is None or raw_label == "":
            continue

        try:
            label = float(raw_label)
        except (TypeError, ValueError):
            continue

        if not question or not chunk:
            continue
        qtext = str(question)
        if apply_query_instruction:
            qtext = f"{query_instruction}{qtext}"

        records.append((qtext, str(chunk), float(label)))

    if not records:
        raise ValueError(
            "No usable training examples were produced. Check column names and label parsing."
        )

    pair_df = pd.DataFrame(records, columns=["sentence1", "sentence2", "label"])
    return pair_df


def to_datasets(
    pairs: pd.DataFrame,
    eval_split: float,
    seed: int,
    max_samples: Optional[int],
) -> Tuple[Dataset, Optional[Dataset]]:
    dataset = Dataset.from_pandas(pairs, preserve_index=False)

    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(max_samples))
    else:
        dataset = dataset.shuffle(seed=seed)

    if eval_split and eval_split > 0.0:
        if not 0.0 < eval_split < 1.0:
            raise ValueError("eval_split must be between 0 and 1 (exclusive)")
        split = dataset.train_test_split(test_size=eval_split, seed=seed)
        return split["train"], split["test"]

    return dataset, None


# ---------------------------------------------------------------------------
# wandb utilities
# ---------------------------------------------------------------------------


def init_wandb(args: argparse.Namespace, train_size: int, eval_size: int) -> Optional[object]:
    if not args.wandb_project:
        return None
    if wandb is None:
        raise ImportError("wandb is not installed but --wandb-project was provided")

    run_name = args.run_name or f"{Path(args.model_name).name}-contrastive"
    config = {
        "model": args.model_name,
        "loss": "ContrastiveLoss",
        "margin": args.margin,
        "epochs": args.num_epochs,
        "batch_size": args.train_batch_size,
        "learning_rate": args.learning_rate,
        "train_samples": train_size,
        "eval_samples": eval_size,
    }
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=config,
        settings=wandb.Settings(init_timeout=240, start_method="thread")
    )


# ---------------------------------------------------------------------------
# Training orchestration
# ---------------------------------------------------------------------------


def create_training_args(args: argparse.Namespace, has_eval: bool, output_dir: Path) -> SentenceTransformerTrainingArguments:
    evaluation_strategy = "steps" if has_eval else "no"
    eval_steps = args.eval_steps if has_eval else None

    return SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        bf16=args.bf16,
        eval_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        run_name=args.run_name,
        report_to=[] if args.wandb_project else None,
        seed=args.seed,
    )


def main() -> None:
    args = parse_args()

    train_path = Path(args.train_file)
    output_root = Path(args.output_dir)
    run_name = args.run_name or f"{Path(args.model_name).name}-contrastive"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {train_path}…")
    df = load_dataframe(train_path)

    is_bge = is_bge_model(args.model_name)
    if is_bge:
        print("finetuning BGE...")
    pairs_df = build_sentence_pairs(
        df,
        question_col=args.question_col,
        chunk_col=args.chunk_col,
        label_col=args.label_col,
        apply_query_instruction=is_bge,
        query_instruction=args.query_instruction,
    )
    label_counts = pairs_df["label"].value_counts().to_dict()
    print(f"Prepared {len(pairs_df)} sentence pairs (label distribution: {label_counts})")

    # Dataset split
    if is_bge and not args.bge_contrastive:
        # Default BGE style (InfoNCE): positives only
        pos_df = pairs_df[pairs_df["label"] > 0].copy()
        if pos_df.empty:
            raise ValueError("BGE InfoNCE needs positive pairs; got none after filtering label>0.")
        train_dataset, eval_dataset = to_datasets(pos_df, args.eval_split, args.seed, args.max_samples)
    else:
        # Contrastive on 0/1 labels (used for non-BGE; or BGE when --bge-contrastive is set)
        train_dataset, eval_dataset = to_datasets(pairs_df, args.eval_split, args.seed, args.max_samples)

    print(f"Training samples: {len(train_dataset)} | Eval samples: {len(eval_dataset) if eval_dataset else 0}")

    wandb_run = init_wandb(args, len(train_dataset), len(eval_dataset) if eval_dataset else 0)

    model = SentenceTransformer(args.model_name)
    # if BGE -> enforce CLS pooling only
    if is_bge:
        from sentence_transformers.models import Pooling
        for m in model.modules():
            if isinstance(m, Pooling):
                m.pooling_mode_cls_token = True
                m.pooling_mode_mean_tokens = False
                m.pooling_mode_max_tokens = False
                break
            
     # Loss: BGELoss (temp=0.02) for BGE, otherwise margin-based ContrastiveLoss
    if is_bge and not args.bge_contrastive:
        loss = MultipleNegativesRankingLoss(model=model, scale=50.0)
    else:
        loss = ContrastiveLoss(model, margin=args.margin)


    evaluator = None
    if eval_dataset and len(eval_dataset) > 0:
        evaluator = EmbeddingSimilarityEvaluator(
            sentences1=list(eval_dataset["sentence1"]),
            sentences2=list(eval_dataset["sentence2"]),
            scores=list(eval_dataset["label"]),
            name="contrastive-eval",
        )

    training_args = create_training_args(args, evaluator is not None, run_dir)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
    )

    print("Starting training…")
    trainer.train()

    final_metrics = trainer.evaluate() if evaluator is not None else {}
    if final_metrics:
        print("Evaluation metrics:", final_metrics)

    if wandb_run:
        wandb_run.log(final_metrics)
        wandb_run.finish()

    save_path = run_dir / "final"
    model.save_pretrained(str(save_path))
    print(f"Saved fine-tuned model to {save_path}")


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    main()
