"""SetFit-based contrastive fine-tuning of nomic-embed-text for education domain.

Uses sentence-transformers SentenceTransformer + TripletLoss for contrastive learning
on education-domain training data (150 rows from classifier_train.jsonl).
Positive pairs: education rows within the same domain.
Negative pairs: sampled from other domains (1:1 ratio).

Output: fine-tuned model saved to models/sentence-transformer-edu/

Usage:
    uv run python scripts/fine_tune_embedding.py
"""

import json
import random
import sys
from pathlib import Path

from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import TripletLoss
from sentence_transformers.sentence_transformer.trainer import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)


def load_education_rows(path: str) -> list[dict]:
    """Load education rows from classifier_train.jsonl."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["domain"] == "education":
                rows.append(row)
    return rows


def create_contrastive_pairs(
    edu_rows: list[dict],
    all_rows: list[dict],
    seed: int = 42,
) -> Dataset:
    """Create (anchor, positive, negative) triplets for contrastive learning.

    Positive pairs: two education rows (same domain).
    Negative pairs: education row + non-education row (different domain).

    Prioritizes negative samples from domains that confuse education most
    (medical, business_economics, general -- identified in Iter39 analysis).
    """
    rng = random.Random(seed)
    edu_queries = [r["query"] for r in edu_rows]
    other_queries = [r["query"] for r in all_rows if r["domain"] != "education"]

    # Prioritize negative samples from domains that confuse education most
    priority_domains = {"medical", "business_economics", "general"}
    priority_negatives = [r["query"] for r in all_rows
                          if r["domain"] in priority_domains and r["domain"] != "education"]
    other_negatives = [r["query"] for r in all_rows
                       if r["domain"] not in priority_domains and r["domain"] != "education"]

    anchors = []
    positives = []
    negatives = []

    for anchor_query in edu_queries:
        # Positive: another education query
        positive_query = rng.choice(edu_queries)
        while positive_query == anchor_query and len(edu_queries) > 1:
            positive_query = rng.choice(edu_queries)

        # Negative: preferentially from confusing domains (60% priority, 40% random)
        if rng.random() < 0.6 and priority_negatives:
            negative_query = rng.choice(priority_negatives)
        elif other_negatives:
            negative_query = rng.choice(other_negatives)
        else:
            negative_query = rng.choice(other_queries)

        anchors.append(anchor_query)
        positives.append(positive_query)
        negatives.append(negative_query)

    return Dataset.from_dict({
        "anchor": anchors,
        "positive": positives,
        "negative": negatives,
    })


def main() -> None:
    """Run SetFit contrastive fine-tuning."""
    # Load data
    train_path = "data/classifier_train.jsonl"
    all_rows = []
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            all_rows.append(json.loads(line))
    edu_rows = [r for r in all_rows if r["domain"] == "education"]
    print(f"[fine_tune_embedding] loaded {len(edu_rows)} education rows, "
          f"{len(all_rows) - len(edu_rows)} other rows", file=sys.stderr)

    # Create contrastive pairs
    train_dataset = create_contrastive_pairs(edu_rows, all_rows)
    print(f"[fine_tune_embedding] created {len(train_dataset)} triplet pairs", file=sys.stderr)

    # Load base model from HuggingFace
    base_model_name = "nomic-ai/nomic-embed-text-v1"
    print(f"[fine_tune_embedding] loading base model: {base_model_name}", file=sys.stderr)
    model = SentenceTransformer(base_model_name, trust_remote_code=True, device="cpu")

    # Training arguments
    output_dir = "models/sentence-transformer-edu"
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        warmup_steps=10,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        fp16=False,
        seed=42,
        no_cuda=True,
    )

    # Train with TripletLoss
    loss = TripletLoss(model)
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.train()

    # Save fine-tuned model
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    print(f"[fine_tune_embedding] saved fine-tuned model to {output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
