"""Dense projection head fine-tuning of nomic-embed-text for education domain.

Applies a learnable linear projection (Dense: W*x + b) to the final 768-dimensional
embedding output from nomic-embed-text-v1. Trains only the Dense parameters using
contrastive learning on education domain pairs.

Base model (Transformer + Pooling) is frozen. Only the Dense module is trained.
This is fundamentally different from LoRA which perturbs attention layers.

Output: Dense projection head saved to models/embedding_projection_education/
Usage:
    uv run python scripts/fine_tune_embedding_projection_head.py
"""

import json
import random
import sys
from pathlib import Path

from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.modules import Dense
from sentence_transformers.training_args import SentenceTransformerTrainingArguments


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
    60% priority from these domains, 40% random from all other domains.
    """
    rng = random.Random(seed)
    edu_queries = [r["query"] for r in edu_rows]
    other_queries = [r["query"] for r in all_rows if r["domain"] != "education"]

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
    """Run Dense projection head fine-tuning of nomic-embed-text for education domain."""
    # Load data
    train_path = "data/classifier_train.jsonl"
    all_rows = []
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            all_rows.append(json.loads(line))
    edu_rows = [r for r in all_rows if r["domain"] == "education"]
    print(f"[fine_tune_projection_head] loaded {len(edu_rows)} education rows, "
          f"{len(all_rows) - len(edu_rows)} other rows", file=sys.stderr)

    # Create contrastive pairs
    train_dataset = create_contrastive_pairs(edu_rows, all_rows)
    print(f"[fine_tune_projection_head] created {len(train_dataset)} triplet pairs",
          file=sys.stderr)

    # Load base model from HuggingFace
    base_model_name = "nomic-ai/nomic-embed-text-v1"
    print(f"[fine_tune_projection_head] loading base model: {base_model_name}",
          file=sys.stderr)
    model = SentenceTransformer(base_model_name, trust_remote_code=True, device="cpu")

    # Inject Dense projection head
    # This is a linear projection (W*x + b) applied to the final 768-dim embedding.
    # activation_function=None -> nn.Identity() (no non-linearity, pure linear projection).
    # This differs from LoRA which adds perturbation to attention layers (12 layers).
    # Dense module is applied AFTER Pooling and Normalize in the SentenceTransformer pipeline,
    # then encode() with normalize_embeddings=True re-normalizes the final output.
    projection_head = Dense(
        in_features=768,
        out_features=768,
        bias=True,
        activation_function=None,  # Pure linear: W*x + b, no Tanh/ReLU
    )
    model.add_module("Dense", projection_head)

    # Freeze base model (Transformer + Pooling) parameters.
    # Only the Dense module parameters should be trainable.
    for param in model.parameters():
        param.requires_grad = False
    for param in projection_head.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        "[fine_tune_projection_head] Dense projection head injected (768->768, no activation)",
        file=sys.stderr,
    )
    print(
        f"[fine_tune_projection_head] Trainable params: {trainable_params:,} / "
        f"{total_params:,} ({100 * trainable_params / total_params:.4f}%)",
        file=sys.stderr,
    )

    # Training arguments
    output_dir = "models/embedding_projection_education"
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
        use_cpu=True,
    )

    # Train with MultipleNegativesRankingLoss
    # SBERT official recommended loss for embedding adaptation.
    loss = MultipleNegativesRankingLoss(model)
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.train()

    # Save the full fine-tuned model (base model + Dense module)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    print(
        f"[fine_tune_projection_head] saved fine-tuned model to {output_dir}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
