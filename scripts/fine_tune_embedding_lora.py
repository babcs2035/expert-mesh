"""PEFT LoRA fine-tuning of nomic-embed-text for education domain.

Applies Low-Rank Adaptation (LoRA) via PEFT to the SentenceTransformer
implementation of nomic-embed-text-v1. Trains only the LoRA adapter
parameters (rank=8) using MultipleNegativesRankingLoss on education
domain contrastive pairs.

Base model parameters are frozen. Only LoRA matrices (A: 768x8, B: 8x768
per target module) are updated, ensuring minimal impact on non-education
domain embeddings (single-lever principle).

Output: LoRA adapter saved to models/embedding_lora_education_r8/ (safetensors)
Usage:
    uv run python scripts/fine_tune_embedding_lora.py
"""

import json
import random
import sys
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, TaskType
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers import SentenceTransformerTrainer


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
    """Run PEFT LoRA fine-tuning of nomic-embed-text for education domain."""
    # Load data
    train_path = "data/classifier_train.jsonl"
    all_rows = []
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            all_rows.append(json.loads(line))
    edu_rows = [r for r in all_rows if r["domain"] == "education"]
    print(f"[fine_tune_embedding_lora] loaded {len(edu_rows)} education rows, "
          f"{len(all_rows) - len(edu_rows)} other rows", file=sys.stderr)

    # Create contrastive pairs
    train_dataset = create_contrastive_pairs(edu_rows, all_rows)
    print(f"[fine_tune_embedding_lora] created {len(train_dataset)} triplet pairs",
          file=sys.stderr)

    # Load base model from HuggingFace
    base_model_name = "nomic-ai/nomic-embed-text-v1"
    print(f"[fine_tune_embedding_lora] loading base model: {base_model_name}",
          file=sys.stderr)
    model = SentenceTransformer(base_model_name, trust_remote_code=True, device="cpu")

    # Configure LoRA adapter
    # rank=8: halved from r=16 (Iter41) to reduce argmax flip rate below 15%.
    # The task (education vs non-education separation) is simpler than full LLM instruction
    # following, so r=8 should be sufficient while maintaining single-lever behavior.
    # alpha=16: alpha = 2 * r (standard setting). Scaling factor = alpha/r = 2.0.
    # dropout=0.1: standard dropout for regularization.
    # target_modules=["Wqkv", "out_proj"]: target all attention projection layers
    # across all 12 encoder layers. nomic-embed-text-v1 uses fused Wqkv (768->3072) and
    # out_proj (768->768) instead of separate q/k/v projections. 24 modules total (2 per
    # layer x 12 layers). Total trainable params: 24 * 2 * (768 * 8 + 768 * 8) = 471,856
    # (~0.34% of base model).
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["Wqkv", "out_proj"],
    )
    model.add_adapter(lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        "[fine_tune_embedding_lora] LoRA config: r=8, alpha=16, dropout=0.1, "
        "target_modules=Wqkv+out_proj (24 modules, ~0.34% of base)", file=sys.stderr
    )
    print(
        f"[fine_tune_embedding_lora] Trainable params: {trainable_params:,} / "
        f"{total_params:,} ({100 * trainable_params / total_params:.2f}%)",
        file=sys.stderr,
    )

    # Training arguments
    output_dir = "models/embedding_lora_education_r8"
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
    # More stable and efficient than TripletLoss for this use case.
    loss = MultipleNegativesRankingLoss(model)
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.train()

    # Save LoRA adapter only (not the full model)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    print(
        f"[fine_tune_embedding_lora] saved LoRA adapter to {output_dir}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
