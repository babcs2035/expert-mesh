"""E10: single-node LoRA fine-tuning for domain-specific expert models.

Extracted from WAFL-PEFT's client.py Thread 3 (Train) and simplified for
single-node supervised fine-tuning. Trains a LoRA adapter on a base model
using domain-specific instruction-tuning data, outputs safetensors format
compatible with Ollama's ADAPTER directive.

Training data format (JSONL):
    {"instruction": "...", "input": "...", "output": "..."}
    or
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Usage:
    uv run python scripts/train_domain_lora.py \\
        --model schroneko/llama-3.1-swallow-8b-instruct-v0.1 \\
        --data data/lora_train/medical.jsonl \\
        --output models/lora_adapters/medical/ \\
        --lora-r 16 --lora-alpha 32 \\
        --epochs 3 --batch-size 2
"""

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path

# CUDA memory fragmentation mitigation (must be set before torch import)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# peft and bitsandbytes are only available when the WAFL-PEFT dependency
# group is installed (see pyproject.toml [project.optional-dependencies] lora).
# Import here so the error is raised at train time, not import time.
from peft import LoraConfig, get_peft_model  # type: ignore[import-not-found]

# Chunked cross-entropy token split size. Materializing full vocab logits in
# fp32 causes hundreds of MB of transient memory per sample with large vocabs
# (e.g. 262144), triggering OOM under GPU contention. Splitting by tokens
# limits the fp32 peak to 1/chunk of the full computation while preserving
# identical gradients through autograd.
_CE_CHUNK_TOKENS = 64


def memory_efficient_causal_lm_loss(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    """Memory-efficient Causal LM loss for large vocabularies.

    Equivalent to transformers' ForCausalLMLoss (right-shift by 1 token,
    mean cross-entropy over non-ignored tokens) but avoids materializing
    the full [seq_len, vocab_size] logits tensor in fp32 at once.
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    vocab_size = shift_logits.size(-1)
    flat_logits = shift_logits.reshape(-1, vocab_size)
    flat_labels = shift_labels.reshape(-1)

    loss_sum = flat_logits.new_zeros((), dtype=torch.float32)
    for start in range(0, flat_logits.size(0), _CE_CHUNK_TOKENS):
        chunk_logits = flat_logits[start : start + _CE_CHUNK_TOKENS].float()
        chunk_labels = flat_labels[start : start + _CE_CHUNK_TOKENS]
        loss_sum = loss_sum + F.cross_entropy(
            chunk_logits, chunk_labels, ignore_index=ignore_index, reduction="sum"
        )
    return loss_sum / flat_labels.ne(ignore_index).sum().clamp(min=1).to(loss_sum.dtype)


def prepare_model_for_training(model, use_gradient_checkpointing: bool = True) -> None:
    """Prepare model parameters for LoRA training.

    Freezes all base model parameters, casts 1D non-4bit params (LayerNorm
    weights) to float32 for numerical stability, and enables gradient
    checkpointing if requested.
    """
    for param in model.parameters():
        param.requires_grad = False

    for param in model.parameters():
        if (
            param.dtype in (torch.float16, torch.bfloat16)
            and param.__class__.__name__ != "Params4bit"
            and param.dim() == 1
        ):
            param.data = param.data.to(torch.float32)

    if use_gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()


def load_model_and_tokenizer(
    model_id: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: str | list[str],
) -> tuple:
    """Load base model with LoRA adapters applied, and its tokenizer.

    Returns (model, tokenizer) tuple. Uses 4-bit QLoRA on GPU, float16 on CPU.
    """
    print(f"[train_domain_lora] Loading model: {model_id}", file=sys.stderr)

    if torch.cuda.is_available():
        from transformers import BitsAndBytesConfig  # type: ignore[attr-defined]

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        device_map = "auto"
        quantization_config = bnb_config
        print("[train_domain_lora] Using 4-bit NF4 quantization (QLoRA)", file=sys.stderr)
    else:
        device_map = "cpu"
        quantization_config = None
        print("[train_domain_lora] No GPU, loading in float16 on CPU", file=sys.stderr)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device_map,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    # Cast lm_head to float32 to match hidden states from gradient checkpointing
    # (LayerNorm weights are cast to float32, making hidden_states float32)
    if quantization_config is not None:
        model.lm_head = model.lm_head.to(torch.float32)
        print("[train_domain_lora] lm_head cast to float32 for dtype consistency", file=sys.stderr)
    print(
        f"[train_domain_lora] Model loaded, device={next(model.parameters()).device}",
        file=sys.stderr,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    if quantization_config is not None:
        prepare_model_for_training(model, use_gradient_checkpointing=True)
    else:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Fix meta tensors that PEFT may leave unresolved
    fallback_device = next(
        (p.device for p in model.parameters() if p.device.type != "meta"),
        torch.device("cpu"),
    )
    for module in model.modules():
        for pname, param in list(module.named_parameters(recurse=False)):
            if param.device.type != "meta":
                continue
            new_data = torch.empty(param.shape, dtype=param.dtype, device=fallback_device)
            if "lora_b" in pname.lower() or new_data.dim() < 2:
                torch.nn.init.zeros_(new_data)
            else:
                torch.nn.init.kaiming_uniform_(new_data, a=math.sqrt(5))
            setattr(module, pname, torch.nn.Parameter(new_data, requires_grad=param.requires_grad))

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[train_domain_lora] LoRA applied and tokenizer loaded", file=sys.stderr)
    return model, tokenizer


def load_training_data(data_path: str) -> list[dict]:
    """Load instruction-tuning data from JSONL file.

    Supports two formats:
    - {"instruction": "...", "input": "...", "output": "..."}
    - {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
    """
    with open(data_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tokenize_sample(
    sample: dict, tokenizer, max_seq_len: int
) -> dict[str, torch.Tensor] | None:
    """Tokenize a single instruction-tuning sample.

    Masks the prompt portion with -100 so loss is only computed on the
    assistant's response tokens. Returns None if the answer is fully
    truncated.
    """
    if "messages" in sample:
        # Chat format
        prompt_text = tokenizer.apply_chat_template(
            sample["messages"][:-1], tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            sample["messages"], tokenize=False
        )
    else:
        # Instruction-input-output format
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        output = sample.get("output", "")
        prompt_text = f"Instruction: {instruction}\n{input_text}\n\nResponse:" if input_text else f"Instruction: {instruction}\n\nResponse:"
        full_text = f"{prompt_text} {output}"

    prompt_ids = tokenizer(prompt_text, truncation=False)["input_ids"]
    tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=max_seq_len,
        padding=False,
    )
    full_ids = tokens["input_ids"]

    if len(full_ids) <= 1:
        return None

    prompt_len = min(len(prompt_ids), len(full_ids))
    if prompt_len >= len(full_ids):
        return None  # answer fully truncated

    input_ids = torch.tensor(full_ids, dtype=torch.long)
    labels = input_ids.clone()
    labels[:prompt_len] = -100
    return {"input_ids": input_ids, "labels": labels}


def tokenize_dataset(
    samples: list[dict], tokenizer, max_seq_len: int
) -> list[dict[str, torch.Tensor]]:
    """Tokenize all samples, filtering out those fully truncated."""
    tokenized = []
    skipped = 0
    for sample in samples:
        result = tokenize_sample(sample, tokenizer, max_seq_len)
        if result is not None:
            tokenized.append(result)
        else:
            skipped += 1
    if skipped > 0:
        print(
            f"[train_domain_lora] Skipped {skipped} samples (answer lost at max_seq_len={max_seq_len})",
            file=sys.stderr,
        )
    return tokenized


def _eta_str(seconds: float) -> str:
    """Convert seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.0f}m{int(seconds%60):02d}s"
    return f"{seconds/3600:.1f}h"


def train(
    model,
    train_data: list[dict[str, torch.Tensor]],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    lr_warmup_steps: int,
    lr_min_ratio: float,
    weight_decay: float,
    max_seq_len: int,
    output_dir: str,
) -> None:
    """Run the LoRA training loop.

    Uses gradient accumulation (micro-batch=1) to keep memory usage low
    while achieving effective batch sizes. Applies cosine LR decay based
    on training progress fraction.
    """
    train_device = next(
        (p.device for p in model.parameters() if p.requires_grad and p.device.type != "meta"),
        torch.device("cpu"),
    )
    print(f"[train_domain_lora] Training device: {train_device}", file=sys.stderr)

    # Paged 8-bit AdamW for memory efficiency under GPU constraints
    try:
        import bitsandbytes as bnb  # type: ignore[import-not-found]

        optimizer = bnb.optim.PagedAdamW8bit(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        print("[train_domain_lora] Using PagedAdamW8bit optimizer", file=sys.stderr)
    except ImportError:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        print("[train_domain_lora] bitsandbytes unavailable, using AdamW", file=sys.stderr)

    base_lr = learning_rate
    total_steps = len(train_data) * epochs
    print(
        f"[train_domain_lora] Training: {len(train_data)} samples, "
        f"{epochs} epochs, {total_steps} total steps",
        file=sys.stderr,
    )

    # Data order: shuffle once per epoch to avoid seeing data in the same
    # sequence every epoch, which encourages overfitting.
    data_order = list(range(len(train_data)))

    optimizer.zero_grad()
    start_time = time.time()
    global_step = 0
    opt_step = 0

    for epoch in range(epochs):
        random.shuffle(data_order)
        epoch_losses = []

        for idx in data_order:
            batch = train_data[idx]
            input_ids = batch["input_ids"].unsqueeze(0).to(train_device)
            labels = batch["labels"].unsqueeze(0).to(train_device)

            # Forward pass without labels to get raw logits (avoid full fp32 materialization)
            outputs = model(input_ids=input_ids)
            loss = memory_efficient_causal_lm_loss(outputs.logits, labels)

            loss.backward()

            # LR scheduling: linear warmup then cosine decay
            if lr_warmup_steps > 0 and opt_step < lr_warmup_steps:
                lr_scale = opt_step / lr_warmup_steps
            else:
                progress = min(1.0, global_step / max(1, total_steps))
                lr_scale = lr_min_ratio + 0.5 * (1.0 - lr_min_ratio) * (1.0 + math.cos(math.pi * progress))

            for group in optimizer.param_groups:
                group["lr"] = base_lr * lr_scale

            optimizer.step()
            optimizer.zero_grad()
            opt_step += 1

            loss_value = loss.item()
            epoch_losses.append(loss_value)
            global_step += 1

            elapsed = time.time() - start_time
            tokens_processed = int(input_ids.numel())
            tokens_per_sec = tokens_processed / (elapsed / max(1, global_step))
            remaining = max(0, (total_steps - global_step) * (elapsed / max(1, global_step)))

            if global_step % 10 == 0 or global_step == 1:
                print(
                    f"[train_domain_lora] Epoch {epoch+1}/{epochs}, Step {global_step}/{total_steps}: "
                    f"loss={loss_value:.4f}, lr={base_lr*lr_scale:.2e}, tok/s={tokens_per_sec:.0f}, "
                    f"remaining={_eta_str(remaining)}",
                    file=sys.stderr,
                )

            # Free GPU memory after each step to prevent fragmentation
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        print(
            f"[train_domain_lora] Epoch {epoch+1} complete: avg_loss={avg_loss:.4f}",
            file=sys.stderr,
        )

    # Save LoRA adapter weights
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    print(f"[train_domain_lora] LoRA adapter saved to {output_path}", file=sys.stderr)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train a domain-specific LoRA adapter for expert-mesh"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Base model ID (HuggingFace or local path)",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to instruction-tuning JSONL file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for LoRA adapter weights (safetensors format)",
    )
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (default: 32)")
    parser.add_argument(
        "--lora-dropout", type=float, default=0.1, help="LoRA dropout (default: 0.1)"
    )
    parser.add_argument(
        "--target-modules",
        default=None,
        help="Comma-separated list of target module names (e.g. q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj) "
             "or a regex pattern. Default: standard attention+MLP projections for Llama-style models.",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Effective batch size (via gradient accumulation)"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-4, help="Base learning rate (default: 2e-4)"
    )
    parser.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=100,
        help="Linear warmup steps (default: 100)",
    )
    parser.add_argument(
        "--lr-min-ratio", type=float, default=0.1, help="Minimum LR ratio for cosine decay"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="Weight decay (default: 0.01)"
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=1024, help="Maximum sequence length (default: 1024)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    # Set deterministic seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load data
    samples = load_training_data(args.data)
    print(f"[train_domain_lora] Loaded {len(samples)} training samples", file=sys.stderr)

    # Load model and tokenizer
    # Resolve target_modules: None -> default list, comma-separated -> list, else -> regex string
    target_modules = args.target_modules
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif "," in target_modules:
        target_modules = [m.strip() for m in target_modules.split(",")]
    # else: keep as regex string

    model, tokenizer = load_model_and_tokenizer(
        args.model, args.lora_r, args.lora_alpha, args.lora_dropout, target_modules
    )

    # Tokenize dataset
    train_data = tokenize_dataset(samples, tokenizer, args.max_seq_len)
    print(f"[train_domain_lora] Tokenized {len(train_data)} samples", file=sys.stderr)

    if not train_data:
        print("[train_domain_lora] ERROR: No valid training samples after tokenization", file=sys.stderr)
        sys.exit(1)

    # Train
    train(
        model=model,
        train_data=train_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_min_ratio=args.lr_min_ratio,
        weight_decay=args.weight_decay,
        max_seq_len=args.max_seq_len,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
