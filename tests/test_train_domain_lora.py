"""Tests for E10's LoRA training script (scripts/train_domain_lora.py).

Tests focus on pure-Python functions that do not require GPU/torch/peft.
Training loop tests require the 'lora' optional dependency group.
"""

import importlib.util
import json
import tempfile

import pytest

_torch_available = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not _torch_available, reason="torch not installed (uv sync --extra lora)")
def test_load_training_data_parses_jsonl() -> None:
    """Load instruction-tuning data from JSONL file."""
    from scripts.train_domain_lora import load_training_data

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"instruction": "test", "output": "answer"}) + "\n")
        f.write(json.dumps({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}) + "\n")
        path = f.name

    samples = load_training_data(path)

    assert len(samples) == 2
    assert samples[0]["instruction"] == "test"
    assert samples[1]["messages"][0]["role"] == "user"


@pytest.mark.skipif(not _torch_available, reason="torch not installed (uv sync --extra lora)")
def test_load_training_data_ignores_empty_lines() -> None:
    """Empty lines in JSONL are skipped."""
    from scripts.train_domain_lora import load_training_data

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"instruction": "a", "output": "b"}) + "\n")
        f.write("\n")
        f.write("  \n")
        f.write(json.dumps({"instruction": "c", "output": "d"}) + "\n")
        path = f.name

    samples = load_training_data(path)

    assert len(samples) == 2
