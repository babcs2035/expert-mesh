"""Tests for E10's Ollama model creation script (scripts/create_lora_model.py)."""

from scripts.create_lora_model import generate_modelfile


def test_generate_modelfile_basic() -> None:
    """Modelfile contains FROM and ADAPTER directives."""
    content = generate_modelfile(
        base_model="schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m",
        adapter_path="models/lora_adapters/medical/",
        system_prompt=None,
    )

    assert "FROM schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m" in content
    assert "ADAPTER models/lora_adapters/medical/" in content
    assert "SYSTEM" not in content


def test_generate_modelfile_with_system_prompt() -> None:
    """Modelfile includes SYSTEM directive when system_prompt is provided."""
    content = generate_modelfile(
        base_model="base-model",
        adapter_path="adapter/path/",
        system_prompt="You are a medical expert.",
    )

    assert "FROM base-model" in content
    assert "ADAPTER adapter/path/" in content
    assert 'SYSTEM """You are a medical expert."""' in content


def test_generate_modelfile_ends_with_newline() -> None:
    """Modelfile content ends with a newline for correct file formatting."""
    content = generate_modelfile(
        base_model="base",
        adapter_path="adapter/",
        system_prompt=None,
    )

    assert content.endswith("\n")
