"""Create Ollama models from LoRA adapters for domain-specific expert specialization.

Generates an Ollama Modelfile that applies a LoRA adapter on top of a base
model using the ADAPTER directive, then registers it with `ollama create`.

Ollama 0.32.4+ supports the ADAPTER directive in Modelfiles, which loads
a LoRA adapter (safetensors directory or GGUF file) at inference time
without permanently merging it into the base weights. This keeps VRAM
usage minimal (~10-30MB per adapter for rank 16) and allows the base
model to be shared across domains.

Usage:
    uv run python scripts/create_lora_model.py \\
        --base schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m \\
        --adapter models/lora_adapters/medical/ \\
        --name expert-mesh-medical-lora \\
        --ollama-host 192.168.15.103 \\
        --ollama-port 11434
"""

import argparse
import sys
from pathlib import Path


def generate_modelfile(base_model: str, adapter_path: str, system_prompt: str | None) -> str:
    """Generate Modelfile content that applies a LoRA adapter to a base model.

    The ADAPTER directive points to the safetensors output directory from
    train_domain_lora.py. Ollama handles the adapter loading at inference
    time without merging weights into the base model.
    """
    lines = [f"FROM {base_model}"]
    lines.append(f"ADAPTER {adapter_path}")

    if system_prompt:
        lines.append(f'SYSTEM """{system_prompt}"""')

    return "\n".join(lines) + "\n"


def create_model(
    name: str,
    modelfile_content: str,
    ollama_host: str,
    ollama_port: int,
    adapter_path: str,
) -> None:
    """Register the model with Ollama using `ollama create`.

    Because the adapter files need to be accessible from within the Ollama
    container, this function handles two scenarios:
    1. If OLLAMA_HOST points to localhost, the adapter path must be accessible
       from the host filesystem (or via Docker volume mount).
    2. If OLLAMA_HOST is remote, the adapter files need to be copied to the
       remote host first (handled by the caller or docker-compose volume mounts).

    The Modelfile is written to a temporary location and passed to
    `ollama create`, which reads it and registers the model.
    """
    # Write Modelfile to a temporary location
    modelfile_path = Path("/tmp") / f"Modelfile_{name.replace(':', '_')}"
    modelfile_path.write_text(modelfile_content, encoding="utf-8")

    try:
        ollama_url = f"http://{ollama_host}:{ollama_port}"

        # Use curl to call Ollama's Create API instead of the `ollama` CLI,
        # which may not be available on the host. The Create API accepts
        # the Modelfile content via POST /api/create.
        import urllib.request
        import urllib.error

        # Read the Modelfile and send it to Ollama's create API
        # The /api/create endpoint expects a stream with the Modelfile content.
        # We use the `modelfile` form field.
        boundary = f"----Boundary_{name}"
        body_parts = []
        # modelfile field
        body_parts.append(f"--{boundary}")
        body_parts.append('Content-Disposition: form-data; name="modelfile"')
        body_parts.append("")
        body_parts.append(modelfile_content)
        # stream field (set to false for simpler response handling)
        body_parts.append(f"--{boundary}")
        body_parts.append('Content-Disposition: form-data; name="stream"')
        body_parts.append("")
        body_parts.append("false")
        body_parts.append(f"--{boundary}--")
        body_parts.append("")

        body = "\r\n".join(body_parts).encode("utf-8")

        req = urllib.request.Request(
            f"{ollama_url}/api/create",
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )

        print(f"[create_lora_model] Creating model '{name}' on {ollama_url}", file=sys.stderr)

        with urllib.request.urlopen(req, timeout=600) as response:
            response.read()

        print(f"[create_lora_model] Model '{name}' created successfully", file=sys.stderr)

    except urllib.error.URLError as e:
        print(
            f"[create_lora_model] ERROR: Failed to create model: {e}\n"
            f"  Ensure Ollama is running at {ollama_url} and the adapter path\n"
            f"  '{adapter_path}' is accessible from the Ollama container.",
            file=sys.stderr,
        )
        sys.exit(1)
    finally:
        modelfile_path.unlink(missing_ok=True)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Create an Ollama model from a LoRA adapter"
    )
    parser.add_argument(
        "--base",
        required=True,
        help="Base model name (e.g., schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m)",
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help="Path to LoRA adapter directory (output from train_domain_lora.py)",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Name for the new Ollama model (e.g., expert-mesh-medical-lora)",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="System prompt to embed in the Modelfile (optional)",
    )
    parser.add_argument(
        "--ollama-host",
        default="localhost",
        help="Ollama host (default: localhost)",
    )
    parser.add_argument(
        "--ollama-port",
        type=int,
        default=11434,
        help="Ollama port (default: 11434)",
    )
    args = parser.parse_args()

    # Validate adapter path exists
    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        print(
            f"[create_lora_model] ERROR: Adapter path does not exist: {adapter_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Check for adapter files
    adapter_files = list(adapter_path.glob("*.safetensors"))
    if not adapter_files:
        # Also check for .bin files (older PEFT format)
        adapter_files = list(adapter_path.glob("*.bin"))
    if not adapter_files:
        print(
            f"[create_lora_model] WARNING: No adapter weight files found in {adapter_path}\n"
            f"  Expected *.safetensors or *.bin files.",
            file=sys.stderr,
        )

    # Generate and create model
    modelfile = generate_modelfile(args.base, str(adapter_path), args.system_prompt)
    print(f"[create_lora_model] Generated Modelfile:\n{modelfile}", file=sys.stderr)

    create_model(
        name=args.name,
        modelfile_content=modelfile,
        ollama_host=args.ollama_host,
        ollama_port=args.ollama_port,
        adapter_path=str(adapter_path),
    )


if __name__ == "__main__":
    main()
