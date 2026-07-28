#!/bin/bash
# Full LoRA training pipeline: train -> convert to GGUF -> register with Ollama
# Usage: bash run_lora_pipeline.sh <domain>
set -e

DOMAIN=$1
WORKDIR="$HOME/workspace/ktakahashi/expert-mesh"
export PATH="$HOME/.local/bin:$PATH"
export TRITON_CACHE_DIR="/tmp/triton_cache"
mkdir -p /tmp/triton_cache
PYTHON="$WORKDIR/.venv/bin/python"

echo "[pipeline] Starting pipeline for domain: $DOMAIN"

# Step 1: Stop Ollama to free VRAM
echo "[pipeline] Stopping Ollama..."
docker stop expert-mesh-ollama-1 2>/dev/null || true
sleep 2

# Step 2: Train LoRA adapter
echo "[pipeline] Training LoRA adapter for $DOMAIN..."
cd "$WORKDIR"
$PYTHON scripts/train_domain_lora.py \
  --model tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.1 \
  --data "data/lora_train/${DOMAIN}.jsonl" \
  --output "models/lora_adapters/${DOMAIN}/" \
  --lora-r 4 --lora-alpha 8 \
  --target-modules q_proj,k_proj \
  --epochs 3 --batch-size 2 \
  --max-seq-len 256 2>&1

echo "[pipeline] Training complete"

# Step 3: Convert to GGUF
echo "[pipeline] Converting to GGUF..."
$PYTHON /tmp/llama.cpp/convert_lora_to_gguf.py \
  "models/lora_adapters/${DOMAIN}/" \
  --outfile "models/lora_adapters/${DOMAIN}/adapter.gguf" 2>&1

echo "[pipeline] GGUF conversion complete"

# Step 4: Recreate containers with updated docker-compose
echo "[pipeline] Recreating containers..."
cd "$WORKDIR"
docker compose down 2>&1
sleep 2
docker compose up -d 2>&1
sleep 5

# Step 5: Copy GGUF to Ollama container
echo "[pipeline] Copying GGUF to Ollama container..."
docker exec expert-mesh-ollama-1 mkdir -p "/root/.ollama/adapters/${DOMAIN}" 2>&1
docker cp "models/lora_adapters/${DOMAIN}/adapter.gguf" \
  expert-mesh-ollama-1:"/root/.ollama/adapters/${DOMAIN}/adapter.gguf" 2>&1

# Step 6: Register model with Ollama
echo "[pipeline] Registering model with Ollama..."
MODEL_NAME="expert-mesh-${DOMAIN}-lora"
docker exec expert-mesh-ollama-1 bash -c "
echo \"FROM schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m
ADAPTER /root/.ollama/adapters/${DOMAIN}/adapter.gguf\" > /tmp/Mf_${DOMAIN}
ollama create ${MODEL_NAME} -f /tmp/Mf_${DOMAIN} 2>&1
"

# Step 7: Verify
echo "[pipeline] Verifying model registration..."
docker exec expert-mesh-ollama-1 ollama list 2>&1 | grep "${MODEL_NAME}" || echo "WARNING: Model not found in list"

echo "[pipeline] Pipeline complete for $DOMAIN"
