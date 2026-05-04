#!/bin/bash
# Per-node setup script — runs inside the Docker container.
set -euo pipefail

# Install miles
cd /workspace/miles
pip install -e .

# Skip download if model + dataset already present (e.g. shared-storage reuse).
if [ ! -d /root/Qwen3-30B-A3B ] || [ -z "$(ls -A /root/Qwen3-30B-A3B 2>/dev/null)" ]; then
  hf download Qwen/Qwen3-30B-A3B --local-dir /root/Qwen3-30B-A3B
else
  echo "[$(hostname)] /root/Qwen3-30B-A3B already populated; skipping download"
fi

if [ ! -d /root/dapo-math-17k ] || [ -z "$(ls -A /root/dapo-math-17k 2>/dev/null)" ]; then
  hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
fi

if [ ! -d /root/aime-2024 ] || [ -z "$(ls -A /root/aime-2024 2>/dev/null)" ]; then
  hf download --repo-type dataset zhuzilin/aime-2024 --local-dir /root/aime-2024
fi

# Convert checkpoint (skip if torch_dist already exists)
if [ ! -f /root/Qwen3-30B-A3B_torch_dist/latest_checkpointed_iteration.txt ]; then
  export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  source /workspace/miles/scripts/models/qwen3-30B-A3B.sh

  MEGATRON_LM_PATH=$(python3 -c \
    "import megatron, os; print(os.path.dirname(os.path.dirname(megatron.__file__)))" \
    2>/dev/null || echo "/app/Megatron-LM")

  PYTHONPATH="${MEGATRON_LM_PATH}" torchrun --nproc-per-node 8 \
    tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --no-gradient-accumulation-fusion \
    --hf-checkpoint /root/Qwen3-30B-A3B \
    --save /root/Qwen3-30B-A3B_torch_dist
else
  echo "[$(hostname)] /root/Qwen3-30B-A3B_torch_dist already exists; skipping conversion"
fi

echo "=== Setup complete on $(hostname) ==="
