#!/bin/bash
set -euo pipefail
SMOKE_WORKDIR=/root/autodl-tmp/a-agent-smoke
source "$SMOKE_WORKDIR/venv/bin/activate"
export HF_HOME="$SMOKE_WORKDIR/hf-cache"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export XDG_CACHE_HOME="$SMOKE_WORKDIR/cache"
export VLLM_CACHE_ROOT="$SMOKE_WORKDIR/vllm-cache"
export TORCHINDUCTOR_CACHE_DIR="$SMOKE_WORKDIR/torch-cache"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
SMOKE_MODEL_PATH=$(cat "$SMOKE_WORKDIR/model_path.txt")
exec vllm serve "$SMOKE_MODEL_PATH" \
  --served-model-name qwen35-9b \
  --dtype bfloat16 \
  --language-model-only \
  --max-model-len 172032 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.90 \
  --offload-backend prefetch \
  --offload-group-size 4 \
  --offload-num-in-group 1 \
  --offload-prefetch-step 1 \
  --enforce-eager \
  --generation-config vllm \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --host 127.0.0.1 --port 8000
