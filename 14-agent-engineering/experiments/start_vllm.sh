#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export VLLM_USE_FLASHINFER_SAMPLER=0

exec "${REPO_ROOT}/.venv/bin/vllm" serve Qwen/Qwen3-1.7B \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype auto \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.70 \
  --generation-config vllm \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --enforce-eager
