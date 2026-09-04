#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "usage: $0 MODEL_PATH SERVED_NAME [extra vllm args...]" >&2
  exit 2
fi
model_path="$1"; served_name="$2"; shift 2
exec python -m vllm.entrypoints.openai.api_server \
  --model "$model_path" \
  --served-model-name "$served_name" \
  --dtype half \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.85 \
  --port 8000 \
  "$@"
