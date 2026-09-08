#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "usage: $0 MODEL_PATH SERVED_NAME [extra vllm args...]" >&2
  exit 2
fi
model_path="$1"; served_name="$2"; shift 2
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_executable="${VLLM_PYTHON:-${repo_root}/.venv-vllm/bin/python}"
if [[ ! -x "$python_executable" ]]; then
  echo "vLLM Python is not executable: $python_executable" >&2
  exit 2
fi
# FlashInfer invokes Ninja by executable name during first-start JIT warmup.
# Prepending the same environment's bin directory keeps both Python imports and
# JIT build tools inside the frozen vLLM environment.
export PATH="$(dirname "$python_executable"):${PATH}"
exec "$python_executable" -m vllm.entrypoints.openai.api_server \
  --model "$model_path" \
  --served-model-name "$served_name" \
  --dtype half \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.85 \
  --port 8000 \
  "$@"
