#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$PWD}"
cd "$ROOT"
python3 -m venv .venv-vllm
.venv-vllm/bin/python -m pip install --upgrade pip
.venv-vllm/bin/pip install -r inference_serving/requirements-vllm.txt
.venv-vllm/bin/python -m pip freeze > inference_serving/environment-lock.vllm.runtime.txt
python3 -m venv .venv-gptq
.venv-gptq/bin/python -m pip install --upgrade pip
.venv-gptq/bin/pip install -r inference_serving/requirements-gptq.txt
.venv-gptq/bin/python -m pip freeze > inference_serving/environment-lock.gptq.runtime.txt
.venv-vllm/bin/python -c "import torch, vllm; print('vLLM', vllm.__version__, 'Torch', torch.__version__, 'CUDA', torch.version.cuda)"
.venv-gptq/bin/python -c "import gptqmodel; print('GPTQModel', gptqmodel.__version__)"
nvidia-smi
