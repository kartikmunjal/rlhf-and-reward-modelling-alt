#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$PWD}"
cd "$ROOT"
python3 -m venv .venv-serving
.venv-serving/bin/python -m pip install --upgrade pip
.venv-serving/bin/pip install -r inference_serving/requirements-wsl.txt
.venv-serving/bin/python -m pip freeze > inference_serving/environment-lock.runtime.txt
nvidia-smi
