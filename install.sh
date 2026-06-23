#!/usr/bin/env bash
# Install the serving environment on a fresh DGX Spark (GB10, sm_121, CUDA 13, py3.12, aarch64).
#
# Reproduces the exact .venv-vllm used to develop this: stock vLLM 0.23.0 + the cu130 PyTorch
# stack + flashinfer/triton/tilelang. The vq2 plugin auto-registers via a vllm.general_plugins
# entry point (no vLLM fork). All versions are pinned in requirements-serve.lock.txt.
#
# Prereqs (NOT installed here): NVIDIA driver for GB10, CUDA 13 toolkit at /usr/local/cuda
# (nvcc, for the vq2_grouped.cu JIT), `uv` (https://docs.astral.sh/uv/), ~110 GB free disk.
#
# NOTE: the index URLs below (PyTorch cu130 + PyPI) are the expected sources; if a wheel fails
# to resolve on your box, that is the thing to debug first — the pins themselves are exact.
set -euo pipefail
cd "$(dirname "$0")"

VENV="${VENV:-.venv-vllm}"
CU130="https://download.pytorch.org/whl/cu130"

echo "== [1/4] uv venv ($VENV, python 3.12) =="
uv venv --python 3.12 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "== [2/4] PyTorch cu130 stack (the sm_121-capable builds) =="
uv pip install --index-url "$CU130" torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0

echo "== [3/4] pinned serving deps (vllm 0.23.0, flashinfer, triton, tilelang, cu13 stack) =="
grep -ivE '^(torch|torchvision|torchaudio|vq2-vllm)==' requirements-serve.lock.txt > /tmp/req-rest.txt
uv pip install --extra-index-url "$CU130" -r /tmp/req-rest.txt

echo "== [4/4] vq2 plugin (auto-registers in vLLM) =="
uv pip install -e .

echo ""
echo "== done. Verify the plugin loads: =="
echo "   $VENV/bin/python -c \"import vllm, vq2_vllm; print('vq2 plugin OK')\""
echo "== then download the model and serve (see README.md):"
echo "   $VENV/bin/python scripts/download_model.py"
echo "   $VENV/bin/python scripts/bench.py   # coherence + tok/s"
