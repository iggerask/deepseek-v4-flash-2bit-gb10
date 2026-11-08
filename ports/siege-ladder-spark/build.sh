#!/usr/bin/env bash
# Build Siege-Ladder's patched llama.cpp on a DGX Spark (GB10, sm_121, aarch64, DGX OS / Ubuntu 24.04).
# Usage: ./build.sh [workdir]   (default ~/siege-ladder)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
W="${1:-$HOME/siege-ladder}"
mkdir -p "$W" && cd "$W"

# CUDA 13.0 - Siege-Ladder found 13.2 miscompiles mul_mat_id for the IQ types on Blackwell (sm_120); assume sm_121 too
nvcc --version | grep -q "release 13.0" || echo "WARNING: nvcc is not CUDA 13.0 - see README (IQ mul_mat_id miscompile on 13.2)"

[ -d Siege-Ladder ] || git clone https://github.com/MasherDon/Siege-Ladder
[ -d llama.cpp ] || git clone https://github.com/JigSawPT/llama.cpp
cd llama.cpp
git checkout -q 8ce0968539b1a6795f3413c2783dfa5646b4688a
git switch -q -C siege-ladder-spark
git -c user.name=build -c user.email=build@localhost am -q ../Siege-Ladder/patches/*.patch "$HERE"/patches/*.patch

cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 \
      -DLLAMA_CURL=OFF -DGGML_NATIVE=ON
cmake --build build --target llama-server llama-cli llama-perplexity llama-logits -j "$(nproc)"
echo "built: $W/llama.cpp/build/bin/llama-server"
