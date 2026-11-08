#!/usr/bin/env bash
# llama-server for Siege-Ladder on one DGX Spark. The PC's ladder (16 GB VRAM slots + 43 GiB pinned RAM tier + NVMe)
# collapses here: VRAM and RAM are the same LPDDR5x, so almost all memory goes to device expert slots and the RAM
# tier is kept small (it is only a landing area for the lens' read-ahead and dropped experts).
#   MODEL=/data/v41-cd/DeepSeek-V4.1-Flash-MXFP4-engram-00001-of-00011.gguf ./run_server.sh
set -euo pipefail
BIN="${BIN:-$HOME/siege-ladder/llama.cpp/build/bin}"
MODEL="${MODEL:?set MODEL to part 1 of the 11 GGUF parts}"
SLOTS="${SLOTS:-230}"   # expert slots per layer: ~0.41 GiB each across 40 layers (cd); 230 -> ~94 GiB
L2="${L2:-6}"           # pinned host tier, GiB
CTX="${CTX:-32768}"
export LLAMA_MOE_STREAM_PRESET="${LLAMA_MOE_STREAM_PRESET:-default}"
# optional: a copy of expert parts 5-11 on an external USB4/Thunderbolt NVMe
# export LLAMA_MOE_STREAM_MIRROR=/mnt/usb-nvme/v41-cd
exec "$BIN/llama-server" -m "$MODEL" -ngl 99 -t "${THREADS:-10}" --parallel 1 --reasoning off \
  --moe-stream-direct --moe-stream-cache "${SLOTS}s" --moe-stream-l2 "$L2" \
  -c "$CTX" -b "$CTX" -ub "${UB:-8192}" -ctk q8_0 -ctv q8_0 --host 127.0.0.1 --port "${PORT:-8080}"
