"""Verify + benchmark DeepSeek-V4-Flash 2-bit: coherence (correct answer, stops on EOS) and
single-stream decode tok/s. Use this to confirm a fresh install works and reproduces ~41 tok/s.

  python scripts/bench.py                       # coherence + decode tok/s (MTP K=3 + FR-Spec)
  python scripts/bench.py --gen 512 --spec-mtp 3
  python scripts/bench.py --no-frspec --spec-mtp 0   # base (non-spec) ~18-20 tok/s

Decode tok/s uses the diff method: time(gen=N+1) - time(gen=1) over N steps, isolating decode
from prefill. Expected on GB10: ~41 tok/s with MTP K=3 + FR-Spec; ~18-20 without spec.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _common  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.environ.get("MODEL", "models/DeepSeek-V4-Flash-2bit"))
ap.add_argument("--max-model-len", type=int, default=int(os.environ.get("MAX_MODEL_LEN", "4096")))
ap.add_argument("--spec-mtp", default=os.environ.get("SPEC_MTP", "3"))
ap.add_argument("--gen", type=int, default=int(os.environ.get("GEN", "256")), help="decode steps to time")
ap.add_argument("--no-frspec", action="store_true")
args = ap.parse_args()

frspec = None if args.no_frspec else os.path.join(args.model, "frspec_nvfp4_ds4.pt")  # ships inside the model
frspec = frspec if (frspec and os.path.exists(frspec)) else None
_common.setup_env(args.model, frspec=frspec)
spec = int(args.spec_mtp) if str(args.spec_mtp).strip() not in ("", "0") else None
print(f"[bench] model={args.model} MTP={spec} FR-Spec={'on' if frspec else 'off'}", flush=True)

from vllm import SamplingParams  # noqa: E402

llm = _common.build_llm(args.model, args.max_model_len, spec_mtp=spec)

# 1) coherence: correct, formatted, stops on EOS (finish_reason=stop)
sp_c = SamplingParams(max_tokens=64, temperature=0.0)
o = llm.chat([[{"role": "user", "content": "What is the capital of France? Then write a haiku about it."}]],
             sp_c, use_tqdm=False)
print("[coherence] finish_reason=", o[0].outputs[0].finish_reason, flush=True)
print("[coherence]", repr(o[0].outputs[0].text), flush=True)

# 2) decode tok/s via the diff method (isolates decode from prefill)
prompt = "Write a long detailed essay about the history of computing. " * 6


def run(n):
    sp = SamplingParams(max_tokens=n, temperature=0.0, ignore_eos=True)
    t0 = time.perf_counter()
    llm.generate([prompt], sp, use_tqdm=False)
    return time.perf_counter() - t0


run(8)                                   # warm CUDA graphs / autotune
t1 = run(1)                              # prefill + 1 step
tN = run(args.gen + 1)                   # prefill + (gen+1) steps
dec = tN - t1
print(f"[bench] decode {args.gen} steps in {dec:.2f}s = {args.gen / dec:.2f} tok/s "
      f"({dec / args.gen * 1000:.2f} ms/tok)", flush=True)
