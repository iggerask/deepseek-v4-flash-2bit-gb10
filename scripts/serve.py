"""Serve DeepSeek-V4-Flash 2-bit interactively (proves the stack runs end-to-end).

  python scripts/serve.py                                  # interactive chat REPL (MTP K=3 + FR-Spec)
  python scripts/serve.py --prompt "What is the capital of France?"
  python scripts/serve.py --model /data/ds4-2bit --spec-mtp 3

For an OpenAI-compatible API server instead, see README.md ("API server"); this offline
entry uses the exact verified recipe and is the quickest way to confirm a fresh install works.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _common  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.environ.get("MODEL", "models/DeepSeek-V4-Flash-2bit"))
ap.add_argument("--max-model-len", type=int, default=int(os.environ.get("MAX_MODEL_LEN", "4096")))
ap.add_argument("--spec-mtp", default=os.environ.get("SPEC_MTP", "3"), help="MTP draft length K; '' or 0 disables")
ap.add_argument("--frspec", default=os.environ.get("VLLM_FRSPEC_NVFP4"),
                help="reduced FR-Spec draft lm_head; default <model>/frspec_nvfp4_ds4.pt (ships with the model)")
ap.add_argument("--max-tokens", type=int, default=512)
ap.add_argument("--prompt", default=None, help="one-shot prompt; omit for an interactive REPL")
args = ap.parse_args()

frspec = args.frspec or os.path.join(args.model, "frspec_nvfp4_ds4.pt")  # ships inside the model repo
frspec = frspec if os.path.exists(frspec) else None
_common.setup_env(args.model, frspec=frspec)
spec = int(args.spec_mtp) if str(args.spec_mtp).strip() not in ("", "0") else None
print(f"[serve] model={args.model} MTP={spec} FR-Spec={'on' if frspec else 'off'} "
      f"max_model_len={args.max_model_len}", flush=True)

from vllm import SamplingParams  # noqa: E402

llm = _common.build_llm(args.model, args.max_model_len, spec_mtp=spec)
sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)


def ask(msg):
    out = llm.chat([[{"role": "user", "content": msg}]], sp, use_tqdm=False)
    return out[0].outputs[0].text


if args.prompt:
    print(ask(args.prompt))
else:
    print("Interactive chat — Ctrl-D / empty line + Ctrl-D to exit.")
    while True:
        try:
            msg = input("\nyou> ")
        except EOFError:
            print()
            break
        if msg.strip():
            print(ask(msg))
