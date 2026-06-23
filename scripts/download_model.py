"""Download the DeepSeek-V4-Flash 2-bit model (~101 GB) from Hugging Face.

    python scripts/download_model.py                      # -> models/DeepSeek-V4-Flash-2bit
    python scripts/download_model.py --local-dir /data/ds4-2bit
"""
import argparse
from huggingface_hub import snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="iggerask/DeepSeek-V4-Flash-2bit-GB10")
ap.add_argument("--local-dir", default="models/DeepSeek-V4-Flash-2bit")
args = ap.parse_args()

print(f"Downloading {args.repo} -> {args.local_dir}  (~101 GB; ensure free disk)...", flush=True)
path = snapshot_download(repo_id=args.repo, local_dir=args.local_dir)
print(f"Done: {path}")
print("Next: python scripts/bench.py --model", args.local_dir)
