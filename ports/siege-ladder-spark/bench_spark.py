#!/usr/bin/env python3
"""Minimal Linux replacement for Siege-Ladder's (Windows-only) bench/harness.py: decode and prompt speeds from
llama-server's own timings. No dependencies.   python3 bench_spark.py [--url http://127.0.0.1:8080] [--n 256]"""
import argparse, json, statistics, urllib.request

PROMPTS = {
    "short": "Explain, step by step, how a B-tree stays balanced on insert and delete.",
    "2k": "Summarize the following text in five bullet points.\n\n" + ("The quick brown fox jumps over the lazy dog. "
          "Systems that stream weights from disk trade latency for capacity. ") * 110,
}

def post(url, body):
    req = urllib.request.Request(url + "/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)["timings"]

def main():
    p = argparse.ArgumentParser(); p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--n", type=int, default=256); p.add_argument("--reps", type=int, default=3); a = p.parse_args()
    post(a.url, {"prompt": "Hi", "n_predict": 8, "cache_prompt": False})  # warm-up
    for name, text in PROMPTS.items():
        dec, pp = [], []
        for _ in range(a.reps):
            t = post(a.url, {"prompt": text, "n_predict": a.n, "temperature": 0, "cache_prompt": False})
            dec.append(t["predicted_per_second"]); pp.append(t["prompt_per_second"])
        print(f"{name:6s} prompt {t['prompt_n']:5d} tok: {statistics.median(pp):7.1f} tok/s   "
              f"decode {statistics.median(dec):5.2f} tok/s (median of {a.reps}, {a.n} tokens)")

if __name__ == "__main__":
    main()
