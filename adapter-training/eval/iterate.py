#!/usr/bin/env python3
"""Train → evaluate → repeat until the held-out score reaches a target.

Each round builds more SFT trajectories from the train split (which never
shares users, ledgers or invented instrument names with the eval split),
continues training the LoRA adapter, runs the full eval set the way the app
runs a local model, grades it, and stops as soon as `record_core` (spending
verdict + amount + currency + merchant + category all exactly right) reaches
`--target`. Every round's adapter, predictions, transcripts and grade are
kept, and `rounds.jsonl` records the trajectory.

    python3 iterate.py --model models/qwen35-08b --out work --target 0.95

Designed for a machine with a GPU (Apple Silicon `mps` or CUDA): each round
is minutes there and hours on a small CPU.
"""

import argparse
import json
import os
import subprocess
import sys

from common import AGENT_EVAL

HERE = os.path.dirname(os.path.abspath(__file__))


def sh(*args):
    print("$", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=HERE, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--start-trajectories", type=int, default=600)
    ap.add_argument("--growth", type=float, default=1.6, help="trajectories multiply by this each round")
    ap.add_argument("--eval-limit", type=int, default=280)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    train = os.path.join(HERE, "data", "train.jsonl")
    if not os.path.exists(train):
        sh("make_eval_set.py", "--split", "train", "--per-locale", "1200", "--users-per-locale", "8", "--seed", "1")
    total = sum(1 for _ in open(train))

    adapter = None
    log = open(os.path.join(out, "rounds.jsonl"), "a")
    for r in range(1, args.rounds + 1):
        n = min(total, int(args.start_trajectories * args.growth ** (r - 1)))
        new_adapter = os.path.join(out, f"adapter_r{r}")
        name = f"iter_r{r}"
        grade_path = os.path.join(HERE, "runs", name, "grade.json")
        if os.path.exists(os.path.join(new_adapter, "adapter_model.safetensors")) and os.path.exists(grade_path):
            adapter = new_adapter
            overall = json.load(open(grade_path))["summary"]["overall"]
            print(f"round {r} already done: record_core {overall['record_core']:.3f}", flush=True)
            if overall["record_core"] >= args.target:
                print(f"target {args.target} reached in round {r}: {adapter}")
                return
            continue
        data = os.path.join(out, f"sft_r{r}.jsonl")
        sh("sft_data.py", "--model", args.model, "--gold", train, "--out", data, "--limit", str(n))
        cmd = [os.path.join(AGENT_EVAL, "train_sft.py"), "--model", args.model, "--data", data, "--out", new_adapter,
               "--accum", "8", "--lr", "2e-4" if adapter is None else "1e-4", "--end-tools-fraction", "0.3"]
        if adapter:
            cmd += ["--init-adapter", adapter]
        sh(*cmd)
        adapter = new_adapter
        sh("run.py", "--backend", "hf", "--model", args.model, "--adapter", adapter, "--gold", "data/eval.jsonl",
           "--name", name, "--limit", str(args.eval_limit), "--batch", str(args.batch))
        sh("grade.py", "--gold", "data/eval.jsonl", "--pred", f"runs/{name}/predictions.jsonl", "--out", grade_path)
        overall = json.load(open(grade_path))["summary"]["overall"]
        entry = {"round": r, "trajectories": n, "adapter": adapter,
                 **{k: overall[k] for k in ("record_core", "record_all", "amount", "currency", "merchant",
                                            "category", "json_valid", "mean_tool_calls")}}
        log.write(json.dumps(entry) + "\n")
        log.flush()
        print(json.dumps(entry), flush=True)
        if overall["record_core"] >= args.target:
            print(f"target {args.target} reached in round {r}: {adapter}")
            return
    print("rounds exhausted below target — see rounds.jsonl and the per-trait breakdowns in runs/iter_r*/grade.json")


if __name__ == "__main__":
    main()
