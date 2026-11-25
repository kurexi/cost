#!/usr/bin/env python3
"""The adapter's multilingual receipt set: the agent's receipt generator
(`agent/eval/receipts.py`) with each sample's `prompt` assembled the way
`SpendingAgent.extractionPrompt` assembles it (via
make_training_data.assemble_prompt), so any model can be run on it with
`common.INSTRUCTIONS` as the system turn. Train and eval use different seeds,
so their users, ledgers and invented instrument names never overlap.

    python3 make_eval_set.py --split eval  --per-locale 40  --seed 7
    python3 make_eval_set.py --split train --per-locale 400 --seed 1

Stdlib only.
"""

import argparse
import os
import random
from collections import Counter
from datetime import date, timedelta

from common import INSTRUCTIONS, assemble_prompt, write_jsonl
from receipts import HOME_LOCALES, build_sample, category_lines, make_profile


def extraction_prompt(profile, today, past, preferences, content):
    return assemble_prompt(today, profile["localCurrency"], category_lines(profile), past, preferences, [], content)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, help="name used in the output files, e.g. eval or train")
    ap.add_argument("--per-locale", type=int, default=40, help="samples per home locale")
    ap.add_argument("--users-per-locale", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    profiles, samples = [], []
    for home in HOME_LOCALES:
        users = []
        for u in range(args.users_per_locale):
            today = date(2026, 1, 15) + timedelta(days=rng.randint(0, 600))
            users.append(make_profile(rng, home, f"{args.seed}{u}", today))
        profiles += users
        for i in range(args.per_locale):
            samples.append(build_sample(rng, users[i % len(users)], i, prompt=extraction_prompt))

    os.makedirs(args.out_dir, exist_ok=True)
    write_jsonl(os.path.join(args.out_dir, f"{args.split}.jsonl"), samples)
    write_jsonl(os.path.join(args.out_dir, f"{args.split}_profiles.jsonl"), profiles)
    with open(os.path.join(args.out_dir, "instructions.txt"), "w", encoding="utf-8") as f:
        f.write(INSTRUCTIONS)

    print(f"{len(samples)} samples, {len(profiles)} users → {args.out_dir}")
    print("languages:", dict(Counter(s["language"] for s in samples)))
    print("styles:   ", dict(Counter(s["style"] for s in samples)))
    print("traits:   ", dict(Counter(t for s in samples for t in s["traits"]).most_common()))


if __name__ == "__main__":
    main()
