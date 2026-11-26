#!/usr/bin/env python3
import argparse
import json
import math
import os
from collections import defaultdict

import grade
from common import read_jsonl

HERE = os.path.dirname(os.path.abspath(__file__))
HEADLINE = ["record_core", "record_all", "is_spending", "amount", "currency", "merchant", "category",
            "instrument", "date", "json_valid"]


def load(arg, gold):
    path = arg
    if not os.path.exists(path):
        path = os.path.join(HERE, "runs", arg)
    if os.path.isdir(path):
        pred = os.path.join(path, "predictions.jsonl")
        path = pred if os.path.exists(pred) else os.path.join(path, "grade.json")
    if not os.path.exists(path):
        raise SystemExit(f"no run found for {arg!r} (looked for runs/{arg}/predictions.jsonl and grade.json)")
    label = os.path.basename(os.path.dirname(os.path.abspath(path))) or arg
    if path.endswith("grade.json"):
        rows = json.load(open(path, encoding="utf-8"))["per_sample"]
    else:
        _, rows = grade.grade(gold, read_jsonl(path))
    return label, {r["id"]: r for r in rows}


def mean(rows, field):
    return sum(r[field] for r in rows) / len(rows) if rows else float("nan")


def groups(rows):
    out = defaultdict(list)
    for r in rows:
        out[("language", r["language"])].append(r)
        out[("style", r["style"])].append(r)
        for t in r["traits"]:
            out[("trait", t)].append(r)
    return out


def sign_test(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="first = baseline")
    ap.add_argument("--gold", default=os.path.join(HERE, "data", "eval.jsonl"))
    ap.add_argument("--metric", default="record_core")
    ap.add_argument("--min-n", type=int, default=3, help="hide breakdown rows with fewer samples")
    args = ap.parse_args()

    gold = read_jsonl(args.gold)
    runs = [load(a, gold) for a in args.runs]
    labels = [l for l, _ in runs]
    if len(set(labels)) < len(labels):
        labels = [f"{i}:{l}" for i, l in enumerate(labels)]
    shared = set.intersection(*(set(rows) for _, rows in runs))
    if not shared:
        raise SystemExit("the runs share no samples")
    order = [g["id"] for g in gold if g["id"] in shared]
    per_run = [[rows[i] for i in order] for _, rows in runs]
    for label, (_, rows) in zip(labels, runs):
        if len(rows) != len(shared):
            print(f"note: {label} has {len(rows)} samples; comparing on the {len(shared)} all runs share")
    w = max(10, *(len(l) for l in labels)) + 1

    print(f"\n{len(shared)} shared samples\n")
    print(f"{'run':{w}}" + "".join(f"{m[:11]:>12}" for m in HEADLINE))
    for label, rows in zip(labels, per_run):
        print(f"{label:{w}}" + "".join(f"{mean(rows, m):>12.2f}" for m in HEADLINE))
    for label, rows in zip(labels[1:], per_run[1:]):
        print(f"{'Δ ' + label:{w}}" + "".join(f"{mean(rows, m) - mean(per_run[0], m):>+12.2f}" for m in HEADLINE))

    print(f"\n{args.metric} by group (Δ = last run − first run; rows with n < {args.min_n} hidden)\n")
    print(f"{'':28}{'n':>4}" + "".join(f"{l[:w - 1]:>{w}}" for l in labels) + f"{'Δ':>8}")
    all_groups = [groups(rows) for rows in per_run]
    for kind in ("language", "style", "trait"):
        print(f"-- {kind}")
        for key in sorted(k for k in all_groups[0] if k[0] == kind):
            n = len(all_groups[0][key])
            if n < args.min_n:
                continue
            vals = [mean(g[key], args.metric) for g in all_groups]
            print(f"{key[1][:28]:28}{n:>4}" + "".join(f"{v:>{w}.2f}" for v in vals) + f"{vals[-1] - vals[0]:>+8.2f}")

    first, last = per_run[0], per_run[-1]
    fixed = [a["id"] for a, b in zip(first, last) if not a[args.metric] and b[args.metric]]
    broke = [a["id"] for a, b in zip(first, last) if a[args.metric] and not b[args.metric]]
    p = sign_test(len(fixed), len(broke))
    print(f"\n{labels[0]} → {labels[-1]} on {args.metric}: {len(fixed)} samples fixed, {len(broke)} broken, "
          f"{len(shared) - len(fixed) - len(broke)} unchanged")
    verdict = ("a real difference" if p < 0.05 else
               "suggestive, not conclusive" if p < 0.2 else "within noise")
    print(f"sign test p = {p:.3f} — {verdict}")
    for name, ids in (("fixed", fixed), ("broken", broke)):
        if ids:
            print(f"  {name}: {', '.join(ids[:15])}{' …' if len(ids) > 15 else ''}")


if __name__ == "__main__":
    main()
