#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
from collections import defaultdict

from grade import BOOTSTRAP_RESAMPLES, grade_run, run_folder

HEADLINE = ["record_core", "record_all", "reward", "amount", "merchant", "category", "tags", "date", "instrument",
            "answered"]


def load(name):
    folder = run_folder(name)
    if not os.path.exists(os.path.join(folder, "meta.json")):
        raise SystemExit(f"no run at {folder}")
    summary, rows = grade_run(folder)
    return os.path.basename(os.path.normpath(folder)), summary["meta"], summary["process"], {r["id"]: r for r in rows}


def paired_interval(base, other, field, seed=0):
    rng = random.Random(seed)
    diffs = [b[field] - a[field] for a, b in zip(base, other)]
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAP_RESAMPLES))
    return sum(diffs) / n, means[int(0.025 * BOOTSTRAP_RESAMPLES)], means[int(0.975 * BOOTSTRAP_RESAMPLES) - 1]


def sign_test(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def mean(rows, field):
    return sum(r[field] for r in rows) / len(rows) if rows else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run names or folders; the first is the baseline")
    ap.add_argument("--metric", default="record_core")
    ap.add_argument("--min-n", type=int, default=5, help="hide breakdown rows with fewer cases")
    args = ap.parse_args()

    runs = [load(name) for name in args.runs]
    base_label, base_meta, _, _ = runs[0]
    for label, meta, _, _ in runs[1:]:
        if meta["suite_fingerprint"] != base_meta["suite_fingerprint"]:
            raise SystemExit(f"{label} ran on suite {meta['suite_fingerprint']}, {base_label} on "
                             f"{base_meta['suite_fingerprint']}: not comparable")
        if meta["contract"] != base_meta["contract"]:
            print(f"warning: {label} ran on contract {meta['contract']}, {base_label} on {base_meta['contract']} — "
                  "the instructions, schema or tools changed too")
        elif meta.get("agent_code") != base_meta.get("agent_code"):
            print(f"note: {label} ran on agent code {meta.get('agent_code')}, {base_label} on "
                  f"{base_meta.get('agent_code')} — check whether a tool's answers changed")
    shared = set.intersection(*(set(rows) for *_, rows in runs))
    if not shared:
        raise SystemExit("the runs share no finished cases")
    order = sorted(shared)
    per_run = [[rows[i] for i in order] for *_, rows in runs]
    labels = [label for label, *_ in runs]
    width = max(12, *(len(l) for l in labels)) + 2

    print(f"\n{len(shared)} shared cases (suite {base_meta['suite']} {base_meta['suite_fingerprint']})\n")
    print(f"{'run':{width}}" + "".join(f"{m[:11]:>12}" for m in HEADLINE) + f"{'s/case':>9}{'tools':>7}")
    for (label, _, process, _), rows in zip(runs, per_run):
        print(f"{label:{width}}" + "".join(f"{mean(rows, m):>12.3f}" for m in HEADLINE)
              + f"{process.get('seconds', 0):>9.1f}{process.get('tool_calls', 0):>7.2f}")
    for label, rows in zip(labels[1:], per_run[1:]):
        cells = []
        for m in HEADLINE:
            delta, lo, hi = paired_interval(per_run[0], rows, m)
            mark = "*" if lo > 0 or hi < 0 else " "
            cells.append(f"{delta:>+11.3f}{mark}")
        print(f"{'Δ ' + label:{width}}" + "".join(cells))
    for label, rows in zip(labels[1:], per_run[1:]):
        delta, lo, hi = paired_interval(per_run[0], rows, args.metric)
        print(f"  {label} − {labels[0]} on {args.metric}: {delta:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}]")
    print("  (* = the 95% paired-bootstrap interval excludes zero)")

    print(f"\n{args.metric} by group (rows with n < {args.min_n} hidden)\n")
    print(f"{'':28}{'n':>5}" + "".join(f"{l[:width - 2]:>{width}}" for l in labels) + f"{'Δ':>8}")
    grouped = []
    for rows in per_run:
        groups = defaultdict(list)
        for r in rows:
            groups[("language", r["language"])].append(r)
            groups[("style", r["style"])].append(r)
            for trait in r["traits"]:
                groups[("trait", trait)].append(r)
        grouped.append(groups)
    for kind in ("language", "style", "trait"):
        print(f"-- {kind}")
        for key in sorted(k for k in grouped[0] if k[0] == kind):
            n = len(grouped[0][key])
            if n < args.min_n:
                continue
            values = [mean(g[key], args.metric) for g in grouped]
            print(f"{key[1][:28]:28}{n:>5}" + "".join(f"{v:>{width}.2f}" for v in values)
                  + f"{values[-1] - values[0]:>+8.2f}")

    first, last = per_run[0], per_run[-1]
    fixed = [a["id"] for a, b in zip(first, last) if not a[args.metric] and b[args.metric]]
    broke = [a["id"] for a, b in zip(first, last) if a[args.metric] and not b[args.metric]]
    p = sign_test(len(fixed), len(broke))
    verdict = "a real difference" if p < 0.05 else "suggestive, not conclusive" if p < 0.2 else "within noise"
    print(f"\n{labels[0]} → {labels[-1]} on {args.metric}: {len(fixed)} fixed, {len(broke)} broken, "
          f"{len(shared) - len(fixed) - len(broke)} unchanged; sign test p = {p:.3f} — {verdict}")
    for name, ids in (("fixed", fixed), ("broken", broke)):
        if ids:
            print(f"  {name}: {', '.join(ids[:20])}{' …' if len(ids) > 20 else ''}")


if __name__ == "__main__":
    main()
