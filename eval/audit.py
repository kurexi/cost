#!/usr/bin/env python3
import argparse
import json
import os
from collections import Counter

from grade import ALL
from suite import RUNS, load_suite, read_jsonl

from spending_agent.money import decimal_string

SHOWN = {
    "is_spending": (lambda g: g["isSpending"], lambda d: d["is_spending"]),
    "amount": (lambda g: g["totalAmount"], lambda d: "(none)" if d["amount_minor"] is None
               else decimal_string(d["amount_minor"], d["currency"] or "USD")),
    "currency": (lambda g: g["currencyCode"], lambda d: d["currency"]),
    "merchant": (lambda g: " | ".join(g["merchantAccepted"]), lambda d: d["merchant"]),
    "category": (lambda g: " | ".join(g["categoryAccepted"]) or f"(proposal) {g['proposedCategory']}",
                 lambda d: d["category"] or f"(proposal) {d['proposed_category']}"),
    "date": (lambda g: g["purchaseDate"], lambda d: d["date"]),
    "time": (lambda g: g["purchaseTime"], lambda d: d["time"]),
    "instrument": (lambda g: " | ".join(g["paymentInstrumentAccepted"]) or "(none)", lambda d: d["payment_instrument"]),
    "tags": (lambda g: ", ".join(g["tags"]) or "(none)", lambda d: ", ".join(d["tags"]) or "(none)"),
    "proposed_tag": (lambda g: g["proposedTagMode"], lambda d: d["proposed_tag"] or "(none)"),
}


def load_run(name):
    folder = os.path.join(RUNS, name)
    with open(os.path.join(folder, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    path = os.path.join(folder, "grade.json")
    if not os.path.exists(path):
        raise SystemExit(f"{name} isn't graded: run grade.py {name} first")
    with open(path, encoding="utf-8") as f:
        per_case = {row["id"]: row for row in json.load(f)["per_case"]}
    results = {row["id"]: row for row in read_jsonl(os.path.join(folder, "results.jsonl"))}
    return meta, per_case, results


def wrong_fields(row):
    if not row.get("answered"):
        return ["no draft"]
    return [field for field in ALL if not row.get(field)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="runs on one suite; the first (the teacher, say) picks the cases shown")
    ap.add_argument("--limit", type=int, default=20, help="cases shown (0: only the counts)")
    ap.add_argument("--field", default="", help="only cases the first run gets wrong on this field")
    ap.add_argument("--characters", type=int, default=700, help="characters of content shown a case")
    args = ap.parse_args()

    loaded = [(name, *load_run(name)) for name in args.runs]
    suite = loaded[0][1]["suite"]
    for name, meta, _, _ in loaded:
        if meta["suite"] != suite or meta["suite_fingerprint"] != loaded[0][1]["suite_fingerprint"]:
            raise SystemExit(f"{name} is on {meta['suite']} {meta['suite_fingerprint']}, not on the first run's suite")
    _, cases, users = load_suite(suite)
    cases = [c for c in cases if all(c["id"] in per_case for _, _, per_case, _ in loaded)]
    print(f"{len(cases)} cases graded by every run (suite {suite})\n")

    first = loaded[0]
    misses = {name: Counter(f for c in cases for f in wrong_fields(per_case[c["id"]]))
              for name, _, per_case, _ in loaded}
    every = Counter(f for c in cases for f in set.intersection(*(set(wrong_fields(p[c["id"]])) for _, _, p, _ in loaded)))
    fields = sorted({f for m in misses.values() for f in m}, key=lambda f: -misses[first[0]][f])
    width = max(len(n) for n in args.runs) + 2
    print("cases wrong on each field (record_all fields)")
    print(" " * width + "".join(f"{f:>13}" for f in fields))
    for name, _, _, _ in loaded:
        print(f"{name:<{width}}" + "".join(f"{misses[name][f]:>13}" for f in fields))
    if len(loaded) > 1:
        print(f"{'all wrong':<{width}}" + "".join(f"{every[f]:>13}" for f in fields))
        print("\n'all wrong' is where every run misses the same field: read those cases first, the gold may be wrong.")

    shown = [c for c in cases if wrong_fields(first[2][c["id"]])
             and (not args.field or args.field in wrong_fields(first[2][c["id"]]))]
    shown.sort(key=lambda c: -sum(bool(wrong_fields(p[c["id"]])) for _, _, p, _ in loaded))
    for case in shown[:args.limit]:
        gold = case["gold"]
        print("\n" + "=" * 100)
        print(f"{case['id']}  user {case['user']}  today {case['today']}  {', '.join(case['traits'])}")
        print("-" * 100)
        print(case["content"][:args.characters] + (" …" if len(case["content"]) > args.characters else ""))
        print("-" * 100)
        wrong = sorted({f for _, _, p, _ in loaded for f in wrong_fields(p[case["id"]]) if f in SHOWN},
                       key=list(SHOWN).index)
        for field in wrong:
            print(f"  {field:<12} gold      {SHOWN[field][0](gold)}")
            for name, _, per_case, results in loaded:
                draft = (results.get(case["id"]) or {}).get("draft")
                mark = "✗" if field in wrong_fields(per_case[case["id"]]) else "✓"
                print(f"  {'':<12} {mark} {name:<{width}} {SHOWN[field][1](draft) if draft else '(no draft)'}")
        for name, _, per_case, results in loaded:
            if "no draft" in wrong_fields(per_case[case["id"]]):
                print(f"  {name}: no draft ({(results.get(case['id']) or {}).get('reason', '')})")
        known = [r for r in users[case["user"]]["records"]
                 if r["merchant"] == gold["merchant"] and r["date"] <= case["today"]]
        print(f"  ledger: {len(known)} earlier records at {gold['merchant']!r}"
              + (f", filed under {sorted({r['category'] for r in known})}" if known else ""))


if __name__ == "__main__":
    main()
