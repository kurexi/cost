#!/usr/bin/env python3
"""Grade a run of the spending agent against its suite's gold records.

    python3 grade.py bonsai-27b                 # runs/bonsai-27b/results.jsonl
    python3 grade.py runs/bonsai-27b --suite dev

What is graded is the draft the session settled — the row that reaches the
user's review queue — not the model's raw text, so the session's own fixes
(snapping a category onto the user's spelling, turning an unknown one into a
proposal) count the way they count in the app.

Per case (1 = right, 0 = wrong; all deterministic, no model in the loop):

- answered: the session produced a draft (not failed);
- status: the session ended as the suite expects (needs_review,
  not_spending, rejected_too_long);
- is_spending, amount (minor units), currency, time;
- date: the date the record gets, so when nothing gives one, an empty
  date and the arrival date (the case's today, which the app falls back
  to) are both right;
- merchant: folded comparison against every accepted spelling, containment
  either way, never equal to the payment instrument;
- instrument: exact where the content names one, empty where it shows
  none, free where it is cash or a bare card scheme;
- category: one of the accepted titles; where the user has no fitting
  category, an empty slot and a proposal (`proposal_wording` says whether
  the proposal is one of the reasonable titles). For goods with no
  category of their own (clothing, home, pets, kids), the user's broad
  Shopping category is also right, when they have one;
- tags: the user's tags the purchase belongs under, as a set. A trip tag on
  a purchase after the trip's last tagged record may be left off, since
  nothing in the content says whether the trip is still going;
- proposed_tag: empty unless the case allows one (a trip abroad with no tag
  yet), and never a restatement of an existing tag;
- note_items: the share of the gold items the note names.

`record_core` = is_spending, amount, currency, merchant, category.
`record_all` adds date, time, instrument, tags and proposed_tag.
`reward` is the scalar for RL: field-weighted, -1 for no draft, -0.03 per
wasted tool call, +0.05 for a fully right record. A draft the app would
approve on its own adds +0.05 when it is fully right and -0.45 when it
isn't, so a confidence of 0.8 or more pays only where the draft is right
at least 9 times in 10 (`APPROVAL_PRECISION`). Without the bonus the cheapest
policy would be to never sound confident; without the penalty, to always.

Safety: `false_approval` is the share of cases the app would have approved
on its own (a spending draft with an existing category and confidence >=
0.8) while the record was wrong — the number the audits must drive to zero.
`right_approval` is the share it would have approved and got right.

Content over the budget is rejected before any model call, the same way for
every model, so those cases are counted under `rejected_ok` and kept out of
the field scores.
"""

import argparse
import json
import os
import random
import unicodedata
from collections import Counter, defaultdict

from suite import RUNS, load_suite, read_jsonl
from locales import CATEGORY_TITLES

FIELDS = ["answered", "status", "is_spending", "amount", "currency", "merchant", "instrument", "category",
          "date", "time", "tags", "proposed_tag", "note_items", "record_core", "record_all"]
REWARD_WEIGHTS = {"is_spending": 0.10, "amount": 0.28, "currency": 0.09, "merchant": 0.14, "category": 0.14,
                  "instrument": 0.07, "date": 0.07, "time": 0.03, "tags": 0.06, "proposed_tag": 0.02}
WASTED_PER_CALL = 0.03
FULL_RECORD_BONUS = 0.05
NO_DRAFT = -1.0
AUTO_APPROVAL_CONFIDENCE = 0.8
APPROVAL_PRECISION = 0.9
RIGHT_APPROVAL_BONUS = 0.05
FALSE_APPROVAL_PENALTY = RIGHT_APPROVAL_BONUS * APPROVAL_PRECISION / (1 - APPROVAL_PRECISION)
BOOTSTRAP_RESAMPLES = 2000
CORE = ("is_spending", "amount", "currency", "merchant", "category")
ALL = CORE + ("date", "time", "instrument", "tags", "proposed_tag")


def fold(text):
    text = unicodedata.normalize("NFKC", text or "")
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return "".join(c for c in text.lower() if c.isalnum())


def names_match(name, accepted):
    folded = fold(name)
    if not folded:
        return False
    for candidate in accepted:
        other = fold(candidate)
        if other and (folded == other or (len(folded) >= 2 and len(other) >= 2
                                          and (folded in other or other in folded))):
            return True
    return False


GOODS_A_SHOPPING_CATEGORY_COVERS = ("clothing", "home", "pets", "kids")


def broad_category_alternative(gold, user):
    """The user's Shopping title, when the gold wants a new category for
    goods that a broad Shopping category also covers; else None."""
    titles = CATEGORY_TITLES.get(user["language"], {})
    kind = next((k for k, title in titles.items() if title == gold["proposedCategory"]), None)
    shopping = titles.get("shopping")
    if kind in GOODS_A_SHOPPING_CATEGORY_COVERS and shopping in user["categories"]:
        return shopping
    return None


def record_date(value, case):
    """The date the record gets: the one given, else the day it arrived."""
    return value or case["today"]


def optional_tags(case, user):
    """Trip tags on a purchase after the trip's last tagged record (up to
    the case's today): tagging and not tagging are both right."""
    day = record_date(case["gold"]["purchaseDate"], case)
    optional = set()
    for tag in case["gold"]["tags"]:
        if tag not in user["trips"].values():
            continue
        dates = [r["date"][:10] for r in user["records"] if tag in r["tags"] and r["date"][:10] <= case["today"]]
        if dates and day > max(dates):
            optional.add(fold(tag))
    return optional


def tags_right(draft_tags, case, user):
    given = {fold(t) for t in draft_tags}
    optional = optional_tags(case, user)
    required = {fold(t) for t in case["gold"]["tags"]} - optional
    return required <= given <= required | optional


def score_case(case, result, user):
    """Per-field scores for one case, or None for a case rejected by design."""
    gold, draft = case["gold"], result.get("draft")
    if case["expectedStatus"] == "rejected_too_long":
        return None
    s = {field: 0.0 for field in FIELDS}
    s["proposal_wording"] = None
    s["status"] = float(result["status"] == case["expectedStatus"])
    s["confidence"] = draft["confidence"] if draft else 0.0
    s["would_approve"] = bool(draft and draft["is_spending"] and draft["category"]
                              and draft["confidence"] >= AUTO_APPROVAL_CONFIDENCE)
    if not draft:
        return s
    s["answered"] = 1.0
    s["is_spending"] = float(draft["is_spending"] == gold["isSpending"])
    if not gold["isSpending"]:
        for field in FIELDS[3:]:
            s[field] = s["is_spending"]
        return s

    s["currency"] = float(draft["currency"] == gold["currencyCode"])
    s["amount"] = float(draft["amount_minor"] == gold["amountMinorUnits"])
    gold_instrument = gold.get("paymentInstrument", "")
    is_instrument = bool(gold_instrument) and fold(draft["merchant"]) == fold(gold_instrument)
    s["merchant"] = float(names_match(draft["merchant"], gold["merchantAccepted"]) and not is_instrument)
    mode = gold["paymentInstrumentMode"]
    if mode == "none":
        s["instrument"] = float(draft["payment_instrument"] == "")
    elif mode == "exact":
        s["instrument"] = float(names_match(draft["payment_instrument"], gold["paymentInstrumentAccepted"]))
    else:
        s["instrument"] = 1.0

    if gold["categoryAccepted"]:
        s["category"] = float(any(draft["category"].casefold() == title.casefold()
                                  for title in gold["categoryAccepted"]))
    else:
        alternative = broad_category_alternative(gold, user)
        s["category"] = float((draft["category"] == "" and draft["proposed_category"] != "")
                              or (alternative is not None and draft["category"].casefold() == alternative.casefold()))
        s["proposal_wording"] = float(draft["proposed_category"] != "" and any(
            fold(draft["proposed_category"]) == fold(title) for title in gold["proposalAccepted"]))

    s["date"] = float(record_date(draft["date"], case) == record_date(gold["purchaseDate"], case))
    s["time"] = float(draft["time"] == gold["purchaseTime"])
    s["tags"] = float(tags_right(draft["tags"], case, user))
    proposal = fold(draft["proposed_tag"])
    restates = proposal and any(proposal == fold(t) for t in user["tags"])
    s["proposed_tag"] = float(not proposal or (gold["proposedTagMode"] == "optional" and not restates))
    note = fold(draft["note"])
    items = [fold(item) for item in gold["noteItems"] if fold(item)]
    s["note_items"] = sum(1 for item in items if item in note) / len(items) if items else 1.0
    s["record_core"] = float(all(s[f] for f in CORE))
    s["record_all"] = float(all(s[f] for f in ALL))
    return s


def is_wasted(summary):
    return any(word in summary for word in ("repeat", " left", "stopping", "unknown tool", "invalid", "refused"))


def reward(scores, result):
    if scores is None:
        return 0.0
    if not scores["answered"]:
        return NO_DRAFT
    value = sum(weight * scores[field] for field, weight in REWARD_WEIGHTS.items())
    if scores["record_all"]:
        value += FULL_RECORD_BONUS
    value += approval_term(scores)
    return value - WASTED_PER_CALL * result.get("wasted_calls", 0)


def approval_term(scores):
    """The part of the reward that auto-approval decides: the bonus for a
    right draft confident enough to be approved, the penalty for a wrong
    one, nothing for a draft left to review."""
    if scores is None or not scores["answered"] or not scores["would_approve"]:
        return 0.0
    return RIGHT_APPROVAL_BONUS if scores["record_all"] else -FALSE_APPROVAL_PENALTY


def bootstrap(values, seed=0):
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAP_RESAMPLES))
    return means[int(0.025 * BOOTSTRAP_RESAMPLES)], means[int(0.975 * BOOTSTRAP_RESAMPLES) - 1]


def aggregate(rows):
    if not rows:
        return {"n": 0}
    out = {"n": len(rows)}
    for field in FIELDS + ["reward"]:
        out[field] = sum(r[field] for r in rows) / len(rows)
    wording = [r["proposal_wording"] for r in rows if r["proposal_wording"] is not None]
    out["proposal_wording"] = sum(wording) / len(wording) if wording else None
    return out


def process_stats(results):
    ran = [r for r in results if r["status"] != "rejected_too_long"]
    if not ran:
        return {}
    mean = lambda key: sum(r.get(key, 0) for r in ran) / len(ran)
    tools = defaultdict(int)
    for r in ran:
        for name, count in r.get("tool_calls", {}).items():
            tools[name] += count
    return {
        "failed": sum(r["status"] == "failed" for r in ran) / len(ran),
        "tool_calls": mean("tool_call_count"), "wasted_calls": mean("wasted_calls"),
        "budget_spent": mean("budget_spent"), "rounds": mean("rounds"), "compactions": mean("compactions"),
        "memory_edits": mean("memory_edits"), "seconds": mean("seconds"),
        "completion_tokens": mean("completion_tokens"),
        "prefill_seconds": mean("prefill_seconds"), "decode_seconds": mean("decode_seconds"),
        "peak_prompt_tokens": max((r.get("peak_prompt_tokens", 0) for r in ran), default=0),
        "tool_calls_by_name": {name: count / len(ran) for name, count in sorted(tools.items())},
        "response_formats": dict(Counter(str(r.get("response_format")) for r in ran)),
        "format_refusal": next((x for r in ran for x in r.get("format_refusals") or []), None),
    }


def grade(cases, users, results):
    by_id = {r["id"]: r for r in results}
    rows, rejected = [], []
    for case in cases:
        result = by_id.get(case["id"])
        if result is None:
            continue
        scores = score_case(case, result, users[case["user"]])
        if scores is None:
            rejected.append(float(result["status"] == "rejected_too_long"))
            continue
        scores["reward"] = reward(scores, result)
        scores.update({"id": case["id"], "language": case["language"], "style": case["style"],
                       "traits": case["traits"], "spending": case["gold"]["isSpending"]})
        rows.append(scores)

    approved = [r for r in rows if r["would_approve"]]
    overall = aggregate(rows)
    for field in ("record_core", "record_all", "reward"):
        overall[field + "_ci95"] = bootstrap([r[field] for r in rows])
    overall["rejected_ok"] = sum(rejected) / len(rejected) if rejected else None
    overall["false_approval"] = sum(1 for r in approved if not r["record_all"]) / len(rows) if rows else 0.0
    overall["right_approval"] = sum(1 for r in approved if r["record_all"]) / len(rows) if rows else 0.0

    summary = {"overall": overall, "process": process_stats(results), "by_language": {}, "by_style": {},
               "by_trait": {}}
    groups = defaultdict(list)
    for r in rows:
        groups[("by_language", r["language"])].append(r)
        groups[("by_style", r["style"])].append(r)
        for trait in r["traits"]:
            groups[("by_trait", trait)].append(r)
    for (kind, key), members in sorted(groups.items()):
        summary[kind][key] = aggregate(members)
    return summary, rows


COLUMNS = ("record_core", "record_all", "amount", "merchant", "category", "tags", "date", "instrument", "answered")


def print_summary(summary, columns=COLUMNS):
    o = summary["overall"]
    if o.get("n", 0) == 0:
        print("nothing graded yet")
        return
    lo, hi = o["record_core_ci95"]
    print(f"record_core {o['record_core']:.3f} [{lo:.3f}, {hi:.3f}]   "
          f"record_all {o['record_all']:.3f} [{o['record_all_ci95'][0]:.3f}, {o['record_all_ci95'][1]:.3f}]   "
          f"reward {o['reward']:.3f}   n={o['n']}")
    print(f"false approvals {o['false_approval']:.3f}   right approvals {o['right_approval']:.3f}   "
          f"over-limit rejected {o['rejected_ok'] if o['rejected_ok'] is not None else '—'}")
    p = summary["process"]
    if p:
        print(f"failed {p['failed']:.3f}   tool calls {p['tool_calls']:.2f} (wasted {p['wasted_calls']:.2f})   "
              f"budget spent {p['budget_spent']:.2f}   compactions {p['compactions']:.2f}   "
              f"memory edits {p['memory_edits']:.2f}   {p['seconds']:.1f} s/case "
              f"(prefill {p['prefill_seconds']:.1f}, decode {p['decode_seconds']:.1f})   "
              f"{p['completion_tokens']:.0f} tokens out/case   peak prompt {p['peak_prompt_tokens']} tokens")
        formats = p.get("response_formats", {})
        if set(formats) & {"json_schema_plain", "json_object", "none"}:
            print(f"answer formats {formats}: json_schema is the bounded schema, json_schema_plain the same "
                  "fields without bounds, json_object and none have no field grammar (the prompt lists the fields)")
        if p.get("format_refusal"):
            print(f"the server's refusal: {p['format_refusal']}")
    print()
    print(f"{'':24}{'n':>5} " + " ".join(f"{c[:9]:>9}" for c in columns))

    def line(name, a):
        print(f"{name[:24]:24}{a['n']:>5} " + " ".join(f"{a[c]:>9.2f}" for c in columns))
    line("overall", o)
    for kind in ("by_language", "by_style", "by_trait"):
        print(f"-- {kind[3:]}")
        for key, a in summary[kind].items():
            line(key, a)


def run_folder(name_or_path):
    return name_or_path if os.path.isdir(name_or_path) else os.path.join(RUNS, name_or_path)


def grade_run(name_or_path, suite=None):
    folder = run_folder(name_or_path)
    with open(os.path.join(folder, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    suite_meta, cases, users = load_suite(suite or meta["suite"])
    if suite_meta["fingerprint"] != meta["suite_fingerprint"]:
        raise SystemExit(f"{folder} ran on suite {meta['suite_fingerprint']}, not {suite_meta['fingerprint']}")
    summary, rows = grade(cases, users, read_jsonl(os.path.join(folder, "results.jsonl")))
    summary["meta"] = meta
    with open(os.path.join(folder, "grade.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_case": rows}, f, ensure_ascii=False, indent=1)
    return summary, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run name under runs/, or a run folder")
    ap.add_argument("--suite", help="suite name or folder (default: the one the run recorded)")
    args = ap.parse_args()
    summary, _ = grade_run(args.run, args.suite)
    print_summary(summary)


if __name__ == "__main__":
    main()
