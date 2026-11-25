#!/usr/bin/env python3
"""Grade extraction outputs against the gold records — and the RL reward.

    python3 grade.py --gold data/eval.jsonl --pred runs/<name>/predictions.jsonl

A prediction line is {"id": ..., "output": <raw model text or an object>} (and
anything else; it is ignored). Raw text is decoded the way `LocalModelAgent`
decodes a local server's reply: the first JSON object in it, with lenient
types — a model that types `"isSpending": "true"` or `12.34` for "12.34" is
not failed for it, because the app wouldn't fail it either.

Fields are judged like this (all deterministic, no model in the loop):

- amount: the value in minor units must equal the gold total; `amount_strict`
  also asks for the exact gold string ("12.34", "1280").
- merchant: folded comparison against every accepted spelling (the printed
  one, the brand, a known alias); a containment either way passes, so
  "TESCO EXPRESS" for Tesco is right — but a name equal to the payment
  instrument never is.
- paymentInstrument: graded only where the content names one ("exact"); must
  be empty where nothing paid is shown ("none"); free where it is a bare
  scheme or cash ("either").
- category: exact title when the user has the category; when they don't, the
  category slot must be empty and a proposal made (`proposal_wording` says
  whether its title was one of the reasonable ones).
- date, time: exact; empty when the gold is empty (never invent a date).

`record_core` = spending verdict + amount + currency + merchant + category;
`record_all` adds date, time and instrument. `reward()` is the scalar used by
train_grpo.py.
"""

import argparse
import json
import re
from collections import defaultdict

from common import ZERO_DECIMAL, fold, read_jsonl

FIELDS = ["json_valid", "is_spending", "amount", "amount_strict", "currency", "merchant", "instrument",
          "category", "date", "time", "record_core", "record_all"]


def first_json_object(text):
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def as_str(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v).strip()


def as_bool(v, default=True):
    if isinstance(v, bool):
        return v
    s = as_str(v).lower()
    if s in ("true", "yes", "1"):
        return True
    if s in ("false", "no", "0"):
        return False
    return default


def parse_minor(text, currency):
    """Lenient amount reader for grading: '12.34', '12,34', '€12.34',
    '1,234.56', '1.234,56', 12.34. Returns None when nothing numeric."""
    s = re.sub(r"[^\d.,\-]", "", as_str(text))
    if not re.search(r"\d", s):
        return None
    if "," in s and "." in s:
        dec = "," if s.rfind(",") > s.rfind(".") else "."
        s = s.replace("." if dec == "," else ",", "").replace(",", ".")
    elif "," in s:
        head, _, tail = s.rpartition(",")
        s = head.replace(",", "") + ("." if len(tail) in (1, 2) else "") + tail
    elif s.count(".") > 1:
        head, _, tail = s.rpartition(".")
        s = head.replace(".", "") + ("." + tail if len(tail) in (1, 2) else tail)
    try:
        value = float(s)
    except ValueError:
        return None
    if currency in ZERO_DECIMAL:
        return round(value)
    return round(value * 100)


def names_match(model, accepted):
    fm = fold(model)
    if not fm:
        return False
    for a in accepted:
        fa = fold(a)
        if not fa:
            continue
        if fm == fa or (len(fm) >= 2 and len(fa) >= 2 and (fm in fa or fa in fm)):
            return True
    return False


def score(gold, output):
    obj = first_json_object(output)
    s = {f: 0.0 for f in FIELDS}
    s["note_items"] = 0.0
    s["proposal_wording"] = None
    if obj is None:
        return s, None
    s["json_valid"] = 1.0
    spending = as_bool(obj.get("isSpending"), default=True)
    s["is_spending"] = float(spending == gold["isSpending"])
    if not gold["isSpending"]:
        ok = float(not spending)
        for f in ("amount", "amount_strict", "currency", "merchant", "instrument", "category", "date", "time",
                  "record_core", "record_all"):
            s[f] = ok
        s["note_items"] = ok
        return s, obj

    cur = as_str(obj.get("currencyCode")).upper()
    s["currency"] = float(cur == gold["currencyCode"])
    minor = parse_minor(obj.get("totalAmount"), gold["currencyCode"])
    s["amount"] = float(minor == gold["amountMinorUnits"])
    s["amount_strict"] = float(as_str(obj.get("totalAmount")) == gold["totalAmount"])

    merchant = as_str(obj.get("merchant"))
    instrument = as_str(obj.get("paymentInstrument"))
    gold_instrument = gold.get("paymentInstrument", "")
    merchant_is_instrument = bool(gold_instrument) and fold(merchant) == fold(gold_instrument)
    s["merchant"] = float(names_match(merchant, gold["merchantAccepted"]) and not merchant_is_instrument)

    mode = gold["paymentInstrumentMode"]
    if mode == "none":
        s["instrument"] = float(instrument == "")
    elif mode == "exact":
        s["instrument"] = float(names_match(instrument, gold["paymentInstrumentAccepted"]))
    else:
        s["instrument"] = 1.0

    category = as_str(obj.get("category"))
    proposal = as_str(obj.get("proposedCategory"))
    if gold["category"]:
        s["category"] = float(category.casefold() == gold["category"].casefold())
    else:
        s["category"] = float(category == "" and proposal != "")
        s["proposal_wording"] = float(proposal != "" and any(
            fold(proposal) == fold(a) for a in gold["proposalAccepted"]))

    s["date"] = float(as_str(obj.get("purchaseDate")) == gold["purchaseDate"])
    s["time"] = float(as_str(obj.get("purchaseTime")) == gold["purchaseTime"])
    note = fold(as_str(obj.get("note")))
    items = gold.get("noteItems") or []
    s["note_items"] = (sum(1 for i in items if fold(i) and fold(i) in note) / len(items)) if items else 1.0

    s["record_core"] = float(all(s[f] for f in ("is_spending", "amount", "currency", "merchant", "category")))
    s["record_all"] = float(s["record_core"] and all(s[f] for f in ("date", "time", "instrument")))
    return s, obj


REWARD_WEIGHTS = {"is_spending": 0.10, "amount": 0.30, "currency": 0.10, "merchant": 0.15,
                  "category": 0.15, "instrument": 0.08, "date": 0.08, "time": 0.04}


def reward(gold, output, trail=None):
    """Scalar reward in [-1, 1.05]: field-weighted correctness against the
    gold labels; -1 for an answer the app couldn't decode at all; a small
    penalty for tool calls that burned budget for nothing (unknown tools,
    refusals, repeats) and a small bonus for a fully correct record. The app's
    own audits are *not* the reward — they are necessary, not sufficient, and
    a policy rewarded on them learns to copy any plausible number."""
    s, _ = score(gold, output)
    if not s["json_valid"]:
        return -1.0, s
    r = sum(w * s[f] for f, w in REWARD_WEIGHTS.items())
    if s["record_all"]:
        r += 0.05
    for step in trail or []:
        summary = step.get("summary", "")
        if "repeat" in summary or "left" in summary or summary == "unknown tool":
            r -= 0.03
    return r, s


def aggregate(rows):
    out = {}
    for f in FIELDS + ["note_items"]:
        out[f] = sum(r[f] for r in rows) / len(rows) if rows else 0.0
    pw = [r["proposal_wording"] for r in rows if r["proposal_wording"] is not None]
    out["proposal_wording"] = sum(pw) / len(pw) if pw else None
    out["n"] = len(rows)
    return out


def grade(gold_rows, preds):
    by_id = {p["id"]: p for p in preds}
    per = []
    for g in gold_rows:
        if g["id"] not in by_id:
            continue
        p = by_id[g["id"]]
        s, _ = score(g["gold"], p.get("output"))
        s.update({"id": g["id"], "language": g["language"], "style": g["style"], "traits": g["traits"],
                  "tool_calls": len(p.get("trail", []))})
        per.append(s)
    summary = {"overall": aggregate(per), "by_language": {}, "by_style": {}, "by_trait": {}}
    groups = defaultdict(list)
    for r in per:
        groups[("by_language", r["language"])].append(r)
        groups[("by_style", r["style"])].append(r)
        for t in r["traits"]:
            groups[("by_trait", t)].append(r)
    for (kind, key), rows in sorted(groups.items()):
        summary[kind][key] = aggregate(rows)
    summary["overall"]["mean_tool_calls"] = sum(r["tool_calls"] for r in per) / len(per) if per else 0
    return summary, per


def print_table(summary, columns=("record_core", "record_all", "amount", "currency", "merchant", "category",
                                  "instrument", "date", "json_valid")):
    head = f"{'':22}{'n':>5} " + " ".join(f"{c[:9]:>9}" for c in columns)
    print(head)

    def line(name, a):
        print(f"{name[:22]:22}{a['n']:>5} " + " ".join(f"{a[c]:>9.2f}" for c in columns))
    line("overall", summary["overall"])
    for kind in ("by_language", "by_style", "by_trait"):
        print(f"-- {kind[3:]}")
        for k, a in summary[kind].items():
            line(k, a)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", help="write the summary JSON here")
    args = ap.parse_args()
    summary, per = grade(read_jsonl(args.gold), read_jsonl(args.pred))
    print_table(summary)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "per_sample": per}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
