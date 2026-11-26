#!/usr/bin/env python3
import argparse
import json
from datetime import date

from common import INSTRUCTIONS, read_jsonl, write_jsonl
from env import Episode, tool_definitions

CLOSING = ("Now give the final answer as a single JSON object and nothing else. "
           "Fill the fields in the order they appear in the schema.")
NO_TOOLS = "No tools needed: the content shows everything required."

KIND = {
    "till": "a till receipt", "email": "an order confirmation email", "notification": "a bank card notification",
    "wallet": "a wallet transaction screenshot", "casual": "a quick personal note",
}


def tool_call_text(name, arg_name, value):
    return f"<tool_call>\n<function={name}>\n<parameter={arg_name}>\n{value}\n</parameter>\n</function>\n</tool_call>"


def planned_calls(sample, profile):
    g, traits = sample["gold"], sample["traits"]
    calls = []
    if "no_printed_total" in traits:
        prices = item_prices(sample)
        if len(prices) == len(g["noteItems"]) >= 2:
            from decimal import Decimal
            if sum(Decimal(p) for p in prices) == Decimal(g["totalAmount"]):
                calls.append(("calculator", "expression", " + ".join(prices)))
    if "ledger_lookup_useful" in traits:
        calls.append(("searchRecords", "query", g["merchant"]))
    return calls


def item_prices(sample):
    g = sample["gold"]
    lines = sample["content"].split("\n")
    out = []
    for item in g["noteItems"]:
        for i, line in enumerate(lines):
            if line.startswith(item):
                rest = line[len(item):].strip() or (lines[i + 1] if i + 1 < len(lines) else "")
                out.append(dot_decimal(rest, g["currencyCode"]))
                break
    return [p for p in out if p]


def dot_decimal(text, currency):
    digits = "".join(c for c in text if c.isdigit() or c in ".,")
    if not digits:
        return ""
    if currency == "JPY":
        return digits.replace(",", "").split(".")[0]
    if "," in digits and "." in digits:
        if digits.rfind(",") > digits.rfind("."):
            digits = digits.replace(".", "").replace(",", ".")
        else:
            digits = digits.replace(",", "")
    elif "," in digits:
        head, _, tail = digits.rpartition(",")
        digits = head.replace(",", "") + ("." + tail if len(tail) == 2 else tail)
    return digits


def reasoning(sample, profile):
    g, t, style = sample["gold"], set(sample["traits"]), sample["style"]
    if not g["isSpending"]:
        return "This content is not a purchase: nothing here shows the user paying money, so it is not spending."
    cur, total = g["currencyCode"], g["totalAmount"]
    s = [f"This is {KIND.get(style, 'content')} from {g['merchant']}."]
    if g["paymentInstrument"] and ("card_heading" in t or style == "notification"):
        s.append(f"{g['paymentInstrument']} is what paid, not who was paid; the merchant is {g['merchant']}.")
    if "no_printed_total" in t:
        s.append(f"No total is printed, so the line items were added with the calculator: {total} {cur}.")
    elif "cash_change" in t:
        s.append(f"The cash handed over and the change are not the price; the total line is {total} {cur}.")
    elif "tip" in t:
        s.append(f"The total includes tax and the tip: {total} {cur}.")
    elif "tax_added" in t:
        s.append(f"Tax is added on top of the subtotal; the total line is {total} {cur}.")
    elif "shipping_added" in t:
        s.append(f"The order total includes delivery: {total} {cur}.")
    elif "column_split" in t:
        s.append(f"The OCR split the columns, so the total's amount is on the line below its label: {total} {cur}.")
    else:
        s.append(f"The amount paid is {total} {cur}.")
    if "was_price" in t:
        s.append("The higher price printed under an item is a comparison price, never part of the total.")
    if "travel" in t:
        s.append(f"The content is from abroad, so the currency is {cur}, not the user's local {profile['localCurrency']}.")
    if "date_without_year" in t and g["purchaseDate"]:
        s.append(f"The date has no year; with today's date it is {g['purchaseDate']}.")
    elif "relative_date" in t and g["purchaseDate"]:
        s.append(f"The note dates it relative to today: {g['purchaseDate']}.")
    elif not g["purchaseDate"]:
        s.append("No date is shown, so none is given.")
    if g["category"]:
        if "mixed_merchant" in t:
            s.append(f"The user files {g['merchant']} more than one way; these items are like the ones filed under {g['category']}.")
        else:
            s.append(f"It belongs in the user's existing category {g['category']}.")
    else:
        s.append(f"None of the user's categories fits, so propose {g['proposedCategory']}.")
    return " ".join(s)


def answer(sample, profile):
    g = sample["gold"]
    if not g["isSpending"]:
        obj = {"reasoning": reasoning(sample, profile), "isSpending": False, "merchant": "", "paymentInstrument": "",
               "totalAmount": "", "currencyCode": g["currencyCode"], "purchaseDate": "", "purchaseTime": "",
               "category": "", "proposedCategory": "", "note": "", "confidence": 0.9}
        return json.dumps(obj, ensure_ascii=False)
    title = g["category"] or g["proposedCategory"]
    items = ", ".join(g["noteItems"][:3])
    confident = (g["category"] and not set(sample["traits"]) & {"no_printed_total", "ocr_noise", "needs_proposal"}
                 and sample["style"] != "casual")
    obj = {
        "reasoning": reasoning(sample, profile),
        "isSpending": True,
        "merchant": g["merchant"],
        "paymentInstrument": g["paymentInstrument"],
        "totalAmount": g["totalAmount"],
        "currencyCode": g["currencyCode"],
        "purchaseDate": g["purchaseDate"],
        "purchaseTime": g["purchaseTime"],
        "category": g["category"],
        "proposedCategory": g["proposedCategory"],
        "note": f"{title}: {items}" if items else title,
        "confidence": 0.9 if confident else 0.7,
    }
    return json.dumps(obj, ensure_ascii=False)


def trajectory(tok, sample, profile, eos="<|im_end|>"):
    tools = tool_definitions()
    episode = Episode(sample["content"], profile["ledger"])
    messages = [{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": sample["prompt"]}]
    pairs = []
    for name, arg, value in planned_calls(sample, profile):
        prompt = tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
        text = tool_call_text(name, arg, value)
        pairs.append({"prompt": prompt, "completion": text + eos, "kind": "tool_call"})
        messages.append({"role": "assistant", "content": "",
                         "tool_calls": [{"type": "function", "function": {"name": name, "arguments": {arg: value}}}]})
        messages.append({"role": "tool", "name": name, "content": episode.call(name, {arg: value})})
    prompt = tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
    pairs.append({"prompt": prompt, "completion": NO_TOOLS + eos, "kind": "end_tools"})
    final = messages + [{"role": "user", "content": CLOSING}]
    prompt = tok.apply_chat_template(final, add_generation_prompt=True, tokenize=False)
    pairs.append({"prompt": prompt, "completion": answer(sample, profile) + eos, "kind": "answer"})
    return pairs, episode.trail


def balanced(samples, limit):
    rare = {"no_printed_total", "ledger_lookup_useful", "tip", "was_price", "cash_change", "card_heading",
            "needs_proposal", "travel", "mixed_merchant"}
    by_home = {}
    for s in samples:
        by_home.setdefault(s["homeLocale"], []).append(s)
    for rows in by_home.values():
        rows.sort(key=lambda s: -len(rare & set(s["traits"])))
    out, i = [], 0
    while len(out) < limit and any(i < len(r) for r in by_home.values()):
        for rows in by_home.values():
            if i < len(rows) and len(out) < limit:
                out.append(rows[i])
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="tokenizer / chat template source")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--profiles")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=240)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    profiles = {p["id"]: p for p in read_jsonl(args.profiles or args.gold.replace(".jsonl", "_profiles.jsonl"))}
    rows = []
    for s in balanced(read_jsonl(args.gold), args.limit):
        pairs, trail = trajectory(tok, s, profiles[s["profile"]])
        rows.append({"id": s["id"], "pairs": pairs, "trail": trail})
    write_jsonl(args.out, rows)
    n_pairs = sum(len(r["pairs"]) for r in rows)
    n_tokens = sum(len(tok(p["prompt"] + p["completion"])["input_ids"]) for r in rows for p in r["pairs"])
    print(f"{len(rows)} trajectories, {n_pairs} sequences, {n_tokens} tokens → {args.out}")


if __name__ == "__main__":
    main()
