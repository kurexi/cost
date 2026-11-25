"""Shared fixtures: a ledger in a temporary folder and a scripted model server."""

import json
import tempfile

from spending_agent.ledger import Ledger, Record


def record(id, date, merchant, amount_minor, category="", tags=(), note="", currency="GBP", source="manual",
           payment_instrument=""):
    return Record(id=id, date=date, merchant=merchant, amount_minor=amount_minor, currency=currency,
                  category=category, tags=list(tags), note=note, source=source,
                  payment_instrument=payment_instrument)


SAMPLE_RECORDS = [
    record("r1", "2026-10-01", "Tesco", 1240, "Groceries", note="Groceries: milk, bread"),
    record("r2", "2026-10-08", "TESCO", 980, "Groceries", note="Groceries: eggs"),
    record("r3", "2026-10-15", "Tesco", 1520, "Groceries"),
    record("r4", "2026-10-20", "Tesco Cafe", 450, "Coffee", note="Coffee: flat white"),
    record("r5", "2026-09-12", "M&S", 899, "Groceries", payment_instrument="Monzo"),
    record("r6", "2026-08-02", "Shinkansen", 14000, "Travel", tags=("Japan Trip",), currency="JPY"),
    record("r7", "2026-08-05", "Lawson", 820, "Groceries", tags=("Japan Trip",), currency="JPY"),
    record("r8", "2026-03-01", "PureGym", 2499, "Fitness", tags=("Health",)),
    record("r9", "2026-09-01", "PureGym", 2499, "Fitness", tags=("Health",)),
]


def sample_ledger(records=SAMPLE_RECORDS, categories=("Groceries", "Coffee", "Travel", "Fitness", "Eating Out"),
                  tags=("Work",)):
    folder = tempfile.mkdtemp(prefix="ledger-")
    ledger = Ledger.create(folder, categories, tags)
    for item in records:
        ledger.append_record(item)
    return ledger


def tool_call(name, arguments, call_id="call_0"):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


def answer_text(**fields):
    draft = {"reasoning": "A till receipt from Tesco.", "isSpending": True, "merchant": "Tesco",
             "paymentInstrument": "", "totalAmount": "12.40", "currencyCode": "GBP", "purchaseDate": "2026-11-17",
             "purchaseTime": "", "category": "Groceries", "proposedCategory": "", "tags": [], "proposedTag": "",
             "note": "Groceries: milk", "confidence": 0.9}
    draft.update(fields)
    return json.dumps(draft)


class ScriptedBackend:
    """Replays tool-round replies in order, then the answer; records every request."""

    def __init__(self, tool_replies=(), answer=None, fail_with=None):
        self.tool_replies = list(tool_replies)
        self.answer_reply = answer if answer is not None else answer_text()
        self.fail_with = fail_with
        self.requests = []
        self.last_format = None

    def complete(self, messages, tools=None, max_tokens=800):
        self.requests.append(("complete", [dict(m) for m in messages]))
        if self.fail_with:
            raise self.fail_with
        if self.tool_replies:
            return self.tool_replies.pop(0), {}
        return {"role": "assistant", "content": "Ready to answer."}, {}

    def answer(self, messages, schema, max_tokens=1500):
        self.requests.append(("answer", [dict(m) for m in messages]))
        self.last_format = "json_schema"
        return {"role": "assistant", "content": self.answer_reply}, {}


def calls(*pairs):
    return {"role": "assistant", "content": "",
            "tool_calls": [tool_call(name, arguments, f"call_{i}") for i, (name, arguments) in enumerate(pairs)]}
