import json


def gold_answer(case):
    gold = case["gold"]
    return {
        "reasoning": "Oracle: the gold record.",
        "isSpending": gold["isSpending"],
        "merchant": gold["merchant"],
        "paymentInstrument": gold["paymentInstrument"],
        "totalAmount": gold["totalAmount"],
        "currencyCode": gold["currencyCode"],
        "purchaseDate": gold["purchaseDate"],
        "purchaseTime": gold["purchaseTime"],
        "category": gold["category"],
        "proposedCategory": gold["proposedCategory"],
        "tags": list(gold["tags"]),
        "proposedTag": "",
        "note": ", ".join(gold["noteItems"]),
        "confidence": 1.0,
    }


class OracleBackend:
    def __init__(self, case):
        self.case = case
        self.last_format = "oracle"

    def complete(self, messages, tools=None, max_tokens=800):
        return {"role": "assistant", "content": ""}, {}

    def answer(self, messages, schema, max_tokens=1500):
        return {"role": "assistant", "content": json.dumps(gold_answer(self.case), ensure_ascii=False)}, {}


class SilentBackend:
    def __init__(self, case=None):
        self.last_format = "silent"

    def complete(self, messages, tools=None, max_tokens=800):
        return {"role": "assistant", "content": ""}, {}

    def answer(self, messages, schema, max_tokens=1500):
        return {"role": "assistant", "content": "I can't tell."}, {}
