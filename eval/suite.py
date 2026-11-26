import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from spending_agent.ledger import Ledger, Record

SUITES = os.path.join(HERE, "suites")
RECORDS = os.path.abspath(os.environ.get("SPENDING_EVAL_RECORDS") or HERE)
RUNS = os.path.join(RECORDS, "runs")


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def suite_folder(name_or_path):
    return name_or_path if os.path.isdir(name_or_path) else os.path.join(SUITES, name_or_path)


def load_suite(name_or_path):
    folder = suite_folder(name_or_path)
    meta_path = os.path.join(folder, "meta.json")
    if not os.path.exists(meta_path):
        raise SystemExit(f"no suite at {folder}; generate it with make_suite.py")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    cases = read_jsonl(os.path.join(folder, "cases.jsonl"))
    users = {u["id"]: u for u in read_jsonl(os.path.join(folder, "users.jsonl"))}
    return meta, cases, users


def materialize(user, today, folder):
    ledger = Ledger.create(folder, user["categories"], user["tags"])
    for row in user["records"]:
        if row["date"] <= today:
            ledger.append_record(Record(id=row["id"], date=row["date"], time=row["time"], merchant=row["merchant"],
                                        payment_instrument=row["payment_instrument"],
                                        amount_minor=row["amount_minor"], currency=row["currency"],
                                        category=row["category"], tags=list(row["tags"]), note=row["note"],
                                        source=row["source"], content=row["content"], created_at=row["date"]))
    if user["memory"]:
        with open(ledger.path("memory.txt"), "w", encoding="utf-8") as f:
            f.write(user["memory"])
    return ledger
