"""Command line for the Python spending agent.

    python -m spending_agent init LEDGER --category Groceries --tag Work
    python -m spending_agent add LEDGER --date 2026-11-01 --merchant Tesco --amount 12.40 --currency GBP
    python -m spending_agent process LEDGER receipt.txt --base-url http://127.0.0.1:8090/v1 --model bonsai
    python -m spending_agent memory LEDGER [--history]
    python -m spending_agent review LEDGER
"""

import argparse
import datetime
import os
import sys
import uuid

from .backend import ChatBackend
from .ledger import SOURCES, Ledger, Record
from .money import decimal_string, minor_units
from .session import SessionConfig, describe_draft, run_session


def _ledger(folder):
    if not os.path.exists(os.path.join(folder, "records.csv")):
        sys.exit(f"{folder} is not a ledger; run `init` first.")
    return Ledger(folder)


def init(arguments):
    ledger = Ledger.create(arguments.ledger, arguments.category, arguments.tag)
    print(f"Ledger at {ledger.folder}: {len(ledger.category_titles())} categories, {len(ledger.tag_names())} tags.")


def add(arguments):
    ledger = _ledger(arguments.ledger)
    currency = arguments.currency.upper()
    amount = minor_units(arguments.amount, currency)
    if amount is None:
        sys.exit(f"Can't read the amount {arguments.amount!r}.")
    record_id = arguments.id or f"manual-{uuid.uuid4().hex[:12]}"
    ledger.append_record(Record(id=record_id, date=arguments.date, time=arguments.time, merchant=arguments.merchant,
                                payment_instrument=arguments.paid_with, amount_minor=amount, currency=currency,
                                category=arguments.category, tags=arguments.tag, note=arguments.note,
                                source=arguments.source))
    print(f"Added {record_id}.")


def process(arguments):
    ledger = _ledger(arguments.ledger)
    with open(arguments.content, encoding="utf-8") as f:
        content = f.read().strip()
    backend = ChatBackend(arguments.base_url, arguments.model, api_key=os.environ.get("SPENDING_AGENT_API_KEY"),
                          timeout=arguments.timeout, reasoning_effort=arguments.reasoning_effort)
    config = SessionConfig(today=arguments.today or datetime.date.today().isoformat(),
                           local_currency=arguments.local_currency, preferences=arguments.preference)
    result = run_session(ledger, content, backend, config)
    print(f"session   {result.session_id}\nstatus    {result.status}\nreason    {result.reason}\n")
    print(describe_draft(result.draft))
    print(f"\nTranscript: {os.path.join(ledger.folder, 'sessions', result.session_id + '.json')}")


def memory(arguments):
    ledger = _ledger(arguments.ledger)
    if arguments.history:
        for row in ledger.memory_history():
            print(f"--- {row['written_at']}  session {row['session']}\n{row['text']}\n")
        return
    text = ledger.memory()
    print(text if text.strip() else "(empty)")


def review(arguments):
    ledger = _ledger(arguments.ledger)
    for row in ledger.review():
        amount = (f"{decimal_string(int(row['amount_minor']), row['currency'] or 'XXX')} {row['currency']}"
                  if row["amount_minor"].isdigit() else "?")
        print(f"{row['created_at']}  {row['session']}  {row['status']:<18} {row['merchant'] or '—':<24} "
              f"{amount:<14} {row['category'] or row['proposed_category'] or '—'}")


def parser():
    root = argparse.ArgumentParser(prog="spending_agent", description="The Cost spending agent over a CSV ledger.")
    commands = root.add_subparsers(dest="command", required=True)

    p = commands.add_parser("init", help="create a ledger folder")
    p.add_argument("ledger")
    p.add_argument("--category", action="append", default=[])
    p.add_argument("--tag", action="append", default=[])
    p.set_defaults(run=init)

    p = commands.add_parser("add", help="add a record by hand")
    p.add_argument("ledger")
    p.add_argument("--date", required=True)
    p.add_argument("--time", default="")
    p.add_argument("--merchant", required=True)
    p.add_argument("--amount", required=True)
    p.add_argument("--currency", required=True)
    p.add_argument("--category", default="")
    p.add_argument("--tag", action="append", default=[])
    p.add_argument("--note", default="")
    p.add_argument("--paid-with", default="")
    p.add_argument("--source", choices=SOURCES, default="manual")
    p.add_argument("--id", default="")
    p.set_defaults(run=add)

    p = commands.add_parser("process", help="run the agent on one piece of content")
    p.add_argument("ledger")
    p.add_argument("content", help="a text file holding the shared text or OCR")
    p.add_argument("--base-url", required=True, help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8090/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--today", default="")
    p.add_argument("--local-currency", default="GBP")
    p.add_argument("--preference", action="append", default=[])
    p.add_argument("--reasoning-effort", default=None, help="none, low, medium or high for reasoning models")
    p.add_argument("--timeout", type=int, default=900)
    p.set_defaults(run=process)

    p = commands.add_parser("memory", help="show the agent's memory")
    p.add_argument("ledger")
    p.add_argument("--history", action="store_true")
    p.set_defaults(run=memory)

    p = commands.add_parser("review", help="list the review queue")
    p.add_argument("ledger")
    p.set_defaults(run=review)
    return root


def main(argv=None):
    arguments = parser().parse_args(argv)
    arguments.run(arguments)
