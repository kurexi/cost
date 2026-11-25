#!/usr/bin/env python3
"""Run the spending agent over a suite, one fresh ledger per case.

    python3 run.py --suite dev --name bonsai-27b --base-url http://127.0.0.1:8090/v1 --model bonsai \\
        --reasoning-effort none
    python3 run.py --suite dev --name oracle --backend oracle        # must grade perfect
    python3 run.py --suite dev --name quick --limit 70 ...           # a balanced prefix

Every case gets its own copy of its user's ledger (records dated after the
case's today left out), so one case's memory edit can't leak into the next and
cases can run in any order. The run is resumable: `runs/<name>/results.jsonl`
gains a line per finished case and a rerun skips them. It refuses to resume
on a different suite or agent contract, because the results would not be one
run any more.

Writes, under `runs/<name>/` in the records folder (`$SPENDING_EVAL_RECORDS`,
or this folder when it is unset):
- `meta.json`: model, server, settings, suite, contract and agent-code
  fingerprints, timing;
- `results.jsonl`: per case, the session's status and settled draft and
  what it cost (tool calls by name, wasted calls, rounds, compactions,
  memory edits, tokens, seconds);
- `transcripts.jsonl`: every session in full — the messages, tool calls and
  answers — which is the material for distilling a teacher into a student;
- `grade.json`: the grade, written at the end (see grade.py).
"""

import argparse
import concurrent.futures
import glob
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time

from suite import RUNS, append_jsonl, load_suite, materialize, read_jsonl
from backends import OracleBackend, SilentBackend
from grade import grade_run, is_wasted, print_summary

from spending_agent import draft as contract
from spending_agent.backend import ChatBackend
from spending_agent.session import SessionConfig, run_session

SETTINGS = ("backend", "model", "base_url", "reasoning_effort", "temperature", "max_tool_rounds", "tool_tokens",
            "answer_tokens")


def agent_code_fingerprint():
    """A hash of the agent's source: a change to what a tool answers changes
    what the model sees as surely as a change to the instructions does."""
    package = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "spending_agent")
    digest = hashlib.sha256()
    for path in sorted(glob.glob(os.path.join(package, "*.py"))):
        with open(path, "rb") as f:
            digest.update(os.path.basename(path).encode() + b"\0" + f.read())
    return digest.hexdigest()[:16]


def session_costs(transcript):
    events = transcript["events"]
    tools = [e for e in events if e["kind"] == "tool"]
    usages = [e.get("usage") or {} for e in events if e["kind"] == "model"]
    by_name = {}
    for event in tools:
        by_name[event["name"]] = by_name.get(event["name"], 0) + 1
    return {
        "tool_calls": by_name,
        "tool_call_count": len(tools),
        "wasted_calls": sum(1 for e in tools if is_wasted(e.get("summary", ""))),
        "budget_spent": int(any(e["kind"] == "note" and "spent" in e.get("text", "") for e in events)),
        "rounds": sum(1 for e in events if e["kind"] == "model" and e.get("phase") == "tools"),
        "compactions": sum(1 for e in events if e["kind"] == "compaction"),
        "memory_edits": sum(1 for e in events if e["kind"] == "memory"),
        "completion_tokens": sum(u.get("completion_tokens", 0) for u in usages),
        "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usages),
        "prefill_seconds": round(sum((u.get("timings") or {}).get("prompt_ms", 0) for u in usages) / 1000, 2),
        "decode_seconds": round(sum((u.get("timings") or {}).get("predicted_ms", 0) for u in usages) / 1000, 2),
        "cached_prompt_tokens": sum((u.get("timings") or {}).get("cache_n", 0) for u in usages),
        "peak_prompt_tokens": max((u.get("prompt_tokens", 0) for u in usages), default=0),
        "response_format": next((e.get("response_format") for e in events if e.get("phase") == "answer"), None),
        "format_refusals": next((e.get("refusals") for e in events if e.get("phase") == "answer"), None) or [],
    }


def run_case(case, user, make_backend, settings):
    folder = tempfile.mkdtemp(prefix="case-")
    try:
        ledger = materialize(user, case["today"], folder)
        config = SessionConfig(today=case["today"], local_currency=user["localCurrency"],
                               preferences=list(user["preferences"]), max_tool_rounds=settings["max_tool_rounds"],
                               tool_tokens=settings["tool_tokens"], answer_tokens=settings["answer_tokens"])
        started = time.time()
        result = run_session(ledger, case["content"], make_backend(case), config, session_id=case["id"])
        seconds = time.time() - started
        with open(os.path.join(folder, "sessions", f"{result.session_id}.json"), encoding="utf-8") as f:
            transcript = json.load(f)
        memory_after = ledger.memory()
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    row = {"id": case["id"], "status": result.status, "reason": result.reason,
           "draft": transcript["draft"], "seconds": round(seconds, 2), **session_costs(transcript)}
    if memory_after != user["memory"]:
        row["memory_after"] = memory_after
    return row, {"id": case["id"], **transcript}


def report(index, total, row, elapsed):
    draft = row["draft"] or {}
    remaining = elapsed / index * (total - index)
    timing = f"{row['seconds']:.1f}s"
    if row.get("decode_seconds"):
        timing += f" (prefill {row['prefill_seconds']:.0f}s, decode {row['decode_seconds']:.0f}s)"
    print(f"[{index}/{total}] {row['id']:<20} {row['status']:<18} {draft.get('merchant', '')[:20]:<20} "
          f"{draft.get('total_amount', ''):>10} {draft.get('currency', ''):<4} tools {row['tool_call_count']}  "
          f"{timing}  ~{remaining / 60:.0f} min left", flush=True)


def backend_factory(args):
    if args.backend == "oracle":
        return OracleBackend
    if args.backend == "silent":
        return SilentBackend
    if not args.base_url or not args.model:
        raise SystemExit("--base-url and --model are required for --backend openai")
    return lambda case: ChatBackend(args.base_url, args.model, api_key=os.environ.get("SPENDING_AGENT_API_KEY"),
                                    timeout=args.timeout, temperature=args.temperature,
                                    reasoning_effort=args.reasoning_effort,
                                    one_request_at_a_time=args.workers == 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="dev")
    ap.add_argument("--name", required=True, help="run name; results go to runs/<name>/")
    ap.add_argument("--backend", choices=("openai", "oracle", "silent"), default="openai")
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    ap.add_argument("--reasoning-effort", default=None, help="none, low, medium or high for reasoning models")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--max-tool-rounds", type=int, default=SessionConfig.max_tool_rounds)
    ap.add_argument("--tool-tokens", type=int, default=SessionConfig.tool_tokens)
    ap.add_argument("--answer-tokens", type=int, default=SessionConfig.answer_tokens)
    ap.add_argument("--limit", type=int, help="run only the first N cases (the suite interleaves locales)")
    ap.add_argument("--ids", nargs="*", help="run only these case ids")
    ap.add_argument("--restart", action="store_true", help="discard an existing run of the same name")
    ap.add_argument("--workers", type=int, default=1,
                    help="cases in flight at once; match the server's parallel slots (llama-server -np)")
    args = ap.parse_args()

    suite_meta, cases, users = load_suite(args.suite)
    if args.ids:
        wanted = set(args.ids)
        cases = [c for c in cases if c["id"] in wanted]
    if args.limit:
        cases = cases[:args.limit]

    folder = os.path.join(RUNS, args.name)
    if args.restart:
        shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder, exist_ok=True)
    meta_path = os.path.join(folder, "meta.json")
    settings = {"backend": args.backend, "model": args.model or args.backend, "base_url": args.base_url,
                "reasoning_effort": args.reasoning_effort, "temperature": args.temperature,
                "max_tool_rounds": args.max_tool_rounds, "tool_tokens": args.tool_tokens,
                "answer_tokens": args.answer_tokens}
    meta = {"name": args.name, "suite": suite_meta["name"], "suite_fingerprint": suite_meta["fingerprint"],
            "contract": contract.contract_fingerprint(), "agent_code": agent_code_fingerprint(), **settings,
            "cases_planned": len(cases), "seconds": 0.0,
            "workers": args.workers}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            previous = json.load(f)
        for key in ("suite_fingerprint", "contract") + SETTINGS:
            if previous.get(key) != meta.get(key):
                raise SystemExit(f"runs/{args.name} was run with {key}={previous.get(key)!r}, not "
                                 f"{meta.get(key)!r}. Use another --name, or --restart to discard it.")
        if previous.get("agent_code") != meta["agent_code"]:
            print(f"warning: the agent's code changed since runs/{args.name} started "
                  f"({previous.get('agent_code')} → {meta['agent_code']}); the finished cases ran on the old code")
            meta["agent_code"] = previous.get("agent_code")
        meta["seconds"] = previous.get("seconds", 0.0)

    results_path = os.path.join(folder, "results.jsonl")
    done = {r["id"] for r in read_jsonl(results_path)}
    todo = [c for c in cases if c["id"] not in done]
    print(f"{len(cases)} cases, {len(done & {c['id'] for c in cases})} already done, {len(todo)} to run "
          f"(suite {suite_meta['name']} {suite_meta['fingerprint']}, contract {meta['contract']})")
    make_backend = backend_factory(args)
    started = time.time()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
    try:
        futures = [pool.submit(run_case, case, users[case["user"]], make_backend, settings) for case in todo]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            row, transcript = future.result()
            append_jsonl(results_path, row)
            append_jsonl(os.path.join(folder, "transcripts.jsonl"), transcript)
            report(index, len(todo), row, time.time() - started)
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        print("\ninterrupted; rerun the same command to resume (cases in flight are run again)")
    finally:
        meta["seconds"] = round(meta["seconds"] + time.time() - started, 1)
        meta["cases_done"] = len(read_jsonl(results_path))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
    summary, _ = grade_run(folder)
    print()
    print_summary(summary)


if __name__ == "__main__":
    sys.exit(main())
