#!/usr/bin/env python3
import argparse
import json
import os

from grade import grade_run
from spending_agent import draft as contract
from suite import HERE, RECORDS, RUNS, load_suite

RESULTS = os.path.join(RECORDS, "RESULTS.md")
BENCHMARK = os.path.join(HERE, "BENCHMARK.md")
README = os.path.join(HERE, "..", "README.md")
HEADLINE = (
    ("27B teacher", "qwen3.8-27b-fp8-v5", "Qwen3.8-27B", "FP8"),
    ("0.8B as released", "qwen3.5-0.8b-q8_0", "Qwen3.5-0.8B", None),
    ("0.8B trained, shipped", "qwen3.5-0.8b-grpo-r2-q8_0", "Qwen3.5-0.8B", None),
    ("0.8B trained, 4-bit", "qwen3.5-0.8b-grpo-r2-q4_k_m", "Qwen3.5-0.8B", None),
)
BEGIN, END = "<!-- board:begin -->", "<!-- board:end -->"
NEW_RESULTS = f"# Spending agent results\n\n{BEGIN}\n{END}\n"
REFERENCE_RUNS = ("oracle", "silent")
BOARD_SUITES = ("dev", "dev-novel")


def table(entries):
    head = ("| run | model | contract | n | record_core (95% CI) | record_all | reward | amount | merchant | "
            "category | tags | false approvals | s/case | tools/case |")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for name, summary in entries:
        o, p, m = summary["overall"], summary["process"], summary["meta"]
        lo, hi = o["record_core_ci95"]
        model = m.get("model") or m.get("backend")
        if m.get("reasoning_effort"):
            model += f" (thinking: {m['reasoning_effort']})"
        lines.append(f"| {name} | {model} | `{m['contract']}` | {o['n']} | {o['record_core']:.3f} ({lo:.2f}–{hi:.2f}) "
                     f"| {o['record_all']:.3f} | {o['reward']:.3f} | {o['amount']:.2f} | {o['merchant']:.2f} "
                     f"| {o['category']:.2f} | {o['tags']:.2f} | {o['false_approval']:.3f} "
                     f"| {p.get('seconds', 0):.1f} | {p.get('tool_calls', 0):.2f} |")
    return "\n".join(lines)


def public_table(entries):
    head = ("| run | GGUF | n | record_core (95% CI) | record_all | reward | amount | merchant | category | tags "
            "| false approvals | tools/case |")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for name, summary in entries:
        o, p = summary["overall"], summary["process"]
        lo, hi = o["record_core_ci95"]
        lines.append(f"| {name} | {gguf_size(name)} | {o['n']} | {o['record_core']:.3f} ({lo:.2f}–{hi:.2f}) "
                     f"| {o['record_all']:.3f} | {o['reward']:.3f} | {o['amount']:.2f} | {o['merchant']:.2f} "
                     f"| {o['category']:.2f} | {o['tags']:.2f} | {o['false_approval']:.3f} "
                     f"| {p.get('tool_calls', 0):.2f} |")
    return "\n".join(lines)


def gguf_size(run):
    path = os.path.join(RUNS, run, "server.json")
    if not os.path.exists(path):
        return "—"
    with open(path, encoding="utf-8") as f:
        served = json.load(f)
    return f"{served['quantization']}, {served['gguf_bytes'] / 1e9:.2f} GB"


def graded_summary(run):
    if not os.path.exists(os.path.join(RUNS, run, "meta.json")):
        return None
    summary, _ = grade_run(os.path.join(RUNS, run))
    return summary if summary["overall"].get("n") else None


def headline_section(rows):
    labels = ", ".join(f'"{label}"' for label, *_ in rows)
    bars = ", ".join(f"{dev['overall']['record_core']:.3f}" for *_, dev, _ in rows)
    chart = ["```mermaid", "xychart-beta", '    title "Records right on dev (record_core)"',
             f"    x-axis [{labels}]", '    y-axis "share of 280 cases" 0 --> 1', f"    bar [{bars}]", "```"]
    head = ("| | weights | records right on `dev` | on `dev-novel` (names never seen) "
            "| wrong records it would file unreviewed | tool calls a case (wasted) | ran out of tool budget |")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for label, weights, dev, novel in rows:
        o, p = dev["overall"], dev["process"]
        lo, hi = o["record_core_ci95"]
        novel_text = f"{novel['overall']['record_core']:.0%}" if novel else "not run"
        lines.append(f"| {label} | {weights} | {o['record_core']:.0%} ({lo:.0%}–{hi:.0%}) "
                     f"| {novel_text} | {o['false_approval']:.0%} "
                     f"| {p['tool_calls']:.1f} ({p['wasted_calls']:.1f}) | {p['budget_spent']:.0%} of cases |")
    return "\n".join(chart) + "\n\n" + "\n".join(lines)


def headline_rows():
    rows = []
    for label, run, model, unquantized in HEADLINE:
        dev = graded_summary(run)
        if dev is None:
            continue
        size = gguf_size(run)
        weights = f"{model}, " + (unquantized if size == "—" else f"{size} GGUF")
        rows.append((label, weights, dev, graded_summary(f"{run}-dev-novel")))
    return rows


def contracts_line(entries):
    current = contract.contract_fingerprint()
    seen = sorted({summary["meta"]["contract"] for _, summary in entries})
    return ", ".join(f"`{c}`" + (" (the current contract)" if c == current else " (an older contract)") for c in seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", nargs="+", default=list(BOARD_SUITES),
                    help="the suites to list, each in its own table (default: every evaluation suite)")
    ap.add_argument("--all", action="store_true", help="include the oracle and silent reference runs")
    ap.add_argument("--write", action="store_true", help="rewrite the table in RESULTS.md")
    ap.add_argument("--publish", action="store_true", help="rewrite the public tables in BENCHMARK.md")
    args = ap.parse_args()

    text = "\n\n".join(suite_section(name, args.all) for name in args.suite)
    print(text)
    if args.write:
        write_board(RESULTS, text)
    if args.publish:
        write_board(BENCHMARK, "\n\n".join(suite_section(name, public=True) for name in args.suite))
        write_board(README, headline_section(headline_rows()))


def write_board(path, text):
    document = NEW_RESULTS
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            document = f.read()
    start, end = document.index(BEGIN) + len(BEGIN), document.index(END)
    with open(path, "w", encoding="utf-8") as f:
        f.write(document[:start] + "\n" + text + "\n" + document[end:])


def suite_section(suite_name, include_references=False, public=False):
    suite_meta, _, _ = load_suite(suite_name)
    current, stale = [], []
    for name in sorted(os.listdir(RUNS)) if os.path.isdir(RUNS) else []:
        folder = os.path.join(RUNS, name)
        meta_path = os.path.join(folder, "meta.json")
        if not os.path.exists(meta_path) or (name.startswith(REFERENCE_RUNS) and not include_references):
            continue
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta["suite"] != suite_meta["name"]:
            continue
        if meta["suite_fingerprint"] != suite_meta["fingerprint"]:
            stale.append(name)
            continue
        summary, _ = grade_run(folder)
        if summary["overall"].get("n"):
            current.append((name, summary))
    current.sort(key=lambda entry: -entry[1]["overall"]["record_core"])

    text = f"Suite `{suite_meta['name']}` (`{suite_meta['fingerprint']}`, {suite_meta['cases']} cases)\n\n"
    if public and current:
        text += f"Agent contract {contracts_line(current)}.\n\n"
    text += (public_table if public else table)(current) if current else "No graded runs yet."
    if stale:
        text += "\n\nOn an older version of this suite (not comparable): " + ", ".join(stale)
    return text


if __name__ == "__main__":
    main()
