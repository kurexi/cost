#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import urllib.request
from dataclasses import dataclass

from suite import RUNS, load_suite, read_jsonl

from spending_agent import draft as contract
from spending_agent.backend import answer_ladder, tool_calls_of, without_thinking
from spending_agent.session import _assistant_turn, compact

FROZEN_SUITES = ("dev", "dev-novel", "dev-varied", "holdout")
END_OF_TURN = "<|im_end|>"
CONFIDENCE = re.compile(r'("confidence"\s*:\s*)(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)')


@dataclass
class Request:
    phase: str
    messages: list
    tools: object
    reply: dict
    response_format: object = None

    @property
    def kind(self):
        if self.phase == "answer":
            return "answer"
        return "tool_call" if tool_calls_of(self.reply) else "end_tools"


class ReplayMismatch(Exception):
    pass


def replay(transcript):
    final = transcript.get("messages") or []
    if len(final) < 2:
        return []
    budget = transcript["config"]["context_character_budget"]
    messages = [dict(m) for m in final[:2]]
    requests, waiting = [], []
    for event in transcript["events"]:
        if event["kind"] == "model":
            if waiting:
                raise ReplayMismatch(f"{len(waiting)} tool calls had no answer before the next model call")
            messages = compact(messages, budget)
            reply = event["message"]
            if event.get("phase") == "tools":
                requests.append(Request("tools", list(messages), contract.TOOL_DEFINITIONS, reply))
                calls = tool_calls_of(reply)
                if calls:
                    messages.append(_assistant_turn(reply, calls))
                    waiting = list(calls)
            else:
                label = event.get("response_format")
                closing = messages + [{"role": "user", "content": contract.CLOSING}]
                ladder = {name: prompt for name, _, prompt in answer_ladder(closing, contract.answer_schema())}
                if label not in ladder:
                    raise ReplayMismatch(f"the answer used {label!r}, which is not a rung of the format ladder")
                requests.append(Request("answer", ladder[label], None, reply, label))
        elif event["kind"] == "tool":
            if not waiting:
                raise ReplayMismatch("a tool answer that no call asked for")
            call = waiting.pop(0)
            if call["name"] != event["name"]:
                raise ReplayMismatch(f"call {call['name']!r} was answered as {event['name']!r}")
            messages.append({"role": "tool", "tool_call_id": call["id"], "name": call["name"],
                             "content": event["answer"]})
    if messages != final:
        raise ReplayMismatch("the replayed conversation doesn't end on the transcript's")
    return requests


def for_template(message):
    if not message.get("tool_calls"):
        return message
    calls = []
    for call in message["tool_calls"]:
        function = dict(call.get("function") or {})
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
        function["arguments"] = arguments if isinstance(arguments, dict) else {}
        calls.append({**call, "function": function})
    return {**message, "tool_calls": calls}


def target_turn(request, end_tools_text="empty"):
    reply = request.reply
    if request.kind == "tool_call":
        return _assistant_turn(reply, tool_calls_of(reply))
    if request.kind == "end_tools" and end_tools_text == "empty":
        return {"role": "assistant", "content": ""}
    return {"role": "assistant", "content": without_thinking(reply.get("content") or "")}


class TemplateMismatch(Exception):
    pass


def pair(render, request, end_tools_text="empty"):
    messages = [for_template(m) for m in request.messages]
    prompt = render(messages, request.tools, True)
    full = render(messages + [for_template(target_turn(request, end_tools_text))], request.tools, False)
    if not full.startswith(prompt):
        raise TemplateMismatch("the conversation with the reply doesn't start with the prompt")
    rest = full[len(prompt):]
    end = rest.rfind(END_OF_TURN)
    if end < 0 or rest[end + len(END_OF_TURN):].strip():
        raise TemplateMismatch("the reply doesn't end the conversation with an end-of-turn token")
    return {"kind": request.kind, "prompt": prompt, "completion": rest[:end + len(END_OF_TURN)]}


class NoConfidence(Exception):
    pass


def confidence_match(text):
    found = list(CONFIDENCE.finditer(text))
    if len(found) != 1:
        raise NoConfidence(f"{len(found)} confidence values in the answer")
    return found[0]


def doubt_pair(render, request, confidence):
    content = without_thinking(request.reply.get("content") or "")
    found = confidence_match(content)
    value = f"{confidence:g}"
    rewritten = content[:found.start(2)] + value + content[found.end(2):]
    doubted = Request(request.phase, request.messages, request.tools, {**request.reply, "content": rewritten},
                      request.response_format)
    built = pair(render, doubted)
    found = confidence_match(built["completion"])
    if found.group(2) != value:
        raise NoConfidence("the rendered answer doesn't carry the replaced confidence")
    return {**built, "kind": "doubt", "train_spans": [list(found.span(2))]}


def _tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def _raise(message):
    raise TemplateMismatch(message)


class ChatTemplate:

    def __init__(self, source):
        from jinja2.ext import loopcontrols
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                    extensions=[loopcontrols])
        environment.filters["tojson"] = _tojson
        environment.globals["raise_exception"] = _raise
        self.source = source
        self.template = environment.from_string(source)

    @classmethod
    def load(cls, where):
        if os.path.isfile(where):
            return cls(_template_text(where))
        if os.path.isdir(where):
            for name in ("chat_template.jinja", "tokenizer_config.json"):
                path = os.path.join(where, name)
                if os.path.exists(path):
                    return cls(_template_text(path))
            raise SystemExit(f"no chat template in {where}")
        url = f"https://huggingface.co/{where}/resolve/main/chat_template.jinja"
        with urllib.request.urlopen(url, timeout=60) as response:
            return cls(response.read().decode("utf-8"))

    def __call__(self, messages, tools=None, add_generation_prompt=False, enable_thinking=False):
        return self.template.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt,
                                    enable_thinking=enable_thinking)


def _template_text(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if path.endswith(".json"):
        template = json.loads(text).get("chat_template")
        if not isinstance(template, str):
            raise SystemExit(f"{path} has no single chat_template")
        return template
    return text


def run_suite(folder):
    with open(os.path.join(folder, "meta.json"), encoding="utf-8") as f:
        return json.load(f)


def kept_case_ids(folder):
    with open(os.path.join(folder, "grade.json"), encoding="utf-8") as f:
        per_case = json.load(f)["per_case"]
    return {row["id"] for row in per_case if row.get("record_all")}


def doubted_case_ids(folder):
    with open(os.path.join(folder, "grade.json"), encoding="utf-8") as f:
        per_case = json.load(f)["per_case"]
    return {row["id"] for row in per_case if row.get("answered") and not row.get("record_all")}


def check_transcripts(name, folder, transcripts, graded):
    missing = set(graded) - {transcript["id"] for transcript in transcripts}
    if missing:
        raise SystemExit(f"{name}: {len(missing)} of its {len(graded)} graded sessions have no transcript in "
                         f"{os.path.join(folder, 'transcripts.jsonl')}. The pairs are built from the transcripts, "
                         f"so copy that file back, or run the teacher again under a new name")


def frozen_contents():
    contents = set()
    for name in FROZEN_SUITES:
        try:
            contents |= {case["content"] for case in load_suite(name)[1]}
        except SystemExit:
            continue
    return contents


def frozen_case_ids(cases, frozen):
    return {case["id"] for case in cases if case["content"] in frozen}


def run_cases(meta):
    suite_meta, cases, _ = load_suite(meta["suite"])
    if suite_meta.get("fingerprint") != meta.get("suite_fingerprint"):
        raise SystemExit(f"run {meta.get('name')} was made on {meta['suite']} {meta.get('suite_fingerprint')}, "
                         f"but suites/{meta['suite']} is {suite_meta.get('fingerprint')}: regenerate it as it was")
    return cases


def combined(all_stats):
    total = {}
    for stats in all_stats:
        for key, value in stats.items():
            if key == "pairs":
                pairs = total.setdefault("pairs", {})
                for kind, n in value.items():
                    pairs[kind] = pairs.get(kind, 0) + n
            else:
                total[key] = total.get(key, 0) + value
    return total


def check_run(meta):
    if meta.get("suite") in FROZEN_SUITES or not str(meta.get("suite", "")).startswith("train"):
        raise SystemExit(f"run {meta.get('name')} is on suite {meta.get('suite')!r}: train only on a train suite, "
                         "never on dev, dev-novel or holdout")
    if meta.get("contract") != contract.contract_fingerprint():
        raise SystemExit(f"run {meta.get('name')} used contract {meta.get('contract')}, not the current "
                         f"{contract.contract_fingerprint()}: its prompts can't be rebuilt from this code")
    if meta.get("reasoning_effort") != "none":
        raise SystemExit(f"run {meta.get('name')} ran with reasoning_effort {meta.get('reasoning_effort')!r}; "
                         "the pairs are built with thinking off")


def build(transcripts, keep, render, end_tools_text="empty", count_tokens=None, max_len=None,
          doubt_ids=(), doubt_confidence=None, frozen_ids=()):
    rows, stats = [], {"sessions": 0, "kept": 0, "doubted": 0, "frozen_content": 0, "not_record_all": 0,
                       "replay_mismatch": 0,
                       "template_mismatch": 0, "no_confidence": 0, "too_long": 0, "pairs": {}, "characters": 0,
                       "tokens": 0}
    for transcript in transcripts:
        stats["sessions"] += 1
        if transcript["id"] in frozen_ids:
            stats["frozen_content"] += 1
            continue
        doubted = transcript["id"] not in keep and doubt_confidence is not None and transcript["id"] in doubt_ids
        if transcript["id"] not in keep and not doubted:
            stats["not_record_all"] += 1
            continue
        try:
            requests = replay(transcript)
        except ReplayMismatch:
            stats["replay_mismatch"] += 1
            continue
        if doubted:
            requests = [r for r in requests if r.kind == "answer"][-1:]
        pairs = []
        for request in requests:
            try:
                built = doubt_pair(render, request, doubt_confidence) if doubted else pair(render, request,
                                                                                              end_tools_text)
            except TemplateMismatch:
                stats["template_mismatch"] += 1
                continue
            except NoConfidence:
                stats["no_confidence"] += 1
                continue
            if count_tokens:
                length = count_tokens(built["prompt"] + built["completion"])
                if max_len and length > max_len:
                    stats["too_long"] += 1
                    continue
                stats["tokens"] += length
            stats["characters"] += len(built["prompt"]) + len(built["completion"])
            stats["pairs"][built["kind"]] = stats["pairs"].get(built["kind"], 0) + 1
            pairs.append(built)
        if pairs:
            stats["doubted" if doubted else "kept"] += 1
            rows.append({"id": transcript["id"], "pairs": pairs})
    return rows, stats


def first_difference(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def verify_template(render, tokenizer, transcripts, sessions=20):
    checked = 0
    for transcript in transcripts[:sessions]:
        for request in replay(transcript):
            messages = [for_template(m) for m in request.messages]
            ours = render(messages, request.tools, True)
            theirs = tokenizer.apply_chat_template(messages, tools=request.tools, add_generation_prompt=True,
                                                   tokenize=False, enable_thinking=False)
            if ours != theirs:
                at = first_difference(ours, theirs)
                raise SystemExit(f"the template renders {transcript['id']} differently from transformers at "
                                 f"character {at}: {ours[at - 40:at + 40]!r} against {theirs[at - 40:at + 40]!r}")
            checked += 1
    return checked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="teacher runs on train suites, under runs/")
    ap.add_argument("--template", required=True,
                    help="the student's chat template: a model folder, a template file or a Hugging Face id")
    ap.add_argument("--out", required=True)
    ap.add_argument("--end-tools-text", choices=("empty", "teacher"), default="empty")
    ap.add_argument("--doubt-confidence", type=float,
                    help="also make `doubt` pairs from the sessions that drafted a wrong record, teaching only "
                         "this confidence on them")
    ap.add_argument("--tokenizer", help="a model folder or id: count tokens and drop pairs over --max-len")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--verify-template", action="store_true",
                    help="check the renderer against transformers' apply_chat_template (needs --tokenizer)")
    args = ap.parse_args()
    if args.verify_template and not args.tokenizer:
        raise SystemExit("--verify-template needs --tokenizer")

    folders = [run if os.path.isdir(run) else os.path.join(RUNS, run) for run in args.runs]
    metas = [run_suite(folder) for folder in folders]
    for meta in metas:
        check_run(meta)
    render = ChatTemplate.load(args.template)
    count_tokens = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        count_tokens = lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"])
    frozen = frozen_contents()
    rows, per_run = [], []
    for folder, meta in zip(folders, metas):
        transcripts = read_jsonl(os.path.join(folder, "transcripts.jsonl"))
        keep = kept_case_ids(folder)
        check_transcripts(meta["name"], folder, transcripts, keep | doubted_case_ids(folder))
        if args.verify_template and not per_run:
            kept = [t for t in transcripts if t["id"] in keep]
            print(f"template agrees with transformers on {verify_template(render, tokenizer, kept)} requests")
        run_rows, stats = build(transcripts, keep, render, args.end_tools_text, count_tokens,
                                args.max_len if count_tokens else None, doubted_case_ids(folder),
                                args.doubt_confidence, frozen_case_ids(run_cases(meta), frozen))
        rows += run_rows
        per_run.append(stats)
        print(f"{meta['name']}: {stats['kept']} kept, {stats['doubted']} doubted, "
              f"{stats['frozen_content']} left out for frozen content, of {stats['sessions']}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats = combined(per_run)
    stats["runs"] = [{"run": meta["name"], "suite": meta["suite"], "suite_fingerprint": meta["suite_fingerprint"],
                      **{key: run[key] for key in ("sessions", "kept", "doubted", "frozen_content")}}
                     for meta, run in zip(metas, per_run)]
    stats["contract"], stats["end_tools_text"], stats["max_len"] = metas[0]["contract"], args.end_tools_text, args.max_len
    stats["doubt_confidence"] = args.doubt_confidence
    with open(os.path.splitext(args.out)[0] + ".stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1))
    if not rows:
        raise SystemExit("no pairs: nothing to train on")


if __name__ == "__main__":
    sys.exit(main())
