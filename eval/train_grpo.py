#!/usr/bin/env python3
"""GRPO for the spending agent, starting from an SFT student's LoRA.

    python3 train_grpo.py --model /path/to/Qwen3.5-0.8B --init-adapter sft/student-sft \\
        --suite train-rl --out grpo/student-grpo --steps 80

Every step takes `--cases` cases from a train suite and runs `--group`
sessions on each with sampling. These are whole sessions of the real loop
(`run.py`'s `run_case`: a fresh ledger, the tools, compaction, the budget,
the answer), with the student generating in this process. Each session is
graded with `grade.reward`: the fields, -1 for no draft, the cost of wasted
calls, and the approval terms, -0.45 for a wrong draft confident enough to
be approved and +0.05 for a right one. The student is then pushed towards
the sessions that beat their group's mean:

    advantage(x)_i = (x_i - mean(x_group)) / (std(x_group) + eps)

The reward is split in two. Its approval term (`grade.approval_term`: the
bonus, the penalty or nothing) is what the confidence number decides, so its
advantage goes to the tokens that write that number and to nothing else; the
rest of the reward (the fields, no draft, wasted calls) goes to every other
completion token. With one advantage over the whole session, the
confidence would be credited with whatever the fields earned and move
nowhere, staying as high on wrong drafts as on right ones.

    loss = - sum over other tokens of A_fields_i * logprob / completion tokens
           - w * sum over sessions of A_approval_i * mean confidence-token logprob / sessions
           + beta * sum over tokens of KL_t / completion tokens

The first sum is averaged per token, not per completion: averaged per
completion, a one-token stop would carry its whole weight on the token that
decides it, which teaches a student to stop after one tool round. The confidence term is averaged per session, so a session's
confidence decision weighs as much as all its other tokens together (`w`,
`--confidence-weight`, 1); averaged per token, three digits among a
thousand tokens would teach almost nothing. The KL term (`--kl`, the k3
estimator, `exp(ref - logp) - (ref - logp) - 1`) holds the student near the
adapter it started from, loaded beside it as a frozen `reference` adapter;
without it, categories drift on real shops. LoRA dropout is off, so the policy that scores the tokens is the
one that sampled them. One update per batch of sessions, so there is no
ratio to clip. A group whose sessions all earned the same teaches nothing
through that part of the reward.

The answer is sampled under the grammar a served model answers under: vLLM
compiles the request's JSON schema with xgrammar (whitespace allowed), and
so does `AnswerGrammar` here, masking the logits of every answer row while
tool rounds stay free text. Sampled without it, most of the reward's rise
is the student learning to stay inside the schema, which a served model's
grammar does for it. An answer that still isn't a JSON object
within the schema (cut off at the token limit) is graded as no draft (-1).
`--no-grammar` samples without it.

Sessions run in threads and their requests are generated in batches
(`Batcher`): each session waits for its reply while the others' requests
join the same batch. A session's tool rounds share their prefix (each
prompt is the previous prompt, its completion and the tool answers), so
they are scored in one forward pass; the answer prompt adds the closing
turn, which re-renders the earlier turns, so it is scored on its own.

Training suites only: `dev`, `dev-novel` and `holdout` are refused, as in
`sft_data.py`, and a train case whose content one of them contains word for
word is left out. Writes `train_log.jsonl` (a line per step), `train_args.json`,
the adapter, and a checkpoint every `--save-every` steps that a rerun
resumes from. With `--keep-every N` the adapter of every N-th step, and of
the last, is also kept as `step-NNN/`, so a later step that trained past
the best one doesn't replace it: the candidates are scored afterwards on a
suite of their own.
"""

import argparse
import concurrent.futures
import json
import math
import os
import random
import re
import shutil
import threading
import time
from dataclasses import dataclass

from grade import NO_DRAFT, approval_term, reward, score_case
from run import run_case
from sft_data import FROZEN_SUITES, ChatTemplate, for_template
from suite import load_suite, suite_folder

from spending_agent import draft as contract
from spending_agent.backend import ChatBackend
from spending_agent.session import ANSWER_TOKENS, MAX_TOOL_ROUNDS, TOOL_TOKENS

END_OF_TURN = "<|im_end|>"
ADVANTAGE_EPSILON = 1e-4
TRAINED_ADAPTER = "default"
REFERENCE_ADAPTER = "reference"
CONFIDENCE = re.compile(r'("confidence"\s*:\s*)(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)')
ANSWER_END = re.compile(r"\s*\}\s*(?:" + re.escape(END_OF_TURN) + r")?\s*")
WHITESPACE_RUN_LIMIT = 32


@dataclass
class Generation:
    phase: str
    prompt_ids: list
    completion_ids: list
    confidence: list = None


def check_suite(meta):
    name = meta.get("name", "")
    if name in FROZEN_SUITES or not name.startswith("train"):
        raise SystemExit(f"suite {name!r} is not a train suite: training on it would spoil every score on it")


def without_frozen_content(cases):
    """The cases whose content no frozen suite contains. The generators draw
    non-spending messages from a small pool, so a train suite repeats some of
    `dev`'s word for word; training on those would score them from memory."""
    frozen = set()
    for name in FROZEN_SUITES:
        if os.path.exists(os.path.join(suite_folder(name), "meta.json")):
            frozen |= {case["content"] for case in load_suite(name)[1]}
    return [case for case in cases if case["content"] not in frozen]


ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def kept_steps(steps, keep_every):
    """The steps whose adapter is kept as `step-NNN/`: every `keep_every`-th
    and the last, which `steps` need not be a multiple of."""
    if not keep_every:
        return []
    return sorted({*range(keep_every, steps + 1, keep_every), steps})


def snapshot_folder(out, step):
    return os.path.join(out, f"step-{step:03d}")


def keep_snapshot(checkpoint, folder):
    """Copy the checkpoint's adapter to `folder`, whole or not at all: it
    is written beside it and renamed into place."""
    partial = folder + ".partial"
    shutil.rmtree(partial, ignore_errors=True)
    os.makedirs(partial)
    for name in ADAPTER_FILES:
        shutil.copy(os.path.join(checkpoint, name), partial)
    shutil.rmtree(folder, ignore_errors=True)
    os.replace(partial, folder)


def group_advantages(rewards, group):
    """Per session, its reward against its group's: `rewards` holds a group's
    sessions next to each other, and None for a session that ran no model
    (content rejected before the first call), whose group teaches nothing."""
    advantages = []
    for start in range(0, len(rewards), group):
        members = rewards[start:start + group]
        if any(r is None for r in members):
            advantages += [0.0] * len(members)
            continue
        mean = sum(members) / len(members)
        std = math.sqrt(sum((r - mean) ** 2 for r in members) / len(members))
        advantages += [0.0 if std < 1e-6 else (r - mean) / (std + ADVANTAGE_EPSILON) for r in members]
    return advantages


def chains(generations):
    """The token sequences that score a session's completions: a generation
    whose prompt starts with the sequence before it, token for token,
    extends that sequence; any other starts its own. Each is
    {"ids", "spans", "confidence"}, a span being the [start, end) of one
    completion and `confidence` the positions that write the answer's
    confidence number."""
    out = []
    for g in generations:
        if not g.completion_ids:
            continue
        if out and g.prompt_ids[:len(out[-1]["ids"])] == out[-1]["ids"]:
            chain = out[-1]
        else:
            chain = {"ids": [], "spans": [], "confidence": []}
            out.append(chain)
        start = len(g.prompt_ids)
        chain["ids"] = g.prompt_ids + g.completion_ids
        chain["spans"].append((start, len(chain["ids"])))
        chain["confidence"] += [start + k for k in g.confidence or []]
    return out


def confidence_positions(decode, completion_ids):
    """Which of an answer's tokens write its confidence number: none unless
    the number is the last field of a finished answer, since a cut-off
    answer is no draft whatever its confidence. Found from the end, where an
    answer is plain ASCII, so a suffix of the tokens decodes to a suffix of
    the text."""
    text = decode(completion_ids)
    found = list(CONFIDENCE.finditer(text))
    if not found or not ANSWER_END.fullmatch(text[found[-1].end():]):
        return []
    start, end = found[-1].span(2)
    positions, after = [], 0
    for k in range(len(completion_ids) - 1, -1, -1):
        suffix = decode(completion_ids[k:])
        if not text.endswith(suffix):
            return []
        token_start, token_end = len(text) - len(suffix), len(text) - after
        if token_start < end and token_end > start:
            positions.append(k)
        if token_start <= start:
            break
        after = len(suffix)
    return sorted(positions)


def token_weights(chain, field_advantage, approval_advantage, tokens, sessions, confidence_weight):
    """What each completion position's logprob is multiplied by in the loss
    (before the minus): the fields' advantage per token over the step's
    completion tokens, and on the confidence positions the approval
    advantage, split over them and averaged per session."""
    confidence = set(chain["confidence"])
    share = confidence_weight * approval_advantage / (max(1, len(confidence)) * sessions)
    return [share if t in confidence else field_advantage / tokens
            for start, end in chain["spans"] for t in range(start, end)]


def completion_tokens(session_chains):
    return sum(end - start for chain in session_chains for start, end in chain["spans"])


def outside_grammar(content):
    """Whether a served model's grammar would have forbidden this answer: it
    must be one JSON object, within the bounded schema."""
    import jsonschema
    try:
        answer = json.loads((content or "").strip())
    except json.JSONDecodeError:
        return True
    return next(jsonschema.Draft202012Validator(contract.answer_schema()).iter_errors(answer), None) is not None


def answer_text(transcript):
    answers = [e for e in transcript["events"] if e["kind"] == "model" and e.get("phase") == "answer"]
    return (answers[-1]["message"].get("content") or "") if answers else None


def episode_reward(case, user, row, transcript):
    """The session's reward, the part of it the approval decided, and its
    scores; None for a case rejected before any model call, which teaches
    nothing."""
    scores = score_case(case, row, user)
    if scores is None:
        return None, None, None
    content = answer_text(transcript)
    if scores["answered"] and content is not None and outside_grammar(content):
        return NO_DRAFT, 0.0, {**scores, "outside_grammar": True}
    return reward(scores, row), approval_term(scores), {**scores, "outside_grammar": False}


def split_advantages(rewards, approvals, group):
    """The advantages of the reward without its approval term, and of the
    approval term alone."""
    fields = [None if r is None else r - a for r, a in zip(rewards, approvals)]
    return group_advantages(fields, group), group_advantages(approvals, group)


class Batcher:
    """Requests from concurrent sessions, generated together. A batch is sent
    when it is full, when every live session is waiting in it, or `wait`
    seconds after its first request, whichever comes first."""

    def __init__(self, generate, batch_size, wait=0.5):
        self.generate, self.batch_size, self.wait = generate, batch_size, wait
        self.pending = []
        self.condition = threading.Condition()
        self.live = 0

    def submit(self, prompt_ids, max_tokens, schema=None):
        slot = {"prompt_ids": prompt_ids, "max_tokens": max_tokens, "schema": schema, "done": threading.Event()}
        with self.condition:
            self.pending.append(slot)
            self.condition.notify_all()
        slot["done"].wait()
        if "error" in slot:
            raise slot["error"]
        return slot["completion"]

    def started(self):
        with self.condition:
            self.live += 1

    def finished(self):
        with self.condition:
            self.live -= 1
            self.condition.notify_all()

    def serve(self):
        """Generate batches until no session is left. Runs on the caller's
        thread, so the model is only ever used from one."""
        while True:
            with self.condition:
                while not self.pending and self.live:
                    self.condition.wait(timeout=self.wait)
                if not self.pending and not self.live:
                    return
                deadline = time.time() + self.wait
                while (len(self.pending) < min(self.batch_size, self.live) and time.time() < deadline):
                    self.condition.wait(timeout=max(0.0, deadline - time.time()))
                batch, self.pending = self.pending[:self.batch_size], self.pending[self.batch_size:]
            try:
                completions = self.generate([(s["prompt_ids"], s["max_tokens"], s["schema"]) for s in batch])
                for slot, completion in zip(batch, completions):
                    slot["completion"] = completion
            except Exception as error:
                for slot in batch:
                    slot["error"] = error
            for slot in batch:
                slot["done"].set()


class PolicyBackend(ChatBackend):
    """The real client with the network replaced by the student: each body
    is rendered with the chat template the way a server renders it, handed
    to the batcher, and the reply returned as a server would return a
    template's text. Every generation is kept for training. With
    `constrain`, an answer asked for on a JSON schema is sampled under it."""

    def __init__(self, batcher, render, tokenizer, tokenizer_lock, constrain=False):
        super().__init__("http://policy/v1", "policy", reasoning_effort="none", one_request_at_a_time=False)
        self.batcher, self.render, self.tokenizer, self.tokenizer_lock = batcher, render, tokenizer, tokenizer_lock
        self.constrain = constrain
        with tokenizer_lock:
            self.end_of_turn = tokenizer.convert_tokens_to_ids(END_OF_TURN)
        self.generations = []

    def _send(self, body):
        messages = [for_template(m) for m in body["messages"]]
        prompt = self.render(messages, body.get("tools"), add_generation_prompt=True, enable_thinking=False)
        with self.tokenizer_lock:
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_format = body.get("response_format") or {}
        schema = None
        if self.constrain and "tools" not in body and response_format.get("type") == "json_schema":
            schema = response_format["json_schema"]["schema"]
        completion = self.batcher.submit(prompt_ids, body["max_tokens"], schema)
        ended = bool(completion) and completion[-1] == self.end_of_turn
        with self.tokenizer_lock:
            text = self.tokenizer.decode(completion[:-1] if ended else completion)
            confidence = None if "tools" in body else confidence_positions(self.tokenizer.decode, completion)
        self.generations.append(Generation("tools" if "tools" in body else "answer", prompt_ids, completion,
                                           confidence))
        return {"choices": [{"message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(completion)}}


def cut(row, max_tokens, end_of_turn):
    row = row[:max_tokens]
    return row[:row.index(end_of_turn) + 1] if end_of_turn in row else row


class AnswerGrammar:
    """The JSON schemas of answer requests, compiled the way vLLM compiles a
    `json_schema` response format (xgrammar, whitespace allowed), once each,
    with one difference: a run of whitespace between elements is capped at
    `WHITESPACE_RUN_LIMIT` characters. Sampled at temperature 1, an answer
    can fall into whitespace for good, and in a static batch that one row
    holds every other for the full answer budget; the student's own JSON
    needs a newline and an indent of a few spaces."""

    def __init__(self, tokenizer_info):
        import xgrammar
        self.xgrammar = xgrammar
        self.vocab_size = tokenizer_info.vocab_size
        self.compiler = xgrammar.GrammarCompiler(tokenizer_info)
        self.compiled = {}

    def compile(self, schema):
        """Compiled from the schema as written: the grammar keeps its key
        order, and `reasoning` must come first."""
        key = json.dumps(schema)
        if key not in self.compiled:
            self.compiled[key] = self.compiler.compile_json_schema(key, any_whitespace=True,
                                                                   max_whitespace_cnt=WHITESPACE_RUN_LIMIT)
        return self.compiled[key]

    def masker(self, schemas, prompt_width):
        return GrammarMask(self, [None if s is None else self.compile(s) for s in schemas], prompt_width)


class GrammarMask:
    """A logits processor for one batch: rows with a grammar have every
    token it forbids masked out, rows without one (tool rounds) are left
    alone, and a row whose grammar has ended is left to the generator, which
    pads it."""

    def __init__(self, grammar, compiled, prompt_width):
        x = grammar.xgrammar
        self.x, self.vocab_size, self.width, self.consumed = x, grammar.vocab_size, prompt_width, 0
        self.matchers = [None if c is None else x.GrammarMatcher(c) for c in compiled]
        self.bitmask = x.allocate_token_bitmask(len(compiled), self.vocab_size)

    def __call__(self, input_ids, scores):
        generated = input_ids.shape[1] - self.width
        rows = []
        for row, matcher in enumerate(self.matchers):
            if matcher is None or matcher.is_terminated():
                continue
            if generated > self.consumed:
                token = int(input_ids[row, -1])
                if not matcher.accept_token(token):
                    raise RuntimeError(f"row {row} sampled token {token}, which its grammar forbids")
                if matcher.is_terminated():
                    continue
            matcher.fill_next_token_bitmask(self.bitmask, row)
            rows.append(row)
        self.consumed = generated
        if rows:
            self.x.apply_token_bitmask_inplace(scores, self.bitmask.to(scores.device), vocab_size=self.vocab_size,
                                               indices=rows)
        return scores


def sampler(model, pad_id, end_of_turn, temperature, device, autocast, grammar=None):
    """Sampling at `temperature` from the plain softmax (no top-k, top-p or
    repetition penalty, whatever the checkpoint's defaults), so the tokens
    are drawn from the distribution the update scores them under; with
    `grammar`, each answer row within its schema, as a server samples it."""
    import torch
    from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList

    class Masking(LogitsProcessor):
        def __init__(self, mask):
            self.mask = mask

        def __call__(self, input_ids, scores):
            return self.mask(input_ids, scores)

    def generate(requests):
        width = max(len(p) for p, _, _ in requests)
        ids = torch.full((len(requests), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(requests), width), dtype=torch.long)
        for i, (prompt, _, _) in enumerate(requests):
            ids[i, width - len(prompt):] = torch.tensor(prompt)
            mask[i, width - len(prompt):] = 1
        config = GenerationConfig(do_sample=temperature > 0, temperature=temperature if temperature > 0 else None,
                                  top_p=1.0, top_k=0, repetition_penalty=1.0,
                                  max_new_tokens=max(m for _, m, _ in requests),
                                  eos_token_id=end_of_turn, pad_token_id=pad_id)
        schemas = [schema for _, _, schema in requests]
        processors = LogitsProcessorList()
        if grammar is not None and any(s is not None for s in schemas):
            processors.append(Masking(grammar.masker(schemas, width)))
        with torch.no_grad(), autocast():
            out = model.generate(input_ids=ids.to(device), attention_mask=mask.to(device), generation_config=config,
                                 logits_processor=processors)
        return [cut(row, m, end_of_turn) for row, (_, m, _) in zip(out[:, width:].tolist(), requests)]

    return generate


def rollouts(cases, users, group, batcher, make_backend, settings, threads):
    """`group` sessions per case, all at once; returns per session the case,
    the result row, the transcript and the generations, in case order."""
    jobs = [case for case in cases for _ in range(group)]

    def one(case):
        backend = make_backend()
        try:
            row, transcript = run_case(case, users[case["user"]], lambda _: backend, settings)
        finally:
            batcher.finished()
        return case, row, transcript, backend.generations

    for _ in jobs:
        batcher.started()
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [pool.submit(one, case) for case in jobs]
        batcher.serve()
        return [f.result() for f in futures]


def chain_logprobs(model, chain, device):
    """The logprob of each of a chain's completion tokens under the active
    adapter, in span order. Logits are kept from the first completion on,
    and the softmax taken only at completion positions: over a 248k
    vocabulary, the tool answers between completions would otherwise cost
    gigabytes."""
    import torch
    ids = chain["ids"]
    first = chain["spans"][0][0]
    logits = model(input_ids=torch.tensor([ids], device=device), logits_to_keep=len(ids) - first + 1,
                   use_cache=False).logits[0, :-1]
    positions = [t for start, end in chain["spans"] for t in range(start, end)]
    index = torch.tensor([t - first for t in positions], device=device)
    target = torch.tensor([ids[t] for t in positions], device=device)
    logp = torch.log_softmax(logits[index].float(), dim=-1)
    return logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)


def use_adapter(model, name):
    """Make `name` the active adapter while only the trained one keeps
    gradients: peft's `set_adapter` would make whichever is active
    trainable."""
    model.set_adapter(name)
    for parameter_name, parameter in model.named_parameters():
        if "lora_" in parameter_name:
            parameter.requires_grad_(f".{TRAINED_ADAPTER}." in parameter_name)


def mean_of(values, places=3):
    return round(sum(values) / len(values), places) if values else None


def step_entry(step, results, rewards, advantages, groups_with_signal, rollout_seconds, train_seconds, kl=None,
               confidence_sessions=0):
    scored = [scores for _, _, scores in results if scores is not None]
    drafts = [s for s in scored if s["answered"] and isinstance(s.get("confidence"), (int, float))]
    rows = [row for _, row, _ in results]
    n = len(scored) or 1
    rewards = [r for r in rewards if r is not None]
    return {
        "step": step,
        "reward_mean": round(sum(rewards) / len(rewards), 4) if rewards else None,
        "record_core": round(sum(s["record_core"] for s in scored) / n, 3),
        "record_all": round(sum(s["record_all"] for s in scored) / n, 3),
        "false_approval": round(sum(1 for s in scored if s["would_approve"] and not s["record_all"]) / n, 3),
        "right_approval": round(sum(1 for s in scored if s["would_approve"] and s["record_all"]) / n, 3),
        "no_draft": round(sum(1 for s in scored if not s["answered"]) / n, 3),
        "outside_grammar": round(sum(1 for s in scored if s["outside_grammar"]) / n, 3),
        "tool_calls": round(sum(r["tool_call_count"] for r in rows) / len(rows), 2),
        "rounds": round(sum(r["rounds"] for r in rows) / len(rows), 2),
        "wasted_calls": round(sum(r["wasted_calls"] for r in rows) / len(rows), 2),
        "budget_spent": round(sum(r["budget_spent"] for r in rows) / len(rows), 3),
        "confidence_right": mean_of([s["confidence"] for s in drafts if s["record_all"]]),
        "confidence_wrong": mean_of([s["confidence"] for s in drafts if not s["record_all"]]),
        "kl": None if kl is None else round(kl, 5),
        "confidence_sessions": confidence_sessions,
        "groups_with_signal": groups_with_signal,
        "sessions_trained": sum(1 for a in advantages if a),
        "rollout_seconds": round(rollout_seconds, 1),
        "train_seconds": round(train_seconds, 1),
    }


def case_order(cases, steps, per_step, seed):
    """The cases each step takes: the suite shuffled, then reshuffled for
    every further pass, the same for a given seed so a resumed run takes
    up where it left off."""
    rng = random.Random(seed)
    order = []
    while len(order) < steps * per_step:
        block = list(cases)
        rng.shuffle(block)
        order += block
    return [order[i * per_step:(i + 1) * per_step] for i in range(steps)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the base checkpoint the adapter was trained on")
    ap.add_argument("--init-adapter", required=True, help="the SFT LoRA adapter to continue")
    ap.add_argument("--suite", required=True, help="a train suite (suites/<name>)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--cases", type=int, default=8, help="cases per step")
    ap.add_argument("--group", type=int, default=8, help="sessions per case")
    ap.add_argument("--gen-batch", type=int, default=32, help="requests generated together")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl", type=float, default=0.04, help="weight of the KL term to the starting adapter; 0 drops it")
    ap.add_argument("--confidence-weight", type=float, default=1.0,
                    help="weight of a session's confidence decision against all its other tokens together")
    ap.add_argument("--no-grammar", action="store_true", help="sample answers without the schema grammar")
    ap.add_argument("--max-tool-rounds", type=int, default=MAX_TOOL_ROUNDS)
    ap.add_argument("--tool-tokens", type=int, default=TOOL_TOKENS)
    ap.add_argument("--answer-tokens", type=int, default=ANSWER_TOKENS)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--keep-every", type=int, default=0,
                    help="keep the adapter of every N-th step (and the last) as step-NNN/; 0 keeps only the last")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="auto")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from train_sft import grad_scaler, pick_dtype, precision

    meta, cases, users = load_suite(args.suite)
    check_suite(meta)
    cases = without_frozen_content(cases)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer_lock = threading.Lock()
    render = ChatTemplate.load(args.model)
    dtype = pick_dtype(device, args.dtype)
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    os.makedirs(args.out, exist_ok=True)
    checkpoint = os.path.join(args.out, "checkpoint")
    resuming = os.path.exists(os.path.join(checkpoint, "state.pt"))
    model = PeftModel.from_pretrained(base, checkpoint if resuming else args.init_adapter, is_trainable=True)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    if args.kl:
        model.load_adapter(args.init_adapter, adapter_name=REFERENCE_ADAPTER, is_trainable=False)
    use_adapter(model, TRAINED_ADAPTER)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    scaler = grad_scaler(device, dtype)
    first_step = 0
    if resuming:
        state = torch.load(os.path.join(checkpoint, "state.pt"), weights_only=False)
        opt.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        first_step = state["step"]
        print(f"resuming from step {first_step}", flush=True)
    kept = kept_steps(args.steps, args.keep_every)
    if resuming and first_step in kept and not os.path.exists(snapshot_folder(args.out, first_step)):
        keep_snapshot(checkpoint, snapshot_folder(args.out, first_step))
    with open(os.path.join(args.out, "train_args.json"), "w", encoding="utf-8") as f:
        json.dump({**vars(args), "suite_fingerprint": meta.get("fingerprint")}, f, indent=1)

    end_of_turn = tokenizer.convert_tokens_to_ids(END_OF_TURN)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else end_of_turn
    autocast = lambda: precision(device, dtype)
    grammar = None
    if not args.no_grammar:
        import xgrammar
        vocab_size = model.get_output_embeddings().weight.shape[0]
        grammar = AnswerGrammar(xgrammar.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size,
                                                                        stop_token_ids=[end_of_turn]))
    batcher = Batcher(sampler(model, pad_id, end_of_turn, args.temperature, device, autocast, grammar),
                      args.gen_batch)
    make_backend = lambda: PolicyBackend(batcher, render, tokenizer, tokenizer_lock, constrain=grammar is not None)
    settings = {"max_tool_rounds": args.max_tool_rounds, "tool_tokens": args.tool_tokens,
                "answer_tokens": args.answer_tokens}
    plan = case_order(cases, args.steps, args.cases, args.seed)
    log_path = os.path.join(args.out, "train_log.jsonl")
    kept_lines = []
    if resuming and os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as f:
            kept_lines = [line for line in f if line.strip() and json.loads(line)["step"] <= first_step]
    log = open(log_path, "w", encoding="utf-8")
    log.writelines(kept_lines)

    for step in range(first_step, args.steps):
        started = time.time()
        model.eval()
        sessions = rollouts(plan[step], users, args.group, batcher, make_backend, settings,
                            threads=args.cases * args.group)
        rollout_seconds = time.time() - started

        results, rewards, approvals = [], [], []
        for case, row, transcript, _ in sessions:
            value, approval, scores = episode_reward(case, users[case["user"]], row, transcript)
            results.append((case, row, scores))
            rewards.append(value)
            approvals.append(approval)
        field_advantages, approval_advantages = split_advantages(rewards, approvals, args.group)
        advantages = [f or a for f, a in zip(field_advantages, approval_advantages)]
        groups_with_signal = sum(1 for i in range(0, len(advantages), args.group) if any(advantages[i:i + args.group]))

        started = time.time()
        model.train()
        trained = [(f, a, chains(session[3])) for f, a, session in zip(field_advantages, approval_advantages, sessions)
                   if f or a]
        tokens = sum(completion_tokens(c) for _, _, c in trained) or 1
        kl_total = 0.0
        confidence_sessions = sum(1 for _, a, c in trained if a and any(chain["confidence"] for chain in c))
        for field_advantage, approval_advantage, session_chains in trained:
            for chain in session_chains:
                weights = torch.tensor(token_weights(chain, field_advantage, approval_advantage, tokens,
                                                     len(sessions), args.confidence_weight), device=device)
                reference = None
                if args.kl:
                    use_adapter(model, REFERENCE_ADAPTER)
                    with torch.no_grad(), autocast():
                        reference = chain_logprobs(model, chain, device)
                    use_adapter(model, TRAINED_ADAPTER)
                with autocast():
                    logp = chain_logprobs(model, chain, device)
                loss = -(weights * logp).sum()
                if reference is not None:
                    gap = reference - logp
                    kl = (torch.exp(gap) - gap - 1).sum()
                    loss = loss + args.kl * kl / tokens
                    kl_total += kl.item()
                scaler.scale(loss).backward()
        if trained:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
        opt.zero_grad(set_to_none=True)
        train_seconds = time.time() - started

        entry = step_entry(step + 1, results, rewards, advantages, groups_with_signal, rollout_seconds, train_seconds,
                           kl_total / tokens if args.kl and trained else None, confidence_sessions)
        log.write(json.dumps(entry) + "\n")
        log.flush()
        print(json.dumps(entry), flush=True)
        keeping = step + 1 in kept
        if keeping or args.save_every and (step + 1) % args.save_every == 0 and step + 1 < args.steps:
            model.save_pretrained(checkpoint, selected_adapters=[TRAINED_ADAPTER])
            torch.save({"optimizer": opt.state_dict(), "scaler": scaler.state_dict(), "step": step + 1},
                       os.path.join(checkpoint, "state.pt.tmp"))
            os.replace(os.path.join(checkpoint, "state.pt.tmp"), os.path.join(checkpoint, "state.pt"))
        if keeping:
            keep_snapshot(checkpoint, snapshot_folder(args.out, step + 1))
            print(f"kept step {step + 1} → {snapshot_folder(args.out, step + 1)}", flush=True)
    model.save_pretrained(args.out, selected_adapters=[TRAINED_ADAPTER])
    tokenizer.save_pretrained(args.out)
    print(f"saved adapter → {args.out}")


if __name__ == "__main__":
    main()
