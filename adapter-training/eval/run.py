#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
import urllib.request

from common import INSTRUCTIONS, read_jsonl, write_jsonl
from env import Episode, extraction_schema, tool_definitions

CLOSING = ("Now give the final answer as a single JSON object and nothing else. "
           "Fill the fields in the order they appear in the schema.")

_XML_CALL = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*(?:</tool_call>|$)", re.S)
_XML_PARAM = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)
_JSON_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse_tool_calls(text):
    calls = []
    for name, body in _XML_CALL.findall(text):
        args = {k.strip(): v for k, v in _XML_PARAM.findall(body)}
        calls.append({"name": name.strip(), "arguments": args})
    if not calls:
        for blob in _JSON_CALL.findall(text):
            try:
                obj = json.loads(blob)
                calls.append({"name": obj.get("name", ""), "arguments": obj.get("arguments", {})})
            except json.JSONDecodeError:
                pass
    return calls


def text_before_calls(text):
    i = text.find("<tool_call>")
    return (text[:i] if i >= 0 else text).strip()


def auto_dtype(device):
    import torch
    if str(device).startswith("cuda") and torch.cuda.get_device_capability(torch.device(device))[0] < 8:
        return "float16"
    return "bfloat16"


def default_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HFBackend:

    def __init__(self, model_path=None, adapter=None, dtype="auto", threads=None, model=None, tokenizer=None,
                 device=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        self.tok = tokenizer or AutoTokenizer.from_pretrained(model_path)
        self.tok.padding_side = "left"
        if model is not None:
            self.model = model
        else:
            device = device or default_device()
            if dtype == "auto":
                dtype = auto_dtype(device)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=getattr(torch, dtype))
            if adapter:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, adapter)
                self.model = self.model.merge_and_unload()
            self.model.eval()
            self.model.to(device)
        self.eos = [self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.eos_token_id]

    def render(self, messages, tools=None, prefix=""):
        text = self.tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
        return text + prefix

    def generate(self, prompts, max_new_tokens, stop=None, temperature=0.0, num_return_sequences=1, schema=None):
        torch = self.torch
        enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        kwargs = dict(max_new_tokens=max_new_tokens, eos_token_id=self.eos, pad_token_id=self.tok.pad_token_id,
                      num_return_sequences=num_return_sequences)
        if schema is not None:
            from transformers import LogitsProcessorList
            from constrained import ExtractionGrammar
            if getattr(self, "_grammar", None) is None:
                self._grammar = ExtractionGrammar(self.tok, schema, 0, self.eos)
            self._grammar.prompt_length = int(enc["input_ids"].shape[1])
            kwargs["logits_processor"] = LogitsProcessorList([self._grammar])
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            kwargs.update(do_sample=False)
        if stop:
            kwargs.update(stop_strings=stop, tokenizer=self.tok)
        with torch.no_grad():
            out = self.model.generate(**enc, **kwargs)
        new = out[:, enc["input_ids"].shape[1]:]
        texts = self.tok.batch_decode(new, skip_special_tokens=True)
        self.last_ids = []
        for row in new.tolist():
            kept = []
            for t in row:
                if t == self.tok.pad_token_id:
                    break
                kept.append(t)
                if t in self.eos:
                    break
            self.last_ids.append(kept)
        counts = [len(k) for k in self.last_ids]
        return texts, counts, int(enc["input_ids"].shape[1])


def describe_error(e):
    import urllib.error
    if isinstance(e, urllib.error.HTTPError):
        try:
            body = e.read().decode("utf-8", "replace")[:300]
        except Exception:
            body = ""
        return f"HTTP {e.code} {body}".strip()
    return f"{type(e).__name__}: {e}"


class OpenAIBackend:

    def __init__(self, base_url, model, api_key=None, timeout=600, template_kwargs=None):
        self.base, self.model, self.key, self.timeout = base_url.rstrip("/"), model, api_key, timeout
        self.template_kwargs = template_kwargs
        self.last_format = None

    def post(self, body):
        req = urllib.request.Request(self.base + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {self.key}"} if self.key else {})})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def chat(self, messages, tools=None, schema=None, max_tokens=800, temperature=0.0):
        body = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if self.template_kwargs:
            body["chat_template_kwargs"] = self.template_kwargs
        if tools:
            body.update(tools=tools, tool_choice="auto")
        if schema:
            errors = []
            for fmt in ({"type": "json_schema", "json_schema": {"name": "receipt_extraction", "strict": True, "schema": schema}},
                        {"type": "json_object"}, None):
                try:
                    if fmt:
                        body["response_format"] = fmt
                    else:
                        body.pop("response_format", None)
                    message = self.post(body)["choices"][0]["message"]
                    self.last_format = fmt["type"] if fmt else "none"
                    return message
                except Exception as e:
                    errors.append(f"{fmt['type'] if fmt else 'none'}: {describe_error(e)}")
            raise RuntimeError("server refused every response format — " + " | ".join(errors))
        return self.post(body)["choices"][0]["message"]


def new_state(sample, ledger, with_tools, with_ledger):
    return {
        "sample": sample,
        "episode": Episode(sample["content"], ledger, with_ledger=with_ledger),
        "messages": [{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": sample["prompt"]}],
        "phase": "tools" if with_tools else "final",
        "rounds": 0,
        "generations": [],
        "tokens": 0,
        "started": time.time(),
    }


def apply_tool_round(state, text, max_rounds, prompt=None, ids=None):
    calls = parse_tool_calls(text)
    state["generations"].append({"phase": "tools", "text": text, "prompt": prompt, "ids": ids})
    if not calls:
        state["phase"] = "final"
        return
    state["messages"].append({"role": "assistant", "content": text_before_calls(text),
                              "tool_calls": [{"type": "function", "function": c} for c in calls]})
    for c in calls:
        answer = state["episode"].call(c["name"], c["arguments"])
        state["messages"].append({"role": "tool", "name": c["name"], "content": answer})
    state["rounds"] += 1
    if state["rounds"] >= max_rounds or state["episode"].budget.is_spent:
        state["phase"] = "final"


def _report(progress, phase, done_batches, total_batches, finished, total, tokens, started):
    if not progress:
        return
    elapsed = time.time() - started
    rate = tokens / elapsed if elapsed else 0.0
    line = (f"[{phase}] batch {done_batches}/{total_batches} | {finished}/{total} samples done | "
            f"{tokens} tokens, {rate:.0f} tok/s | {elapsed / 60:.1f} min")
    if phase == "answer" and done_batches:
        line += f" | ~{elapsed / done_batches * (total_batches - done_batches) / 60:.1f} min of answers left"
    print(line, flush=True)


def run_hf(backend, states, args, tools, schema):
    progress = getattr(args, "progress", True)
    started, total = time.time(), len(states)
    pending = [s for s in states]
    tool_round = 0
    while any(s["phase"] == "tools" for s in pending):
        tool_round += 1
        active = [s for s in pending if s["phase"] == "tools"]
        active.sort(key=lambda s: len(json.dumps(s["messages"])))
        batches = (len(active) + args.batch - 1) // args.batch
        for b, i in enumerate(range(0, len(active), args.batch), 1):
            chunk = active[i:i + args.batch]
            prompts = [backend.render(s["messages"], tools=tools) for s in chunk]
            texts, counts, _ = backend.generate(prompts, args.tool_tokens, stop=["</tool_call>"],
                                                temperature=getattr(args, "temperature", 0.0))
            for s, t, n, p, ids in zip(chunk, texts, counts, prompts, backend.last_ids):
                s["tokens"] += n
                apply_tool_round(s, t, args.max_rounds, prompt=p, ids=ids)
            still = sum(s["phase"] == "tools" for s in pending)
            _report(progress, f"tools round {tool_round}, {still} still calling", b, batches, 0, total,
                    sum(s["tokens"] for s in states), started)
    finals = sorted(pending, key=lambda s: len(json.dumps(s["messages"])))
    batches = (len(finals) + args.batch - 1) // args.batch
    for b, i in enumerate(range(0, len(finals), args.batch), 1):
        chunk = finals[i:i + args.batch]
        prompts = [backend.render(s["messages"] + [{"role": "user", "content": CLOSING}], tools=None)
                   for s in chunk]
        texts, counts, _ = backend.generate(prompts, args.answer_tokens, schema=schema,
                                            temperature=getattr(args, "temperature", 0.0))
        for s, t, n, p, ids in zip(chunk, texts, counts, prompts, backend.last_ids):
            s["tokens"] += n
            s["output"] = t
            s["generations"].append({"phase": "answer", "text": s["output"], "prompt": p, "ids": ids})
            s["phase"] = "done"
        _report(progress, "answer", b, batches, sum(s["phase"] == "done" for s in states), total,
                sum(s["tokens"] for s in states), started)


def run_openai(backend, states, args, tools, schema):
    started = time.time()
    for n, s in enumerate(states, 1):
        t0 = time.time()
        try:
            while s["phase"] == "tools":
                msg = backend.chat(s["messages"], tools=tools, max_tokens=args.tool_tokens)
                calls = msg.get("tool_calls") or []
                s["generations"].append({"phase": "tools", "message": msg})
                if not calls:
                    s["phase"] = "final"
                    break
                s["messages"].append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
                for c in calls:
                    fn = c.get("function", {})
                    answer = s["episode"].call(fn.get("name", ""), fn.get("arguments", ""))
                    s["messages"].append({"role": "tool", "tool_call_id": c.get("id", ""), "name": fn.get("name", ""),
                                          "content": answer})
                s["rounds"] += 1
                if s["rounds"] >= args.max_rounds or s["episode"].budget.is_spent:
                    s["phase"] = "final"
            msg = backend.chat(s["messages"] + [{"role": "user", "content": CLOSING}], schema=schema,
                               max_tokens=args.answer_tokens)
            s["output"] = msg.get("content") or ""
            s["generations"].append({"phase": "answer", "text": s["output"], "message": msg,
                                     "response_format": backend.last_format})
            status = f"format {backend.last_format}"
        except Exception as e:
            s["output"] = ""
            s["error"] = describe_error(e)
            s["generations"].append({"phase": "error", "error": s["error"]})
            status = f"FAILED: {s['error'][:160]}"
        s["phase"] = "done"
        elapsed = time.time() - started
        print(f"[{n}/{len(states)}] {s['sample']['id']}: {len(s['episode'].trail)} tool calls, "
              f"{time.time() - t0:.0f} s, {status} | ~{elapsed / n * (len(states) - n) / 60:.0f} min left", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "openai"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", help="hf: a PEFT/LoRA adapter directory")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--api-key")
    ap.add_argument("--timeout", type=int, default=600, help="openai: seconds per request")
    ap.add_argument("--no-think", action="store_true",
                    help="openai: send chat_template_kwargs enable_thinking=false (Qwen-family templates on "
                         "llama-server/vLLM); the app's answer carries its reasoning in the JSON instead")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--profiles", help="default: <gold>_profiles.jsonl")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"))
    ap.add_argument("--limit", type=int, help="first N samples per home locale, round-robin")
    ap.add_argument("--ids", help="file with one sample id per line")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--device", help="cuda, mps (Apple Silicon) or cpu; default: the best available")
    ap.add_argument("--dtype", default="auto",
                    help="auto (bfloat16; float16 on pre-Ampere GPUs such as the T4), bfloat16, float16 or float32")
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--tool-tokens", type=int, default=160)
    ap.add_argument("--answer-tokens", type=int, default=600)
    ap.add_argument("--no-tools", action="store_true", help="the app's tool-less retry path")
    ap.add_argument("--no-ledger", action="store_true", help="offer only calculator and searchReceipt")
    args = ap.parse_args()

    gold = read_jsonl(args.gold)
    if args.ids:
        wanted = [l.strip() for l in open(args.ids) if l.strip()]
        gold = [g for g in gold if g["id"] in set(wanted)]
    elif args.limit:
        by_home = {}
        for g in gold:
            by_home.setdefault(g["homeLocale"], []).append(g)
        per = max(1, args.limit // len(by_home))
        gold = [g for rows in by_home.values() for g in rows[:per]]
    profiles_path = args.profiles or args.gold.replace(".jsonl", "_profiles.jsonl")
    ledgers = {p["id"]: p["ledger"] for p in read_jsonl(profiles_path)}

    with_tools, with_ledger = not args.no_tools, not args.no_ledger
    tools = tool_definitions(with_ledger=with_ledger) if with_tools else None
    states = [new_state(g, ledgers[g["profile"]], with_tools, with_ledger) for g in gold]

    started = time.time()
    if args.backend == "hf":
        backend = HFBackend(args.model, adapter=args.adapter, threads=args.threads, device=args.device,
                            dtype=args.dtype)
        run_hf(backend, states, args, tools, extraction_schema())
    else:
        backend = OpenAIBackend(args.base_url, args.model, args.api_key, timeout=args.timeout,
                                template_kwargs={"enable_thinking": False} if args.no_think else None)
        run_openai(backend, states, args, tools, extraction_schema())
    elapsed = time.time() - started

    out = os.path.join(args.out_dir, args.name)
    os.makedirs(out, exist_ok=True)
    write_jsonl(os.path.join(out, "predictions.jsonl"), [
        {"id": s["sample"]["id"], "output": s.get("output", ""), "trail": s["episode"].trail,
         "tokens": s["tokens"]} for s in states])
    write_jsonl(os.path.join(out, "transcripts.jsonl"), [
        {"id": s["sample"]["id"], "messages": s["messages"],
         "generations": [{k: v for k, v in g.items() if k not in ("prompt", "ids")} for g in s["generations"]],
         "trail": s["episode"].trail, **({"error": s["error"]} if "error" in s else {})} for s in states])
    meta = {"model": args.model, "adapter": args.adapter, "backend": args.backend, "samples": len(states),
            "seconds": round(elapsed, 1), "seconds_per_sample": round(elapsed / max(1, len(states)), 1),
            "tools": with_tools, "ledger_tool": with_ledger, "max_rounds": args.max_rounds,
            "tool_tokens": args.tool_tokens, "answer_tokens": args.answer_tokens,
            "failed_requests": sum("error" in s for s in states)}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta))

    import grade
    preds = read_jsonl(os.path.join(out, "predictions.jsonl"))
    summary, per = grade.grade(gold, preds)
    with open(os.path.join(out, "grade.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_sample": per}, f, ensure_ascii=False, indent=1)
    print()
    grade.print_table(summary)
    print(f"\nscores → {os.path.join(out, 'grade.json')}   compare: python3 compare.py <other run> {args.name}")


if __name__ == "__main__":
    main()
