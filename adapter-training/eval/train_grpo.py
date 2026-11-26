#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import time
from types import SimpleNamespace

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import read_jsonl
from env import extraction_schema, tool_definitions
from grade import reward
from run import HFBackend, new_state, run_hf
from train_sft import grad_scaler, pick_dtype, precision


def completion_logprob(model, prompt_ids, completion_ids, device):
    ids = torch.tensor([prompt_ids + completion_ids], device=device)
    keep = len(completion_ids) + 1
    logits = model(input_ids=ids, logits_to_keep=keep).logits[:, :-1].float()
    target = torch.tensor([completion_ids], device=device)
    lp = torch.log_softmax(logits, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return lp.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--init-adapter", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--profiles")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--prompts", type=int, default=2, help="training samples per step")
    ap.add_argument("--group", type=int, default=4, help="episodes per sample")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--tool-tokens", type=int, default=120)
    ap.add_argument("--answer-tokens", type=int, default=450)
    ap.add_argument("--skip", type=int, default=0, help="skip the first N training samples (those SFT saw)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--dtype", default="auto",
                    help="auto (bf16; float16 on pre-Ampere GPUs such as the T4; float32 on mps), "
                         "bfloat16, float16 or float32")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = pick_dtype(args.device, args.dtype)
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device)
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    model = PeftModel.from_pretrained(base, args.init_adapter, is_trainable=True)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    scaler = grad_scaler(args.device, dtype)
    backend = HFBackend(model=model, tokenizer=tok)
    schema, tools = extraction_schema(), tool_definitions()

    gold = read_jsonl(args.gold)[args.skip:]
    gold = [g for g in gold if g["gold"]["isSpending"] or random.random() < 0.3]
    random.shuffle(gold)
    ledgers = {p["id"]: p["ledger"] for p in read_jsonl(args.profiles or args.gold.replace(".jsonl", "_profiles.jsonl"))}
    rollout_args = SimpleNamespace(progress=False, batch=args.prompts * args.group, max_rounds=args.max_rounds,
                                   tool_tokens=args.tool_tokens, answer_tokens=args.answer_tokens,
                                   temperature=args.temperature)
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "train_log.jsonl"), "w")
    started = time.time()
    for step in range(args.steps):
        samples = gold[step * args.prompts:(step + 1) * args.prompts]
        if len(samples) < args.prompts:
            break
        model.eval()
        states = [new_state(s, ledgers[s["profile"]], True, True) for s in samples for _ in range(args.group)]
        with torch.no_grad():
            run_hf(backend, states, rollout_args, tools, schema)
        rewards, details = [], []
        for st in states:
            r, sc = reward(st["sample"]["gold"], st.get("output", ""), st["episode"].trail)
            rewards.append(r)
            details.append(sc)

        advantages = []
        for gi in range(len(samples)):
            rs = rewards[gi * args.group:(gi + 1) * args.group]
            mean = sum(rs) / len(rs)
            std = math.sqrt(sum((x - mean) ** 2 for x in rs) / len(rs))
            advantages += [0.0 if std < 1e-6 else (x - mean) / (std + 1e-4) for x in rs]

        model.train()
        n_terms = sum(1 for a in advantages if a != 0.0) or 1
        for st, adv in zip(states, advantages):
            if adv == 0.0:
                continue
            gens = [g for g in st["generations"] if g.get("ids")]
            lps = []
            for g in gens:
                p_ids = tok(g["prompt"], add_special_tokens=False)["input_ids"]
                with precision(args.device, dtype):
                    lps.append(completion_logprob(model, p_ids, g["ids"], args.device))
            if not lps:
                continue
            loss = -adv * torch.stack(lps).mean() / n_terms
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        entry = {
            "step": step + 1,
            "reward_mean": round(sum(rewards) / len(rewards), 4),
            "reward_max": round(max(rewards), 4),
            "record_core": round(sum(d["record_core"] for d in details) / len(details), 3),
            "json_valid": round(sum(d["json_valid"] for d in details) / len(details), 3),
            "tool_calls": round(sum(len(st["episode"].trail) for st in states) / len(states), 2),
            "groups_with_signal": sum(1 for gi in range(len(samples))
                                      if any(advantages[gi * args.group:(gi + 1) * args.group])),
            "elapsed": round(time.time() - started, 1),
        }
        log.write(json.dumps(entry) + "\n")
        log.flush()
        print(json.dumps(entry), flush=True)
        if (step + 1) % 5 == 0:
            model.save_pretrained(args.out)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"saved adapter → {args.out}")


if __name__ == "__main__":
    main()
