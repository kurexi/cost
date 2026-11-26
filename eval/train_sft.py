#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

def pick_dtype(device, requested="auto"):
    if requested != "auto":
        return getattr(torch, requested)
    if str(device).startswith("mps"):
        return torch.float32
    if str(device).startswith("cuda") and torch.cuda.get_device_capability(torch.device(device))[0] < 8:
        return torch.float16
    return torch.bfloat16


def grad_scaler(device, dtype):
    return torch.amp.GradScaler(str(device).split(":")[0], enabled=dtype == torch.float16)


def precision(device, dtype):
    import contextlib
    if dtype == torch.float32:
        return contextlib.nullcontext()
    return torch.autocast(device_type=str(device).split(":")[0], dtype=dtype)


TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b",
           "out_proj", "gate_proj", "up_proj", "down_proj"]


def load_sequences(tok, path, max_len, kinds=None, end_tools_fraction=1.0, rng=random):
    seqs = []
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        for pair in row["pairs"]:
            if kinds and pair["kind"] not in kinds:
                continue
            if pair["kind"] == "end_tools" and rng.random() >= end_tools_fraction:
                continue
            p = tok(pair["prompt"], add_special_tokens=False)["input_ids"]
            c, trained = completion_targets(tok, pair)
            ids = (p + c)[-max_len:]
            n_c = min(len(c), len(ids))
            labels = [-100] * (len(ids) - n_c) + [t if keep else -100 for t, keep in zip(c, trained)][len(c) - n_c:]
            seqs.append({"id": row["id"], "kind": pair["kind"], "input_ids": ids, "labels": labels})
    return seqs


def completion_targets(tok, pair):
    spans = pair.get("train_spans")
    if not spans:
        ids = tok(pair["completion"], add_special_tokens=False)["input_ids"]
        return ids, [True] * len(ids)
    encoded = tok(pair["completion"], add_special_tokens=False, return_offsets_mapping=True)
    trained = [any(start < end_span and end > start_span for start_span, end_span in spans)
               for start, end in encoded["offset_mapping"]]
    if not any(trained):
        raise ValueError(f"no token of the completion falls in {spans}")
    return encoded["input_ids"], trained


def completion_nll(model, seq, device):
    ids = torch.tensor([seq["input_ids"]], device=device)
    n_completion = completion_length(seq)
    tail = scored_tail(seq)
    logits = model(input_ids=ids, logits_to_keep=tail + 1).logits[:, :-1].float()
    target = torch.tensor([seq["labels"][-tail:]], device=device)
    mask = target != -100
    logp = torch.log_softmax(logits, dim=-1)
    tok_lp = logp.gather(-1, target.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return -(tok_lp * mask).sum(), n_completion


def completion_length(seq):
    return max(1, sum(1 for x in seq["labels"] if x != -100))


def scored_tail(seq):
    first = next((i for i, x in enumerate(seq["labels"]) if x != -100), len(seq["labels"]) - 1)
    return len(seq["labels"]) - first


LOSS_MODES = ("sequence", "token")


def loss_weights(completion_lengths, mode):
    if mode == "sequence":
        return [1 / (n * len(completion_lengths)) for n in completion_lengths]
    if mode == "token":
        return [1 / sum(completion_lengths)] * len(completion_lengths)
    raise ValueError(f"unknown loss mode {mode!r}: use one of {LOSS_MODES}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--max-steps", type=int, help="optimizer steps cap (for a smoke run)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--dtype", default="auto",
                    help="auto (bf16; float16 on pre-Ampere GPUs such as the T4; float32 on mps), "
                         "bfloat16, float16 or float32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss", choices=LOSS_MODES, default="sequence",
                    help="average the completion loss per sequence (default) or per token")
    ap.add_argument("--end-tools-fraction", type=float, default=1.0,
                    help="keep this share of the end-of-tool-phase sequences")
    ap.add_argument("--no-checkpointing", action="store_true", help="faster, more memory")
    ap.add_argument("--init-adapter", help="continue training this LoRA adapter")
    ap.add_argument("--save-every", type=int, default=4,
                    help="checkpoint (adapter + optimizer + step) every N steps; a rerun resumes from it")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = pick_dtype(args.device, args.dtype)
    print(f"device {args.device}, dtype {dtype}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device)
    if not args.no_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    if args.init_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                                 target_modules=TARGETS, task_type="CAUSAL_LM"))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    seqs = load_sequences(tok, args.data, args.max_len, end_tools_fraction=args.end_tools_fraction,
                          rng=random.Random(args.seed))
    total_seqs = int(len(seqs) * args.epochs)
    order = []
    while len(order) < total_seqs:
        block = list(range(len(seqs)))
        random.shuffle(block)
        order += block
    order = order[:total_seqs]
    steps = math.ceil(len(order) / args.accum)
    if args.max_steps:
        steps = min(steps, args.max_steps)
        order = order[:steps * args.accum]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    scaler = grad_scaler(args.device, dtype)
    warmup = max(1, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps)))))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "train_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=1)
    ckpt = os.path.join(args.out, "checkpoint")
    first_step = 0
    if os.path.exists(os.path.join(ckpt, "state.pt")):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model, load_file(os.path.join(ckpt, "adapter_model.safetensors")))
        state = torch.load(os.path.join(ckpt, "state.pt"), weights_only=False)
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        first_step = state["step"]
        print(f"resuming from step {first_step}", flush=True)
    log = open(os.path.join(args.out, "train_log.jsonl"), "a" if first_step else "w")
    model.train()
    started = time.time()
    tokens = 0
    for step in range(first_step, steps):
        batch = order[step * args.accum:(step + 1) * args.accum]
        losses = {}
        weights = loss_weights([completion_length(seqs[i]) for i in batch], args.loss)
        for i, weight in zip(batch, weights):
            with precision(args.device, dtype):
                nll, n_completion = completion_nll(model, seqs[i], args.device)
            scaler.scale(nll * weight).backward()
            losses.setdefault(seqs[i]["kind"], []).append(nll.item() / n_completion)
            tokens += len(seqs[i]["input_ids"])
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        opt.zero_grad(set_to_none=True)
        entry = {"step": step + 1, "steps": steps, "lr": sched.get_last_lr()[0],
                 "loss": sum(sum(v) for v in losses.values()) / len(batch),
                 **{f"loss_{k}": sum(v) / len(v) for k, v in losses.items()},
                 "elapsed": round(time.time() - started, 1), "tok_per_s": round(tokens / (time.time() - started), 1)}
        log.write(json.dumps(entry) + "\n")
        log.flush()
        print(json.dumps(entry), flush=True)
        if args.save_every and (step + 1) % args.save_every == 0 and step + 1 < steps:
            model.save_pretrained(ckpt)
            torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "step": step + 1,
                        "scaler": scaler.state_dict()},
                       os.path.join(ckpt, "state.pt.tmp"))
            os.replace(os.path.join(ckpt, "state.pt.tmp"), os.path.join(ckpt, "state.pt"))
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"saved adapter → {args.out}")


if __name__ == "__main__":
    main()
