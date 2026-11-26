#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import struct

import numpy as np

QK = 64
BLOCK_BYTES = 2 + QK // 4


def read_gguf(path, keep_arrays=("prism.",)):
    f = open(path, "rb")
    types = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def u(fmt):
        return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]

    def s():
        return f.read(u("Q")).decode("utf-8", "replace")

    def val(t, keep):
        if t in types:
            return u(types[t])
        if t == 8:
            return s()
        et, n = u("I"), u("Q")
        items = [val(et, keep) for _ in range(n)]
        return items if keep else f"<{n}>"

    assert f.read(4) == b"GGUF"
    u("I")
    n_tensors, n_kv = u("Q"), u("Q")
    meta = {}
    for _ in range(n_kv):
        k = s()
        meta[k] = val(u("I"), k.startswith(keep_arrays))
    tensors = {}
    for _ in range(n_tensors):
        name = s()
        dims = [u("Q") for _ in range(u("I"))]
        tensors[name] = (dims, u("I"), u("Q"))
    align = meta.get("general.alignment", 32)
    start = (f.tell() + align - 1) // align * align
    return meta, tensors, start


def load_tensor(path, start, dims, ttype, offset):
    n = int(np.prod(dims))
    with open(path, "rb") as f:
        f.seek(start + offset)
        if ttype == 0:
            arr = np.frombuffer(f.read(4 * n), dtype="<f4")
        elif ttype == 1:
            arr = np.frombuffer(f.read(2 * n), dtype="<f2").astype(np.float32)
        elif ttype == 42:
            raw = np.frombuffer(f.read(n // QK * BLOCK_BYTES), dtype=np.uint8).reshape(-1, BLOCK_BYTES)
            d = raw[:, :2].copy().view("<f2").astype(np.float32)
            qs = raw[:, 2:]
            codes = np.stack([(qs >> (2 * k)) & 3 for k in range(4)], axis=-1).reshape(-1, QK)
            arr = ((codes.astype(np.int8) - 1).astype(np.float32) * d).reshape(-1)
        else:
            raise ValueError(f"unsupported ggml type {ttype}")
    return arr.reshape(list(reversed(dims)))


def fwht_blocks(x, block):
    shape = x.shape
    y = x.reshape(-1, shape[-1] // block, block).astype(np.float32).copy()
    h = 1
    while h < block:
        y = y.reshape(y.shape[0], y.shape[1], block // (2 * h), 2, h)
        a, b = y[..., 0, :].copy(), y[..., 1, :].copy()
        y[..., 0, :], y[..., 1, :] = a + b, a - b
        y = y.reshape(y.shape[0], y.shape[1], block)
        h *= 2
    return (y / np.float32(np.sqrt(block))).reshape(shape)


def fold(w, signs, block, order):
    signs = signs.astype(np.float32)
    out = np.empty(w.shape, dtype=np.float32)
    rows = w.reshape(-1, w.shape[-1])
    view = out.reshape(-1, w.shape[-1])
    for i in range(0, rows.shape[0], 16384):
        chunk = rows[i:i + 16384]
        if order == "hs":
            view[i:i + 16384] = fwht_blocks(chunk, block) * signs
        else:
            view[i:i + 16384] = fwht_blocks(chunk * signs, block)
    return out


def hf_name(gguf_name, prefix):
    if gguf_name == "token_embd.weight":
        return f"{prefix}embed_tokens.weight"
    if gguf_name == "output_norm.weight":
        return f"{prefix}norm.weight"
    _, i, rest = gguf_name.split(".", 2)
    L = f"{prefix}layers.{i}."
    table = {
        "attn_norm.weight": "input_layernorm.weight",
        "post_attention_norm.weight": "post_attention_layernorm.weight",
        "ffn_gate.weight": "mlp.gate_proj.weight", "ffn_up.weight": "mlp.up_proj.weight",
        "ffn_down.weight": "mlp.down_proj.weight",
        "attn_q.weight": "self_attn.q_proj.weight", "attn_k.weight": "self_attn.k_proj.weight",
        "attn_v.weight": "self_attn.v_proj.weight", "attn_output.weight": "self_attn.o_proj.weight",
        "attn_q_norm.weight": "self_attn.q_norm.weight", "attn_k_norm.weight": "self_attn.k_norm.weight",
        "attn_qkv.weight": "linear_attn.in_proj_qkv.weight", "attn_gate.weight": "linear_attn.in_proj_z.weight",
        "ssm_out.weight": "linear_attn.out_proj.weight", "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
        "ssm_beta.weight": "linear_attn.in_proj_b.weight", "ssm_conv1d.weight": "linear_attn.conv1d.weight",
        "ssm_dt.bias": "linear_attn.dt_bias", "ssm_a": "linear_attn.A_log", "ssm_norm.weight": "linear_attn.norm.weight",
    }
    return L + table[rest]


def corr(a, b):
    a = a.reshape(-1)[:4_000_000].astype(np.float64)
    b = b.reshape(-1)[:4_000_000].astype(np.float64)
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--base", required=True, help="Qwen3.5-2B (config, tokenizer, layout, check)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--order", choices=["auto", "hs", "sh"], default="auto")
    args = ap.parse_args()

    import torch
    from safetensors.torch import load_file, save_file

    meta, tensors, start = read_gguf(args.gguf)
    block = int(meta["prism.hadamard.block_size"])
    widths = [int(w) for w in meta["prism.hadamard.sign_widths"]]
    flat = np.array(meta["prism.hadamard.sign_values"], dtype=np.float64)
    signs, off = {}, 0
    for w in widths:
        signs[w] = flat[off:off + w]
        off += w
    rotated = set(meta["prism.hadamard.weight_names"]) | set(meta["prism.hadamard.inverse_weight_names"])
    print(f"block {block}, sign widths {widths}, {len(rotated)} rotated tensors, "
          f"transform {meta['prism.hadamard.transform']}, signs {set(np.unique(flat))}")

    base = {}
    for fn in sorted(os.listdir(args.base)):
        if fn.endswith(".safetensors"):
            base.update(load_file(os.path.join(args.base, fn)))
    prefix = next(k for k in base if k.endswith("embed_tokens.weight")).replace("embed_tokens.weight", "")
    print(f"base tensors: {len(base)}, text prefix: {prefix!r}")

    def unpack(name, order):
        dims, ttype, offset = tensors[name]
        w = load_tensor(args.gguf, start, dims, ttype, offset)
        if name in rotated:
            w = fold(w, signs[w.shape[-1]], block, order)
        return w

    order = args.order
    if order == "auto":
        probe = "blk.0.ffn_up.weight"
        ref = base[hf_name(probe, prefix)].float().numpy()
        scores = {o: corr(unpack(probe, o), ref) for o in ("hs", "sh")}
        order = max(scores, key=scores.get)
        print(f"sign order probe on {probe}: {scores} → {order}")

    out, report = {}, []
    for name, (dims, ttype, offset) in tensors.items():
        if name == "output.weight":
            continue
        try:
            key = hf_name(name, prefix)
        except KeyError:
            print("unmapped", name)
            continue
        ref = base[key].float().numpy()
        w = unpack(name, order)
        if name.endswith("ssm_a"):
            w = np.log(-w)
        elif name.endswith(("attn_norm.weight", "post_attention_norm.weight", "q_norm.weight", "k_norm.weight")) \
                or name == "output_norm.weight":
            if abs(corr(w - 1.0, ref)) >= abs(corr(w, ref)) and np.abs(w - 1 - ref).mean() < np.abs(w - ref).mean():
                w = w - 1.0
        w = w.reshape(ref.shape)
        out[key] = torch.from_numpy(np.ascontiguousarray(w)).to(base[key].dtype)
        report.append((name, key, ttype, round(corr(w, ref), 4)))
    missing = [k for k in base if k not in out and k.startswith(prefix)]
    for k in missing:
        out[k] = base[k]
    for k in base:
        if k not in out:
            out[k] = base[k]

    if args.verify:
        lows = [r for r in report if r[3] < 0.5]
        by_kind = {}
        for name, key, ttype, c in report:
            by_kind.setdefault(name.split(".", 2)[-1], []).append(c)
        for kind, cs in sorted(by_kind.items()):
            print(f"  {kind:28} n={len(cs):3}  corr min {min(cs):.3f}  mean {sum(cs) / len(cs):.3f}")
        print(f"{len(lows)} tensors below 0.5 correlation", [l[0] for l in lows[:10]])

    os.makedirs(args.out, exist_ok=True)
    save_file(out, os.path.join(args.out, "model.safetensors"), metadata={"format": "pt"})
    for fn in os.listdir(args.base):
        if fn.endswith((".json", ".jinja", ".txt", ".model")) and "safetensors.index" not in fn:
            shutil.copy(os.path.join(args.base, fn), os.path.join(args.out, fn))
    with open(os.path.join(args.out, "bonsai_unpack.json"), "w") as f:
        json.dump({"source": os.path.basename(args.gguf), "order": order, "block": block,
                   "correlations": report}, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
