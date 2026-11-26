#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import sys

ADAPTER_PREFIX = "base_model.model."
NOT_THE_TEXT_MODEL = ("visual", "mtp")


def adapted_modules(adapter_keys):
    modules = {}
    for name in adapter_keys:
        for side in ("lora_A", "lora_B"):
            marker = f".{side}."
            if marker in name:
                module = name.split(marker)[0]
                if module.startswith(ADAPTER_PREFIX):
                    module = module[len(ADAPTER_PREFIX):]
                modules.setdefault(module, {})[side] = name
    incomplete = [m for m, sides in modules.items() if len(sides) != 2]
    if incomplete:
        raise SystemExit(f"adapter weights without their pair: {incomplete[:3]}")
    return {m: (sides["lora_A"], sides["lora_B"]) for m, sides in modules.items()}


def checkpoint_key(module, checkpoint_keys):
    if "layers." not in module:
        raise SystemExit(f"{module} is not inside a decoder layer; this merge only maps layer weights")
    suffix = "layers." + module.split("layers.", 1)[1] + ".weight"
    matches = [k for k in checkpoint_keys
               if (k == suffix or k.endswith("." + suffix)) and not any(part in k.split(".") for part in NOT_THE_TEXT_MODEL)]
    if len(matches) != 1:
        raise SystemExit(f"{module} matches {len(matches)} checkpoint weights ({matches[:3]}), not one")
    return matches[0]


def scaling(config):
    rank, alpha = config["r"], config.get("lora_alpha", config["r"])
    if config.get("use_rslora"):
        return alpha / rank ** 0.5
    return alpha / rank


def shards(folder):
    index = os.path.join(folder, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index, encoding="utf-8") as f:
            return sorted(set(json.load(f)["weight_map"].values()))
    names = sorted(n for n in os.listdir(folder) if n.endswith(".safetensors"))
    if not names:
        raise SystemExit(f"no safetensors in {folder}")
    return names


def merge(base, adapter, out):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    with open(os.path.join(adapter, "adapter_config.json"), encoding="utf-8") as f:
        config = json.load(f)
    if config.get("fan_in_fan_out"):
        raise SystemExit("fan_in_fan_out adapters are not supported")
    scale = scaling(config)
    lora = load_file(os.path.join(adapter, "adapter_model.safetensors"))
    modules = adapted_modules(lora)

    names = shards(base)
    keys_by_shard = {}
    for name in names:
        with safe_open(os.path.join(base, name), "pt") as f:
            keys_by_shard[name] = list(f.keys())
    all_keys = [k for keys in keys_by_shard.values() for k in keys]
    targets = {checkpoint_key(module, all_keys): pair for module, pair in modules.items()}
    if len(targets) != len(modules):
        raise SystemExit("two adapted modules map onto the same checkpoint weight")

    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    for entry in os.listdir(base):
        source = os.path.join(base, entry)
        if entry in names or not os.path.isfile(source):
            continue
        shutil.copy2(source, os.path.join(out, entry))
    merged, largest = 0, 0.0
    for name in names:
        with safe_open(os.path.join(base, name), "pt") as f:
            metadata = f.metadata() or {"format": "pt"}
            tensors = {k: f.get_tensor(k) for k in keys_by_shard[name]}
        for key in keys_by_shard[name]:
            if key not in targets:
                continue
            a_key, b_key = targets[key]
            weight = tensors[key]
            delta = (lora[b_key].float() @ lora[a_key].float()) * scale
            if delta.shape != weight.shape:
                raise SystemExit(f"{key}: the adapter's update is {tuple(delta.shape)}, the weight "
                                 f"{tuple(weight.shape)}")
            tensors[key] = (weight.float() + delta).to(weight.dtype)
            largest = max(largest, delta.abs().max().item())
            merged += 1
        save_file(tensors, os.path.join(out, name), metadata=metadata)
    return {"merged": merged, "modules": len(modules), "scale": scale, "largest_update": largest,
            "rank": config["r"], "alpha": config.get("lora_alpha")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="the base model's Hugging Face folder")
    ap.add_argument("--adapter", required=True, help="the folder train_sft.py saved")
    ap.add_argument("--out", required=True, help="the merged model's folder (replaced if it exists)")
    args = ap.parse_args()
    report = merge(args.base, args.adapter, args.out)
    with open(os.path.join(args.out, "merge.json"), "w", encoding="utf-8") as f:
        json.dump({"base": args.base, "adapter": args.adapter, **report}, f, indent=1)
    print(json.dumps(report, indent=1))
    if report["merged"] != report["modules"]:
        raise SystemExit("not every adapted module was merged")


if __name__ == "__main__":
    sys.exit(main())
