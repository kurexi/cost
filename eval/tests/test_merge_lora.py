import importlib.util
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import merge_lora

CHECKPOINT = [
    "model.language_model.embed_tokens.weight",
    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
    "model.language_model.layers.0.linear_attn.out_proj.weight",
    "model.language_model.layers.3.self_attn.q_proj.weight",
    "model.language_model.layers.3.mlp.down_proj.weight",
    "model.visual.blocks.0.attn.proj.weight",
    "model.visual.blocks.0.mlp.linear_fc1.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.mlp.down_proj.weight",
]

TRAINER = ["base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight",
           "base_model.model.model.layers.3.self_attn.q_proj.lora_B.weight",
           "base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_A.weight",
           "base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_B.weight"]


class KeyMappingTests(unittest.TestCase):
    def test_the_text_models_layer_maps_onto_the_language_model_in_the_checkpoint(self):
        self.assertEqual(merge_lora.checkpoint_key("model.layers.3.self_attn.q_proj", CHECKPOINT),
                         "model.language_model.layers.3.self_attn.q_proj.weight")
        self.assertEqual(merge_lora.checkpoint_key("model.layers.0.linear_attn.in_proj_qkv", CHECKPOINT),
                         "model.language_model.layers.0.linear_attn.in_proj_qkv.weight")

    def test_the_mtp_head_and_vision_tower_are_never_targets(self):
        self.assertEqual(merge_lora.checkpoint_key("model.layers.0.mlp.down_proj",
                                                   CHECKPOINT + ["model.language_model.layers.0.mlp.down_proj.weight"]),
                         "model.language_model.layers.0.mlp.down_proj.weight")

    def test_a_module_matching_nothing_stops_the_merge(self):
        with self.assertRaises(SystemExit):
            merge_lora.checkpoint_key("model.layers.9.self_attn.q_proj", CHECKPOINT)

    def test_a_module_outside_the_layers_stops_the_merge(self):
        with self.assertRaises(SystemExit):
            merge_lora.checkpoint_key("lm_head", CHECKPOINT)

    def test_adapter_weights_pair_up_by_module(self):
        modules = merge_lora.adapted_modules(TRAINER)
        self.assertEqual(sorted(modules), ["model.layers.0.linear_attn.in_proj_qkv", "model.layers.3.self_attn.q_proj"])
        self.assertTrue(modules["model.layers.3.self_attn.q_proj"][0].endswith("lora_A.weight"))

    def test_a_lone_half_of_a_pair_stops_the_merge(self):
        with self.assertRaises(SystemExit):
            merge_lora.adapted_modules(TRAINER[:3])

    def test_scaling_is_alpha_over_rank_or_its_root(self):
        self.assertEqual(merge_lora.scaling({"r": 16, "lora_alpha": 32}), 2.0)
        self.assertEqual(merge_lora.scaling({"r": 16, "lora_alpha": 32, "use_rslora": True}), 8.0)


def has(*modules):
    return all(importlib.util.find_spec(m) for m in modules)


@unittest.skipUnless(has("torch", "transformers", "peft", "safetensors"), "needs torch, transformers, peft")
class MergeMatchesPeftTests(unittest.TestCase):

    def test_merged_checkpoint_equals_peft_merge(self):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

        folder = tempfile.mkdtemp()
        base, adapter, out = (os.path.join(folder, n) for n in ("base", "adapter", "merged"))
        config = AutoConfig.for_model(
            "qwen3_5", text_config=dict(hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                                        num_attention_heads=2, num_key_value_heads=1, head_dim=32,
                                        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16,
                                        linear_value_head_dim=16, vocab_size=512,
                                        layer_types=["linear_attention"] * 3 + ["full_attention"]),
            vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64))
        torch.manual_seed(0)
        AutoModelForImageTextToText.from_config(config).to(torch.bfloat16).save_pretrained(base)
        targets = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b",
                   "out_proj", "gate_proj", "up_proj", "down_proj"]
        trained = get_peft_model(AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16),
                                 LoraConfig(r=4, lora_alpha=8, target_modules=targets, task_type="CAUSAL_LM"))
        for name, parameter in trained.named_parameters():
            if "lora_B" in name:
                parameter.data.normal_(0, 0.5)
        trained.save_pretrained(adapter)
        expected = trained.merge_and_unload().state_dict()

        report = merge_lora.merge(base, adapter, out)
        self.assertEqual((report["merged"], report["modules"]), (31, 31))
        with open(os.path.join(out, "config.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["architectures"], ["Qwen3_5ForConditionalGeneration"])
        merged = AutoModelForCausalLM.from_pretrained(out, dtype=torch.bfloat16).state_dict()
        self.assertEqual(set(merged), set(expected))
        for key, value in expected.items():
            bound = 0.02 * max(1.0, value.float().abs().max().item())
            self.assertLessEqual((merged[key].float() - value.float()).abs().max().item(), bound, key)


if __name__ == "__main__":
    unittest.main()
