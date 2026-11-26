import importlib.util
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def has(*modules):
    return all(importlib.util.find_spec(m) for m in modules)


@unittest.skipUnless(has("torch", "transformers", "peft"), "needs torch, transformers, peft")
class LossWeightTests(unittest.TestCase):

    def setUp(self):
        import train_sft
        self.weights = train_sft.loss_weights

    def test_per_sequence_every_sequence_weighs_the_same_whatever_its_length(self):
        lengths = [1, 25, 300]
        totals = [w * n for w, n in zip(self.weights(lengths, "sequence"), lengths)]
        for total in totals:
            self.assertAlmostEqual(total, 1 / 3)

    def test_per_sequence_a_stop_token_outweighs_the_token_that_decides_to_go_on(self):
        stop, go_on = self.weights([1, 25], "sequence")
        self.assertAlmostEqual(stop / go_on, 25)

    def test_per_token_every_token_weighs_the_same(self):
        lengths = [1, 25, 300]
        weights = self.weights(lengths, "token")
        self.assertEqual(len(set(weights)), 1)
        self.assertAlmostEqual(sum(w * n for w, n in zip(weights, lengths)), 1)

    def test_an_unknown_mode_is_refused(self):
        with self.assertRaises(ValueError):
            self.weights([1, 2], "pair")


class CharTokenizer:

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        encoded = {"input_ids": [ord(c) % 90 + 1 for c in text]}
        if return_offsets_mapping:
            encoded["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return encoded


@unittest.skipUnless(has("torch", "transformers", "peft"), "needs torch, transformers, peft")
class SpanTests(unittest.TestCase):

    def setUp(self):
        import json
        import tempfile
        import train_sft
        self.train_sft = train_sft
        completion = '{"merchant": "Wrong", "confidence": 0.5}'
        start = completion.index("0.5")
        pairs = [{"kind": "answer", "prompt": "P:", "completion": '{"confidence": 0.9}'},
                 {"kind": "doubt", "prompt": "P:", "completion": completion, "train_spans": [[start, start + 3]]}]
        self.path = os.path.join(tempfile.mkdtemp(), "pairs.jsonl")
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"id": "c", "pairs": pairs}) + "\n")
        self.answer, self.doubt = train_sft.load_sequences(CharTokenizer(), self.path, max_len=100)

    def targets(self, seq):
        return "".join(chr(0) if x == -100 else "x" for x in seq["labels"])

    def test_a_pair_without_spans_trains_its_whole_completion(self):
        self.assertEqual(self.targets(self.answer), chr(0) * 2 + "x" * len('{"confidence": 0.9}'))

    def test_a_doubt_pair_trains_only_the_tokens_in_its_spans(self):
        self.assertEqual(self.train_sft.completion_length(self.doubt), 3)
        trained = [i for i, x in enumerate(self.doubt["labels"]) if x != -100]
        self.assertEqual(trained, list(range(len("P:") + 36, len("P:") + 39)))

    def test_logits_reach_back_to_the_first_target_not_just_the_last_n_tokens(self):
        self.assertEqual(self.train_sft.scored_tail(self.doubt), 4)

    def test_the_nll_is_the_targets_nll_under_the_full_forward(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        torch.manual_seed(0)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=96, hidden_size=16, intermediate_size=32,
                                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2))
        nll, n = self.train_sft.completion_nll(model, self.doubt, "cpu")
        logits = model(input_ids=torch.tensor([self.doubt["input_ids"]])).logits[0, :-1]
        targets = torch.tensor(self.doubt["labels"][1:])
        keep = targets != -100
        expected = -torch.log_softmax(logits, -1)[keep].gather(-1, targets[keep].unsqueeze(-1)).sum()
        self.assertEqual(n, 3)
        self.assertAlmostEqual(nll.item(), expected.item(), places=4)

    def test_spans_that_cover_no_token_are_refused(self):
        with self.assertRaises(ValueError):
            self.train_sft.completion_targets(CharTokenizer(), {"completion": "abc", "train_spans": [[5, 6]]})


if __name__ == "__main__":
    unittest.main()
