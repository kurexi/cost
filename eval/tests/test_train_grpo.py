import importlib.util
import json
import os
import sys
import tempfile
import threading
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import grade
import make_suite
import train_grpo
from sft_data import ChatTemplate
from suite import load_suite

TEMPLATE = os.path.join(HERE, "fixtures", "qwen3.5-chat_template.jinja")
END = 1_000_000
CALL = ("I'll look the shop up.\n\n<tool_call>\n<function=queryMerchants>\n<parameter=pattern>\nSHOP\n"
        "</parameter>\n</function>\n</tool_call>")


class CharTokenizer:
    """One id per character, and `<|im_end|>` as one id, the way a real
    tokenizer keeps a special token whole: enough for prompts to extend
    each other token for token."""

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for i, piece in enumerate(text.split(train_grpo.END_OF_TURN)):
            if i:
                ids.append(END)
            ids += [ord(c) for c in piece]
        return {"input_ids": ids}

    def decode(self, ids):
        return "".join(train_grpo.END_OF_TURN if i == END else chr(i) for i in ids)

    def convert_tokens_to_ids(self, token):
        return END


def answer_for(case, confidence=0.9, **changes):
    gold = case["gold"]
    answer = {"reasoning": "A receipt; the total line holds the amount.", "isSpending": gold["isSpending"],
              "merchant": gold["merchant"], "paymentInstrument": gold.get("paymentInstrument", ""),
              "totalAmount": gold["amountMinorUnits"] / 100, "currencyCode": gold["currencyCode"],
              "purchaseDate": gold["purchaseDate"], "purchaseTime": gold["purchaseTime"],
              "category": (gold["categoryAccepted"] or [""])[0], "proposedCategory": gold["proposedCategory"],
              "tags": gold["tags"], "proposedTag": "", "note": "", "confidence": confidence}
    return {**answer, **changes}


class ScriptedPolicy:
    """Stands in for the model: a tool round asks for one merchant lookup,
    the next ends the phase, the answer is the given JSON. It records the
    size of every batch it is handed."""

    def __init__(self, tokenizer, answer):
        self.tokenizer, self.answer, self.batches, self.schemas = tokenizer, answer, [], []
        self.lock = threading.Lock()

    def __call__(self, requests):
        with self.lock:
            self.batches.append(len(requests))
            self.schemas.append([schema is not None for _, _, schema in requests])
        out = []
        for prompt_ids, _, _ in requests:
            prompt = self.tokenizer.decode(prompt_ids)
            if "# Tools" not in prompt:
                text = json.dumps(self.answer)
            elif "<tool_response>" not in prompt:
                text = CALL
            else:
                text = ""
            out.append(self.tokenizer(text)["input_ids"] + [END])
        return out


def spending_cases():
    cases, users = make_suite.build(seed=3, per_locale=6, users_per_locale=2)
    users = {u["id"]: u for u in users}
    picked = [c for c in cases if c["gold"]["isSpending"] and c["gold"]["categoryAccepted"]
              and c["expectedStatus"] == "needs_review"][:2]
    return picked, users


def run_rollouts(cases, users, group, answer, constrain=True):
    tokenizer = CharTokenizer()
    policy = ScriptedPolicy(tokenizer, answer)
    batcher = train_grpo.Batcher(policy, batch_size=64, wait=2.0)
    lock = threading.Lock()
    render = ChatTemplate.load(TEMPLATE)
    make_backend = lambda: train_grpo.PolicyBackend(batcher, render, tokenizer, lock, constrain=constrain)
    settings = {"max_tool_rounds": 10, "tool_tokens": 800, "answer_tokens": 1500}
    return train_grpo.rollouts(cases, users, group, batcher, make_backend, settings, threads=len(cases) * group), policy


class RolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases, cls.users = spending_cases()
        cls.sessions, cls.policy = run_rollouts(cls.cases, cls.users, 3, None)

    def test_every_session_runs_the_real_loop(self):
        self.assertEqual(len(self.sessions), 6)
        for case, row, transcript, generations in self.sessions:
            self.assertEqual([g.phase for g in generations], ["tools", "tools", "answer"])
            self.assertEqual(row["tool_calls"], {"queryMerchants": 1})
            self.assertEqual(transcript["messages"][2]["tool_calls"][0]["function"]["name"], "queryMerchants")

    def test_sessions_waiting_together_are_generated_together(self):
        self.assertEqual(self.policy.batches, [6, 6, 6])

    def test_only_the_answer_is_sampled_under_its_schema(self):
        self.assertEqual(self.policy.schemas, [[False] * 6, [False] * 6, [True] * 6])
        _, unconstrained = run_rollouts(self.cases[:1], self.users, 1, None, constrain=False)
        self.assertEqual(unconstrained.schemas, [[False], [False], [False]])

    def test_only_the_answer_is_searched_for_a_confidence(self):
        for _, _, _, generations in self.sessions:
            self.assertEqual(generations[-1].confidence, [])
            self.assertTrue(all(g.confidence is None for g in generations[:-1]))

    def test_sessions_come_back_in_case_order(self):
        self.assertEqual([s[0]["id"] for s in self.sessions], [c["id"] for c in self.cases for _ in range(3)])

    def test_the_tool_rounds_share_one_sequence_and_the_answer_has_its_own(self):
        generations = self.sessions[0][3]
        built = train_grpo.chains(generations)
        self.assertEqual([len(c["spans"]) for c in built], [2, 1])
        for chain in built:
            self.assertEqual(chain["ids"][:chain["spans"][0][0]], generations[0 if len(chain["spans"]) == 2 else 2]
                             .prompt_ids)
        scored = [chain["ids"][start:end] for chain in built for start, end in chain["spans"]]
        self.assertEqual(scored, [g.completion_ids for g in generations])
        self.assertEqual(train_grpo.completion_tokens(built), sum(len(g.completion_ids) for g in generations))


class RewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases, cls.users = spending_cases()
        cls.case = cls.cases[0]

    def reward_for(self, answer):
        sessions, _ = run_rollouts([self.case], self.users, 1, answer)
        case, row, transcript, generations = sessions[0]
        value, approval, scores = train_grpo.episode_reward(case, self.users[case["user"]], row, transcript)
        self.generations, self.approval = generations, approval
        return value, scores

    def test_a_right_confident_draft_earns_the_approval_bonus(self):
        sure, scores = self.reward_for(answer_for(self.case, 0.9))
        held, _ = self.reward_for(answer_for(self.case, 0.5))
        self.assertTrue(scores["record_all"] and scores["would_approve"])
        self.assertAlmostEqual(sure - held, grade.RIGHT_APPROVAL_BONUS)

    def test_the_approval_term_is_reported_apart_from_the_rest(self):
        self.reward_for(answer_for(self.case, 0.9))
        self.assertAlmostEqual(self.approval, grade.RIGHT_APPROVAL_BONUS)
        wrong = self.case["gold"]["amountMinorUnits"] / 100 + 1
        self.reward_for(answer_for(self.case, 0.9, totalAmount=wrong))
        self.assertAlmostEqual(self.approval, -grade.FALSE_APPROVAL_PENALTY)
        self.reward_for(answer_for(self.case, 0.5, totalAmount=wrong))
        self.assertEqual(self.approval, 0.0)

    def test_the_confidence_tokens_are_the_number_in_the_answer(self):
        self.reward_for(answer_for(self.case, 0.85))
        answer = self.generations[-1]
        self.assertEqual(CharTokenizer().decode([answer.completion_ids[k] for k in answer.confidence]), "0.85")

    def test_a_wrong_confident_draft_costs_the_false_approval_penalty(self):
        wrong = self.case["gold"]["amountMinorUnits"] / 100 + 1
        sure, scores = self.reward_for(answer_for(self.case, 0.9, totalAmount=wrong))
        held, _ = self.reward_for(answer_for(self.case, 0.5, totalAmount=wrong))
        self.assertTrue(scores["would_approve"] and not scores["record_all"])
        self.assertAlmostEqual(held - sure, grade.FALSE_APPROVAL_PENALTY)
        self.assertLess(sure, held)

    def test_approving_pays_only_above_the_precision_bar(self):
        at_bar = (grade.APPROVAL_PRECISION * grade.RIGHT_APPROVAL_BONUS
                  - (1 - grade.APPROVAL_PRECISION) * grade.FALSE_APPROVAL_PENALTY)
        self.assertAlmostEqual(at_bar, 0.0)

    def test_an_answer_a_grammar_would_forbid_is_no_draft(self):
        value, scores = self.reward_for(answer_for(self.case, 0.5, reasoning="x" * 701))
        self.assertTrue(scores["answered"] and scores["outside_grammar"])
        self.assertEqual(value, grade.NO_DRAFT)
        self.assertEqual(self.approval, 0.0)


class GrammarTests(unittest.TestCase):
    def setUp(self):
        case = spending_cases()[0][0]
        self.answer = answer_for(case)

    def test_an_answer_within_the_schema_passes(self):
        self.assertFalse(train_grpo.outside_grammar(json.dumps(self.answer)))

    def test_text_around_the_object_is_outside(self):
        self.assertTrue(train_grpo.outside_grammar("```json\n" + json.dumps(self.answer) + "\n```"))

    def test_a_broken_bound_is_outside(self):
        for change in ({"currencyCode": "gbp"}, {"purchaseDate": "2 May"}, {"merchant": "M" * 81}):
            self.assertTrue(train_grpo.outside_grammar(json.dumps({**self.answer, **change})), change)


class AdvantageTests(unittest.TestCase):
    def test_a_group_with_equal_rewards_teaches_nothing(self):
        self.assertEqual(train_grpo.group_advantages([0.5, 0.5, 1.0, 0.0], 2)[:2], [0.0, 0.0])

    def test_advantages_are_centred_and_scaled_within_a_group(self):
        a = train_grpo.group_advantages([1.0, 0.0, 1.0, 0.0], 4)
        self.assertAlmostEqual(sum(a), 0.0)
        self.assertAlmostEqual(a[0], 1.0, places=3)

    def test_a_group_of_rejected_content_teaches_nothing(self):
        self.assertEqual(train_grpo.group_advantages([None, None, 1.0, 0.0], 2)[:2], [0.0, 0.0])

    def test_the_approval_term_gets_an_advantage_of_its_own(self):
        rewards = [1.05, 0.55, 1.0, 0.5]
        approvals = [0.05, -0.45, 0.0, 0.0]
        fields, approval = train_grpo.split_advantages(rewards, approvals, 2)
        self.assertEqual(fields[:2], [0.0, 0.0])
        self.assertAlmostEqual(approval[0], 1.0, places=3)
        self.assertAlmostEqual(approval[1], -1.0, places=3)
        self.assertAlmostEqual(fields[2], 1.0, places=3)
        self.assertEqual(approval[2:], [0.0, 0.0])


class ChainTests(unittest.TestCase):
    def test_a_prompt_that_does_not_extend_the_sequence_starts_a_new_one(self):
        g = train_grpo.Generation
        built = train_grpo.chains([g("tools", [1, 2], [3]), g("tools", [1, 2, 3, 4], [5]), g("answer", [1, 9], [6])])
        self.assertEqual([(c["ids"], c["spans"]) for c in built],
                         [([1, 2, 3, 4, 5], [(2, 3), (4, 5)]), ([1, 9, 6], [(2, 3)])])

    def test_a_truncated_completion_that_the_next_prompt_rewrites_breaks_the_chain(self):
        g = train_grpo.Generation
        built = train_grpo.chains([g("tools", [1, 2], [3, 4]), g("tools", [1, 2, 3, 7, 8], [9])])
        self.assertEqual(len(built), 2)


class ConfidenceTokenTests(unittest.TestCase):
    def test_the_number_is_found_from_the_end(self):
        tokenizer = CharTokenizer()
        text = '{\n  "reasoning": "confidence 1.0 here",\n  "confidence": 0.95\n}'
        ids = tokenizer(text)["input_ids"] + [END]
        positions = train_grpo.confidence_positions(tokenizer.decode, ids)
        self.assertEqual(tokenizer.decode([ids[k] for k in positions]), "0.95")

    def test_a_confidence_that_is_not_the_end_of_a_finished_answer_has_none(self):
        tokenizer = CharTokenizer()
        for text in ('{"confidence": 0.9, "note": ""}', '{"reasoning": "x", "confidence": 0.9'):
            ids = tokenizer(text)["input_ids"]
            self.assertEqual(train_grpo.confidence_positions(tokenizer.decode, ids), [], text)

    def test_an_answer_without_a_confidence_has_none(self):
        tokenizer = CharTokenizer()
        self.assertEqual(train_grpo.confidence_positions(tokenizer.decode, tokenizer("{}")["input_ids"]), [])

    def test_a_token_that_holds_part_of_the_number_counts(self):
        pieces = ['{"confidence": 0', '.9', '5}']
        decode = lambda ids: "".join(pieces[i] for i in ids)
        self.assertEqual(train_grpo.confidence_positions(decode, [0, 1, 2]), [0, 1, 2])


class WeightTests(unittest.TestCase):
    def chain(self):
        return {"ids": list(range(10)), "spans": [(2, 5), (6, 10)], "confidence": [8, 9]}

    def test_the_fields_advantage_is_per_token_and_the_approval_one_per_session(self):
        weights = train_grpo.token_weights(self.chain(), 0.5, -2.0, tokens=100, sessions=4, confidence_weight=1.0)
        self.assertEqual(len(weights), 7)
        self.assertEqual(weights[:5], [0.005] * 5)
        self.assertEqual(weights[5:], [-0.25, -0.25])
        self.assertAlmostEqual(sum(weights[5:]), -2.0 / 4)

    def test_a_chain_without_the_answer_takes_only_the_fields_advantage(self):
        chain = {**self.chain(), "confidence": []}
        self.assertEqual(train_grpo.token_weights(chain, 1.0, 3.0, 10, 4, 1.0), [0.1] * 7)


class KeptStepTests(unittest.TestCase):
    def test_every_twentieth_step_and_the_last(self):
        self.assertEqual(train_grpo.kept_steps(80, 20), [20, 40, 60, 80])
        self.assertEqual(train_grpo.kept_steps(90, 20), [20, 40, 60, 80, 90])

    def test_none_without_keep_every(self):
        self.assertEqual(train_grpo.kept_steps(80, 0), [])

    def test_snapshot_copies_the_adapter_and_replaces_an_older_one(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint = os.path.join(root, "checkpoint")
            os.makedirs(checkpoint)
            for name in (*train_grpo.ADAPTER_FILES, "state.pt"):
                with open(os.path.join(checkpoint, name), "w") as f:
                    f.write(name + " new")
            folder = train_grpo.snapshot_folder(root, 20)
            os.makedirs(folder)
            with open(os.path.join(folder, "stale"), "w") as f:
                f.write("old")
            train_grpo.keep_snapshot(checkpoint, folder)
            self.assertEqual(folder, os.path.join(root, "step-020"))
            self.assertEqual(sorted(os.listdir(folder)), sorted(train_grpo.ADAPTER_FILES))
            with open(os.path.join(folder, "adapter_model.safetensors")) as f:
                self.assertEqual(f.read(), "adapter_model.safetensors new")
            self.assertFalse(os.path.exists(folder + ".partial"))


class PlanTests(unittest.TestCase):
    def test_frozen_suites_are_refused(self):
        for name in ("dev", "dev-novel", "holdout", "smoke"):
            with self.assertRaises(SystemExit):
                train_grpo.check_suite({"name": name})
        train_grpo.check_suite({"name": "train-rl"})

    def test_a_case_a_frozen_suite_contains_word_for_word_is_left_out(self):
        _, dev_cases, _ = load_suite("dev")
        kept = train_grpo.without_frozen_content([dev_cases[0], {"content": "a message no suite has"}])
        self.assertEqual([c["content"] for c in kept], ["a message no suite has"])

    def test_the_case_order_is_the_same_for_a_seed_and_covers_every_case(self):
        cases = [{"id": str(i)} for i in range(10)]
        plan = train_grpo.case_order(cases, 5, 4, seed=1)
        self.assertEqual(plan, train_grpo.case_order(cases, 5, 4, seed=1))
        self.assertEqual({c["id"] for step in plan[:2] for c in step} | {c["id"] for c in plan[2][:2]},
                         {c["id"] for c in cases})


class CutTests(unittest.TestCase):
    def test_a_completion_ends_at_its_end_of_turn_or_its_limit(self):
        self.assertEqual(train_grpo.cut([5, 6, END, 0, 0], 10, END), [5, 6, END])
        self.assertEqual(train_grpo.cut([5, 6, 7, 8], 2, END), [5, 6])


@unittest.skipUnless(all(importlib.util.find_spec(m) for m in ("torch", "xgrammar")), "needs torch and xgrammar")
class GrammarMaskTests(unittest.TestCase):
    """An answer row may only write what its schema allows; a tool row is
    left alone; a row whose grammar has ended is the generator's to pad."""

    SCHEMA = {"type": "object", "properties": {"confidence": {"type": "number"}}, "required": ["confidence"],
              "additionalProperties": False}

    def setUp(self):
        import torch
        import xgrammar
        self.torch = torch
        self.vocab = [chr(c) for c in range(32, 127)] + ["\n", "<|im_end|>"]
        self.end = len(self.vocab) - 1
        info = xgrammar.TokenizerInfo(self.vocab, xgrammar.VocabType.RAW, stop_token_ids=[self.end])
        self.mask = train_grpo.AnswerGrammar(info).masker([self.SCHEMA, None], prompt_width=1)
        self.ids = torch.zeros((2, 1), dtype=torch.long)

    def allowed(self):
        scores = self.mask(self.ids, self.torch.zeros((2, len(self.vocab))))
        return [{self.vocab[i] for i in range(len(self.vocab)) if scores[row, i] > -1e9} for row in (0, 1)]

    def write(self, answer, tool):
        self.ids = self.torch.cat([self.ids, self.torch.tensor([[answer], [tool]])], dim=1)

    def test_the_answer_row_follows_its_schema_and_the_tool_row_is_free(self):
        answer, tool = self.allowed()
        self.assertIn("{", answer)
        self.assertNotIn("a", answer)
        self.assertEqual(len(tool), len(self.vocab))
        for c in '{"confidence": 0.5}':
            self.write(self.vocab.index(c), self.vocab.index("a"))
            answer, tool = self.allowed()
        self.assertIn("<|im_end|>", answer)
        self.assertNotIn(",", answer)
        self.write(self.end, self.vocab.index("a"))
        answer, _ = self.allowed()
        self.assertEqual(len(answer), len(self.vocab))

    def test_the_keys_come_in_the_order_the_schema_declares_them(self):
        import xgrammar
        info = xgrammar.TokenizerInfo(self.vocab, xgrammar.VocabType.RAW, stop_token_ids=[self.end])
        schema = {"type": "object", "properties": {"z": {"type": "number"}, "a": {"type": "number"}},
                  "required": ["z", "a"], "additionalProperties": False}
        self.mask = train_grpo.AnswerGrammar(info).masker([schema, None], prompt_width=1)
        for c in '{"':
            self.allowed()
            self.write(self.vocab.index(c), 0)
        answer, _ = self.allowed()
        self.assertEqual(answer, {"z"})

    def test_a_token_the_grammar_forbids_is_refused(self):
        self.allowed()
        self.write(self.vocab.index("a"), 0)
        with self.assertRaises(RuntimeError):
            self.allowed()


if __name__ == "__main__":
    unittest.main()
