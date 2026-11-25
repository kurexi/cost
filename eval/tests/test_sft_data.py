import copy
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import sft_data
from spending_agent import draft as contract
from spending_agent.backend import ChatBackend
from spending_agent.ledger import Ledger, Record
from spending_agent.session import COMPACTED_HEADER, SessionConfig, run_session

TEMPLATE = os.path.join(HERE, "fixtures", "qwen3.5-chat_template.jinja")
GENERATION_PROMPT = "<|im_start|>assistant\n<think>\n\n</think>\n\n"

ANSWER = {"reasoning": "A till receipt from Tesco; the total line says 12.40.", "isSpending": True,
          "merchant": "Tesco", "paymentInstrument": "", "totalAmount": 12.4, "currencyCode": "GBP",
          "purchaseDate": "2026-05-02", "purchaseTime": "", "category": "Groceries", "proposedCategory": "",
          "tags": [], "proposedTag": "", "note": "Groceries: milk, bread", "confidence": 0.9}


def native_call(index, name, arguments):
    return {"id": f"call-{index}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


SCRIPT = [
    {"role": "assistant", "content": "Let me see how the user names this shop.",
     "tool_calls": [native_call(0, "queryMerchants", {"pattern": "TESCO STORES"}),
                    native_call(1, "memory", {"action": "read"})]},
    {"role": "assistant", "content": "Checking the total.\n<tool_call>\n<function=calculator>\n"
                                     "<parameter=expression>\n4.20 + 8.20\n</parameter>\n</function>\n</tool_call>"},
    {"role": "assistant", "content": None,
     "tool_calls": [native_call(2, "queryRecords", {"merchant": "Tesco", "amount": "12.40"})]},
    {"role": "assistant", "content": "Here is the record I drafted: Tesco, 12.40 GBP, Groceries."},
]


class FakeServer(ChatBackend):
    """The real client, with the network replaced: every body it would send
    is recorded, tool rounds follow the script, and the answer formats in
    `refuses` are refused the way a server refuses them."""

    def __init__(self, script, refuses=()):
        super().__init__("http://fake/v1", "qwen", reasoning_effort="none", one_request_at_a_time=False)
        self.script, self.refuses, self.bodies = list(script), set(refuses), []

    def _send(self, body):
        self.bodies.append(copy.deepcopy(body))
        if "tools" in body:
            message = self.script.pop(0)
        else:
            response_format = body.get("response_format") or {}
            label = response_format.get("type", "none")
            schema = (response_format.get("json_schema") or {}).get("schema", {})
            if label == "json_schema" and "pattern" not in json.dumps(schema):
                label = "json_schema_plain"
            if label in self.refuses:
                raise urllib.error.HTTPError("http://fake/v1", 400, "refused", {}, io.BytesIO(b"unsupported"))
            message = {"role": "assistant", "content": json.dumps(ANSWER, indent=2)}
        return {"choices": [{"message": message}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


def ledger_in(folder):
    ledger = Ledger.create(folder, ["Groceries", "Coffee"], ["Work"])
    ledger.append_record(Record(id="r1", date="2026-04-20", amount_minor=1180, currency="GBP", merchant="Tesco",
                                category="Groceries", note="Groceries: eggs"))
    return ledger


def run_scripted(budget=16000, refuses=(), script=SCRIPT):
    folder = tempfile.mkdtemp()
    server = FakeServer(script, refuses)
    config = SessionConfig(today="2026-05-02", local_currency="GBP", context_character_budget=budget)
    content = "TESCO STORES 3297\nMILK 4.20\nBREAD 8.20\nTOTAL\n12.40"
    result = run_session(ledger_in(folder), content, server, config, session_id="s1")
    with open(os.path.join(folder, "sessions", f"{result.session_id}.json"), encoding="utf-8") as f:
        transcript = json.load(f)
    transcript["id"] = "case-1"
    return transcript, server


def sent(server):
    """The bodies the server accepted: refused answer formats are dropped."""
    answers = [i for i, b in enumerate(server.bodies) if "tools" not in b]
    return [b for i, b in enumerate(server.bodies) if "tools" in b or i == answers[-1]]


class ReplayTests(unittest.TestCase):
    def assert_replay_matches(self, transcript, server):
        requests = sft_data.replay(transcript)
        bodies = sent(server)
        self.assertEqual(len(requests), len(bodies))
        for request, body in zip(requests, bodies):
            self.assertEqual(request.messages, body["messages"])
            self.assertEqual(request.tools, body.get("tools"))
        return requests

    def test_rebuilds_every_request_the_backend_sent(self):
        transcript, server = run_scripted()
        requests = self.assert_replay_matches(transcript, server)
        self.assertEqual([r.kind for r in requests], ["tool_call", "tool_call", "tool_call", "end_tools", "answer"])
        self.assertEqual(requests[-1].messages[-1]["content"], contract.CLOSING)
        self.assertEqual(requests[-1].response_format, "json_schema")

    def test_rebuilds_requests_across_compaction(self):
        transcript, server = run_scripted(budget=200)
        self.assertTrue(any(e["kind"] == "compaction" for e in transcript["events"]))
        requests = self.assert_replay_matches(transcript, server)
        self.assertTrue(any(m["content"].startswith(COMPACTED_HEADER) for m in requests[-1].messages
                            if m["role"] == "user"))

    def test_an_answer_on_a_fallback_format_carries_the_field_guide(self):
        transcript, server = run_scripted(refuses={"json_schema", "json_schema_plain"})
        requests = self.assert_replay_matches(transcript, server)
        self.assertEqual(requests[-1].response_format, "json_object")
        self.assertIn("Answer with one JSON object with exactly these keys", requests[-1].messages[-1]["content"])

    def test_a_transcript_that_does_not_replay_is_refused(self):
        transcript, _ = run_scripted()
        transcript["messages"][-1]["content"] = "edited afterwards"
        with self.assertRaises(sft_data.ReplayMismatch):
            sft_data.replay(transcript)

    def test_a_rejected_session_has_no_requests(self):
        transcript, server = run_scripted()
        transcript["messages"], transcript["events"] = [], [{"kind": "stop", "status": "rejected_too_long"}]
        self.assertEqual(sft_data.replay(transcript), [])

    def test_tool_call_arguments_reach_the_template_as_objects(self):
        message = {"role": "assistant", "content": "", "tool_calls": [native_call(0, "calculator",
                                                                                 {"expression": "1 + 2"})]}
        shaped = sft_data.for_template(message)
        self.assertEqual(shaped["tool_calls"][0]["function"]["arguments"], {"expression": "1 + 2"})
        self.assertIsInstance(message["tool_calls"][0]["function"]["arguments"], str)


def has_jinja():
    try:
        import jinja2
        return True
    except ImportError:
        return False


@unittest.skipUnless(has_jinja(), "jinja2 is not installed")
class PairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.render = sft_data.ChatTemplate.load(TEMPLATE)
        cls.transcript, _ = run_scripted()
        cls.requests = sft_data.replay(cls.transcript)

    def pairs(self, end_tools_text="empty"):
        return [sft_data.pair(self.render, r, end_tools_text) for r in self.requests]

    def test_every_prompt_ends_in_the_empty_think_block_and_every_completion_in_end_of_turn(self):
        for built in self.pairs():
            self.assertTrue(built["prompt"].endswith(GENERATION_PROMPT))
            self.assertTrue(built["completion"].endswith("<|im_end|>"))
            self.assertEqual(built["completion"].count("<|im_end|>"), 1)

    def test_tool_round_prompts_list_the_tools_and_the_answer_prompt_does_not(self):
        pairs = self.pairs()
        for built in pairs[:-1]:
            self.assertIn("# Tools", built["prompt"])
            self.assertIn('"name": "queryRecords"', built["prompt"])
        self.assertNotIn("# Tools", pairs[-1]["prompt"])

    def test_a_tool_call_completion_keeps_the_words_before_it(self):
        first = self.pairs()[0]["completion"]
        self.assertEqual(first, "Let me see how the user names this shop.\n\n<tool_call>\n<function=queryMerchants>\n"
                                "<parameter=pattern>\nTESCO STORES\n</parameter>\n</function>\n</tool_call>\n"
                                "<tool_call>\n<function=memory>\n<parameter=action>\nread\n</parameter>\n"
                                "</function>\n</tool_call><|im_end|>")

    def test_a_call_written_as_text_becomes_the_same_canonical_call(self):
        second = self.pairs()[1]["completion"]
        self.assertEqual(second, "Checking the total.\n\n<tool_call>\n<function=calculator>\n<parameter=expression>\n"
                                 "4.20 + 8.20\n</parameter>\n</function>\n</tool_call><|im_end|>")

    def test_the_end_of_the_tool_phase_is_an_empty_reply_unless_asked(self):
        self.assertEqual(self.pairs()[3]["completion"], "<|im_end|>")
        self.assertEqual(self.pairs("teacher")[3]["completion"],
                         "Here is the record I drafted: Tesco, 12.40 GBP, Groceries.<|im_end|>")

    def test_the_answer_completion_is_the_teachers_json(self):
        answer = self.pairs()[-1]
        self.assertEqual(answer["kind"], "answer")
        self.assertEqual(json.loads(answer["completion"][:-len("<|im_end|>")]), ANSWER)

    def test_a_later_prompt_contains_the_earlier_completion(self):
        pairs = self.pairs()
        self.assertIn(pairs[0]["completion"], pairs[1]["prompt"])

    def test_build_keeps_only_record_all_cases_and_counts_what_it_dropped(self):
        broken = copy.deepcopy(self.transcript)
        broken["id"] = "case-2"
        broken["messages"][-1]["content"] = "edited afterwards"
        wrong = copy.deepcopy(self.transcript)
        wrong["id"] = "case-3"
        rows, stats = sft_data.build([self.transcript, broken, wrong], {"case-1", "case-2"}, self.render)
        self.assertEqual([r["id"] for r in rows], ["case-1"])
        self.assertEqual((stats["kept"], stats["replay_mismatch"], stats["not_record_all"]), (1, 1, 1))
        self.assertEqual(stats["pairs"], {"tool_call": 3, "end_tools": 1, "answer": 1})

    def test_a_wrong_session_teaches_only_a_low_confidence_on_its_answer(self):
        wrong = copy.deepcopy(self.transcript)
        wrong["id"] = "case-3"
        rows, stats = sft_data.build([self.transcript, wrong], {"case-1"}, self.render,
                                     doubt_ids={"case-3"}, doubt_confidence=0.5)
        self.assertEqual([r["id"] for r in rows], ["case-1", "case-3"])
        self.assertEqual((stats["kept"], stats["doubted"], stats["not_record_all"]), (1, 1, 0))
        [doubt] = rows[1]["pairs"]
        self.assertEqual(doubt["kind"], "doubt")
        self.assertEqual(json.loads(doubt["completion"][:-len("<|im_end|>")]), {**ANSWER, "confidence": 0.5})
        [(start, end)] = doubt["train_spans"]
        self.assertEqual(doubt["completion"][start:end], "0.5")
        self.assertEqual(doubt["prompt"], rows[0]["pairs"][-1]["prompt"])

    def test_without_a_doubt_confidence_a_wrong_session_is_left_out(self):
        rows, stats = sft_data.build([self.transcript], set(), self.render, doubt_ids={"case-1"})
        self.assertEqual((rows, stats["not_record_all"]), ([], 1))

    def test_an_answer_without_one_confidence_number_is_counted_not_guessed(self):
        wrong = copy.deepcopy(self.transcript)
        for event in wrong["events"]:
            if event["kind"] == "model" and event.get("phase") != "tools":
                event["message"]["content"] = event["message"]["content"].replace('"confidence": 0.9', '"sure": 1')
        rows, stats = sft_data.build([wrong], set(), self.render, doubt_ids={"case-1"}, doubt_confidence=0.5)
        self.assertEqual((rows, stats["no_confidence"]), ([], 1))

    def test_a_case_whose_content_a_frozen_suite_repeats_is_left_out_even_when_right(self):
        rows, stats = sft_data.build([self.transcript], {"case-1"}, self.render, frozen_ids={"case-1"})
        self.assertEqual((rows, stats["frozen_content"], stats["kept"]), ([], 1, 0))

    def test_pairs_over_the_length_limit_are_dropped(self):
        rows, stats = sft_data.build([self.transcript], {"case-1"}, self.render, count_tokens=len, max_len=10)
        self.assertEqual((rows, stats["too_long"]), ([], 5))


class SuiteTests(unittest.TestCase):
    def test_frozen_case_ids_match_on_content(self):
        cases = [{"id": "a", "content": "Thanks for lunch!"}, {"id": "b", "content": "TESCO 12.40"}]
        self.assertEqual(sft_data.frozen_case_ids(cases, {"Thanks for lunch!"}), {"a"})

    def test_the_frozen_suites_are_read(self):
        frozen = sft_data.frozen_contents()
        with open(os.path.join(os.path.dirname(HERE), "suites", "dev", "cases.jsonl"), encoding="utf-8") as f:
            first = json.loads(f.readline())["content"]
        self.assertIn(first, frozen)

    def test_runs_add_up_kind_by_kind(self):
        total = sft_data.combined([{"kept": 2, "tokens": 10, "pairs": {"answer": 2, "tool_call": 3}},
                                   {"kept": 1, "tokens": 5, "pairs": {"answer": 1, "doubt": 4}}])
        self.assertEqual(total, {"kept": 3, "tokens": 15, "pairs": {"answer": 3, "tool_call": 3, "doubt": 4}})

    def test_a_suite_regenerated_differently_from_the_run_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            with open(os.path.join(folder, "meta.json"), "w") as f:
                json.dump({"name": "train-x", "fingerprint": "1111111111111111"}, f)
            for name in ("cases.jsonl", "users.jsonl"):
                with open(os.path.join(folder, name), "w") as f:
                    f.write("")
            run = {"name": "teacher-x", "suite": folder}
            self.assertEqual(sft_data.run_cases({**run, "suite_fingerprint": "1111111111111111"}), [])
            with self.assertRaises(SystemExit):
                sft_data.run_cases({**run, "suite_fingerprint": "2222222222222222"})


class RunCheckTests(unittest.TestCase):
    def meta(self, **changes):
        return {"name": "teacher-train", "suite": "train", "contract": contract.contract_fingerprint(),
                "reasoning_effort": "none", **changes}

    def test_a_train_run_passes(self):
        sft_data.check_run(self.meta())

    def test_frozen_suites_are_refused(self):
        for suite in ("dev", "dev-novel", "holdout", "smoke"):
            with self.assertRaises(SystemExit):
                sft_data.check_run(self.meta(suite=suite))

    def test_another_contract_is_refused(self):
        with self.assertRaises(SystemExit):
            sft_data.check_run(self.meta(contract="0000000000000000"))

    def test_thinking_runs_are_refused(self):
        with self.assertRaises(SystemExit):
            sft_data.check_run(self.meta(reasoning_effort="low"))


if __name__ == "__main__":
    unittest.main()
