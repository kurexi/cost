import json
import os
import unittest

from spending_agent.session import COMPACTED_HEADER, SessionConfig, compact, run_session, settle, unreadable_answer_reason
from spending_agent.draft import RecordDraft
from tests.fixtures import ScriptedBackend, answer_text, calls, sample_ledger

CONTENT = "TESCO EXPRESS\nMILK 1.20\nBREAD 1.10\nTOTAL 2.30\n17/11/2026"


def config(**overrides):
    return SessionConfig(today="2026-11-18", **overrides)


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.ledger = sample_ledger()

    def transcript(self, result):
        with open(os.path.join(self.ledger.folder, "sessions", result.session_id + ".json")) as f:
            return json.load(f)

    def test_answer_without_tools_parks_in_review(self):
        backend = ScriptedBackend()
        result = run_session(self.ledger, CONTENT, backend, config())
        self.assertEqual(result.status, "needs_review")
        [row] = self.ledger.review()
        self.assertEqual((row["merchant"], row["amount_minor"], row["category"]), ("Tesco", "1240", "Groceries"))
        self.assertEqual([kind for kind, _ in backend.requests], ["complete", "answer"])

    def test_tool_results_reach_the_model(self):
        backend = ScriptedBackend([calls(("queryMerchants", {"pattern": "Tesco"}), ("calculator", {"expression": "1.20 + 1.10"}))])
        run_session(self.ledger, CONTENT, backend, config())
        answered = backend.requests[-1][1]
        tool_messages = [m for m in answered if m["role"] == "tool"]
        self.assertIn("filed under Groceries", tool_messages[0]["content"])
        self.assertEqual(tool_messages[1]["content"], "1.20 + 1.10 = 2.3")
        self.assertEqual(answered[-1]["role"], "user")

    def test_content_over_limit_is_rejected_before_the_model(self):
        backend = ScriptedBackend()
        result = run_session(self.ledger, "x" * 50, backend, config(content_character_limit=10))
        self.assertEqual(result.status, "rejected_too_long")
        self.assertIn("Record it manually", result.reason)
        self.assertEqual(backend.requests, [])
        self.assertEqual(self.ledger.review()[0]["status"], "rejected_too_long")

    def test_server_failure_parks_with_reason(self):
        result = run_session(self.ledger, CONTENT, ScriptedBackend(fail_with=OSError("refused")), config())
        self.assertEqual(result.status, "failed")
        self.assertIn("refused", result.reason)
        self.assertEqual(self.ledger.review()[0]["content"], CONTENT)

    def test_undecodable_answer_fails(self):
        result = run_session(self.ledger, CONTENT, ScriptedBackend(answer="no idea"), config())
        self.assertEqual(result.status, "failed")

    def test_not_spending(self):
        result = run_session(self.ledger, "Your code is 1234", ScriptedBackend(answer=answer_text(isSpending=False)),
                             config())
        self.assertEqual(result.status, "not_spending")

    def test_memory_edit_is_written_at_once_with_history(self):
        backend = ScriptedBackend([calls(("memory", {"action": "append", "text": "Tesco Cafe is Coffee."}))])
        result = run_session(self.ledger, CONTENT, backend, config())
        self.assertEqual(self.ledger.memory(), "Tesco Cafe is Coffee.")
        self.assertEqual(self.ledger.memory_history()[0]["session"], result.session_id)

    def test_memory_appears_in_the_next_prompt(self):
        self.ledger.write_memory("Monzo is a card.", "earlier")
        backend = ScriptedBackend()
        run_session(self.ledger, CONTENT, backend, config())
        self.assertIn("Monzo is a card.", backend.requests[0][1][1]["content"])

    def test_refused_memory_edit_keeps_the_chance_for_a_valid_one(self):
        backend = ScriptedBackend([
            calls(("memory", {"action": "append", "text": "x" * 1200})),
            calls(("memory", {"action": "append", "text": "short note"})),
        ])
        run_session(self.ledger, CONTENT, backend, config())
        self.assertEqual(self.ledger.memory(), "short note")

    def test_only_one_memory_edit_per_session(self):
        backend = ScriptedBackend([
            calls(("memory", {"action": "append", "text": "one"})),
            calls(("memory", {"action": "append", "text": "two"})),
        ])
        run_session(self.ledger, CONTENT, backend, config())
        self.assertEqual(self.ledger.memory(), "one")

    def test_spent_budget_ends_the_tool_phase(self):
        many = [calls(("memory", {"action": "read"})) for _ in range(6)]
        backend = ScriptedBackend(many)
        result = run_session(self.ledger, CONTENT, backend, config())
        completes = [kind for kind, _ in backend.requests if kind == "complete"]
        self.assertEqual(len(completes), 4)
        self.assertEqual(result.status, "needs_review")

    def test_round_limit(self):
        backend = ScriptedBackend([calls(("calculator", {"expression": f"{i} + 1"})) for i in range(20)])
        run_session(self.ledger, CONTENT, backend, config(max_tool_rounds=3))
        self.assertEqual(len([k for k, _ in backend.requests if k == "complete"]), 3)

    def test_unknown_tool_answers_instead_of_failing(self):
        backend = ScriptedBackend([calls(("searchWeb", {"q": "tesco"}))])
        result = run_session(self.ledger, CONTENT, backend, config())
        self.assertEqual(result.status, "needs_review")
        tool_message = [m for m in backend.requests[-1][1] if m["role"] == "tool"][0]
        self.assertIn('no tool called "searchWeb"', tool_message["content"])

    def test_transcript_records_contract_and_events(self):
        result = run_session(self.ledger, CONTENT, ScriptedBackend([calls(("queryTags", {"pattern": "trip"}))]),
                             config())
        transcript = self.transcript(result)
        self.assertEqual(len(transcript["contract"]), 16)
        self.assertIn("tool", [event["kind"] for event in transcript["events"]])


class SettleTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = sample_ledger().snapshot()

    def test_category_snaps_to_the_users_spelling(self):
        self.assertEqual(settle(RecordDraft(category="groceries"), self.snapshot).category, "Groceries")

    def test_unknown_category_becomes_a_proposal(self):
        settled = settle(RecordDraft(category="Pet Supplies"), self.snapshot)
        self.assertEqual((settled.category, settled.proposed_category), ("", "Pet Supplies"))

    def test_proposal_that_restates_a_category_is_the_category(self):
        settled = settle(RecordDraft(proposed_category="coffee"), self.snapshot)
        self.assertEqual((settled.category, settled.proposed_category), ("Coffee", ""))

    def test_tags(self):
        settled = settle(RecordDraft(tags=["japan trip", "Paris Trip", "Nope"]), self.snapshot)
        self.assertEqual((settled.tags, settled.proposed_tag), (["Japan Trip"], "Paris Trip"))


class CompactionTests(unittest.TestCase):
    def exchange(self, index, size):
        return [{"role": "assistant", "content": "<think>long</think>",
                 "tool_calls": [{"id": f"c{index}", "type": "function",
                                 "function": {"name": "queryRecords", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": f"c{index}", "name": "queryRecords", "content": f"r{index} " * size}]

    def test_thinking_is_never_sent_back(self):
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}] + self.exchange(0, 1)
        self.assertEqual(compact(messages, 10_000)[2]["content"], "")

    def test_old_exchanges_fold_into_one_bounded_note(self):
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        for index in range(4):
            messages += self.exchange(index, 300)
        compacted = compact(messages, 2500)
        notes = [m for m in compacted if (m.get("content") or "").startswith(COMPACTED_HEADER)]
        self.assertEqual(len(notes), 1)
        self.assertEqual(compacted[-1]["content"], messages[-1]["content"])
        self.assertTrue(compacted[-2].get("tool_calls"))
        self.assertLess(len(notes[0]["content"]), 1200)

    def test_latest_exchange_is_kept_even_over_budget(self):
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}] + self.exchange(0, 2000)
        self.assertEqual(len(compact(messages, 100)), 4)


class UnreadableAnswerTests(unittest.TestCase):
    def test_a_whitespace_runaway_names_where_it_stalled(self):
        reason = unreadable_answer_reason('{\n  "reasoning": "a shop",\n  "totalAmount":' + " " * 1200)
        self.assertIn("ran into whitespace", reason)
        self.assertIn('"totalAmount":', reason)

    def test_other_garbage_is_not_a_runaway(self):
        self.assertIn("not a JSON object", unreadable_answer_reason("This is a receipt from Tesco."))


if __name__ == "__main__":
    unittest.main()
