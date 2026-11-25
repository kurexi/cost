import os
import unittest

from spending_agent.ledger import Ledger
from tests.fixtures import record, sample_ledger


class LedgerTests(unittest.TestCase):
    def test_record_round_trip(self):
        ledger = sample_ledger(records=[record("a", "2026-11-01", "Lawson", 820, "Groceries",
                                               tags=("Japan Trip", "Work"), currency="JPY", note="onigiri, tea")])
        [back] = ledger.records()
        self.assertEqual((back.amount_minor, back.currency, back.tags), (820, "JPY", ["Japan Trip", "Work"]))
        self.assertEqual(back.note, "onigiri, tea")

    def test_categories_and_tags_include_those_used(self):
        snapshot = sample_ledger(categories=("Groceries",), tags=()).snapshot()
        self.assertIn("Travel", snapshot.categories)
        self.assertIn("Japan Trip", snapshot.tags)

    def test_create_is_idempotent(self):
        ledger = sample_ledger(records=[])
        again = Ledger.create(ledger.folder, ["groceries", "Pets"], ["work"])
        self.assertEqual(again.category_titles().count("Groceries"), 1)
        self.assertIn("Pets", again.category_titles())
        self.assertEqual(len(again.tag_names()), 1)

    def test_memory_writes_keep_history(self):
        ledger = sample_ledger(records=[])
        ledger.write_memory("first", "s1")
        ledger.write_memory("second", "s2")
        self.assertEqual(ledger.memory(), "second")
        self.assertEqual([row["text"] for row in ledger.memory_history()], ["first", "second"])

    def test_session_transcript_saved(self):
        ledger = sample_ledger(records=[])
        ledger.save_session("abc", {"status": "ok"})
        self.assertTrue(os.path.exists(os.path.join(ledger.folder, "sessions", "abc.json")))


if __name__ == "__main__":
    unittest.main()
