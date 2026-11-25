import unittest

from spending_agent import tools
from spending_agent.ledger import Snapshot
from tests.fixtures import record, sample_ledger


class CalculatorTests(unittest.TestCase):
    def test_exact_decimal_sum(self):
        self.assertEqual(tools.calculator({"expression": "4.50 + 3.25 + 1.20"}).message, "4.50 + 3.25 + 1.20 = 8.95")

    def test_precedence_and_parentheses(self):
        self.assertIn("= 14", tools.calculator({"expression": "2 + 3 * 4"}).message)
        self.assertIn("= 20", tools.calculator({"expression": "(2 + 3) * 4"}).message)

    def test_currency_symbol_is_an_instructive_error(self):
        answer = tools.calculator({"expression": "£4 + 2"})
        self.assertEqual(answer.summary, "invalid expression")
        self.assertIn("Unsupported character", answer.message)

    def test_division_by_zero_answers(self):
        self.assertEqual(tools.calculator({"expression": "1 / 0"}).summary, "invalid expression")


class QueryTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = sample_ledger().snapshot()

    def test_records_need_a_condition(self):
        self.assertEqual(tools.query_records(self.snapshot, {}).summary, "invalid query")

    def test_records_by_merchant_group_spellings(self):
        answer = tools.query_records(self.snapshot, {"merchant": "tesco"})
        self.assertIn("Tesco, filed under Groceries: 3 records", answer.message)
        self.assertIn("Tesco Cafe, filed under Coffee", answer.message)
        self.assertIn("about weekly", answer.message)

    def test_records_show_tags(self):
        self.assertIn("tagged Health (2)", tools.query_records(self.snapshot, {"merchant": "PureGym"}).message)

    def test_records_by_exact_amount(self):
        answer = tools.query_records(self.snapshot, {"amount": "24.99"})
        self.assertIn("PureGym", answer.message)
        self.assertNotIn("Tesco", answer.message)

    def test_records_rejects_bad_date(self):
        self.assertIn("2026-03-31", tools.query_records(self.snapshot, {"from": "last week"}).message)

    def test_records_no_match_discourages_variations(self):
        self.assertIn("don't try variations", tools.query_records(self.snapshot, {"merchant": "Waitrose"}).message)

    def test_categories_listed_without_pattern(self):
        answer = tools.query_categories(self.snapshot, {})
        self.assertIn("- Eating Out: no records yet.", answer.message)
        self.assertIn("- Groceries: 5 records", answer.message)

    def test_categories_fuzzy(self):
        self.assertIn("Groceries", tools.query_categories(self.snapshot, {"pattern": "grocery"}).message)

    def test_merchants_show_filing_and_instrument(self):
        answer = tools.query_merchants(self.snapshot, {"pattern": "M&S"})
        self.assertIn("filed under Groceries (1)", answer.message)
        self.assertIn("paid with Monzo", answer.message)

    def test_merchants_need_a_pattern(self):
        self.assertEqual(tools.query_merchants(self.snapshot, {}).summary, "invalid query")

    def test_tags_have_a_shape(self):
        answer = tools.query_tags(self.snapshot, {})
        self.assertIn("Japan Trip: 2 records; an event, 2026-08-02 to 2026-08-05", answer.message)
        self.assertIn("Health: 2 records; standing, since 2026-03-01", answer.message)
        self.assertIn("- Work: no records yet.", answer.message)

    def test_bad_dates_never_raise(self):
        snapshot = Snapshot(records=[record("x", "someday", "Tesco", 100, "Groceries", tags=("Odd",))],
                            categories=["Groceries"], tags=["Odd"])
        self.assertIn("undated", tools.query_tags(snapshot, {"pattern": "odd"}).message)
        self.assertIn("Tesco", tools.query_merchants(snapshot, {"pattern": "tesco"}).message)
        self.assertIn("Tesco", tools.query_records(snapshot, {"merchant": "tesco"}).message)


class MemoryTests(unittest.TestCase):
    def test_append_to_empty(self):
        memory, answer = tools.memory_append("", {"text": "Monzo is a card, not a shop."})
        self.assertEqual(memory, "Monzo is a card, not a shop.")
        self.assertEqual(answer.summary, "appended")

    def test_append_adds_a_line(self):
        memory, _ = tools.memory_append("one", {"text": "two"})
        self.assertEqual(memory, "one\ntwo")

    def test_append_over_limit_is_refused_not_truncated(self):
        memory, answer = tools.memory_append("x" * 990, {"text": "y" * 20})
        self.assertIsNone(memory)
        self.assertIn("1000-character limit", answer.message)

    def test_replace_exact_passage(self):
        memory, _ = tools.memory_replace("Tesco cafe is Coffee.\nUser is in UK.", {"old": "UK", "new": "Japan"})
        self.assertEqual(memory, "Tesco cafe is Coffee.\nUser is in Japan.")

    def test_replace_empty_new_removes_the_line(self):
        memory, answer = tools.memory_replace("a\nb\nc", {"old": "b", "new": ""})
        self.assertEqual(memory, "a\nc")
        self.assertEqual(answer.summary, "removed")

    def test_replace_missing_or_ambiguous_passage_refused(self):
        self.assertIsNone(tools.memory_replace("abc", {"old": "zzz", "new": "y"})[0])
        memory, answer = tools.memory_replace("ab ab", {"old": "ab", "new": "y"})
        self.assertIsNone(memory)
        self.assertEqual(answer.summary, "refused, ambiguous")


if __name__ == "__main__":
    unittest.main()
