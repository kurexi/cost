import unittest

from spending_agent.budget import SPENT, ToolBudget
from spending_agent.tools import ToolAnswer


def answer():
    return ToolAnswer("found", "1 hit", "evidence")


class BudgetTests(unittest.TestCase):
    def test_repeat_query_is_cached_and_says_so(self):
        budget = ToolBudget()
        runs = []
        budget.call("queryMerchants", {"pattern": "Tesco"}, lambda: runs.append(1) or answer(), "q")
        repeat = budget.call("queryMerchants", {"pattern": "TESCO"}, lambda: runs.append(1) or answer(), "q")
        self.assertEqual(len(runs), 1)
        self.assertIn("already asked", repeat.message)

    def test_last_call_is_announced(self):
        budget = ToolBudget(limits={"calculator": 1})
        self.assertIn("last calculator call", budget.call("calculator", {}, answer, "c").message)

    def test_second_refusal_spends_the_budget(self):
        budget = ToolBudget(limits={"calculator": 1})
        budget.call("calculator", {"expression": "1"}, answer, "c")
        first = budget.call("calculator", {"expression": "2"}, answer, "c")
        self.assertFalse(budget.is_spent)
        self.assertIn("No more calculations", first.message)
        second = budget.call("calculator", {"expression": "3"}, answer, "c")
        self.assertTrue(budget.is_spent)
        self.assertEqual(second.message, SPENT)

    def test_session_ceiling(self):
        budget = ToolBudget(limits={"calculator": 10}, total=2)
        budget.call("calculator", {"expression": "1"}, answer, "c")
        budget.call("calculator", {"expression": "2"}, answer, "c")
        self.assertIn("No more tool calls", budget.call("calculator", {"expression": "3"}, answer, "c").message)

    def test_refund_restores_the_slot_but_not_the_ceiling(self):
        budget = ToolBudget(limits={"memoryEdit": 1}, total=5)
        budget.call("memoryEdit", {"text": "a"}, answer, "m")
        budget.refund("memoryEdit")
        self.assertEqual(budget.used["memoryEdit"], 0)
        self.assertEqual(budget.calls, 1)

    def test_findings_keep_evidence(self):
        budget = ToolBudget()
        budget.call("queryRecords", {"merchant": "Tesco"}, answer, "queryRecords(merchant='Tesco')")
        self.assertEqual(budget.findings, [("queryRecords(merchant='Tesco')", "evidence")])


if __name__ == "__main__":
    unittest.main()
