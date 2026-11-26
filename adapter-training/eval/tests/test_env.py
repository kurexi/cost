import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import env
from grade import parse_minor, reward, score


class CalculatorTests(unittest.TestCase):
    def test_sums_exactly(self):
        self.assertEqual(env.calculator_answer("4.50 + 3.25 + 1.20")[0], "4.50 + 3.25 + 1.20 = 8.95")
        self.assertEqual(env.calculator_answer("0.1 + 0.2")[1], "= 0.3")

    def test_refuses_with_instructions(self):
        self.assertIn("Commas are ambiguous", env.calculator_answer("1,5 + 2")[0])
        self.assertIn('Unsupported character "£"', env.calculator_answer("£4 + 2")[0])
        self.assertIn("Division by zero", env.calculator_answer("1/0")[0])
        self.assertIn("empty", env.calculator_answer("  ")[0])


class ReceiptSearchTests(unittest.TestCase):
    RECEIPT = "ASDA\nMILK 1.20\nBREAD 0.75\nTOTAL:\n£11.92\nVISA"

    def test_returns_neighbours(self):
        msg, summary = env.search_answer("total", self.RECEIPT)
        self.assertIn("4: TOTAL:", msg)
        self.assertIn("5: £11.92", msg)
        self.assertEqual(summary, "1 line")

    def test_folds_typography_and_retries_literals(self):
        self.assertEqual(env.search_answer("sainsbury's", "SAINSBURY’S\nx")[1], "1 line")
        self.assertEqual(env.search_answer("sainsbury's", "SAINSBURYS\nx")[1], "1 line")

    def test_no_match_says_stop(self):
        msg, summary = env.search_answer("avocado", self.RECEIPT)
        self.assertEqual(summary, "no match")
        self.assertIn("don't try variations", msg)


def _gym(month, printed=""):
    return {"merchant": "Harbourside Sports", "category": "Gym", "note": "Membership", "amountMinorUnits": 2799,
            "currencyCode": "GBP", "date": f"2026-{month:02d}-08", "printed": printed}


def sample_ledger():
    return [
        _gym(7), _gym(8), _gym(9),
        _gym(10, printed="WWW.HBRSD.AC.UK\nHARBOURSIDE GB\n£ 27.99"),
        {"merchant": "McDonald's", "category": "Food", "note": "Lunch", "amountMinorUnits": 649,
         "currencyCode": "GBP", "date": "2026-10-20", "printed": "MCDONALDS\nHARBOURSIDE GB\n£ 6.49"},
        {"merchant": "Tesco", "category": "Groceries", "note": "Groceries: milk, bread", "amountMinorUnits": 270,
         "currencyCode": "GBP", "date": "2026-10-02", "printed": "TESCO STORES 3297\nTOTAL 2.70"},
        {"merchant": "Tesco", "category": "Groceries", "note": "weekly shop", "amountMinorUnits": 6120,
         "currencyCode": "GBP", "date": "2026-10-09"},
    ]


class LedgerSearchTests(unittest.TestCase):

    LEDGER = None

    @classmethod
    def setUpClass(cls):
        cls.LEDGER = sample_ledger()

    def test_query_words(self):
        self.assertEqual(env.ledger_tokens("WWW.HBRSD.AC.UK"), ["hbrsd"])
        self.assertEqual(env.ledger_tokens("TESCO STORES 3297"), ["tesco", "stores", "3297"])
        self.assertEqual(env.ledger_tokens("  "), [])

    def test_exact_amount(self):
        hits = env.ledger_hits("", "27.99", self.LEDGER)
        self.assertEqual(len(hits), 4)
        self.assertTrue(all(h["entry"]["merchant"] == "Harbourside Sports" and h["amount"] for h in hits))
        self.assertEqual(env.ledger_hits("", "27.98", self.LEDGER), [])
        self.assertEqual(len(env.ledger_hits("", "£ 27.99", self.LEDGER)), 4)

    def test_descriptor_finds_record(self):
        hits = env.ledger_hits("WWW.HBRSD.AC.UK", "", self.LEDGER)
        self.assertEqual(len(hits), 1)
        self.assertTrue(hits[0]["printed"] and not hits[0]["name"])
        self.assertEqual({h["entry"]["merchant"] for h in env.ledger_hits("TESCO STORES 3297", "", self.LEDGER)}, {"Tesco"})
        self.assertEqual({h["entry"]["merchant"] for h in env.ledger_hits("HARBOURSIDE GB", "", self.LEDGER)},
                         {"Harbourside Sports", "McDonald's"})
        self.assertEqual(env.ledger_hits("www", " ", self.LEDGER), [])

    def test_cadence(self):
        self.assertEqual(env.ledger_cadence(["2026-07-08", "2026-08-08", "2026-09-09", "2026-10-08"]), "about monthly")
        self.assertEqual(env.ledger_cadence(["2026-10-01", "2026-10-08", "2026-10-15"]), "about weekly")
        self.assertEqual(env.ledger_cadence(["2024-03-01", "2025-03-01", "2026-03-02"]), "about yearly")
        self.assertIsNone(env.ledger_cadence(["2026-09-08", "2026-10-08"]))
        self.assertIsNone(env.ledger_cadence(["2026-07-08", "2026-08-08", "2026-08-20"]))

    def test_answer(self):
        msg, summary = env.ledger_answer("", "27.99", self.LEDGER)
        self.assertEqual(summary, "4 records")
        self.assertTrue(msg.endswith('- Harbourside Sports, filed under Gym: 4 records, each 27.99 GBP, about monthly, '
                                     'latest 2026-10-08, 2026-09-08, 2026-08-08. Note: "Membership".'))
        msg, summary = env.ledger_answer("hbrsd", "", self.LEDGER)
        self.assertEqual(summary, "1 record")
        self.assertIn("1 record, 27.99 GBP on 2026-10-08", msg)
        self.assertIn('Its content printed "WWW.HBRSD.AC.UK"', msg)
        msg, _ = env.ledger_answer("tesco", "", self.LEDGER)
        self.assertIn("- Tesco, filed under Groceries: 2 records, from 2.70 GBP to 61.20 GBP", msg)
        self.assertIn('Note: "weekly shop"', msg)

    def test_ranking(self):
        lines = env.ledger_lines(env.ledger_hits("HARBOURSIDE GB", "27.99", self.LEDGER), "HARBOURSIDE GB")
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("- Harbourside Sports, filed under Gym"))
        self.assertTrue(lines[1].startswith("- McDonald's, filed under Food"))
        many = [{"merchant": f"Shop {i} Harbourside", "amountMinorUnits": 100, "currencyCode": "GBP",
                 "date": "2026-10-01"} for i in range(10)]
        self.assertEqual(len(env.ledger_lines(env.ledger_hits("harbourside", "", many), "harbourside")),
                         env.LEDGER_MAX_GROUPS)
        corner = [{"merchant": "Corner Shop", "amountMinorUnits": 120, "currencyCode": "GBP", "date": "2026-10-01"}]
        self.assertIn("- Corner Shop, not filed under a category: 1 record, 1.20 GBP", env.ledger_answer("corner", "", corner)[0])

    def test_degenerate(self):
        self.assertEqual(env.ledger_answer("netflix", "", self.LEDGER)[1], "no records")
        self.assertEqual(env.ledger_answer("", "", self.LEDGER)[1], "invalid lookup")


class BudgetTests(unittest.TestCase):
    def test_search_budget_and_cache(self):
        b = env.ToolBudget(max_searches=2)
        self.assertNotIn("already", b.search("total", "TOTAL 1.00")[0])
        msg, summary = b.search(" TOTAL: ", "TOTAL 1.00")
        self.assertIn("already searched", msg)
        self.assertIn("last available search", msg)
        self.assertEqual(b.search("x", "y")[1], "no searches left")

    def test_lookups_bounded_and_cached(self):
        b = env.ToolBudget(max_lookups=2)
        self.assertEqual(b.lookup("", "27.99", sample_ledger())[1], "4 records")
        msg, summary = b.lookup("", " 27.99 ", sample_ledger())
        self.assertIn("repeat", summary)
        self.assertIn("last available lookup", msg)
        self.assertEqual(b.lookup("tesco", "", sample_ledger())[1], "no lookups left")
        self.assertFalse(b.is_spent)
        self.assertEqual(b.searches, 0)

    def test_second_refusal_ends_the_tool_session(self):
        b = env.ToolBudget(max_searches=1)
        b.search("total", "TOTAL 11.92")
        self.assertEqual(b.search("bus", "x")[1], "no searches left")
        self.assertFalse(b.is_spent)
        self.assertEqual(b.search("taxi", "x")[1], "stopping tool use")
        self.assertTrue(b.is_spent)

    def test_refusals_count_across_tools(self):
        b = env.ToolBudget(max_searches=0, max_calculations=0)
        b.search("total", "x")
        self.assertFalse(b.is_spent)
        b.calculate("1 + 1")
        self.assertTrue(b.is_spent)

    def test_episode_offers_lookup_only_with_records(self):
        e = env.Episode("x", sample_ledger())
        self.assertIn("Harbourside Sports", e.call("searchRecords", '{"query": "", "amount": "27.99"}'))
        self.assertIn("no tool", env.Episode("x", []).call("searchRecords", {"query": "tesco", "amount": ""}))
        names = [t["function"]["name"] for t in env.tool_definitions()]
        self.assertEqual(names, ["calculator", "searchReceipt", "searchRecords"])


class SchemaTests(unittest.TestCase):
    def test_schema_is_read_from_the_app_in_order(self):
        s = env.extraction_schema()
        self.assertEqual(s["required"][0], "reasoning")
        self.assertEqual(s["required"][-1], "confidence")
        self.assertIn("paymentInstrument", s["required"])


class GraderTests(unittest.TestCase):
    GOLD = {"isSpending": True, "merchant": "Tesco", "merchantAccepted": ["Tesco", "TESCO EXPRESS"],
            "paymentInstrument": "Monzo", "paymentInstrumentMode": "exact", "paymentInstrumentAccepted": ["Monzo"],
            "totalAmount": "12.34", "amountMinorUnits": 1234, "currencyCode": "GBP", "purchaseDate": "2026-10-01",
            "purchaseTime": "", "category": "Groceries", "proposedCategory": "", "proposalAccepted": [],
            "noteItems": ["milk"]}

    def out(self, **kw):
        base = {"reasoning": "", "isSpending": True, "merchant": "TESCO EXPRESS", "paymentInstrument": "Monzo",
                "totalAmount": "12.34", "currencyCode": "GBP", "purchaseDate": "2026-10-01", "purchaseTime": "",
                "category": "Groceries", "proposedCategory": "", "note": "Groceries: milk", "confidence": 0.9}
        base.update(kw)
        import json
        return json.dumps(base)

    def test_correct_record(self):
        s, _ = score(self.GOLD, self.out())
        self.assertEqual(s["record_all"], 1.0)
        self.assertAlmostEqual(reward(self.GOLD, self.out())[0], 1.05)

    def test_card_as_merchant_is_wrong(self):
        s, _ = score(self.GOLD, self.out(merchant="Monzo"))
        self.assertEqual(s["merchant"], 0.0)

    def test_amount_forms(self):
        self.assertEqual(score(self.GOLD, self.out(totalAmount="£12.34"))[0]["amount"], 1.0)
        self.assertEqual(score(self.GOLD, self.out(totalAmount="£12.34"))[0]["amount_strict"], 0.0)
        self.assertEqual(parse_minor("1.234,56", "EUR"), 123456)
        self.assertEqual(parse_minor("¥1,280", "JPY"), 1280)

    def test_undecodable_is_minus_one(self):
        self.assertEqual(reward(self.GOLD, "I think it's Tesco")[0], -1.0)


if __name__ == "__main__":
    unittest.main()
