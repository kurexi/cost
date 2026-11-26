import os
import re
import sys
import unittest
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import make_suite
from backends import OracleBackend
from grade import grade
from run import run_case
from suite import load_suite

SETTINGS = {"max_tool_rounds": 10, "tool_tokens": 800, "answer_tokens": 1500}
GB_AMOUNT = re.compile(r"(-?)(?:£|GBP )?(\d+\.\d\d)(-?)(?: [AB])?$")


def paper_total_gb(content):
    block = content.split("\n\n")[0].split("\n")
    total = 0
    for line in block:
        if line.startswith("SUBTOTAL"):
            continue
        match = GB_AMOUNT.search(line)
        if match and "  " in line:
            sign = -1 if (match.group(1) or match.group(3)) else 1
            pounds, pence = match.group(2).split(".")
            total += sign * (int(pounds) * 100 + int(pence))
    return total


class VariedTillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases, users = make_suite.build(seed=7, per_locale=60, users_per_locale=4, novel_share=0.5, variety=0.8)
        cls.users = {u["id"]: u for u in users}
        cls.varied = [c for c in cls.cases if "varied_till" in c["traits"]]

    def test_generation_is_deterministic(self):
        again, users = make_suite.build(seed=7, per_locale=60, users_per_locale=4, novel_share=0.5, variety=0.8)
        self.assertEqual(make_suite.fingerprint(self.cases, list(self.users.values())),
                         make_suite.fingerprint(again, users))

    def test_variety_off_leaves_a_suite_as_it_was(self):
        plain, users = make_suite.build(seed=7, per_locale=20, users_per_locale=2)
        again, again_users = make_suite.build(seed=7, per_locale=20, users_per_locale=2, variety=0.0)
        self.assertEqual(make_suite.fingerprint(plain, users), make_suite.fingerprint(again, again_users))

    def test_only_till_receipts_change(self):
        plain, _ = make_suite.build(seed=7, per_locale=60, users_per_locale=4, novel_share=0.5)
        before = {c["id"]: c for c in plain}
        for case in self.cases:
            if "varied_till" not in case["traits"]:
                self.assertEqual(case["content"], before[case["id"]]["content"], case["id"])
        self.assertTrue(all(c["style"] == "till" for c in self.varied))

    def test_the_hard_shapes_all_occur(self):
        traits = Counter(t for c in self.varied for t in c["traits"])
        for trait in ("reduction_line", "quantity_line", "savings_line", "vat_table", "card_slip",
                      "no_printed_total", "product_code", "column_split", "charge_line"):
            self.assertGreater(traits[trait], 0, trait)
        hard = [c for c in self.varied if {"reduction_line", "no_printed_total"} <= set(c["traits"])]
        self.assertTrue(hard)

    def test_the_gold_total_is_what_the_paper_adds_up_to(self):
        checked = 0
        for case in self.varied:
            simple = (case["homeLocale"] == "en-GB" and "travel" not in case["traits"]
                      and "no_printed_total" in case["traits"] and "column_split" not in case["traits"]
                      and "tax_added" not in case["traits"])
            if simple:
                self.assertEqual(paper_total_gb(case["content"]), case["gold"]["amountMinorUnits"], case["id"])
                checked += 1
        self.assertGreater(checked, 0)

    def test_a_printed_total_is_the_gold(self):
        for case in self.varied:
            if case["homeLocale"] == "en-GB" and "no_printed_total" not in case["traits"] \
                    and "travel" not in case["traits"]:
                self.assertIn(case["gold"]["totalAmount"], case["content"], case["id"])

    def test_the_vat_table_adds_up_to_the_total(self):
        checked = 0
        row = re.compile(r"^[AB]  \d+ %  (\d+\.\d\d)  (\d+\.\d\d)$")
        for case in self.varied:
            if case["homeLocale"] == "en-GB" and "vat_table" in case["traits"] and "column_split" not in case["traits"] \
                    and "travel" not in case["traits"]:
                sales = [m.group(1) for m in map(row.match, case["content"].split("\n")) if m]
                self.assertEqual(sum(round(float(s) * 100) for s in sales), case["gold"]["amountMinorUnits"],
                                 case["id"])
                checked += 1
        self.assertGreater(checked, 0)

    def test_a_misread_logo_is_never_an_accepted_merchant(self):
        for case in self.varied:
            if "logo_misread" in case["traits"]:
                first_line = case["content"].split("\n")[0]
                self.assertNotIn(first_line, case["gold"]["merchantAccepted"])

    def test_oracle_is_perfect_on_varied_receipts(self):
        sample = self.varied[:30]
        results = [run_case(c, self.users[c["user"]], OracleBackend, SETTINGS)[0] for c in sample]
        summary, _ = grade(sample, self.users, results)
        self.assertEqual(summary["overall"]["record_all"], 1.0)


class CommittedVariedSuiteTests(unittest.TestCase):
    def test_dev_varied_regenerates_identically(self):
        meta, cases, users = load_suite("dev-varied")
        self.assertEqual(make_suite.fingerprint(cases, list(users.values())), meta["fingerprint"])
        again, again_users = make_suite.build(meta["seed"], meta["perLocale"], meta["usersPerLocale"],
                                              novel_share=meta["novelShare"], novel_split=meta["novelSplit"],
                                              variety=meta["variety"])
        self.assertEqual(make_suite.fingerprint(again, again_users), meta["fingerprint"])


if __name__ == "__main__":
    unittest.main()
