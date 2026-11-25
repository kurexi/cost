import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import make_suite
import novel_names
from locales import LOCALES
from suite import load_suite


def merchants(world):
    return [m for loc in world.values() for m in loc["merchants"]]


def item_names(merchant):
    pools = [merchant["items"]] + ([merchant["mixed"][1]] if merchant["mixed"] else [])
    return [name for pool in pools for name, _, _ in pool]


class WorldTests(unittest.TestCase):
    def test_share_zero_is_the_real_list_itself(self):
        self.assertIs(novel_names.world_for(1, "en-GB", 0, 0.0), LOCALES)

    def test_a_world_is_the_same_every_time(self):
        self.assertEqual(novel_names.world_for(3, "de", 1, 0.5), novel_names.world_for(3, "de", 1, 0.5))

    def test_every_merchant_is_invented_at_share_one_and_keeps_what_it_sells(self):
        world = novel_names.world_for(3, "fr", 0, 1.0)
        real = novel_names.real_names(LOCALES)
        for code, loc in LOCALES.items():
            for original, renamed in zip(loc["merchants"], world[code]["merchants"]):
                self.assertTrue(renamed["invented"])
                self.assertNotIn(renamed["name"].casefold(), real)
                self.assertEqual((renamed["cat"], renamed["online"]), (original["cat"], original["online"]))
                self.assertEqual([(lo, hi) for _, lo, hi in renamed["items"]],
                                 [(lo, hi) for _, lo, hi in original["items"]])
                self.assertIn(renamed["name"], renamed["printed"] + [renamed["name"]])

    def test_rewritten_items_are_never_a_real_items_name(self):
        real = novel_names.real_names(LOCALES)
        for merchant in merchants(novel_names.world_for(3, "ja", 0, 1.0)):
            for name in item_names(merchant):
                self.assertNotIn(name.casefold(), real)

    def test_about_the_share_of_merchants_is_invented(self):
        invented = [m.get("invented", False) for i in range(10) for m in merchants(novel_names.world_for(5, "es", i, 0.5))]
        self.assertAlmostEqual(sum(invented) / len(invented), 0.5, delta=0.1)

    def test_names_in_one_world_are_distinct(self):
        for loc in novel_names.world_for(7, "zh", 0, 1.0).values():
            names = [m["name"].casefold() for m in loc["merchants"]]
            self.assertEqual(len(names), len(set(names)))

    def test_the_two_splits_never_share_a_name(self):
        train = {n.casefold() for i in range(4) for m in merchants(novel_names.world_for(9, "en-GB", i, 1.0, "train"))
                 for n in [m["name"]] + item_names(m)}
        held = {n.casefold() for i in range(4) for m in merchants(novel_names.world_for(9, "en-GB", i, 1.0, "eval"))
                for n in [m["name"]] + item_names(m)}
        self.assertEqual(train & held, set())
        self.assertTrue(all(novel_names.in_split(n, "train") for n in train))

    def test_french_contracts_du_before_a_vowel(self):
        self.assertEqual(novel_names.elided("Rail du Ardbridge"), "Rail d'Ardbridge")
        self.assertEqual(novel_names.elided("Animalerie du Wynton"), "Animalerie du Wynton")

    def test_a_resized_item_keeps_its_unit(self):
        rng = random.Random(0)
        for _ in range(20):
            self.assertRegex(novel_names.resized(rng, "Penne 500g"), r"^Penne [0-9.]+g$")


class NovelSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases, users = make_suite.build(seed=5, per_locale=12, users_per_locale=3, novel_share=1.0,
                                            novel_split="eval")
        cls.users = {u["id"]: u for u in users}

    def test_spending_cases_are_at_invented_merchants(self):
        spending = [c for c in self.cases if c["gold"]["isSpending"]]
        self.assertTrue(all("invented_merchant" in c["traits"] for c in spending))

    def test_the_ledger_knows_the_users_invented_merchants(self):
        for case in self.cases:
            if "mixed_merchant" in case["traits"]:
                user = self.users[case["user"]]
                self.assertIn(case["gold"]["merchant"], {r["merchant"] for r in user["records"]})

    def test_the_gold_merchant_is_printed_in_the_content(self):
        for case in self.cases:
            if case["gold"]["isSpending"] and case["style"] != "casual" and "ocr_noise" not in case["traits"]:
                self.assertTrue(any(name in case["content"] for name in case["gold"]["merchantAccepted"]), case["id"])


class CommittedNovelSuiteTests(unittest.TestCase):
    def test_dev_novel_regenerates_identically(self):
        meta, cases, users = load_suite("dev-novel")
        again, again_users = make_suite.build(meta["seed"], meta["perLocale"], meta["usersPerLocale"],
                                              novel_share=meta["novelShare"], novel_split=meta["novelSplit"])
        self.assertEqual(make_suite.fingerprint(again, again_users), meta["fingerprint"])

    def test_dev_novel_draws_from_the_eval_half(self):
        meta, cases, _ = load_suite("dev-novel")
        self.assertEqual((meta["novelShare"], meta["novelSplit"]), (1.0, "eval"))
        for case in cases:
            if case["gold"]["isSpending"]:
                self.assertTrue(novel_names.in_split(case["gold"]["merchant"], "eval"))


if __name__ == "__main__":
    unittest.main()
