import copy
import json
import os
import sys
import tempfile
import unittest
import unittest.mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import make_suite
from backends import OracleBackend, SilentBackend
import board
from board import BEGIN, END, write_board
from compare import paired_interval, sign_test
from grade import broad_category_alternative, grade, score_case
from run import run_case
from suite import load_suite, materialize

SETTINGS = {"max_tool_rounds": 10, "tool_tokens": 800, "answer_tokens": 1500}


def first(cases, predicate):
    return next(case for case in cases if predicate(case))


class SuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases, users = make_suite.build(seed=5, per_locale=12, users_per_locale=2)
        cls.users = {u["id"]: u for u in users}

    def test_generation_is_deterministic(self):
        again, users = make_suite.build(seed=5, per_locale=12, users_per_locale=2)
        self.assertEqual(make_suite.fingerprint(self.cases, list(self.users.values())),
                         make_suite.fingerprint(again, users))

    def test_prefixes_are_balanced_across_locales(self):
        self.assertEqual(len({c["homeLocale"] for c in self.cases[:7]}), 7)

    def test_gold_categories_and_tags_are_the_users_own(self):
        for case in self.cases:
            user = self.users[case["user"]]
            for title in case["gold"]["categoryAccepted"]:
                self.assertIn(title, user["categories"])
            for tag in case["gold"]["tags"]:
                self.assertIn(tag, user["tags"])

    def test_over_limit_cases_exceed_the_budget(self):
        for case in self.cases:
            if case["expectedStatus"] == "rejected_too_long":
                self.assertGreater(len(case["content"]), make_suite.CONTENT_CHARACTER_LIMIT)
            else:
                self.assertLessEqual(len(case["content"]), make_suite.CONTENT_CHARACTER_LIMIT)

    def test_long_receipt_gold_total_is_printed(self):
        case = first(self.cases, lambda c: "many_items" in c["traits"] and "no_printed_total" not in c["traits"]
                     and c["gold"]["currencyCode"] == "GBP")
        self.assertIn(case["gold"]["totalAmount"], case["content"])

    def test_misleading_memory_contradicts_the_records(self):
        user = first(list(self.users.values()), lambda u: u["memoryWrongAbout"])
        filed = {r["category"] for r in user["records"] if r["merchant"] == user["memoryWrongAbout"]}
        line = next(l for l in user["memory"].split("\n") if l.startswith(user["memoryWrongAbout"] + " purchases"))
        self.assertFalse(any(line.endswith(f" {title}.") for title in filed))

    def test_materialize_leaves_out_future_records(self):
        user = next(iter(self.users.values()))
        cut = sorted(r["date"] for r in user["records"])[len(user["records"]) // 2]
        ledger = materialize(user, cut, tempfile.mkdtemp())
        self.assertTrue(all(r.date <= cut for r in ledger.records()))
        self.assertEqual(ledger.memory(), user["memory"])

    def test_oracle_is_perfect_and_silent_is_the_floor(self):
        sample = self.cases[:30]
        for backend, expected in ((OracleBackend, 1.0), (SilentBackend, 0.0)):
            results = [run_case(c, self.users[c["user"]], backend, SETTINGS)[0] for c in sample]
            summary, _ = grade(sample, self.users, results)
            self.assertEqual(summary["overall"]["record_all"], expected, backend.__name__)


class ScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases, users = make_suite.build(seed=5, per_locale=12, users_per_locale=2)
        cls.users = {u["id"]: u for u in users}

    def draft_for(self, case, **changes):
        answer = OracleBackend(case)
        result, _ = run_case(case, self.users[case["user"]], lambda c: answer, SETTINGS)
        result["draft"].update(changes)
        return result

    def scores(self, case, **changes):
        return score_case(case, self.draft_for(case, **changes), self.users[case["user"]])

    def spending(self, predicate=lambda c: True):
        return first(self.cases, lambda c: c["expectedStatus"] == "needs_review" and predicate(c))

    def test_wrong_amount(self):
        case = self.spending()
        s = self.scores(case, amount_minor=case["gold"]["amountMinorUnits"] + 1)
        self.assertEqual((s["amount"], s["record_core"]), (0.0, 0.0))

    def test_instrument_as_merchant_is_wrong(self):
        case = self.spending(lambda c: c["gold"]["paymentInstrumentMode"] == "exact")
        self.assertEqual(self.scores(case, merchant=case["gold"]["paymentInstrument"])["merchant"], 0.0)

    def test_printed_merchant_spelling_is_right(self):
        case = self.spending(lambda c: len(c["gold"]["merchantAccepted"]) > 1)
        spelling = next(a for a in case["gold"]["merchantAccepted"] if a != case["gold"]["merchant"])
        self.assertEqual(self.scores(case, merchant=spelling)["merchant"], 1.0)

    def test_forcing_a_category_on_a_proposal_case_is_wrong(self):
        case = self.spending(lambda c: not c["gold"]["categoryAccepted"])
        user = self.users[case["user"]]
        title = next(t for t in user["categories"] if t != broad_category_alternative(case["gold"], user))
        s = self.scores(case, category=title, proposed_category="")
        self.assertEqual((s["category"], s["record_core"]), (0.0, 0.0))

    def test_missing_tag_breaks_record_all_not_core(self):
        case = self.spending(lambda c: c["gold"]["tags"])
        s = self.scores(case, tags=[])
        self.assertEqual((s["tags"], s["record_core"], s["record_all"]), (0.0, 1.0, 0.0))

    def test_proposed_tag_only_where_allowed(self):
        trip = self.spending(lambda c: c["gold"]["proposedTagMode"] == "optional")
        home = self.spending(lambda c: c["gold"]["proposedTagMode"] == "none" and not c["gold"]["tags"])
        self.assertEqual(self.scores(trip, proposed_tag="Lisbon Trip")["proposed_tag"], 1.0)
        self.assertEqual(self.scores(home, proposed_tag="Lisbon Trip")["proposed_tag"], 0.0)

    def test_confident_wrong_record_is_a_false_approval(self):
        case = self.spending(lambda c: c["gold"]["categoryAccepted"])
        result = self.draft_for(case, amount_minor=1, confidence=0.95)
        summary, _ = grade([case], self.users, [result])
        self.assertEqual(summary["overall"]["false_approval"], 1.0)

    def test_saying_not_spending_fails_a_receipt(self):
        case = self.spending()
        s = self.scores(case, is_spending=False)
        self.assertEqual((s["is_spending"], s["record_core"]), (0.0, 0.0))


class GradingPolicyTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _, cases, cls.users = load_suite("dev")
        cls.cases = {c["id"]: c for c in cases}

    def scores(self, case, user=None, **changes):
        answer = OracleBackend(case)
        user = user or self.users[case["user"]]
        result, _ = run_case(case, user, lambda c: answer, SETTINGS)
        result["draft"].update(changes)
        return score_case(case, result, user)

    def goods_proposal_case(self, shopping_title):
        case = copy.deepcopy(self.cases["es-u110-0000"])
        user = copy.deepcopy(self.users[case["user"]])
        user["categories"] = [t for t in user["categories"] if t != "Compras"] + ([shopping_title] if shopping_title else [])
        return case, user

    def test_a_broad_shopping_category_is_right_for_goods_with_no_category(self):
        case, user = self.goods_proposal_case("Compras")
        self.assertEqual(case["gold"]["proposedCategory"], "Ropa")
        self.assertEqual(self.scores(case, user, category="Compras", proposed_category="")["category"], 1.0)
        self.assertEqual(self.scores(case, user)["category"], 1.0)

    def test_without_a_shopping_category_the_goods_still_want_a_proposal(self):
        case, user = self.goods_proposal_case(None)
        self.assertEqual(self.scores(case, user, category="Café", proposed_category="")["category"], 0.0)

    def test_shopping_does_not_cover_what_is_not_goods(self):
        case = copy.deepcopy(self.cases["es-u110-0026"])
        user = copy.deepcopy(self.users[case["user"]])
        self.assertEqual(case["gold"]["proposedCategory"], "Gasolina")
        user["categories"] = sorted(set(user["categories"]) | {"Compras"})
        self.assertEqual(self.scores(case, user, category="Compras", proposed_category="")["category"], 0.0)

    def test_with_no_date_given_the_record_gets_the_arrival_date(self):
        case = next(c for c in self.cases.values() if c["expectedStatus"] == "needs_review"
                    and not c["gold"]["purchaseDate"])
        self.assertEqual(self.scores(case, date="")["date"], 1.0)
        self.assertEqual(self.scores(case, date=case["today"])["date"], 1.0)
        self.assertEqual(self.scores(case, date="2020-01-01")["date"], 0.0)

    def test_a_printed_date_is_still_required(self):
        case = next(c for c in self.cases.values() if c["expectedStatus"] == "needs_review"
                    and c["gold"]["purchaseDate"] and c["gold"]["purchaseDate"] != c["today"])
        self.assertEqual(self.scores(case, date=case["today"])["date"], 0.0)

    def test_a_trip_tag_after_the_trips_last_record_is_optional(self):
        case = self.cases["es-u110-0002"]
        tag = case["gold"]["tags"][0]
        self.assertEqual(self.scores(case, tags=[])["tags"], 1.0)
        self.assertEqual(self.scores(case, tags=[tag])["tags"], 1.0)
        self.assertEqual(self.scores(case, tags=[tag, "Gifts"])["tags"], 0.0)

    def test_a_trip_tag_inside_the_trip_is_required(self):
        case = self.cases["en-GB-u110-0022"]
        self.assertEqual(self.scores(case, tags=[])["tags"], 0.0)


class CommittedSuiteTests(unittest.TestCase):
    def test_dev_suite_fingerprint_matches_its_files(self):
        meta, cases, users = load_suite("dev")
        self.assertEqual(make_suite.fingerprint(cases, list(users.values())), meta["fingerprint"])

    def test_dev_suite_regenerates_identically(self):
        meta, cases, users = load_suite("dev")
        again, again_users = make_suite.build(meta["seed"], meta["perLocale"], meta["usersPerLocale"],
                                              novel_share=meta.get("novelShare", 0.0),
                                              novel_split=meta.get("novelSplit") or "train")
        self.assertEqual(make_suite.fingerprint(again, again_users), meta["fingerprint"])


class StatisticsTests(unittest.TestCase):
    def test_sign_test(self):
        self.assertEqual(sign_test(0, 0), 1.0)
        self.assertLess(sign_test(12, 1), 0.01)
        self.assertGreater(sign_test(5, 4), 0.5)

    def test_paired_interval_on_identical_runs_is_zero(self):
        rows = [{"x": float(i % 2)} for i in range(50)]
        self.assertEqual(paired_interval(rows, copy.deepcopy(rows), "x"), (0.0, 0.0, 0.0))


class BoardTests(unittest.TestCase):
    def summary(self):
        overall = {"n": 2, "record_core": 0.5, "record_core_ci95": (0.1, 0.9), "record_all": 0.5, "reward": 0.4,
                   "amount": 1.0, "merchant": 1.0, "category": 0.5, "tags": 1.0, "false_approval": 0.0}
        return {"overall": overall, "process": {"seconds": 123.4, "tool_calls": 1.5},
                "meta": {"model": "served-alias", "contract": "c", "backend": "openai"}}

    def test_public_table_says_nothing_about_where_a_run_was_served(self):
        with tempfile.TemporaryDirectory() as folder, unittest.mock.patch.object(board, "RUNS", folder):
            text = board.public_table([("some-run", self.summary())])
        self.assertNotIn("served-alias", text)
        self.assertNotIn("123.4", text)
        self.assertIn("| some-run | — | 2 |", text)

    def test_public_table_gives_the_size_of_a_served_gguf(self):
        with tempfile.TemporaryDirectory() as folder, unittest.mock.patch.object(board, "RUNS", folder):
            os.makedirs(os.path.join(folder, "small-q4_k_m"))
            with open(os.path.join(folder, "small-q4_k_m", "server.json"), "w", encoding="utf-8") as f:
                json.dump({"quantization": "Q4_K_M", "gguf_bytes": 534702208, "gpu": "a machine detail"}, f)
            text = board.public_table([("small-q4_k_m", self.summary())])
        self.assertIn("| small-q4_k_m | Q4_K_M, 0.53 GB |", text)
        self.assertNotIn("a machine detail", text)

    def test_writing_a_board_starts_a_results_file_when_there_is_none(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "RESULTS.md")
            write_board(path, "first table")
            with open(path, encoding="utf-8") as f:
                self.assertIn(f"{BEGIN}\nfirst table\n{END}", f.read())

    def test_writing_a_board_keeps_the_notes_around_it(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "RESULTS.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"intro\n{BEGIN}\nold\n{END}\nnotes\n")
            write_board(path, "new")
            with open(path, encoding="utf-8") as f:
                self.assertEqual(f.read(), f"intro\n{BEGIN}\nnew\n{END}\nnotes\n")


if __name__ == "__main__":
    unittest.main()
