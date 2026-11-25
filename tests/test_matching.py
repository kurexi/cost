import unittest

from spending_agent.matching import key, matches, rank


class MatchingTests(unittest.TestCase):
    def test_typography_folds(self):
        self.assertEqual(key("Sainsbury’s"), key("SAINSBURYS"))

    def test_short_name_only_as_whole_token(self):
        self.assertTrue(matches("M&S", "M&S Simply Food"))
        self.assertFalse(matches("BP", "BPM Studios"))

    def test_prefix_either_way(self):
        self.assertTrue(matches("Tesco", "Tesco Express"))
        self.assertTrue(matches("Tesco Express Southampton", "Tesco Express"))

    def test_near_spelling(self):
        self.assertTrue(matches("Grocries", "Groceries"))

    def test_singular_finds_plural(self):
        self.assertTrue(matches("grocery", "Groceries"))
        self.assertTrue(matches("Gift", "Gifts"))
        self.assertFalse(matches("Bus", "Bu"))

    def test_unrelated(self):
        self.assertFalse(matches("Fitness", "Groceries"))

    def test_rank_best_first(self):
        ranked = [name for name, _ in rank("tesco", ["Tesco Cafe", "Tesco", "Costa"])]
        self.assertEqual(ranked, ["Tesco", "Tesco Cafe"])


if __name__ == "__main__":
    unittest.main()
