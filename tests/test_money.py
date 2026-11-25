import unittest

from spending_agent.money import decimal_string, minor_units, money


class MoneyTests(unittest.TestCase):
    def test_plain_decimal(self):
        self.assertEqual(minor_units("12.34", "GBP"), 1234)

    def test_decimal_comma(self):
        self.assertEqual(minor_units("12,34", "EUR"), 1234)

    def test_thousands_separators(self):
        self.assertEqual(minor_units("1,234.50", "USD"), 123450)
        self.assertEqual(minor_units("1.234,50", "EUR"), 123450)

    def test_zero_decimal_currency(self):
        self.assertEqual(minor_units("1400", "JPY"), 1400)
        self.assertEqual(decimal_string(1400, "JPY"), "1400")

    def test_symbols_are_ignored(self):
        self.assertEqual(minor_units("£11.92", "GBP"), 1192)

    def test_unreadable_and_negative(self):
        self.assertIsNone(minor_units("", "GBP"))
        self.assertIsNone(minor_units("-3.00", "GBP"))

    def test_round_trip(self):
        self.assertEqual(money(905, "GBP"), "9.05 GBP")


if __name__ == "__main__":
    unittest.main()
