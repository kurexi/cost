import json
import unittest

from spending_agent import draft
from spending_agent.backend import arguments_of, tool_calls_of, without_thinking


class DraftTests(unittest.TestCase):
    def test_reasoning_is_the_first_field(self):
        self.assertEqual(list(draft.answer_schema()["properties"])[0], "reasoning")

    def test_every_text_field_is_bounded(self):
        for name, node in draft.answer_schema()["properties"].items():
            for option in node.get("anyOf", [node]):
                if option["type"] == "string":
                    self.assertTrue("maxLength" in option or "pattern" in option, name)

    def test_a_pattern_carries_its_own_length_bound(self):
        for name, node in draft.answer_schema()["properties"].items():
            if "pattern" in node:
                self.assertNotIn("maxLength", node, f"{name}: xgrammar refuses a pattern beside a length cap")
                self.assertNotRegex(node["pattern"], r"[+*]", f"{name}: an unbounded repeat leaves it uncapped")

    def test_shapes_admit_good_values_and_refuse_bad_ones(self):
        import re
        properties = draft.answer_schema()["properties"]
        cases = {                 "currencyCode": (["GBP", ""], ["gbp", "£"]),
                 "purchaseDate": (["2026-11-18", ""], ["18/11/2026"]),
                 "purchaseTime": (["09:05", ""], ["9:05 am"])}
        for name, (good, bad) in cases.items():
            pattern = re.compile(properties[name]["pattern"])
            for value in good:
                self.assertTrue(pattern.fullmatch(value), (name, value))
            for value in bad:
                self.assertFalse(pattern.fullmatch(value), (name, value))

    def test_the_amount_takes_a_number_or_any_short_string(self):
        node = draft.answer_schema()["properties"]["totalAmount"]
        self.assertNotIn("type", node)
        self.assertEqual([option["type"] for option in node["anyOf"]], ["number", "string"])
        self.assertNotIn("pattern", json.dumps(node))

    def test_amounts_decode_as_the_model_wrote_them(self):
        cases = [(46.28, "EUR", 4628), (1181, "JPY", 1181), ("€6,17", "EUR", 617), ("¥1,181", "JPY", 1181),
                 (1e21, "EUR", 10 ** 23), (True, "EUR", None), ("-46,28", "EUR", None)]
        for value, currency, minor in cases:
            decoded = draft.decode_draft(json.dumps({"totalAmount": value, "currencyCode": currency}))
            self.assertEqual(decoded.amount_minor, minor, value)

    def test_tool_names(self):
        self.assertEqual(draft.TOOL_NAMES, ["calculator", "queryRecords", "queryCategories", "queryMerchants",
                                            "queryTags", "memory"])

    def test_fingerprint_is_stable(self):
        self.assertEqual(draft.contract_fingerprint(), draft.contract_fingerprint())

    def test_prompt_lists_categories_tags_and_memory(self):
        prompt = draft.assemble_prompt("2026-11-18", "GBP", ["Groceries"], ["Work"], ["Use EUR in Dublin."],
                                       "Monzo is a card.", "TESCO 12.40")
        for part in ("- Groceries", "- Work", "Use EUR in Dublin.", "Monzo is a card.", "TESCO 12.40"):
            self.assertIn(part, prompt)
        self.assertIn("Your memory is empty.", draft.assemble_prompt("2026-11-18", "GBP", [], [], [], " ", "x"))

    def test_decode_is_lenient(self):
        text = 'Sure! {"isSpending": "true", "merchant": "Tesco", "totalAmount": "£12.40", "currencyCode": "gbp", ' \
               '"tags": "Work, Home", "confidence": "1.4"} done'
        decoded = draft.decode_draft(text)
        self.assertEqual((decoded.amount_minor, decoded.currency, decoded.tags), (1240, "GBP", ["Work", "Home"]))
        self.assertEqual(decoded.confidence, 1.0)
        self.assertTrue(any("missing fields" in p for p in decoded.problems))

    def test_decode_without_object(self):
        self.assertIsNone(draft.decode_draft("I can't help with that."))

    def test_unreadable_amount_is_a_problem(self):
        decoded = draft.decode_draft(json.dumps({"totalAmount": "twelve", "currencyCode": "GBP"}))
        self.assertIsNone(decoded.amount_minor)
        self.assertTrue(any("unreadable amount" in p for p in decoded.problems))


class BackendParsingTests(unittest.TestCase):
    def test_native_tool_calls(self):
        message = {"tool_calls": [{"id": "a", "function": {"name": "calculator",
                                                           "arguments": '{"expression": "1+1"}'}}]}
        [call] = tool_calls_of(message)
        self.assertEqual((call["name"], arguments_of(call)), ("calculator", {"expression": "1+1"}))

    def test_xml_tool_calls_in_content(self):
        message = {"content": "<tool_call>\n<function=queryMerchants>\n<parameter=pattern>\nTesco\n</parameter>\n"
                              "</function>\n</tool_call>"}
        [call] = tool_calls_of(message)
        self.assertEqual((call["name"], arguments_of(call)), ("queryMerchants", {"pattern": "Tesco"}))

    def test_json_tool_calls_in_content(self):
        message = {"content": '<tool_call>{"name": "queryTags", "arguments": {"pattern": "trip"}}</tool_call>'}
        [call] = tool_calls_of(message)
        self.assertEqual((call["name"], arguments_of(call)), ("queryTags", {"pattern": "trip"}))

    def test_thinking_is_stripped(self):
        self.assertEqual(without_thinking("<think>hmm</think> {}"), "{}")
        self.assertEqual(without_thinking("hmm</think>{}"), "{}")


if __name__ == "__main__":
    unittest.main()
