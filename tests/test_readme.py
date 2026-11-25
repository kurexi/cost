import os
import re
import unittest

from spending_agent import draft
from spending_agent.budget import LIMITS, MAX_REFUSALS, TOTAL_CALLS
from spending_agent.session import CONTENT_CHARACTER_LIMIT, MAX_TOOL_ROUNDS
from spending_agent.tools import MEMORY_CHARACTER_LIMIT

SPELLED_OUT = {6: "six"}
README = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md")


def read_readme():
    with open(README, encoding="utf-8") as f:
        return f.read()


def flat(text):
    return " ".join(text.split())


def tool_rows(text):
    rows = re.findall(r"^\| `(\w+)` \| .* \| ([^|]+) \|$", text, re.MULTILINE)
    return {name: limit.strip() for name, limit in rows}


class ReadmeTests(unittest.TestCase):
    def test_the_tool_table_lists_exactly_the_tools_the_model_is_given(self):
        self.assertEqual(sorted(tool_rows(read_readme())), sorted(draft.TOOL_NAMES))

    def test_the_tool_table_gives_each_tool_its_budget(self):
        rows = tool_rows(read_readme())
        for name, limit in rows.items():
            if name == "memory":
                self.assertEqual(limit, f"{LIMITS['memoryRead']} reads, {LIMITS['memoryEdit']} edit")
            else:
                self.assertEqual(limit, str(LIMITS[name]), name)

    def test_the_session_limits_are_the_agents(self):
        text = flat(read_readme())
        self.assertIn(f"At most {TOTAL_CALLS} calls a session in all, over at most {MAX_TOOL_ROUNDS} model rounds",
                      text)
        self.assertEqual(MAX_REFUSALS, 1, "the README says a second refusal ends the tool phase")
        self.assertIn(f"Over {CONTENT_CHARACTER_LIMIT:,} characters?", text)
        self.assertIn(f"or {MAX_TOOL_ROUNDS} rounds", text)
        self.assertIn(f"at most {MEMORY_CHARACTER_LIMIT:,} characters", text)

    def test_the_count_of_tools_is_the_agents(self):
        text = flat(read_readme())
        count = SPELLED_OUT[len(draft.TOOL_NAMES)]
        self.assertIn(f"## {count.capitalize()} tools, a small budget", text)
        self.assertIn(f"The model has only these {count} tools", text)


if __name__ == "__main__":
    unittest.main()
