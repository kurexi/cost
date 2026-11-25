"""The tool budget, a port of `AgentToolBudgetState` widened to the v2 tools.

Each tool has its own limit and every call counts towards a ceiling for the
session. A repeated query is answered from the cache with a note that it
reveals nothing new. Past a limit a call is refused; the second refusal
spends the budget, and the loop then ends the tool phase."""

from . import matching
from .tools import ToolAnswer

LIMITS = {
    "calculator": 8,
    "queryRecords": 4,
    "queryCategories": 3,
    "queryMerchants": 3,
    "queryTags": 3,
    "memoryRead": 2,
    "memoryEdit": 1,
}
TOTAL_CALLS = 12
MAX_REFUSALS = 1

REFUSALS = {
    "calculator": "No more calculations are available. Use the result you already computed and fill in the answer fields now.",
    "queryRecords": "No more record queries are available. Fill in the answer fields now, using what the queries already told you.",
    "queryCategories": "No more category queries are available. Fill in the answer fields now.",
    "queryMerchants": "No more merchant queries are available. Fill in the answer fields now.",
    "queryTags": "No more tag queries are available. Fill in the answer fields now.",
    "memoryRead": "Your memory is already in the prompt; there are no more reads. Fill in the answer fields now.",
    "memoryEdit": "You have already edited your memory in this session. Fill in the answer fields now.",
}
TOTAL_REFUSAL = "No more tool calls are available in this session. Fill in the answer fields now."
SPENT = "Tool use is over. Fill in the answer fields now."
CACHED_SLOTS = ("queryRecords", "queryCategories", "queryMerchants", "queryTags")


class ToolBudget:
    def __init__(self, limits=None, total=TOTAL_CALLS, max_refusals=MAX_REFUSALS):
        self.limits = dict(LIMITS if limits is None else limits)
        self.total, self.max_refusals = total, max_refusals
        self.used = {slot: 0 for slot in self.limits}
        self.calls = 0
        self.refusals = 0
        self.findings = []
        self._cache = {}

    @property
    def is_spent(self):
        return self.refusals > self.max_refusals

    def _refuse(self, message, summary):
        self.refusals += 1
        if self.is_spent:
            return ToolAnswer(SPENT, "stopping tool use")
        return ToolAnswer(message, summary)

    @staticmethod
    def cache_key(slot, arguments):
        parts = sorted((name, matching.key(str(value))) for name, value in (arguments or {}).items() if value)
        return slot, tuple(parts)

    def refund(self, slot):
        """An edit the memory rules refused (too long, passage not found) is
        not an edit: the agent keeps its chance to make a valid one. The call
        still counts towards the session's ceiling, so retries stay bounded."""
        self.used[slot] = max(0, self.used[slot] - 1)

    def call(self, slot, arguments, run, description):
        if self.calls >= self.total:
            return self._refuse(TOTAL_REFUSAL, "no calls left")
        if self.used[slot] >= self.limits[slot]:
            return self._refuse(REFUSALS[slot], f"no {slot} left")
        self.calls += 1
        self.used[slot] += 1
        key = self.cache_key(slot, arguments)
        if slot in CACHED_SLOTS and key in self._cache:
            cached = self._cache[key]
            answer = ToolAnswer(cached.message + "\nYou already asked that — repeating it will not reveal anything new.",
                                cached.summary + ", repeat")
        else:
            answer = run()
            if slot in CACHED_SLOTS:
                self._cache[key] = answer
            if answer.evidence:
                self.findings.append((description, answer.evidence))
        if self.used[slot] >= self.limits[slot] and slot != "memoryEdit":
            answer = ToolAnswer(answer.message + f"\nThat was the last {slot} call available.",
                                answer.summary, answer.evidence)
        return answer
