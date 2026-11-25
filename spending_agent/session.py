"""One session: the spending agent working on one piece of content.

The loop is `LocalModelClient.complete`'s: tool rounds while the model asks
for tools, then one structured answer. Everything around it is a hook, the
way LangChain's agent middleware works:

    on_start      before anything is sent (content over budget is rejected)
    before_model  before every model call (compaction)
    after_model   after every model call
    after_tools   after every round of tool calls (a spent budget ends the phase)

No session ends without a row in review.csv and a transcript: a rejected,
failed or undecodable session parks there with its reason, so nothing the
user shared is lost."""

import re
import json
import uuid
from dataclasses import asdict, dataclass, field

from . import draft as contract
from . import tools
from .backend import arguments_of, describe_error, tool_calls_of, without_thinking
from .budget import ToolBudget
from .matching import key
from .money import decimal_string

CONTENT_CHARACTER_LIMIT = 4000
CONTEXT_CHARACTER_BUDGET = 16000
MAX_TOOL_ROUNDS = 10
TOOL_TOKENS = 800
ANSWER_TOKENS = 1500
FINDING_CHARACTER_LIMIT = 280
FINDINGS_CHARACTER_BUDGET = 900
COMPACTED_HEADER = "Earlier tool results, kept short to save space (the calls themselves were removed):"

QUERY_TOOLS = {
    "queryRecords": tools.query_records,
    "queryCategories": tools.query_categories,
    "queryMerchants": tools.query_merchants,
    "queryTags": tools.query_tags,
}


@dataclass
class SessionConfig:
    today: str
    local_currency: str = "GBP"
    preferences: list = field(default_factory=list)
    max_tool_rounds: int = MAX_TOOL_ROUNDS
    content_character_limit: int = CONTENT_CHARACTER_LIMIT
    context_character_budget: int = CONTEXT_CHARACTER_BUDGET
    tool_tokens: int = TOOL_TOKENS
    answer_tokens: int = ANSWER_TOKENS


@dataclass
class SessionResult:
    session_id: str
    status: str
    reason: str
    draft: object = None


class Session:
    def __init__(self, ledger, content, config, session_id=None):
        self.id = session_id or uuid.uuid4().hex[:12]
        self.ledger, self.content, self.config = ledger, content, config
        self.snapshot = ledger.snapshot()
        self.memory = ledger.memory()
        self.budget = ToolBudget()
        self.messages = []
        self.events = []
        self.phase = "tools"
        self.rounds = 0
        self.status, self.reason = "", ""
        self.draft = None

    def log(self, kind, **data):
        self.events.append({"kind": kind, **data})

    def end_tool_phase(self, why):
        if self.phase == "tools":
            self.phase = "answer"
            self.log("note", text=why)

    def stop(self, status, reason):
        self.phase = "done"
        self.status, self.reason = status, reason
        self.log("stop", status=status, reason=reason)


class Hook:
    def on_start(self, session):
        pass

    def before_model(self, session):
        pass

    def after_model(self, session, message):
        pass

    def after_tools(self, session):
        pass


class RejectLongContent(Hook):
    def on_start(self, session):
        length, limit = len(session.content), session.config.content_character_limit
        if length > limit:
            session.stop("rejected_too_long",
                         f"The content is {length} characters, over the agent's {limit}-character limit. "
                         "Record it manually.")


class Compaction(Hook):
    def before_model(self, session):
        before = conversation_size(session.messages)
        session.messages = compact(session.messages, session.config.context_character_budget)
        after = conversation_size(session.messages)
        if after < before:
            session.log("compaction", before=before, after=after)


class EndToolPhaseWhenSpent(Hook):
    def after_tools(self, session):
        if session.budget.is_spent:
            session.end_tool_phase("Tool budget spent; taking the answer.")


def default_hooks():
    return [RejectLongContent(), Compaction(), EndToolPhaseWhenSpent()]


def conversation_size(messages):
    return sum(len(m.get("content") or "") + len(json.dumps(m.get("tool_calls") or "")) for m in messages)


def _strip_thinking(message):
    if message.get("role") != "assistant" or not message.get("content"):
        return message
    return {**message, "content": without_thinking(message["content"])}


def _exchanges(messages):
    """(start, end) of each assistant tool-call turn with the tool answers after it."""
    spans, index = [], 0
    while index < len(messages):
        message = messages[index]
        if message.get("role") == "assistant" and message.get("tool_calls"):
            end = index + 1
            while end < len(messages) and messages[end].get("role") == "tool":
                end += 1
            spans.append((index, end))
            index = end
        else:
            index += 1
    return spans


def _findings_lines(tool_messages):
    lines = []
    for message in tool_messages:
        text = " ".join((message.get("content") or "").split())
        if len(text) > FINDING_CHARACTER_LIMIT:
            text = text[:FINDING_CHARACTER_LIMIT] + "…"
        lines.append(f"- {message.get('name', 'tool')}: {text}")
    return lines


def _bounded(lines):
    kept, used = [], 0
    for line in reversed(lines):
        if used + len(line) > FINDINGS_CHARACTER_BUDGET:
            break
        kept.insert(0, line)
        used += len(line)
    return kept


def compact(messages, budget):
    """Never send earlier reasoning back; past the budget, replace the oldest
    tool exchanges (never the latest) with a bounded summary of what their
    tools answered. Deterministic: no model is asked to summarise."""
    messages = [_strip_thinking(m) for m in messages]
    while conversation_size(messages) > budget:
        spans = _exchanges(messages)
        if len(spans) < 2:
            break
        start, end = spans[0]
        lines = _findings_lines(messages[start + 1:end])
        previous = messages[start - 1] if start > 0 else None
        if previous and previous.get("role") == "user" and (previous.get("content") or "").startswith(COMPACTED_HEADER):
            earlier = previous["content"].split("\n")[1:]
            summary = {"role": "user", "content": "\n".join([COMPACTED_HEADER] + _bounded(earlier + lines))}
            messages = messages[:start - 1] + [summary] + messages[end:]
        else:
            summary = {"role": "user", "content": "\n".join([COMPACTED_HEADER] + _bounded(lines))}
            messages = messages[:start] + [summary] + messages[end:]
    return messages


def _describe_call(name, arguments):
    shown = ", ".join(f"{k}={v!r}" for k, v in arguments.items() if v not in ("", None))
    return f"{name}({shown})"


def run_tool(session, name, arguments):
    description = _describe_call(name, arguments)
    if name == "calculator":
        return session.budget.call("calculator", arguments, lambda: tools.calculator(arguments), description)
    if name in QUERY_TOOLS:
        query = QUERY_TOOLS[name]
        return session.budget.call(name, arguments, lambda: query(session.snapshot, arguments), description)
    if name == "memory":
        action = str(arguments.get("action", "")).strip().lower()
        if action == "read":
            return session.budget.call("memoryRead", arguments, lambda: tools.memory_read(session.memory), description)
        if action in ("append", "replace"):
            return _edit_memory(session, action, arguments, description)
        return tools.ToolAnswer('Give action "read", "append" or "replace".', "invalid action")
    return tools.ToolAnswer(f'There is no tool called "{name}". Answer with the fields instead.', "unknown tool")


def _edit_memory(session, action, arguments, description):
    edit = tools.memory_append if action == "append" else tools.memory_replace
    outcome = {}

    def run():
        new_memory, answer = edit(session.memory, arguments)
        outcome["new"] = new_memory
        return answer

    answer = session.budget.call("memoryEdit", arguments, run, description)
    if "new" not in outcome:
        return answer
    if outcome["new"] is None:
        session.budget.refund("memoryEdit")
        return answer
    session.memory = outcome["new"]
    session.ledger.write_memory(session.memory, session.id)
    session.log("memory", action=action, memory=session.memory)
    return answer


def _assistant_turn(message, calls):
    content = without_thinking(message.get("content") or "")
    cut = content.find("<tool_call>")
    if cut >= 0:
        content = content[:cut].strip()
    return {"role": "assistant", "content": content, "tool_calls": [
        {"id": call["id"], "type": "function",
         "function": {"name": call["name"], "arguments": json.dumps(arguments_of(call), ensure_ascii=False)}}
        for call in calls]}


def _canonical(name, names):
    wanted = key(name)
    return next((n for n in names if key(n) == wanted), None) if wanted else None


def settle(draft, snapshot):
    """Snap names onto the user's own spelling; a category or tag that isn't
    theirs becomes a proposal, never a new category or tag."""
    if draft.category:
        canonical = _canonical(draft.category, snapshot.categories)
        if canonical:
            draft.category = canonical
        else:
            draft.problems.append(f"category {draft.category!r} is not one of the user's; kept as a proposal")
            draft.proposed_category = draft.proposed_category or draft.category
            draft.category = ""
    if draft.proposed_category:
        canonical = _canonical(draft.proposed_category, snapshot.categories)
        if canonical:
            draft.category = draft.category or canonical
            draft.proposed_category = ""
    kept = []
    for tag in draft.tags:
        canonical = _canonical(tag, snapshot.tags)
        if canonical and canonical not in kept:
            kept.append(canonical)
        elif not canonical and not draft.proposed_tag:
            draft.problems.append(f"tag {tag!r} is not one of the user's; kept as a proposal")
            draft.proposed_tag = tag
        elif not canonical:
            draft.problems.append(f"tag {tag!r} is not one of the user's; dropped")
    draft.tags = kept
    if draft.proposed_tag:
        canonical = _canonical(draft.proposed_tag, snapshot.tags)
        if canonical:
            if canonical not in draft.tags:
                draft.tags.append(canonical)
            draft.proposed_tag = ""
    return draft


def _review_row(session):
    d = session.draft
    row = {"session": session.id, "status": session.status, "reason": session.reason, "content": session.content}
    if d is not None:
        row.update({
            "is_spending": str(d.is_spending).lower(), "merchant": d.merchant,
            "payment_instrument": d.payment_instrument,
            "amount_minor": "" if d.amount_minor is None else str(d.amount_minor), "currency": d.currency,
            "date": d.date, "time": d.time, "category": d.category, "proposed_category": d.proposed_category,
            "tags": ";".join(d.tags), "proposed_tag": d.proposed_tag, "note": d.note,
            "confidence": f"{d.confidence:.2f}", "reasoning": d.reasoning,
        })
    return row


def _tool_rounds(session, backend, hooks):
    while session.phase == "tools" and session.rounds < session.config.max_tool_rounds:
        for hook in hooks:
            hook.before_model(session)
        message, usage = backend.complete(session.messages, contract.TOOL_DEFINITIONS, session.config.tool_tokens)
        session.log("model", phase="tools", message=message, usage=usage)
        for hook in hooks:
            hook.after_model(session, message)
        calls = tool_calls_of(message)
        if not calls:
            session.end_tool_phase("The model asked for no tool; taking the answer.")
            break
        session.messages.append(_assistant_turn(message, calls))
        for call in calls:
            arguments = arguments_of(call)
            answer = run_tool(session, call["name"], arguments)
            session.messages.append({"role": "tool", "tool_call_id": call["id"], "name": call["name"],
                                     "content": answer.message})
            session.log("tool", name=call["name"], arguments=arguments, summary=answer.summary,
                        answer=answer.message)
        session.rounds += 1
        for hook in hooks:
            hook.after_tools(session)
    session.end_tool_phase("Tool round limit reached; taking the answer.")


WHITESPACE_RUNAWAY = re.compile(r"\s{200,}")


def unreadable_answer_reason(content):
    """Why an answer didn't decode. A long run of whitespace is the grammar's
    signature: the model wanted a token the schema forbade, and whitespace was
    the only thing left to write until the token limit."""
    runaway = WHITESPACE_RUNAWAY.search(content)
    if runaway:
        before = content[:runaway.start()].rstrip()[-40:]
        return (f"The model's answer ran into whitespace after {before!r} until the token limit: the answer "
                "schema forbade what it meant to write there. Record it manually or retry.")
    return "The model's answer was not a JSON object. Record it manually or retry."


def _final_answer(session, backend, hooks):
    for hook in hooks:
        hook.before_model(session)
    closing = session.messages + [{"role": "user", "content": contract.CLOSING}]
    message, usage = backend.answer(closing, contract.answer_schema(), session.config.answer_tokens)
    session.log("model", phase="answer", message=message, usage=usage, response_format=backend.last_format,
                refusals=list(getattr(backend, "last_refusals", None) or []))
    for hook in hooks:
        hook.after_model(session, message)
    draft = contract.decode_draft(without_thinking(message.get("content") or ""))
    if draft is None:
        session.stop("failed", unreadable_answer_reason(message.get("content") or ""))
        return
    session.draft = settle(draft, session.snapshot)
    if not draft.is_spending:
        session.stop("not_spending", "The agent read this as not spending.")
        return
    problems = "; ".join(draft.problems)
    session.stop("needs_review", "Every draft is reviewed until the audits are ported."
                 + (f" Problems: {problems}." if problems else ""))


def run_session(ledger, content, backend, config, hooks=None, session_id=None):
    hooks = default_hooks() if hooks is None else hooks
    session = Session(ledger, content, config, session_id)
    for hook in hooks:
        hook.on_start(session)
    if session.phase != "done":
        prompt = contract.assemble_prompt(config.today, config.local_currency, session.snapshot.categories,
                                          session.snapshot.tags, config.preferences, session.memory, content)
        session.messages = [{"role": "system", "content": contract.INSTRUCTIONS},
                            {"role": "user", "content": prompt}]
        try:
            _tool_rounds(session, backend, hooks)
            _final_answer(session, backend, hooks)
        except Exception as error:
            session.stop("failed", f"The model server failed: {describe_error(error)}. Record it manually or retry.")
    ledger.add_to_review(_review_row(session))
    ledger.save_session(session.id, {
        "id": session.id, "contract": contract.contract_fingerprint(), "config": asdict(config),
        "status": session.status, "reason": session.reason, "content": content,
        "draft": asdict(session.draft) if session.draft else None,
        "messages": session.messages, "events": session.events,
        "tool_calls": session.budget.calls, "findings": session.budget.findings,
    })
    return SessionResult(session.id, session.status, session.reason, session.draft)


def describe_draft(draft):
    if draft is None:
        return "(no draft)"
    amount = decimal_string(draft.amount_minor, draft.currency or "XXX") if draft.amount_minor is not None else "?"
    lines = [f"merchant           {draft.merchant or '—'}",
             f"amount             {amount} {draft.currency}",
             f"date / time        {draft.date or '—'} {draft.time}",
             f"paid with          {draft.payment_instrument or '—'}",
             f"category           {draft.category or '—'}"
             + (f"   (proposed: {draft.proposed_category})" if draft.proposed_category else ""),
             f"tags               {', '.join(draft.tags) or '—'}"
             + (f"   (proposed: {draft.proposed_tag})" if draft.proposed_tag else ""),
             f"note               {draft.note or '—'}",
             f"confidence         {draft.confidence:.2f}",
             f"reasoning          {draft.reasoning}"]
    lines += [f"problem            {p}" for p in draft.problems]
    return "\n".join(lines)
