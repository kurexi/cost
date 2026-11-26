import json
import os
import re
import unicodedata
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

from common import fold

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOCAL_MODEL_AGENT = os.path.join(REPO, "cost 2", "Intelligence", "LocalModel", "LocalModelAgent.swift")
LEDGER_SEARCH = os.path.join(REPO, "cost 2", "Intelligence", "LedgerSearch.swift")


def _swift_string(literal):
    return literal.replace('\\"', '"').replace("\\\\", "\\")


def extraction_schema():
    source = open(LOCAL_MODEL_AGENT, encoding="utf-8").read()
    block = source[source.index("static let extractionSchema"):]
    block = block[:block.index("])")]
    props, order = {}, []
    for name, kind, desc in re.findall(r'\.init\("(\w+)", \.(string|boolean|number)Schema\("((?:[^"\\]|\\.)*)"', block):
        node = {"type": {"string": "string", "boolean": "boolean", "number": "number"}[kind],
                "description": _swift_string(desc)}
        if kind == "number":
            node.update({"minimum": 0, "maximum": 1})
        props[name] = node
        order.append(name)
    assert order and order[0] == "reasoning", order
    return {"type": "object", "properties": props, "required": order, "additionalProperties": False}


def _swift_multiline(source, marker):
    start = source.index(marker)
    start = source.index('"""', start) + 3
    end = source.index('"""', start)
    lines = [l.strip() for l in source[start:end].strip("\n").split("\n")]
    text = ""
    for l in lines:
        text += l[:-1] if l.endswith("\\") else l + "\n"
    return text.strip().replace('\\"', '"').replace("\\\\", "\\")


def _tool_descriptions():
    agent = open(LOCAL_MODEL_AGENT, encoding="utf-8").read()
    calc = _swift_multiline(agent, 'name: "calculator"')
    search = _swift_multiline(agent, 'name: "searchReceipt"')
    ledger_src = open(LEDGER_SEARCH, encoding="utf-8").read()
    ledger = _swift_multiline(ledger_src, "static let toolDescription")
    guide = {name: re.search(r'static let %s = "((?:[^"\\]|\\.)*)"' % name, ledger_src).group(1).replace('\\"', '"')
             for name in ("queryGuide", "amountGuide")}
    return calc, search, ledger, guide["queryGuide"], guide["amountGuide"]


CALC_DESCRIPTION, SEARCH_DESCRIPTION, LEDGER_DESCRIPTION, LEDGER_QUERY_GUIDE, LEDGER_AMOUNT_GUIDE = _tool_descriptions()


def tool_definitions(with_ledger=True):
    tools = [
        {"type": "function", "function": {"name": "calculator", "description": CALC_DESCRIPTION, "parameters": {
            "type": "object", "properties": {"expression": {"type": "string", "description":
                'The expression to evaluate: plain numbers with dot decimals and + - * / ( ) only. No currency symbols, no thousands separators, e.g. "12.99 + 3.10 - 2.00".'}},
            "required": ["expression"]}}},
        {"type": "function", "function": {"name": "searchReceipt", "description": SEARCH_DESCRIPTION, "parameters": {
            "type": "object", "properties": {"pattern": {"type": "string", "description":
                'A short case-insensitive regular expression, e.g. "total|balance" or "[0-9]+\\.[0-9]{2}".'}},
            "required": ["pattern"]}}},
    ]
    if with_ledger:
        tools.append({"type": "function", "function": {"name": "searchRecords", "description": LEDGER_DESCRIPTION,
                      "parameters": {"type": "object", "properties": {
                          "query": {"type": "string", "description": LEDGER_QUERY_GUIDE},
                          "amount": {"type": "string", "description": LEDGER_AMOUNT_GUIDE}},
                                     "required": ["query", "amount"]}}})
    return tools


class CalcError(Exception):
    pass


CALC_MESSAGES = {
    "empty": 'The expression is empty. Provide one like "4.50 + 3.25".',
    "commas": "Commas are ambiguous. Rewrite every number with a dot decimal point, e.g. 1234.56.",
    "malformed": 'The expression is malformed. Check numbers, operators and parentheses, e.g. "(4.50 + 3.25) * 2".',
    "div0": "Division by zero is undefined. Check the divisor.",
}


def _tokenize(expr):
    tokens, buf = [], ""
    for ch in expr:
        if ch.isdigit() or ch == ".":
            buf += ch
            continue
        if buf:
            tokens.append(Decimal(buf) if buf.count(".") <= 1 and buf != "." else _bad("malformed"))
            buf = ""
        if ch in "+-*/()":
            tokens.append(ch)
        elif ch == ",":
            raise CalcError("commas")
        elif ch.isspace():
            continue
        else:
            raise CalcError(("char", ch))
    if buf:
        tokens.append(Decimal(buf) if buf.count(".") <= 1 and buf != "." else _bad("malformed"))
    return tokens


def _bad(kind):
    raise CalcError(kind)


def evaluate(expr):
    expr = expr.strip()
    if not expr:
        raise CalcError("empty")
    tokens = _tokenize(expr)
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def take():
        nonlocal pos
        pos += 1
        return tokens[pos - 1]

    def primary():
        t = peek()
        if t == "-":
            take()
            return -primary()
        if t == "+":
            take()
            return primary()
        if t == "(":
            take()
            v = sum_()
            if take_if(")"):
                return v
            raise CalcError("malformed")
        if isinstance(t, Decimal):
            return take()
        raise CalcError("malformed")

    def take_if(sym):
        if peek() == sym:
            take()
            return True
        return False

    def product():
        v = primary()
        while peek() in ("*", "/"):
            op = take()
            r = primary()
            if op == "/":
                if r == 0:
                    raise CalcError("div0")
                v = v / r
            else:
                v = v * r
        return v

    def sum_():
        v = product()
        while peek() in ("+", "-"):
            op = take()
            r = product()
            v = v + r if op == "+" else v - r
        return v

    value = sum_()
    if pos != len(tokens):
        raise CalcError("malformed")
    return value


def result_string(value):
    rounded = value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP).normalize()
    text = format(rounded, "f")
    return text


def calculator_answer(expression):
    try:
        value = evaluate(expression)
    except CalcError as e:
        kind = e.args[0]
        if isinstance(kind, tuple):
            msg = f'Unsupported character "{kind[1]}". Use only numbers with dot decimals and + - * / ( ).'
        else:
            msg = CALC_MESSAGES[kind]
        return msg, "invalid expression"
    except (InvalidOperation, RecursionError):
        return CALC_MESSAGES["malformed"], "invalid expression"
    echo = expression.strip()
    result = result_string(value)
    return f"{echo} = {result}", f"= {result}"


_TYPO = [("‘", "'"), ("’", "'"), ("‛", "'"), ("′", "'"), ("´", "'"),
         ("“", '"'), ("”", '"'), ("„", '"'),
         ("–", "-"), ("—", "-"), ("−", "-"), (" ", " ")]


def normalized_for_matching(text):
    for a, b in _TYPO:
        text = text.replace(a, b)
    text = unicodedata.normalize("NFKC", text)
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def punctuation_stripped(text):
    return fold(text)


def receipt_lines(text):
    return [l.strip() for l in text.splitlines() if l.strip()]


def _capped(line, n=200):
    return line if len(line) <= n else line[:n] + "…"


def is_literal(pattern):
    return not any(c in "\\^$.|?*+()[]{}" for c in pattern)


def search_matches(pattern, text, max_matches=12):
    regex = re.compile(normalized_for_matching(pattern), re.IGNORECASE)
    lines = receipt_lines(text)
    out = []
    for i, line in enumerate(lines):
        if regex.search(normalized_for_matching(line)):
            out.append((i + 1, _capped(line)))
            if len(out) >= max_matches:
                break
    if out or not is_literal(pattern):
        return out
    needle = punctuation_stripped(pattern)
    if not needle:
        return out
    for i, line in enumerate(lines):
        if needle in punctuation_stripped(line):
            out.append((i + 1, _capped(line)))
            if len(out) >= max_matches:
                break
    return out


def search_report(matches, lines, max_emitted=30):
    wanted = set()
    for number, _ in matches:
        wanted.update(range(max(1, number - 1), min(len(lines), number + 1) + 1))
    rendered, previous = [], None
    for number in sorted(wanted):
        if len(rendered) >= max_emitted:
            rendered.append("…")
            break
        if previous is not None and number > previous + 1:
            rendered.append("…")
        rendered.append(f"{number}: {_capped(lines[number - 1])}")
        previous = number
    return "\n".join(rendered)


def search_answer(pattern, text):
    trimmed = pattern.strip()
    if not trimmed or len(trimmed) > 200:
        return 'Provide a short regular expression pattern, e.g. "total" or "[0-9]+\\.[0-9]{2}".', "invalid pattern"
    try:
        hits = search_matches(trimmed, text)
    except re.error:
        return 'That is not a valid regular expression. Try a simpler pattern, e.g. "total" or "[0-9]+\\.[0-9]{2}".', "invalid pattern"
    if not hits:
        return ("No lines match that pattern. The text really doesn't contain it — don't try variations of the same search.",
                "no match")
    body = search_report(hits, receipt_lines(text))
    return ("Numbered lines, each match shown with the line before and after it (this receipt's OCR often puts an "
            f"amount on its own line, away from the label it belongs to):\n{body}",
            "1 line" if len(hits) == 1 else f"{len(hits)} lines")


ZERO_DECIMAL_CURRENCIES = {"JPY", "KRW", "VND", "CLP", "ISK", "UGX", "RWF", "XOF", "XAF"}


def minor_units(raw, currency):
    cleaned = "".join(c for c in raw if c.isdigit() or c in ".,-")
    if not cleaned:
        return None
    if "." in cleaned and "," in cleaned:
        if cleaned.rindex(".") > cleaned.rindex(","):
            cleaned = cleaned.replace(",", "")
        else:
            cleaned = cleaned.replace(".", "").replace(",", ".")
    elif "," in cleaned:
        parts = [x for x in cleaned.split(",") if x]
        cleaned = cleaned.replace(",", "." if len(parts) == 2 and len(parts[1]) == 2 else "")
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None
    if value < 0:
        return None
    scale = 0 if currency.upper() in ZERO_DECIMAL_CURRENCIES else 2
    return int((value * (Decimal(10) ** scale)).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def decimal_string(minor, currency):
    return str(minor) if currency.upper() in ZERO_DECIMAL_CURRENCIES else f"{minor / 100:.2f}"


def purchase_description(note):
    t = note.strip()
    if ":" not in t:
        return t
    label, rest = t.split(":", 1)
    rest = rest.strip()
    if rest and len(label.split()) <= 3 and not any(c.isdigit() for c in label):
        return rest
    return t


def common_spelling(names):
    counts = {}
    for n in names:
        counts[n] = counts.get(n, 0) + 1
    return min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0] if counts else ""


LEDGER_MIN_TOKEN = 3
LEDGER_IGNORED = {"www", "com", "http", "https", "the", "and", "ltd", "limited", "inc"}
LEDGER_MAX_GROUPS = 4
LEDGER_MAX_DATES = 3
LEDGER_QUOTE_LIMIT = 40
LEDGER_MIN_CADENCE = 3


def ledger_tokens(query):
    folded = normalized_for_matching(query).lower()
    out = []
    for piece in re.split(r"[^\w]+|_", folded):
        if len(piece) >= LEDGER_MIN_TOKEN and piece not in LEDGER_IGNORED and piece not in out:
            out.append(piece)
    return out


def _entry_day(entry):
    return entry["date"][:10]


def ledger_hits(query, amount, entries):
    tokens = ledger_tokens(query)
    wants_amount = bool(amount.strip())
    if not tokens and not wants_amount:
        return []
    hits = []
    for e in entries:
        amount_matches = False
        if wants_amount:
            minor = minor_units(amount, e["currencyCode"])
            amount_matches = minor is not None and minor > 0 and minor == e["amountMinorUnits"]
        name_matches = printed_matches = False
        matched = 0
        if tokens:
            named = punctuation_stripped(e["merchant"] + " " + e.get("note", ""))
            printed = punctuation_stripped(e.get("printed", ""))
            in_name = sum(1 for t in tokens if t in named)
            matched = sum(1 for t in tokens if t in named or t in printed)
            name_matches = in_name > 0
            printed_matches = not name_matches and matched > 0
        if amount_matches or name_matches or printed_matches:
            text = name_matches or printed_matches
            rank = (3 if text else 2) if amount_matches else 1
            hits.append({"entry": e, "amount": amount_matches, "name": name_matches, "printed": printed_matches,
                         "words": matched, "rank": rank})
    return hits


def ledger_cadence(days):
    from datetime import date as _date
    if len(days) < LEDGER_MIN_CADENCE:
        return None
    ds = sorted(_date.fromisoformat(d) for d in days)
    gaps = [(b - a).days for a, b in zip(ds, ds[1:])]
    if all(6 <= g <= 8 for g in gaps):
        return "about weekly"
    if all(26 <= g <= 35 for g in gaps):
        return "about monthly"
    if all(350 <= g <= 380 for g in gaps):
        return "about yearly"
    return None


def _quoted(text):
    flat = text.replace('"', "'")
    return flat if len(flat) <= LEDGER_QUOTE_LIMIT else flat[:LEDGER_QUOTE_LIMIT] + "…"


def _ledger_line(group, tokens):
    entries = sorted((h["entry"] for h in group), key=_entry_day, reverse=True)
    merchant = common_spelling([e["merchant"] for e in entries])
    category = entries[0].get("category", "") if entries else ""
    line = "- " + (merchant or "(no merchant)")
    line += ", not filed under a category" if not category else f", filed under {category}"

    def money(e):
        return decimal_string(e["amountMinorUnits"], e["currencyCode"]) + " " + e["currencyCode"]

    if len(entries) == 1:
        line += f": 1 record, {money(entries[0])} on {_entry_day(entries[0])}"
    else:
        line += f": {len(entries)} records"
        if len({e['currencyCode'] for e in entries}) == 1:
            low = min(entries, key=lambda e: e["amountMinorUnits"])
            high = max(entries, key=lambda e: e["amountMinorUnits"])
            line += (f", each {money(low)}" if len({e['amountMinorUnits'] for e in entries}) == 1
                     else f", from {money(low)} to {money(high)}")
        cadence = ledger_cadence([_entry_day(e) for e in entries])
        if cadence:
            line += f", {cadence}"
        line += ", latest " + ", ".join(_entry_day(e) for e in entries[:LEDGER_MAX_DATES])
    note = next((d for d in (purchase_description(e.get("note", "")) for e in entries) if d), "")
    if note:
        line += f'. Note: "{_quoted(note)}"'
    if any(h["printed"] for h in group):
        printed_line = next((l for e in entries if e.get("printed")
                             for l in receipt_lines(e["printed"])
                             if any(t in punctuation_stripped(l) for t in tokens)), None)
        if printed_line:
            line += f'. Its content printed "{_quoted(printed_line)}"'
    return line + "."


def ledger_lines(hits, query):
    order, groups = [], {}
    for h in hits:
        key = punctuation_stripped(h["entry"]["merchant"]) + "|" + h["entry"].get("category", "").lower()
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(h)
    ranked = sorted((groups[k] for k in order), key=lambda g: (
        -max(h["rank"] for h in g), -max(h["words"] for h in g), -len(g),
        tuple(-ord(c) for c in max(_entry_day(h["entry"]) for h in g))))
    tokens = ledger_tokens(query)
    return [_ledger_line(g, tokens) for g in ranked[:LEDGER_MAX_GROUPS]]


def ledger_answer(query, amount, entries):
    amount = amount.strip()
    if not ledger_tokens(query) and not amount:
        return 'Give a word or name from the content, the amount printed in it (e.g. "27.99"), or both.', "invalid lookup"
    found = ledger_hits(query, amount, entries)
    if not found:
        return ("No past records match. This may be a merchant or a kind of payment the user has not recorded "
                "before — don't try variations of the same lookup.", "no records")
    body = "\n".join(ledger_lines(found, query))
    return ("The user's past records that match, most relevant first. A payment of the same amount at a regular "
            "interval is usually the same thing again; a different amount at the same shop may be something else:\n"
            + body, "1 record" if len(found) == 1 else f"{len(found)} records")


class ToolBudget:
    def __init__(self, max_searches=6, max_calculations=8, max_lookups=3, max_refusals=1):
        self.max_searches, self.max_calculations = max_searches, max_calculations
        self.max_lookups, self.max_refusals = max_lookups, max_refusals
        self.searches = self.calculations = self.lookups = self.refusals = 0
        self.search_cache, self.lookup_cache = {}, {}

    @property
    def is_spent(self):
        return self.refusals > self.max_refusals

    def _refusal(self, message, summary):
        self.refusals += 1
        if self.is_spent:
            return "Tool use is over. Fill in the answer fields now.", "stopping tool use"
        return message, summary

    def search(self, pattern, text):
        if self.searches >= self.max_searches:
            return self._refusal("No more receipt searches are available. Fill in the answer fields now, using the "
                                 "excerpt and the search results you already have.", "no searches left")
        self.searches += 1
        key = punctuation_stripped(pattern)
        if key in self.search_cache:
            msg, summary = self.search_cache[key]
            msg += "\nYou already searched that pattern — repeating searches will not reveal anything new."
            summary += ", repeat"
        else:
            msg, summary = search_answer(pattern, text)
            self.search_cache[key] = (msg, summary)
        left = self.max_searches - self.searches
        if left == 0:
            msg += "\nThat was the last available search. Fill in the answer fields now."
        elif left <= 2:
            msg += f"\n{left} search{'' if left == 1 else 'es'} left — finish reading and answer."
        return msg, summary

    def calculate(self, expression):
        if self.calculations >= self.max_calculations:
            return self._refusal("No more calculations are available. Use the result you already computed and fill "
                                 "in the answer fields now.", "no calculations left")
        self.calculations += 1
        return calculator_answer(expression)

    def lookup(self, query, amount, ledger):
        if self.lookups >= self.max_lookups:
            return self._refusal("No more record lookups are available. Fill in the answer fields now, using what "
                                 "the lookups already told you.", "no lookups left")
        self.lookups += 1
        query, amount = query.strip(), amount.strip()
        key = punctuation_stripped(query) + "|" + amount
        if key in self.lookup_cache:
            msg, summary = self.lookup_cache[key]
            msg += "\nYou already made that lookup — repeating it will not reveal anything new."
            summary += ", repeat"
        else:
            msg, summary = ledger_answer(query, amount, ledger)
            self.lookup_cache[key] = (msg, summary)
        if self.lookups >= self.max_lookups:
            msg += "\nThat was the last available lookup. Fill in the answer fields now."
        return msg, summary


def argument_value(name, arguments):
    if isinstance(arguments, dict):
        if name in arguments:
            return str(arguments[name])
        if len(arguments) == 1:
            return str(next(iter(arguments.values())))
        return ""
    if isinstance(arguments, str):
        try:
            return argument_value(name, json.loads(arguments))
        except (json.JSONDecodeError, TypeError):
            return arguments
    return str(arguments)


class Episode:

    def __init__(self, content, ledger, with_ledger=True):
        self.content, self.ledger, self.with_ledger = content, ledger, with_ledger
        self.budget = ToolBudget()
        self.trail = []

    def call(self, name, arguments):
        if name == "calculator":
            msg, summary = self.budget.calculate(argument_value("expression", arguments))
        elif name == "searchReceipt":
            msg, summary = self.budget.search(argument_value("pattern", arguments), self.content)
        elif name == "searchRecords" and self.with_ledger and self.ledger:
            fields = arguments
            if isinstance(fields, str):
                try:
                    fields = json.loads(fields)
                except (json.JSONDecodeError, TypeError):
                    fields = {}
            fields = fields if isinstance(fields, dict) else {}
            msg, summary = self.budget.lookup(str(fields.get("query", "") or ""), str(fields.get("amount", "") or ""),
                                              self.ledger)
        else:
            msg, summary = f'There is no tool called "{name}". Answer with the fields instead.', "unknown tool"
        self.trail.append({"tool": name, "arguments": arguments, "summary": summary})
        return msg
