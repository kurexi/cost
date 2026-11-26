import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

from . import matching
from .money import minor_units, money

MAX_RECORD_GROUPS = 4
MAX_DATES_PER_GROUP = 3
MAX_MATCHES = 5
MAX_LISTED = 30
TOP_NAMES = 3
QUOTE_LIMIT = 40
MINIMUM_CADENCE_COUNT = 3
EVENT_SPAN_DAYS = 45
MEMORY_CHARACTER_LIMIT = 1000
IGNORED_WORDS = frozenset({"www", "com", "http", "https", "the", "and", "ltd", "limited", "inc"})
DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass
class ToolAnswer:
    message: str
    summary: str
    evidence: str = ""


def _arg(arguments, name):
    value = (arguments or {}).get(name, "")
    return "" if value is None else str(value).strip()


class CalculatorError(Exception):
    pass


CALCULATOR_MESSAGES = {
    "empty": 'The expression is empty. Provide one like "4.50 + 3.25".',
    "commas": "Commas are ambiguous. Rewrite every number with a dot decimal point, e.g. 1234.56.",
    "malformed": 'The expression is malformed. Check numbers, operators and parentheses, e.g. "(4.50 + 3.25) * 2".',
    "div0": "Division by zero is undefined. Check the divisor.",
}


def _number(text):
    if text.count(".") > 1 or text == ".":
        raise CalculatorError("malformed")
    return Decimal(text)


def _tokens(expression):
    tokens, digits = [], ""
    for ch in expression:
        if ch.isdigit() or ch == ".":
            digits += ch
            continue
        if digits:
            tokens.append(_number(digits))
            digits = ""
        if ch in "+-*/()":
            tokens.append(ch)
        elif ch == ",":
            raise CalculatorError("commas")
        elif not ch.isspace():
            raise CalculatorError(("char", ch))
    if digits:
        tokens.append(_number(digits))
    return tokens


def evaluate(expression):
    expression = expression.strip()
    if not expression:
        raise CalculatorError("empty")
    tokens = _tokens(expression)
    position = 0

    def peek():
        return tokens[position] if position < len(tokens) else None

    def take():
        nonlocal position
        position += 1
        return tokens[position - 1]

    def primary():
        token = peek()
        if token == "-":
            take()
            return -primary()
        if token == "+":
            take()
            return primary()
        if token == "(":
            take()
            value = total()
            if peek() == ")":
                take()
                return value
            raise CalculatorError("malformed")
        if isinstance(token, Decimal):
            return take()
        raise CalculatorError("malformed")

    def product():
        value = primary()
        while peek() in ("*", "/"):
            operator, right = take(), primary()
            if operator == "/":
                if right == 0:
                    raise CalculatorError("div0")
                value = value / right
            else:
                value = value * right
        return value

    def total():
        value = product()
        while peek() in ("+", "-"):
            operator, right = take(), product()
            value = value + right if operator == "+" else value - right
        return value

    value = total()
    if position != len(tokens):
        raise CalculatorError("malformed")
    return value


def result_string(value):
    return format(value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP).normalize(), "f")


def calculator(arguments):
    expression = _arg(arguments, "expression")
    try:
        value = evaluate(expression)
    except CalculatorError as error:
        kind = error.args[0]
        message = (f'Unsupported character "{kind[1]}". Use only numbers with dot decimals and + - * / ( ).'
                   if isinstance(kind, tuple) else CALCULATOR_MESSAGES[kind])
        return ToolAnswer(message, "invalid expression")
    except (InvalidOperation, RecursionError):
        return ToolAnswer(CALCULATOR_MESSAGES["malformed"], "invalid expression")
    line = f"{expression} = {result_string(value)}"
    return ToolAnswer(line, f"= {result_string(value)}", line)


def purchase_description(note):
    text = (note or "").strip()
    if ":" not in text:
        return text
    label, rest = text.split(":", 1)
    rest = rest.strip()
    if rest and len(label.split()) <= 3 and not any(c.isdigit() for c in label):
        return rest
    return text


def common_spelling(names):
    counts = Counter(name for name in names if name)
    return min(counts.items(), key=lambda pair: (-pair[1], pair[0]))[0] if counts else ""


def day(text):
    if not DAY.match(text or ""):
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def cadence(days):
    ordered = sorted(d for d in (day(text) for text in days) if d)
    if len(ordered) < MINIMUM_CADENCE_COUNT:
        return ""
    gaps = [(b - a).days for a, b in zip(ordered, ordered[1:])]
    if all(6 <= g <= 8 for g in gaps):
        return "about weekly"
    if all(26 <= g <= 35 for g in gaps):
        return "about monthly"
    if all(350 <= g <= 380 for g in gaps):
        return "about yearly"
    return ""


def _quoted(text):
    flat = text.replace('"', "'").replace("\n", " ")
    return flat if len(flat) <= QUOTE_LIMIT else flat[:QUOTE_LIMIT] + "…"


def _amount_range(records):
    by_currency = {}
    for record in records:
        by_currency.setdefault(record.currency, []).append(record.amount_minor)
    parts = []
    for currency, amounts in sorted(by_currency.items(), key=lambda pair: -len(pair[1])):
        low, high = min(amounts), max(amounts)
        parts.append(f"each {money(low, currency)}" if low == high
                     else f"{money(low, currency)} to {money(high, currency)}")
    return ", ".join(parts)


def _latest(records):
    return sorted({r.date for r in records if day(r.date)}, reverse=True)


def _last_seen(records):
    dates = _latest(records)
    return f"; last {dates[0]}" if dates else ""


def _top(names, limit=TOP_NAMES, counted=False):
    counts = Counter(name for name in names if name)
    ordered = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]
    return ", ".join(f"{name} ({count})" if counted else name for name, count in ordered)


def _words(text):
    return [w for w in (matching.key(word) for word in matching.words(text))
            if len(w) >= matching.MINIMUM_WORD_LENGTH and w not in IGNORED_WORDS]


RECORD_CONDITIONS = ("merchant", "category", "tag", "text", "amount", "minAmount", "maxAmount", "currency", "from", "to")


def _record_filter(arguments):
    given = {name: _arg(arguments, name) for name in RECORD_CONDITIONS}
    given = {name: value for name, value in given.items() if value}
    if not given:
        return None, "Give at least one condition: merchant, category, tag, text, amount, minAmount, maxAmount, currency, from or to."
    for name in ("from", "to"):
        if name in given and not DAY.match(given[name]):
            return None, f'Give {name} as a date like "2026-03-31".'
    for name in ("amount", "minAmount", "maxAmount"):
        if name in given and minor_units(given[name], given.get("currency") or "XXX") is None:
            return None, f'Give {name} as a plain decimal amount, e.g. "27.99".'
    text_words = _words(given.get("text", ""))
    if "text" in given and not text_words:
        return None, "Give text as a word or name of at least three letters from the content."

    def keep(record):
        if "merchant" in given and not matching.matches(given["merchant"], record.merchant):
            return False
        if "category" in given and not matching.matches(given["category"], record.category):
            return False
        if "tag" in given and not any(matching.matches(given["tag"], tag) for tag in record.tags):
            return False
        if "currency" in given and record.currency != given["currency"].upper():
            return False
        if "from" in given and record.date < given["from"]:
            return False
        if "to" in given and record.date > given["to"]:
            return False
        currency = record.currency
        if "amount" in given and minor_units(given["amount"], currency) != record.amount_minor:
            return False
        if "minAmount" in given and record.amount_minor < minor_units(given["minAmount"], currency):
            return False
        if "maxAmount" in given and record.amount_minor > minor_units(given["maxAmount"], currency):
            return False
        if text_words:
            haystack = matching.key(" ".join((record.merchant, record.note, record.content)))
            if not any(word in haystack for word in text_words):
                return False
        return True

    return keep, ""


def _record_group_line(records):
    records = sorted(records, key=lambda r: r.date, reverse=True)
    merchant = common_spelling([r.merchant for r in records]) or "(no merchant)"
    category = records[0].category
    line = f"- {merchant}, " + (f"filed under {category}" if category else "not filed under a category")
    count = "1 record" if len(records) == 1 else f"{len(records)} records"
    line += f": {count}, {_amount_range(records)}"
    rhythm = cadence([r.date for r in records if r.date])
    if rhythm:
        line += f", {rhythm}"
    dates = _latest(records)[:MAX_DATES_PER_GROUP]
    if dates:
        line += (" on " if len(records) == 1 else ", latest ") + ", ".join(dates)
    tags = _top([tag for r in records for tag in r.tags], counted=True)
    if tags:
        line += f", tagged {tags}"
    note = next((purchase_description(r.note) for r in records if purchase_description(r.note)), "")
    if note:
        line += f'. Note: "{_quoted(note)}"'
    return line + "."


def query_records(snapshot, arguments):
    keep, problem = _record_filter(arguments)
    if keep is None:
        return ToolAnswer(problem, "invalid query")
    found = [r for r in snapshot.records if keep(r)]
    if not found:
        return ToolAnswer("No records match. The user may not have recorded anything like this before — "
                          "don't try variations of the same query.", "no records")
    groups = {}
    for record in found:
        groups.setdefault((matching.key(record.merchant), record.category.casefold()), []).append(record)
    ordered = sorted(groups.values(), key=lambda g: (len(g), max(r.date for r in g)), reverse=True)
    lines = [_record_group_line(group) for group in ordered[:MAX_RECORD_GROUPS]]
    if len(ordered) > MAX_RECORD_GROUPS:
        lines.append(f"… and {len(ordered) - MAX_RECORD_GROUPS} more groups; add a condition to narrow it.")
    body = "\n".join(lines)
    count = "1 record matches" if len(found) == 1 else f"{len(found)} records match"
    return ToolAnswer(f"{count}, grouped by merchant and category, most first. They are past purchases, "
                      f"not this one:\n{body}", count.split(" match")[0], body)


def _category_line(title, records):
    mine = [r for r in records if r.category.casefold() == title.casefold()]
    if not mine:
        return f"- {title}: no records yet."
    merchants = _top([common_spelling([r.merchant]) for r in mine])
    count = "1 record" if len(mine) == 1 else f"{len(mine)} records"
    line = f"- {title}: {count}"
    if merchants:
        line += f"; merchants {merchants}"
    return line + f"; {_amount_range(mine)}{_last_seen(mine)}."


def query_categories(snapshot, arguments):
    pattern = _arg(arguments, "pattern")
    if not snapshot.categories:
        return ToolAnswer("The user has no categories yet.", "no categories")
    if not pattern:
        titles = snapshot.categories[:MAX_LISTED]
        body = "\n".join(_category_line(t, snapshot.records) for t in titles)
        return ToolAnswer(f"The user's categories:\n{body}", f"{len(titles)} categories", body)
    hits = matching.rank(pattern, snapshot.categories, limit=MAX_MATCHES)
    if not hits:
        return ToolAnswer(f'No category matches "{pattern}". Call with an empty pattern to list them all.',
                          "no match")
    body = "\n".join(_category_line(title, snapshot.records) for title, _ in hits)
    return ToolAnswer(f"Matching categories, best first (copy a title exactly):\n{body}",
                      f"{len(hits)} categories", body)


def query_merchants(snapshot, arguments):
    pattern = _arg(arguments, "pattern")
    if not pattern:
        return ToolAnswer("Give a merchant name, or part of one, as it appears in the content.", "invalid query")
    groups = {}
    for record in snapshot.records:
        if record.merchant:
            groups.setdefault(matching.key(record.merchant), []).append(record)
    spellings = {common_spelling([r.merchant for r in group]): group for group in groups.values()}
    hits = matching.rank(pattern, list(spellings), limit=MAX_MATCHES)
    if not hits:
        return ToolAnswer(f'No merchant in the user\'s records matches "{pattern}". This may be a merchant '
                          "they haven't bought from before — don't try variations of the same name.", "no match")
    lines = []
    for name, _ in hits:
        group = spellings[name]
        count = "1 record" if len(group) == 1 else f"{len(group)} records"
        line = f"- {name}: {count}"
        filed = _top([r.category for r in group], limit=TOP_NAMES, counted=True)
        line += f"; filed under {filed}" if filed else "; not filed under a category"
        line += f"; {_amount_range(group)}"
        paid = _top([r.payment_instrument for r in group])
        if paid:
            line += f"; paid with {paid}"
        lines.append(line + f"{_last_seen(group)}.")
    body = "\n".join(lines)
    return ToolAnswer(f"Merchants in the user's records, best match first, spelled as the user records them:\n"
                      f"{body}", f"{len(hits)} merchants", body)


def _tag_line(name, records):
    mine = [r for r in records if any(t.casefold() == name.casefold() for t in r.tags)]
    if not mine:
        return f"- {name}: no records yet."
    days = sorted(d for d in (day(r.date) for r in mine) if d)
    if not days:
        shape = "undated"
    elif (days[-1] - days[0]).days <= EVENT_SPAN_DAYS:
        shape = f"an event, {days[0].isoformat()} to {days[-1].isoformat()}"
    else:
        shape = f"standing, since {days[0].isoformat()}, latest {days[-1].isoformat()}"
    count = "1 record" if len(mine) == 1 else f"{len(mine)} records"
    line = f"- {name}: {count}; {shape}"
    merchants = _top([r.merchant for r in mine])
    if merchants:
        line += f"; merchants {merchants}"
    categories = _top([r.category for r in mine])
    if categories:
        line += f"; categories {categories}"
    return line + "."


def query_tags(snapshot, arguments):
    pattern = _arg(arguments, "pattern")
    if not snapshot.tags:
        return ToolAnswer("The user has no tags yet.", "no tags")
    if not pattern:
        names = snapshot.tags[:MAX_LISTED]
        body = "\n".join(_tag_line(n, snapshot.records) for n in names)
        return ToolAnswer(f"The user's tags:\n{body}", f"{len(names)} tags", body)
    hits = matching.rank(pattern, snapshot.tags, limit=MAX_MATCHES)
    if not hits:
        return ToolAnswer(f'No tag matches "{pattern}". Call with an empty pattern to list them all.', "no match")
    body = "\n".join(_tag_line(name, snapshot.records) for name, _ in hits)
    return ToolAnswer(f"Matching tags, best first (copy a name exactly):\n{body}", f"{len(hits)} tags", body)


def memory_read(memory):
    if not memory.strip():
        return ToolAnswer("Your memory is empty.", "empty")
    return ToolAnswer(f"Your memory ({len(memory)} of {MEMORY_CHARACTER_LIMIT} characters):\n{memory}",
                      f"{len(memory)} characters")


def _too_long(result, memory):
    return ToolAnswer(f"Refused: that would make your memory {len(result)} characters, over the "
                      f"{MEMORY_CHARACTER_LIMIT}-character limit (it is {len(memory)} now). Condense it with "
                      "replace instead.", "refused, too long")


def memory_append(memory, arguments):
    text = _arg(arguments, "text")
    if not text:
        return None, ToolAnswer("Give the text to append.", "invalid edit")
    result = f"{memory.rstrip()}\n{text}" if memory.strip() else text
    if len(result) > MEMORY_CHARACTER_LIMIT:
        return None, _too_long(result, memory)
    return result, ToolAnswer(f"Appended. Your memory is now {len(result)} of {MEMORY_CHARACTER_LIMIT} characters.",
                              "appended", text)


def memory_replace(memory, arguments):
    old, new = _arg(arguments, "old"), _arg(arguments, "new")
    if not old:
        return None, ToolAnswer("Give old, the exact passage to replace, copied from your memory.", "invalid edit")
    occurrences = memory.count(old)
    if occurrences != 1:
        problem = "isn't in your memory" if occurrences == 0 else f"appears {occurrences} times; give a longer passage"
        return None, ToolAnswer(f"Refused: that passage {problem}. Copy it exactly from your memory.",
                                "refused, passage not found" if occurrences == 0 else "refused, ambiguous")
    result = memory.replace(old, new)
    result = "\n".join(line for line in result.split("\n") if line.strip()) if not new else result
    if len(result) > MEMORY_CHARACTER_LIMIT:
        return None, _too_long(result, memory)
    verb = "Replaced" if new else "Removed"
    return result, ToolAnswer(f"{verb}. Your memory is now {len(result)} of {MEMORY_CHARACTER_LIMIT} characters.",
                              verb.lower(), new)
