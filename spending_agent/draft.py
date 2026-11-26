import hashlib
import json
from decimal import Decimal
from dataclasses import dataclass, field

from .money import minor_units

INSTRUCTIONS = (
    "You are the spending agent inside a personal expense tracking app: you help the user keep their spending "
    "records. The user shares receipts (OCR text from screenshots or photos), order confirmations, bank "
    "notifications and free-form notes, and from each you draft one spending record for them to keep. "
    "Be precise with numbers: when a grand total is printed, copy it exactly. Receipt OCR reads the two columns "
    "as separate lines, so the amount belonging to a label is often on the line just after it — a bare \"TOTAL:\" "
    "line means the total is the amount printed below it. Only when no total is printed, add up the line items "
    "(plus any tax) with the calculator — never do arithmetic in your head, and never add the total, subtotal or "
    "payment lines into that sum. A price printed for comparison — an \"ORIGINAL PRICE\" or \"WAS\" price, an RRP, "
    "a saving or a discount line — is not money paid. "
    "A payment has two sides: the merchant is who was paid; the payment instrument is the card, bank, wallet or "
    "app that paid them. A bank or card name is never the merchant, even when it is the largest text on a screen. "
    "The user's records are your memory of what they have bought before. Use the query tools when the content "
    "alone doesn't settle something: queryMerchants to see how the user names a shop and where they file it, "
    "queryRecords when a notice prints only an amount, a web address or a reference, or to see whether the same "
    "payment recurs, queryCategories to see what a category holds, queryTags to see what a tag covers. Tool "
    "answers describe past records, never this purchase. Don't call a tool to re-check what the content already "
    "shows, and stop after a few calls. "
    "Category: pick the user's category this purchase belongs in and copy its title exactly; when none fits, "
    "leave category empty and propose a new title — a proposal the user can accept is far better than a wrong "
    "category. The same shop can be filed under different categories depending on what was bought. "
    "Tags: most spending belongs under no tag. Put a purchase under one of the user's tags only when it plainly "
    "belongs to that trip, event, project or purpose, and propose a new tag only when the content shows a "
    "distinct trip, event or project beginning. "
    "Your memory is the notes you keep for yourself across sessions. Edit it with the memory tool when you learn "
    "something that will help with future records: append a line, or replace a passage that turned out wrong. "
    "Keep it short — it is limited to 1,000 characters — and never use it as a log of single purchases. "
    "The user's standing preferences are their own instructions: follow them unless the content plainly "
    "contradicts them. "
    "Start the answer with the reasoning field: what the content is, who was paid and what paid them, where the "
    "total is, how you know the currency, and why the category and any tag fit, naming any records you relied on."
)

CLOSING = ("Now give the final answer as a single JSON object and nothing else. "
           "Fill the fields in the order they appear in the schema.")

FIELDS = [
    ("reasoning", "string", "Think step by step before the other fields, in two or three short sentences: what "
     "kind of content this is, who was paid (and, if shown, what card, bank or app paid them — that is not the "
     "merchant), which line holds the grand total, the currency evidence, why the category and any tag fit, and "
     "which past records you relied on."),
    ("isSpending", "boolean", "True only if the content genuinely describes money being spent (a receipt, order "
     "confirmation, bill or purchase). False for refunds, money received, codes, deliveries and anything else."),
    ("merchant", "string", "The shop, restaurant, company or person that was paid — the counterparty. Never the "
     "bank, card, wallet or payment app that paid them. Spell it as the user's records do when it is plainly the "
     "same shop. Empty string if unknown."),
    ("paymentInstrument", "string", "The card, bank, wallet or payment app the money left, when the content names "
     "one — what paid, as opposed to who was paid. Empty string if none is shown."),
    ("totalAmount", "number", "The final total amount paid, as a plain decimal number like 12.34, never negative. "
     "Use the grand total including tax, not a line item."),
    ("currencyCode", "string", "ISO 4217 currency code such as USD, GBP, EUR, JPY, CNY. Infer it from currency "
     "symbols, the merchant's country or town, the language, or anything else in the content. Only when the "
     "content carries no clue at all, fall back to the user's local currency given in the prompt — unless a "
     "standing preference says what to do instead, which wins."),
    ("purchaseDate", "string", "Purchase date in yyyy-MM-dd format if it appears in the content, resolving a "
     "missing year or a relative day from today's date; otherwise an empty string. Never invent a date."),
    ("purchaseTime", "string", "Purchase time in 24-hour HH:mm format if present, otherwise an empty string."),
    ("category", "string", "The user's existing category this purchase belongs in, copied EXACTLY as listed. "
     "When past purchases at this merchant are known, the items decide: pick the category of the past purchase "
     "most like this one. Empty string when none of the existing categories genuinely fits; never force a poor "
     "fit."),
    ("proposedCategory", "string", "Only when category is empty: a short new category title of one to three words "
     "that this purchase belongs in, e.g. \"Pet Supplies\". Empty string whenever category is filled, and never "
     "one that restates an existing category."),
    ("tags", "array", "Names of the user's existing tags this purchase belongs under, copied EXACTLY. Usually "
     "empty; sometimes one."),
    ("proposedTag", "string", "A short name for a NEW tag only when this purchase clearly starts a distinct trip, "
     "event or project the user has no tag for yet, e.g. \"Japan Trip\". Empty string otherwise — never for "
     "ordinary spending, and never one that restates the category."),
    ("note", "string", "A short one-line note summarizing what was bought, e.g. \"Groceries: milk, bread, eggs\"."),
    ("confidence", "number", "Your overall confidence in this record from 0.0 to 1.0. Use below 0.6 when the "
     "total, merchant or category is uncertain."),
]


FIELD_SHAPES = {
    "reasoning": {"maxLength": 700},
    "merchant": {"maxLength": 80},
    "paymentInstrument": {"maxLength": 60},
    "totalAmount": {"anyOf": [{"type": "number", "minimum": 0}, {"type": "string", "maxLength": 24}]},
    "currencyCode": {"pattern": "^([A-Z]{3})?$"},
    "purchaseDate": {"pattern": "^([0-9]{4}-[0-9]{2}-[0-9]{2})?$"},
    "purchaseTime": {"pattern": "^([0-9]{2}:[0-9]{2})?$"},
    "category": {"maxLength": 60},
    "proposedCategory": {"maxLength": 40},
    "tags": {"maxItems": 3, "items": {"type": "string", "maxLength": 40}},
    "proposedTag": {"maxLength": 40},
    "note": {"maxLength": 160},
    "confidence": {"minimum": 0, "maximum": 1},
}


def answer_schema():
    properties = {}
    for name, kind, description in FIELDS:
        node = {"type": kind, "description": description}
        if kind == "array":
            node["items"] = {"type": "string"}
        node.update(FIELD_SHAPES.get(name, {}))
        if "anyOf" in node:
            del node["type"]
        properties[name] = node
    return {"type": "object", "properties": properties, "required": [name for name, _, _ in FIELDS],
            "additionalProperties": False}


def _function(name, description, parameters, required=()):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object",
        "properties": {p: {"type": "string", "description": d} for p, d in parameters},
        "required": list(required)}}}


TOOL_DEFINITIONS = [
    _function("calculator", "Evaluates one arithmetic expression over plain decimal numbers with + - * / and "
              "parentheses, e.g. \"4.50 + 3.25 + 1.20\". Use it whenever amounts must be added or adjusted — never "
              "do arithmetic in your head.",
              [("expression", "Plain numbers with dot decimals and + - * / ( ) only. No currency symbols, no "
                "thousands separators, e.g. \"12.99 + 3.10 - 2.00\".")], ["expression"]),
    _function("queryRecords", "Looks up the user's past spending records by conditions, all optional but at least "
              "one: merchant, category and tag match loosely; amount is exact; dates are yyyy-MM-dd. Answers with "
              "the matching records grouped by merchant and category: counts, amounts, how regularly they recur, "
              "latest dates. Use it for a notice that prints only an amount, a web address or a reference.",
              [("merchant", "A merchant name, matched loosely."),
               ("category", "A category title, matched loosely."),
               ("tag", "A tag name, matched loosely."),
               ("text", "A word from the content to find in past merchants, notes and receipts."),
               ("amount", "An exact amount as a plain decimal, e.g. \"27.99\"."),
               ("minAmount", "The smallest amount, as a plain decimal."),
               ("maxAmount", "The largest amount, as a plain decimal."),
               ("currency", "An ISO 4217 code, e.g. \"GBP\"."),
               ("from", "The earliest date, yyyy-MM-dd."),
               ("to", "The latest date, yyyy-MM-dd.")]),
    _function("queryCategories", "Finds the user's categories by a loose match on the title; an empty pattern "
              "lists them all. Answers with each category's exact title, how many records it holds, the merchants "
              "filed under it, its amounts and when it was last used.",
              [("pattern", "Part of a category title, or empty for all.")], ["pattern"]),
    _function("queryMerchants", "Finds merchants in the user's records by a loose, OCR-tolerant match on the name "
              "— \"TESCO STORES 3297\" finds \"Tesco\". Answers with the name as the user spells it, how often they "
              "buy there, the categories they file it under, amounts, the cards used and the last date.",
              [("pattern", "The merchant name as printed in the content, or part of it.")], ["pattern"]),
    _function("queryTags", "Finds the user's tags by a loose match on the name; an empty pattern lists them all. "
              "Answers with each tag's exact name, whether it is an event (with its dates) or standing, and the "
              "merchants and categories under it.",
              [("pattern", "Part of a tag name, or empty for all.")], ["pattern"]),
    _function("memory", "Reads or edits your memory, the notes you keep across sessions (at most 1,000 "
              "characters). action \"read\" returns it; \"append\" adds text as a new line; \"replace\" swaps the "
              "passage old for new (an empty new removes it). One edit per session.",
              [("action", "\"read\", \"append\" or \"replace\"."),
               ("text", "For append: the line to add."),
               ("old", "For replace: the exact passage to change, copied from your memory."),
               ("new", "For replace: the text to put in its place; empty to remove the passage.")], ["action"]),
]

TOOL_NAMES = [definition["function"]["name"] for definition in TOOL_DEFINITIONS]


def assemble_prompt(today, local_currency, categories, tags, preferences, memory, content):
    prompt = f"Today's date: {today}\nThe user's local currency: {local_currency}\n\n"
    if categories:
        prompt += "The user's categories:\n" + "\n".join(f"- {c}" for c in categories) + "\n\n"
    else:
        prompt += "The user has no categories yet.\n\n"
    if tags:
        prompt += "The user's tags:\n" + "\n".join(f"- {t}" for t in tags) + "\n\n"
    if preferences:
        prompt += ("The user's standing preferences — follow these rules:\n"
                   + "\n".join(f"- {p}" for p in preferences) + "\n\n")
    if memory.strip():
        prompt += f'Your memory (notes you wrote for yourself):\n"""\n{memory.strip()}\n"""\n\n'
    else:
        prompt += "Your memory is empty.\n\n"
    prompt += f'Content to record:\n"""\n{content}\n"""'
    return prompt


def contract_fingerprint():
    blob = json.dumps({"instructions": INSTRUCTIONS, "closing": CLOSING, "schema": answer_schema(),
                       "tools": TOOL_DEFINITIONS}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass
class RecordDraft:
    reasoning: str = ""
    is_spending: bool = True
    merchant: str = ""
    payment_instrument: str = ""
    total_amount: str = ""
    amount_minor: int = None
    currency: str = ""
    date: str = ""
    time: str = ""
    category: str = ""
    proposed_category: str = ""
    tags: list = field(default_factory=list)
    proposed_tag: str = ""
    note: str = ""
    confidence: float = 0.0
    problems: list = field(default_factory=list)


def _parse_object(text):
    text = (text or "").strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(text[start:end + 1])
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None


def _amount_text(value):
    if isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        return format(Decimal(repr(value)), "f")
    return _text(value)


def _text(value):
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return ""
    return str(value).strip()


def _boolean(value, default=True):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    return default


def _names(value):
    if isinstance(value, list):
        return [_text(v) for v in value if _text(v)]
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return []


def _confidence(value):
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def decode_draft(text):
    obj = _parse_object(text)
    if obj is None:
        return None
    draft = RecordDraft(
        reasoning=_text(obj.get("reasoning")),
        is_spending=_boolean(obj.get("isSpending")),
        merchant=_text(obj.get("merchant")),
        payment_instrument=_text(obj.get("paymentInstrument")),
        total_amount=_amount_text(obj.get("totalAmount")),
        currency=_text(obj.get("currencyCode")).upper(),
        date=_text(obj.get("purchaseDate")),
        time=_text(obj.get("purchaseTime")),
        category=_text(obj.get("category")),
        proposed_category=_text(obj.get("proposedCategory")),
        tags=_names(obj.get("tags")),
        proposed_tag=_text(obj.get("proposedTag")),
        note=_text(obj.get("note")),
        confidence=_confidence(obj.get("confidence")),
    )
    missing = [name for name, _, _ in FIELDS if name not in obj]
    if missing:
        draft.problems.append("missing fields: " + ", ".join(missing))
    if draft.total_amount:
        draft.amount_minor = minor_units(draft.total_amount, draft.currency or "XXX")
        if draft.amount_minor is None:
            draft.problems.append(f"unreadable amount {draft.total_amount!r}")
    return draft
