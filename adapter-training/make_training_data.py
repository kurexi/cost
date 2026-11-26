#!/usr/bin/env python3
import argparse
import json
import random
from datetime import date, timedelta

INSTRUCTIONS = (
    "You are the spending assistant inside a personal expense tracking app. "
    "The user shares receipts (as OCR text from screenshots or photos), order "
    "confirmations and free-form notes. Extract one spending record from the "
    "content you are given. Be precise with numbers: when a grand total is "
    "printed, copy it exactly, never compute or invent amounts. Receipt OCR "
    "reads the two columns as separate lines, so the amount belonging to a "
    "label is often on the line just after it — a bare \"TOTAL:\" line means the "
    "total is the amount printed below it. A payment has two sides: the "
    "merchant is who was paid, the payment instrument is what paid them. A "
    "bank, card issuer, wallet or payment app is never the merchant, even "
    "when it is the largest text on a screenshot — on a wallet or bank screen "
    "it is usually the heading, and the merchant is the name below it that "
    "the money went to; put it in paymentInstrument. Only when no readable grand total "
    "is printed, add up the line items (plus any tax) with the calculator tool "
    "— never do arithmetic in your head, and never add the total, subtotal or "
    "payment lines into that sum, they repeat the items rather than adding to "
    "them. A price printed for comparison is not money paid: an \"ORIGINAL "
    "PRICE\" or \"WAS\" price above a price reduction, an RRP, a saving or a "
    "discount line is never part of the total — only the reduced price the "
    "item actually rang up at is. Use the "
    "searchReceipt tool sparingly: only to re-read a line you can see is "
    "garbled or cut off, never to re-check something the excerpt already "
    "shows you and never once per line item. It searches this one receipt's "
    "text and nothing else, and it already ignores differences in "
    "apostrophes, accents and punctuation, so if it reports no match the "
    "text truly isn't there — trying variants of the same search only wastes "
    "it. The user's past records are your memory of what they have paid for "
    "before, and the searchRecords tool reads them. When the content does not "
    "plainly say who was paid or what was bought — a bank or card notice that "
    "prints only a web address, a city or a reference — look it up before "
    "answering: search by the exact amount, and by a word from the content. A "
    "payment of the same amount at a regular interval is usually the same thing "
    "again, so record it under that merchant and category; say in your "
    "reasoning which past records you relied on. Name a merchant the way the "
    "user's past records name it when it is plainly the same shop — a till "
    "prints \"TESCO STORES 3297\", the records say \"Tesco\". "
    "After a few tool calls you must stop and fill in the fields with "
    "your best answer. Start with the reasoning field: think through what "
    "the content is, who was paid and what paid them, where the grand total "
    "is, and how you know the currency "
    "and category, before filling in the other fields. Each of the user's "
    "categories is listed with the merchants they file under it: pick the one "
    "this purchase genuinely belongs in and copy its title exactly, and when "
    "none fits leave category empty and propose a new title instead — a "
    "proposal the user can accept is far better than a wrong category saved "
    "to the ledger. When the user's past purchases at a merchant named in the "
    "content are listed, judge this purchase by what was bought: the same shop "
    "can be filed under different categories depending on the items, and the "
    "past purchase most like this one is the best guide. Any standing "
    "preferences listed for you are the user's own instructions about how they "
    "want their spending recorded: follow them unless the content in front of "
    "you plainly contradicts them. The knowledge base "
    "facts about this user are context for interpreting what you have read."
)

REAL_ISSUERS = [
    "Monzo", "Starling", "Revolut", "Barclays", "Lloyds", "HSBC", "NatWest",
    "Nationwide", "Santander", "Halifax", "First Direct", "Chase", "Amex",
    "Wise", "Klarna", "PayPal", "Curve", "Zopa", "Tide", "Capital One", "Citi",
    "Wells Fargo", "Bank of America", "Discover", "N26", "Bunq", "ING", "BBVA",
    "Rabobank", "Nordea", "Sparkasse", "Apple Card", "Google Pay", "Apple Pay",
]
INVENTED_ONSETS = ["Vel", "Tor", "Mar", "Ken", "Bri", "Sol", "Lan", "Ard", "Fen",
                   "Cor", "Hal", "Nor", "Wes", "Ald", "Pen", "Rav", "Ost", "Elm"]
INVENTED_CODAS = ["mar", "dis", "ton", "by", "ford", "wick", "stone", "field",
                  "gate", "holm", "wood", "bridge", "mere", "dale", "bank"]
KIND_SUFFIXES = ["", "", " Debit", " Credit", " Card", " Bank", " Platinum",
                 " Gold", " Current Account", " Everyday", " Pay", " Wallet"]
GENERIC_INSTRUMENTS = ["Debit card", "Credit card", "Visa Debit", "Mastercard",
                       "Visa", "Amex", "Maestro"]
CARD_SCHEMES = ["VISA", "MASTERCARD", "AMEX", "MAESTRO"]


def instrument_name(rng):
    roll = rng.random()
    if roll < 0.15:
        return rng.choice(GENERIC_INSTRUMENTS)
    if roll < 0.55:
        stem = rng.choice(REAL_ISSUERS)
        if stem in ("Apple Card", "Google Pay", "Apple Pay"):
            return stem
    else:
        stem = rng.choice(INVENTED_ONSETS) + rng.choice(INVENTED_CODAS)
        if rng.random() < 0.15:
            return stem + rng.choice(["Pay", "Card", "Bank"])
    return stem + rng.choice(KIND_SUFFIXES)


def till_payment_line(rng):
    roll = rng.random()
    last4 = str(rng.randrange(1000, 10000))
    if roll < 0.2:
        scheme = rng.choice(CARD_SCHEMES)
        return f"CONTACTLESS {scheme}", scheme.title()
    if roll < 0.4:
        scheme = rng.choice(CARD_SCHEMES)
        kind = rng.choice(["DEBIT", "CREDIT"])
        return f"{scheme} {kind}", f"{scheme.title()} {kind.title()}"
    if roll < 0.55:
        return f"CARD ****{last4}", "Card"
    if roll < 0.7:
        return "CASH", "Cash"
    if roll < 0.8:
        wallet = rng.choice(["APPLE PAY", "GOOGLE PAY"])
        return wallet, wallet.title()
    issuer = instrument_name(rng)
    return f"{issuer.upper()} ****{last4}", issuer

MERCHANTS = [
    ("Tesco", "Groceries", "GBP", ["milk", "bread", "eggs", "bananas", "pasta", "cheddar"]),
    ("Sainsbury's", "Groceries", "GBP", ["chicken breast", "rice", "yoghurt", "spinach"]),
    ("Waitrose", "Groceries", "GBP", ["sourdough", "olive oil", "tomatoes"]),
    ("Whole Foods Market", "Groceries", "USD", ["avocados", "almond milk", "granola"]),
    ("Aldi", "Groceries", "EUR", ["butter", "flour", "apples", "oats"]),
    ("Costa Coffee", "Coffee", "GBP", ["flat white", "latte", "almond croissant"]),
    ("Starbucks", "Coffee", "USD", ["caffe latte", "cold brew", "banana bread"]),
    ("Pret A Manger", "Coffee", "GBP", ["americano", "chicken wrap"]),
    ("Caffè Nero", "Coffee", "GBP", ["cappuccino", "brownie"]),
    ("McDonald's", "Dining", "GBP", ["Big Mac meal", "McFlurry", "fries"]),
    ("Nando's", "Dining", "GBP", ["half chicken", "peri chips", "bottomless drink"]),
    ("Wagamama", "Dining", "GBP", ["katsu curry", "gyoza", "ramen"]),
    ("Pizza Express", "Dining", "GBP", ["margherita", "dough balls", "tiramisu"]),
    ("Ichiran", "Dining", "JPY", ["tonkotsu ramen", "extra noodles", "green tea"]),
    ("Uber", "Transport", "GBP", ["trip"]),
    ("Trainline", "Transport", "GBP", ["off-peak return", "railcard booking fee"]),
    ("Shell", "Fuel", "GBP", ["unleaded 95"]),
    ("BP", "Fuel", "USD", ["regular unleaded"]),
    ("Amazon", "Shopping", "GBP", ["USB-C cable", "phone case", "notebook", "desk lamp"]),
    ("ASOS", "Clothing", "GBP", ["t-shirt", "jeans", "hoodie"]),
    ("Uniqlo", "Clothing", "JPY", ["airism tee", "chino trousers"]),
    ("IKEA", "Home", "EUR", ["LACK side table", "tealights", "frame"]),
    ("Boots", "Health", "GBP", ["paracetamol", "vitamin D", "plasters"]),
    ("CVS Pharmacy", "Health", "USD", ["ibuprofen", "band-aids"]),
    ("Odeon Cinemas", "Entertainment", "GBP", ["2x adult ticket", "popcorn"]),
    ("Steam", "Entertainment", "USD", ["game purchase"]),
    ("Netflix", "Subscriptions", "GBP", ["monthly plan"]),
    ("Spotify", "Subscriptions", "GBP", ["premium individual"]),
    ("Vodafone", "Bills", "GBP", ["monthly plan"]),
    ("British Gas", "Bills", "GBP", ["energy bill"]),
    ("Luckin Coffee", "Coffee", "CNY", ["iced coconut latte"]),
    ("Meituan", "Dining", "CNY", ["delivery order"]),
]

ALL_CATEGORIES = sorted({m[1] for m in MERCHANTS})

MIXED_MERCHANTS = {
    "Tesco": ("Coffee", ["latte", "flat white", "cappuccino", "pain au chocolat"]),
    "Sainsbury's": ("Coffee", ["americano", "croissant", "hot chocolate"]),
    "Waitrose": ("Coffee", ["flat white", "cinnamon swirl"]),
    "Whole Foods Market": ("Coffee", ["cold brew", "blueberry muffin"]),
    "Shell": ("Groceries", ["sandwich", "crisps", "still water", "milk"]),
    "BP": ("Groceries", ["coffee to go", "energy drink", "chewing gum"]),
    "Amazon": ("Home", ["cushion covers", "storage boxes", "lamp shade"]),
    "IKEA": ("Dining", ["meatballs", "cinnamon bun", "lingonberry drink"]),
    "Boots": ("Shopping", ["hair dryer", "perfume", "electric toothbrush"]),
}

NON_SPENDING = [
    "Hey, are we still on for Friday? I can pick you up at 7 if that works.\nSure! See you then 🙂",
    "Weather today: 21°C, partly cloudy. Chance of rain this evening 40%.",
    "Reminder: Dentist appointment tomorrow at 09:30, Riverside Dental Practice.",
    "Your one-time verification code is 482913. It expires in 10 minutes. Do not share this code.",
    "Meeting notes:\n- Q3 roadmap review\n- Hiring update\n- AOB: office move in September",
    "Boarding pass\nPassenger: A. KHAN\nFlight BA341 LHR→CDG\nGate B42 Seat 17A Boards 18:05",
    "Top 10 hiking trails in the Lake District — number 4 will surprise you. Read more…",
    "Your parcel is out for delivery and will arrive today between 14:00 and 16:00.",
    "shopping list: milk, eggs, bread, washing up liquid, bin bags",
    "Payment received: John Smith sent you £25.00 — 'thanks for lunch!'",
]

ZERO_DECIMAL = {"JPY", "KRW"}
SYMBOL = {"GBP": "£", "USD": "$", "EUR": "€", "JPY": "¥", "CNY": "¥"}

DATE_STYLES = [
    lambda d: d.strftime("%d/%m/%Y"),
    lambda d: d.strftime("%m/%d/%Y"),
    lambda d: d.strftime("%d %b %Y"),
    lambda d: d.strftime("%B %d, %Y"),
    lambda d: d.strftime("%Y-%m-%d"),
    lambda d: d.strftime("%d.%m.%y"),
]

OCR_SUBS = [("O", "0"), ("0", "O"), ("l", "1"), ("I", "1"), ("S", "5"), ("B", "8")]


def minor_units(rng, currency):
    if currency in ZERO_DECIMAL:
        return rng.randrange(300, 12000, 10)
    return rng.randrange(199, 15000)


def amount_string(minor, currency):
    if currency in ZERO_DECIMAL:
        return str(minor)
    return f"{minor // 100}.{minor % 100:02d}"


def display_amount(rng, minor, currency):
    plain = amount_string(minor, currency)
    sym = SYMBOL[currency]
    style = rng.random()
    if currency == "EUR" and style < 0.5:
        return plain.replace(".", ",") + " €"
    if currency in ZERO_DECIMAL and style < 0.5:
        return sym + f"{minor:,}"
    if style < 0.6:
        return sym + plain
    if style < 0.8:
        return f"{currency} {plain}"
    return plain


def ocr_noise(rng, text):
    chars = list(text)
    for _ in range(rng.randint(1, 4)):
        i = rng.randrange(len(chars))
        for a, b in rng.sample(OCR_SUBS, len(OCR_SUBS)):
            if chars[i] == a:
                chars[i] = b
                break
    text = "".join(chars)
    if rng.random() < 0.4:
        text = text.replace("  ", " ").replace(": ", ":", 1)
    if rng.random() < 0.2:
        lines = text.split("\n")
        i = rng.randrange(len(lines))
        lines[i] = lines[i] + " ~"
        text = "\n".join(lines)
    return text


def render_till_receipt(rng, merchant, items_priced, total_display, subtotal_display,
                        tax_display, date_str, time_str, column_split=False,
                        reduction=None):
    def row(label, amount):
        if column_split:
            return [f"{label}:", amount]
        return [f"{label[:22].ljust(24)}{amount}"]

    lines = [merchant.upper(), "-" * rng.randint(16, 28)]
    for index, (name, price) in enumerate(items_priced):
        if reduction and reduction[0] == index:
            lines.append("PRICE REDUCTION")
            lines.extend(row("ORIGINAL PRICE", reduction[1]))
        lines.extend(row(name, price))
    lines.append("-" * rng.randint(16, 28))
    if subtotal_display:
        lines.extend(row("SUBTOTAL", subtotal_display))
        lines.extend(row("VAT", tax_display))
    if total_display:
        lines.extend(row("TOTAL", total_display))
    payment_line, instrument = till_payment_line(rng)
    lines.append(payment_line)
    if date_str:
        stamp = date_str + (f" {time_str}" if time_str else "")
        lines.append(stamp)
    lines.append(rng.choice(["THANK YOU FOR SHOPPING", "Please retain receipt", ""]))
    return "\n".join(l for l in lines if l is not None), instrument


def render_order_email(rng, merchant, items_priced, total_display, date_str):
    order_no = rng.randrange(10 ** 6, 10 ** 9)
    lines = [
        f"Your {merchant} order confirmation",
        f"Order #{order_no}",
    ]
    if date_str:
        lines.append(f"Order placed: {date_str}")
    lines.append("")
    for name, price in items_priced:
        lines.append(f"  {name} — {price}")
    lines.append("")
    lines.append(f"Order total: {total_display}")
    lines.append("Thanks for your order!")
    return "\n".join(lines)


def render_bank_notification(rng, merchant, total_display, date_str, time_str):
    card = instrument_name(rng)
    stamp = f" on {date_str}" if date_str else ""
    stamp += f" at {time_str}" if time_str else ""
    shop = merchant.upper()
    templates = [
        f"{card}: You spent {total_display} at {shop}{stamp}.",
        f"You paid {total_display} to {shop} using your {card}{stamp}.",
        f"New transaction on {card}: {total_display} at {shop}{stamp}.",
        f"{total_display} spent at {shop} with {card}{stamp}.",
        f"{shop} · {total_display} · {card}" + (f" ·{stamp}" if stamp else ""),
    ]
    return rng.choice(templates), card


def render_wallet_screenshot(rng, merchant, total_display, date_str, time_str):
    card = instrument_name(rng)
    lines = []
    if rng.random() < 0.3:
        lines.append(rng.choice(["Wallet", "Latest Transaction", "Transaction"]))
    lines += [card, merchant.upper(), total_display]
    if date_str:
        lines.append((date_str + (f" {time_str}" if time_str else "")).strip())
    lines += ["Card Number", "•••• " + str(rng.randrange(1000, 10000))]
    lines += ["Status", rng.choice(["Approved", "Approved", "Pending"])]
    return "\n".join(lines), card


def render_casual_note(rng, merchant, total_display):
    templates = [
        f"{merchant.lower()} {total_display}",
        f"paid {total_display} at {merchant}",
        f"{total_display} {merchant} today",
        f"{merchant}: {total_display} (split with Sam later)",
    ]
    return rng.choice(templates)


def exact_parts(rng, total, n):
    if n <= 1 or total <= n:
        return [total]
    cuts = sorted(rng.sample(range(1, total), n - 1))
    return [b - a for a, b in zip([0] + cuts, cuts + [total])]


def build_spending_sample(rng, today):
    name, category, currency, item_pool = rng.choice(MERCHANTS)

    primary_category, primary_items = category, item_pool
    secondary = MIXED_MERCHANTS.get(name)
    mixed = secondary is not None and rng.random() < 0.6
    if mixed and rng.random() < 0.45:
        category, item_pool = secondary

    style = rng.random()
    is_till_receipt = style < 0.40
    omits_total = is_till_receipt and rng.random() < 0.25

    total = minor_units(rng, currency)
    n_items = min(len(item_pool), rng.randint(1, 4))
    items = rng.sample(item_pool, n_items)
    parts = None
    if omits_total:
        parts = exact_parts(rng, total, n_items)
        items_priced = [(it, display_amount(rng, p, currency))
                        for it, p in zip(items, parts)]
        part_values = list(parts)
    else:
        shares = sorted(rng.random() for _ in range(n_items - 1))
        bounds = [0.0] + shares + [1.0]
        items_priced = []
        part_values = []
        for i, it in enumerate(items):
            part = max(1, int(total * (bounds[i + 1] - bounds[i])))
            items_priced.append((it, display_amount(rng, part, currency)))
            part_values.append(part)

    reduction = None
    if is_till_receipt and rng.random() < 0.2:
        index = rng.randrange(len(items_priced))
        was = int(part_values[index] * rng.uniform(1.3, 2.5)) + 1
        reduction = (index, display_amount(rng, was, currency))

    has_tax_split = not omits_total and rng.random() < 0.35
    subtotal_display = tax_display = None
    if has_tax_split:
        tax = int(total * 0.2 / 1.2)
        subtotal_display = display_amount(rng, total - tax, currency)
        tax_display = display_amount(rng, tax, currency)
    total_display = None if omits_total else display_amount(rng, total, currency)

    has_date = rng.random() < 0.75
    has_time = has_date and rng.random() < 0.5
    purchase_date = today - timedelta(days=rng.randint(0, 30)) if has_date else None
    date_str = rng.choice(DATE_STYLES)(purchase_date) if purchase_date else ""
    hh, mm = rng.randint(7, 22), rng.randint(0, 59)
    if has_time and rng.random() < 0.3:
        ampm = "PM" if hh >= 12 else "AM"
        time_str = f"{(hh - 12) if hh > 12 else hh}:{mm:02d} {ampm}"
    else:
        time_str = f"{hh:02d}:{mm:02d}" if has_time else ""

    column_split = is_till_receipt and rng.random() < 0.3
    if is_till_receipt:
        content, instrument = render_till_receipt(
            rng, name, items_priced, total_display,
            subtotal_display, tax_display, date_str, time_str,
            column_split=column_split, reduction=reduction)
    elif style < 0.58:
        content = render_order_email(rng, name, items_priced, total_display, date_str)
        instrument = ""
    elif style < 0.73:
        content, instrument = render_bank_notification(rng, name, total_display, date_str, time_str)
    elif style < 0.88:
        content, instrument = render_wallet_screenshot(rng, name, total_display, date_str, time_str)
    else:
        content = render_casual_note(rng, name, total_display)
        instrument = ""
        date_str = ""
        purchase_date = None
        time_str = ""

    noisy = is_till_receipt and rng.random() < 0.6
    if noisy:
        content = ocr_noise(rng, content)

    category_known = rng.random() < 0.8
    others = [c for c in ALL_CATEGORIES if c != category]
    existing = rng.sample(others, rng.randint(3, 6))
    if category_known:
        existing.insert(rng.randrange(len(existing) + 1), category)
    existing = [category_prompt_line(rng, c, exclude=name) for c in existing]

    past_purchases = []
    if mixed:
        past_purchases.append(purchase_examples_line(
            rng, name, currency,
            [(primary_category, primary_items, rng.randint(3, 20)),
             (secondary[0], secondary[1], rng.randint(1, 6))]))
    elif category_known and rng.random() < 0.35:
        past_purchases.append(purchase_examples_line(
            rng, name, currency, [(category, item_pool, rng.randint(1, 12))]))

    knowledge = []
    if not mixed and rng.random() < 0.3:
        knowledge.append(f'Purchases at "{name}" belong to the category "{category}".')
    if rng.random() < 0.2:
        knowledge.append(f'Purchases at "{name}" are usually in {currency}.')
    if rng.random() < 0.25:
        other = rng.choice(MERCHANTS)
        knowledge.append(f'Purchases at "{other[0]}" belong to the category "{other[1]}".')
    if is_till_receipt and not omits_total and rng.random() < 0.15:
        knowledge.append(
            f'On a receipt from "{name}" the amount to record is the one printed under "total".'
        )

    confidence = rng.uniform(0.85, 0.97)
    if noisy or not category_known:
        confidence = rng.uniform(0.6, 0.8)
    if omits_total:
        confidence = min(confidence, rng.uniform(0.65, 0.85))
    if style >= 0.88:
        confidence = rng.uniform(0.55, 0.75)

    kind_phrase = ("a till receipt" if is_till_receipt
                   else "an order confirmation" if style < 0.58
                   else "a bank card notification" if style < 0.73
                   else "a wallet transaction screenshot" if style < 0.88
                   else "a quick personal note")
    sentences = [f"This is {kind_phrase} from {name}."]
    if instrument and not is_till_receipt:
        sentences.append(f"{instrument} is the card that paid, not the merchant; "
                         f"the merchant is {name}.")
    if omits_total and len(parts) > 1:
        expr = " + ".join(amount_string(p, currency) for p in parts)
        sentences.append("No grand total is printed, so the calculator summed the "
                         f"line items: {expr} = {amount_string(total, currency)} {currency}.")
    elif omits_total:
        sentences.append("No grand total is printed and there is a single line item, "
                         f"so the amount paid is {amount_string(total, currency)} {currency}.")
    elif is_till_receipt and column_split:
        sentences.append("The OCR split the receipt's columns, so the TOTAL label is on its "
                         f"own line and its amount {total_display} on the line below it — "
                         f"that is the {currency} total, copied exactly, not a sum of the items.")
    elif is_till_receipt and subtotal_display:
        sentences.append(f"The TOTAL line shows {total_display} in {currency} — "
                         "copy it exactly, not the SUBTOTAL.")
    elif is_till_receipt or style < 0.58:
        sentences.append(f"The total line shows {total_display} in {currency}; copy it exactly.")
    else:
        sentences.append(f"The amount paid is {total_display} in {currency}.")
    if reduction:
        sentences.append(f"The PRICE REDUCTION block quotes an original price of "
                         f"{reduction[1]}, which was not paid, so it is not part of the total.")
    if category_known and any(name in k and category in k for k in knowledge):
        sentences.append(f'The knowledge base maps "{name}" to the existing category "{category}".')
    elif category_known and mixed:
        other = secondary[0] if category == primary_category else primary_category
        bought = ", ".join(i for i, _ in items_priced[:2])
        sentences.append(f'The user files {name} under both {primary_category} and {secondary[0]}; '
                         f'this purchase ({bought}) is like the ones filed under "{category}", '
                         f'not {other}.')
    elif category_known:
        sentences.append(f'"{category}" matches one of the user\'s existing categories.')
    else:
        sentences.append(f'No existing category fits, so propose "{category}".')
    reasoning = " ".join(sentences)

    note_items = ", ".join(i for i, _ in items_priced[:3])
    extraction = {
        "reasoning": reasoning,
        "isSpending": True,
        "merchant": name,
        "paymentInstrument": instrument,
        "totalAmount": amount_string(total, currency),
        "currencyCode": currency,
        "purchaseDate": purchase_date.strftime("%Y-%m-%d") if purchase_date else "",
        "purchaseTime": f"{hh:02d}:{mm:02d}" if time_str else "",
        "category": category if category_known else "",
        "proposedCategory": "" if category_known else category,
        "note": f"{category}: {note_items}" if note_items != "trip" else f"{name} trip",
        "confidence": round(confidence, 2),
    }
    preferences = []
    if rng.random() < 0.3:
        preferences.append("Record spending in the local currency of the place it happened in.")
    if rng.random() < 0.15 and not has_date:
        preferences.append("Never guess a purchase date the content does not print.")
    if rng.random() < 0.15 and category_known:
        preferences.append("Prefer one of the existing categories over inventing a new one.")

    local_currency = currency if rng.random() < 0.7 else rng.choice(list(SYMBOL))
    return content, existing, past_purchases, preferences, knowledge, extraction, local_currency


def build_non_spending_sample(rng):
    content = rng.choice(NON_SPENDING)
    existing = [category_prompt_line(rng, c) for c in rng.sample(ALL_CATEGORIES, rng.randint(3, 6))]
    local_currency = rng.choice(list(SYMBOL))
    reasoning = rng.choice([
        "This content is not a purchase: no merchant is charging money and no total was paid.",
        "Nothing here shows money being spent — it is not a receipt, order or bill.",
        "This is unrelated to spending; no payment or amount due appears.",
    ])
    extraction = {
        "reasoning": reasoning,
        "isSpending": False,
        "merchant": "",
        "paymentInstrument": "",
        "totalAmount": "",
        "currencyCode": local_currency,
        "purchaseDate": "",
        "purchaseTime": "",
        "category": "",
        "proposedCategory": "",
        "note": "",
        "confidence": round(rng.uniform(0.85, 0.95), 2),
    }
    return content, existing, [], [], [], extraction, local_currency


PROMPT_PURCHASE_EXAMPLE_LIMIT = 2


def purchase_examples_line(rng, merchant, currency, filed):
    parts = []
    for category, pool, count in sorted(filed, key=lambda f: (-f[2], f[0])):
        cited = []
        for _ in range(min(count, PROMPT_PURCHASE_EXAMPLE_LIMIT)):
            items = ", ".join(rng.sample(pool, min(len(pool), rng.randint(1, 3))))
            amount = SYMBOL[currency] + amount_string(minor_units(rng, currency), currency)
            cited.append(f'"{items}" {amount}')
        parts.append(f"{count} under {category} (e.g. {'; '.join(cited)})")
    return f"{merchant}: {', '.join(parts)}"


PROMPT_MERCHANT_LIMIT = 4


def category_prompt_line(rng, category, exclude=None):
    pool = [m[0] for m in MERCHANTS if m[1] == category]
    if exclude in pool and rng.random() < 0.5:
        pool = [m for m in pool if m != exclude]
    if not pool or rng.random() < 0.15:
        return category
    merchants = rng.sample(pool, min(len(pool), rng.randint(1, PROMPT_MERCHANT_LIMIT)))
    return f"{category} (e.g. {', '.join(merchants)})"


def assemble_prompt(today, local_currency, existing, past_purchases, preferences, knowledge, content):
    prompt = f"Today's date: {today.strftime('%Y-%m-%d')}\n"
    prompt += f"The user's local currency: {local_currency}\n\n"
    if existing:
        prompt += "The user's existing categories, each with the merchants they file under it:\n"
        prompt += "\n".join(f"- {c}" for c in existing) + "\n\n"
    if past_purchases:
        prompt += "The user's past purchases at merchants named in the content, by the category each was filed under:\n"
        prompt += "\n".join(f"- {p}" for p in past_purchases) + "\n\n"
    if knowledge:
        prompt += "Knowledge base about this user:\n"
        prompt += "\n".join(f"- {k}" for k in knowledge) + "\n\n"
    if preferences:
        prompt += "The user's standing preferences — follow these rules:\n"
        prompt += "\n".join(f"- {p}" for p in preferences) + "\n\n"
    prompt += f'Content to analyze:\n"""\n{content}\n"""'
    return prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--negative-fraction", type=float, default=0.15,
                    help="fraction of isSpending=false samples")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    with open(args.out, "w", encoding="utf-8") as f:
        for _ in range(args.count):
            today = date(2026, 1, 1) + timedelta(days=rng.randint(0, 700))
            if rng.random() < args.negative_fraction:
                content, existing, past_purchases, preferences, knowledge, extraction, local_currency = build_non_spending_sample(rng)
            else:
                content, existing, past_purchases, preferences, knowledge, extraction, local_currency = build_spending_sample(rng, today)
            sample = [
                {"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": assemble_prompt(today, local_currency, existing, past_purchases, preferences, knowledge, content)},
                {"role": "assistant", "content": json.dumps(extraction, ensure_ascii=False)},
            ]
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Wrote {args.count} samples to {args.out}")


if __name__ == "__main__":
    main()
