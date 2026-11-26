import random
import unicodedata
from datetime import date, timedelta

from locales import CATEGORY_TITLES, LOCALES, PROPOSAL_SYNONYMS, instrument_name

ZERO_DECIMAL = {"JPY", "KRW"}


def fold(text):
    text = unicodedata.normalize("NFKC", text or "")
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return "".join(c for c in text.lower() if c.isalnum())


def gold_amount_string(minor, currency):
    if currency in ZERO_DECIMAL:
        return str(minor)
    return f"{minor // 100}.{minor % 100:02d}"


HOME_LOCALES = ["en-GB", "en-US", "zh", "ja", "fr", "es", "de"]
TRAVEL = {
    "en-GB": ["fr", "es", "de", "ja"], "en-US": ["es-MX", "fr", "ja"],
    "zh": ["ja", "en-GB", "fr"], "ja": ["zh", "en-US", "fr"],
    "fr": ["es", "de", "en-GB"], "es": ["fr", "de", "en-GB"], "de": ["fr", "es", "en-GB"],
}
LOCAL_CURRENCY_PREFERENCE = "Record spending in the local currency of the place it happened in."


def ledger_amount(loc, minor):
    cur = loc["currency"]
    p = gold_amount_string(minor, cur)
    if cur == "GBP":
        return f"£{p}"
    if cur in ("USD", "MXN"):
        return f"${p}"
    if cur == "CNY":
        return f"¥{p}"
    if cur == "JPY":
        return f"¥{minor:,}"
    return p.replace(".", ",") + " €"


def make_profile(rng, home, index, today, locales=None):
    loc = (locales or LOCALES)[home]
    titles = CATEGORY_TITLES[loc["language"]]
    cats = sorted({m["cat"] for m in loc["merchants"]} | {m["mixed"][0] for m in loc["merchants"] if m["mixed"]})
    droppable = [c for c in cats if c not in ("groceries", "dining", "coffee")]
    missing = set(rng.sample(droppable, min(len(droppable), rng.randint(1, 3))))
    kept = [c for c in cats if c not in missing]
    rng.shuffle(kept)

    ledger = []
    for merchant in loc["merchants"]:
        kinds = []
        if merchant["cat"] in kept:
            kinds.append((merchant["cat"], merchant["items"], rng.randint(2, 8)))
        if merchant["mixed"] and merchant["mixed"][0] in kept and merchant["cat"] in kept:
            kinds.append((merchant["mixed"][0], merchant["mixed"][1], rng.randint(1, 5)))
        if rng.random() < 0.15:
            continue
        for cat, items, count in kinds:
            for _ in range(count):
                picked = rng.sample(items, min(len(items), rng.randint(1, 3)))
                amount = sum(rng.randint(lo, hi) for _, lo, hi in picked)
                when = today - timedelta(days=rng.randint(3, 180))
                ledger.append({
                    "merchant": merchant["name"],
                    "category": titles[cat],
                    "note": f"{titles[cat]}: " + ", ".join(name for name, _, _ in picked),
                    "paymentInstrument": rng.choice(["", "", instrument_name(rng, loc)]),
                    "amountMinorUnits": amount,
                    "currencyCode": loc["currency"],
                    "date": when.isoformat(),
                })
    ledger.sort(key=lambda e: e["date"])
    return {
        "id": f"{home}-u{index}",
        "home": home,
        "language": loc["language"],
        "localCurrency": loc["currency"],
        "today": today.isoformat(),
        "categoryKeys": kept,
        "missingCategoryKeys": sorted(missing),
        "categories": [titles[c] for c in kept],
        "ledger": ledger,
        "prefersLocalCurrencyRule": rng.random() < 0.3,
    }


def category_lines(profile):
    lines = []
    for title in profile["categories"]:
        counts = {}
        for e in profile["ledger"]:
            if e["category"] == title:
                counts[e["merchant"]] = counts.get(e["merchant"], 0) + 1
        top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:4]
        lines.append(f"{title} (e.g. {', '.join(m for m, _ in top)})" if top else title)
    return lines


def past_purchase_lines(profile, content):
    loc = LOCALES[profile["home"]]
    hay = fold(content)
    by_merchant = {}
    for e in profile["ledger"]:
        by_merchant.setdefault(e["merchant"], []).append(e)
    named = [(m, es) for m, es in by_merchant.items() if fold(m) and fold(m) in hay]
    named.sort(key=lambda kv: (-len(kv[1]), kv[0]))
    lines = []
    for merchant, entries in named[:2]:
        counts = {}
        for e in entries:
            counts[e["category"]] = counts.get(e["category"], 0) + 1
        parts = []
        for title, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]:
            filed = sorted((e for e in entries if e["category"] == title), key=lambda e: e["date"], reverse=True)
            cites = []
            for e in filed[:2]:
                desc = e["note"].split(": ", 1)[-1]
                if len(desc) > 36:
                    desc = desc[:35].rstrip() + "…"
                cites.append(f'"{desc}" {ledger_amount(loc, e["amountMinorUnits"])}')
            parts.append(f"{count} under {title} (e.g. {'; '.join(cites)})")
        lines.append(f"{merchant}: {', '.join(parts)}")
    return lines


def light_ocr_noise(rng, line):
    swaps = {"O": "0", "l": "1", "I": "l", "S": "5", "B": "8", "e": "c", "rn": "m"}
    for a, b in rng.sample(list(swaps.items()), 2):
        if a in line and rng.random() < 0.5:
            i = line.index(a)
            if " " in line and i < line.rindex(" "):
                line = line[:i] + b + line[i + len(a):]
    return line


def pick_items(rng, items, n):
    picked = rng.sample(items, min(len(items), n))
    return [(name, rng.randint(lo, hi)) for name, lo, hi in picked]


def cash_tendered(total, currency):
    if currency == "JPY":
        steps = [1000, 5000, 10000]
    elif currency in ("CNY", "MXN"):
        steps = [5000, 10000, 20000, 50000]
    else:
        steps = [500, 1000, 2000, 5000, 10000]
    for s in steps:
        if s > total:
            return s
    return ((total // steps[-1]) + 1) * steps[-1]


def render_till(rng, loc, merchant, items, when, traits):
    L = loc["labels"]
    amt = lambda m: loc["amount"](rng, m)
    cur = loc["currency"]
    split = rng.random() < 0.3
    if split:
        traits.append("column_split")

    def row(label, value):
        return [label, value] if split else [f"{label}  {value}"]

    lines = [rng.choice(merchant["printed"])]
    if rng.random() < 0.8:
        lines.append(rng.choice(L["addresses"]))
    has_date = rng.random() < 0.85
    has_time = has_date and rng.random() < 0.75
    if has_date:
        stamp = loc["date"](rng, when)
        if has_time:
            stamp += " " + loc["time"](when.hour, when.minute)
        lines.append(stamp)
    if rng.random() < 0.5:
        lines.append(L["receipt_no"].format(n=rng.randint(1000, 99999)))
    lines.append("")

    subtotal = sum(p for _, p in items)
    reduction = len(items) >= 1 and rng.random() < 0.18
    for i, (name, price) in enumerate(items):
        lines += row(name, amt(price))
        if reduction and i == 0:
            was = int(price * rng.uniform(1.3, 2.2)) + 1
            lines.append("  " + L["was"].format(amt=amt(was)))
            traits.append("was_price")

    total = subtotal
    omit_total = not loc["tax_added"] and len(items) >= 2 and rng.random() < 0.12
    if loc["tax_added"]:
        rate = rng.choice([0.0725, 0.08875, 0.06])
        tax = round(subtotal * rate)
        total = subtotal + tax
        lines += row(rng.choice(L["subtotal"]), amt(subtotal))
        lines += row(rng.choice(L["tax"]), amt(tax))
        traits.append("tax_added")
        if merchant["cat"] in ("dining",) and rng.random() < 0.5:
            tip = round(subtotal * 0.18)
            total += tip
            lines += row(L["tip"], amt(tip))
            traits.append("tip")
    if omit_total:
        traits.append("no_printed_total")
    else:
        if not loc["tax_added"] and rng.random() < 0.3:
            lines += row(rng.choice(L["subtotal"]), amt(subtotal))
            traits.append("subtotal_line")
        lines += row(rng.choice(L["total"]), amt(total))
        if L.get("tax_included") and rng.random() < 0.5:
            rate = {"JPY": 0.08, "EUR": 0.2, "GBP": 0.2, "MXN": 0.16}.get(cur, 0.1)
            lines.append(L["tax_included"].format(amt=amt(round(total * rate / (1 + rate)))))

    roll = rng.random()
    if roll < 0.3 and not omit_total:
        tendered = cash_tendered(total, cur)
        lines += row(L["tendered"], amt(tendered))
        lines += row(L["change"], amt(tendered - total))
        traits.append("cash_change")
        instrument = {"mode": "either", "gold": L["cash_name"], "accepted": [L["cash_name"], "Cash"]}
    elif roll < 0.6:
        issuer = instrument_name(rng, loc)
        last4 = rng.randint(1000, 9999)
        lines.append(f"{issuer.upper() if loc['language'] in ('en', 'fr', 'es', 'de') else issuer} ****{last4}")
        instrument = {"mode": "exact", "gold": issuer, "accepted": [issuer]}
        traits.append("issuer_on_receipt")
    else:
        scheme = rng.choice(["VISA", "MASTERCARD", L["card"]])
        lines += row(f"{scheme}", amt(total) if not omit_total else "")
        instrument = {"mode": "either", "gold": scheme.title(), "accepted": [scheme, scheme.title()]}
    lines.append("")
    lines.append(rng.choice(L["thanks"]))

    if rng.random() < 0.3 and loc["language"] in ("en", "fr", "es", "de"):
        lines = [light_ocr_noise(rng, l) for l in lines]
        traits.append("ocr_noise")
    content = "\n".join(l for l in lines if l is not None).replace("\n\n\n", "\n\n").strip()
    return content, total, has_date, has_time, instrument


def render_email(rng, loc, merchant, items, when, traits):
    shipping = rng.choice([0, 0, 199, 299, 399, 499]) if loc["currency"] != "JPY" else rng.choice([0, 0, 350, 500])
    if loc["currency"] in ("CNY", "MXN"):
        shipping = rng.choice([0, 0, 600, 1000, 9900])
    total = sum(p for _, p in items) + shipping
    lines = "\n".join(f"{name} × 1  {loc['amount'](rng, price)}" for name, price in items)
    content = loc["email"].format(
        merchant=merchant["name"], order=f"{rng.randint(100, 999)}-{rng.randint(1000000, 9999999)}",
        date=loc["date"](rng, when), items=lines, shipping=loc["amount"](rng, shipping),
        total=loc["amount"](rng, total))
    if shipping:
        traits.append("shipping_added")
    return content, total, True, False, {"mode": "none", "gold": "", "accepted": []}


def render_notification(rng, loc, merchant, total, when, traits):
    instrument = instrument_name(rng, loc)
    content = rng.choice(loc["notification"]).format(
        instrument=instrument, merchant=rng.choice(merchant["printed"]) if rng.random() < 0.5 else merchant["name"],
        amount=loc["amount"](rng, total), plain=gold_amount_string(total, loc["currency"]),
        date_no_year=loc["date_no_year"](when), time=loc["time"](when.hour, when.minute),
        last4=rng.randint(1000, 9999))
    traits.append("date_without_year")
    has_time = loc["time"](when.hour, when.minute) in content
    has_date = loc["date_no_year"](when) in content
    return content, total, has_date, has_time, {"mode": "exact", "gold": instrument, "accepted": [instrument]}


def render_wallet(rng, loc, merchant, total, when, traits):
    instrument = instrument_name(rng, loc)
    content = loc["wallet"].format(
        instrument=instrument, merchant=merchant["name"], amount=loc["amount"](rng, total),
        plain=gold_amount_string(total, loc["currency"]), date=loc["date"](rng, when),
        time=loc["time"](when.hour, when.minute), last4=rng.randint(1000, 9999))
    traits.append("card_heading")
    return content, total, True, True, {"mode": "exact", "gold": instrument, "accepted": [instrument]}


def render_casual(rng, loc, merchant, total, today, traits):
    day_roll = rng.random()
    if day_roll < 0.4:
        day, when = loc["yesterday"], today - timedelta(days=1)
        traits.append("relative_date")
    elif day_roll < 0.55:
        day, when = loc["today_word"], today
        traits.append("relative_date")
    else:
        day, when = "", None
    content = rng.choice(loc["casual"]).format(
        amount=loc["amount"](rng, total), plain=gold_amount_string(total, loc["currency"]),
        merchant=merchant["name"], day=day).strip()
    content = " ".join(content.split())
    return content, total, when


def build_sample(rng, profile, index, locales=None, prompt=None):
    locales = locales or LOCALES
    home = profile["home"]
    hloc = locales[home]
    titles = CATEGORY_TITLES[hloc["language"]]
    today = date.fromisoformat(profile["today"]) + timedelta(days=rng.randint(0, 3))
    traits = []
    preferences = [LOCAL_CURRENCY_PREFERENCE] if profile["prefersLocalCurrencyRule"] else []

    if rng.random() < 0.1:
        content = rng.choice(hloc["non_spending"])
        traits.append("non_spending")
        gold = {
            "isSpending": False, "merchant": "", "merchantAccepted": [], "paymentInstrument": "",
            "paymentInstrumentMode": "none", "paymentInstrumentAccepted": [], "totalAmount": "",
            "amountMinorUnits": 0, "currencyCode": profile["localCurrency"], "purchaseDate": "",
            "purchaseTime": "", "category": "", "proposedCategory": "", "proposalAccepted": [],
            "noteItems": [],
        }
        return finish(profile, index, today, content, gold, traits, hloc["language"], "non_spending", preferences, [],
                      prompt)

    content_locale = home
    if rng.random() < 0.15:
        content_locale = rng.choice(TRAVEL[home])
        traits.append("travel")
    loc = locales[content_locale]
    in_profile = [m for m in loc["merchants"] if m["cat"] in profile["categoryKeys"]]
    out_profile = [m for m in loc["merchants"] if m["cat"] in profile["missingCategoryKeys"]]
    if out_profile and rng.random() < 0.18:
        merchant = rng.choice(out_profile)
        traits.append("needs_proposal")
    else:
        merchant = rng.choice(in_profile or loc["merchants"])
    if merchant.get("invented"):
        traits.append("invented_merchant")

    cat_key, items_pool = merchant["cat"], merchant["items"]
    filed_as = {e["category"] for e in profile["ledger"] if e["merchant"] == merchant["name"]}
    if merchant["mixed"] and content_locale == home \
            and {titles[merchant["cat"]], titles.get(merchant["mixed"][0], "")} <= filed_as:
        traits.append("mixed_merchant")
        if rng.random() < 0.5:
            cat_key, items_pool = merchant["mixed"]
            traits.append("secondary_kind")

    items = pick_items(rng, items_pool, rng.randint(1, 4))
    when_day = today - timedelta(days=rng.randint(0, 20))
    from datetime import datetime
    when = datetime(when_day.year, when_day.month, when_day.day, rng.randint(7, 22), rng.randint(0, 59))

    if merchant["online"]:
        style = rng.choices(["email", "notification", "wallet"], [0.6, 0.2, 0.2])[0]
    else:
        style = rng.choices(["till", "notification", "wallet", "casual"], [0.55, 0.17, 0.15, 0.13])[0]

    purchase_date = ""
    purchase_time = ""
    if style == "till":
        content, total, has_date, has_time, instr = render_till(rng, loc, merchant, items, when, traits)
    elif style == "email":
        content, total, has_date, has_time, instr = render_email(rng, loc, merchant, items, when, traits)
    else:
        total = sum(p for _, p in items)
        if style == "notification":
            content, total, has_date, has_time, instr = render_notification(rng, loc, merchant, total, when, traits)
        elif style == "wallet":
            content, total, has_date, has_time, instr = render_wallet(rng, loc, merchant, total, when, traits)
        else:
            content, total, day = render_casual(rng, loc, merchant, total, today, traits)
            has_date, has_time = day is not None, False
            when = datetime(day.year, day.month, day.day) if day else when
            instr = {"mode": "either", "gold": "", "accepted": []}
            items = []
    if has_date:
        purchase_date = when.date().isoformat()
    if has_time:
        purchase_time = f"{when.hour:02d}:{when.minute:02d}"

    category = titles[cat_key] if cat_key in profile["categoryKeys"] else ""
    proposal = titles[cat_key] if not category else ""
    accepted_merchants = sorted({merchant["name"], *merchant["printed"], *merchant["aliases"]})

    past = past_purchase_lines(profile, content) if content_locale == home else []
    if "mixed_merchant" in traits and not past:
        traits.append("ledger_lookup_useful")
    gold = {
        "isSpending": True,
        "merchant": merchant["name"],
        "merchantAccepted": accepted_merchants,
        "paymentInstrument": instr["gold"] if instr["mode"] == "exact" else "",
        "paymentInstrumentMode": instr["mode"],
        "paymentInstrumentAccepted": instr["accepted"],
        "totalAmount": gold_amount_string(total, loc["currency"]),
        "amountMinorUnits": total,
        "currencyCode": loc["currency"],
        "purchaseDate": purchase_date,
        "purchaseTime": purchase_time,
        "category": category,
        "proposedCategory": proposal,
        "proposalAccepted": sorted({proposal, *PROPOSAL_SYNONYMS.get(cat_key, [])}) if proposal else [],
        "noteItems": [name for name, _ in items],
    }
    return finish(profile, index, today, content, gold, traits, loc["language"], style, preferences, past, prompt)


def finish(profile, index, today, content, gold, traits, language, style, preferences, past, prompt):
    sample = {
        "id": f"{profile['id']}-{index:04d}",
        "profile": profile["id"],
        "language": language,
        "homeLocale": profile["home"],
        "style": style,
        "traits": sorted(set(traits)),
        "today": today.isoformat(),
        "content": content,
    }
    if prompt:
        sample["prompt"] = prompt(profile, today, past, preferences, content)
    sample["gold"] = gold
    return sample
