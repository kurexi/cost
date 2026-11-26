#!/usr/bin/env python3
"""Generate a frozen evaluation suite for the spending agent.

    python3 make_suite.py --name dev     --per-locale 40 --seed 11
    python3 make_suite.py --name holdout --per-locale 20 --seed 23

Content and gold records come from the receipt generator in `receipts.py`
(multilingual till receipts, order emails, bank notifications, wallet
screens, casual notes and non-spending messages, with their traits). On top
of that each simulated user gets what the agent is judged on:

- a ledger in the agent's own CSV shape, with provenance on every record;
- tags: a standing one on one merchant's records ("Work"), an event one per
  trip the user has already started recording, and an unrelated decoy
  ("Gifts") — with gold tags on the cases that belong under them, and cases
  abroad with no trip tag yet, where proposing one is allowed;
- a memory text: empty, helpful, or helpful with one wrong line, so whether
  the agent trusts a bad memory over the records is measured;
- realistic content length: card slips, VAT tables, policies and legal
  footers around the receipt, and a few cases over the content budget that
  must be rejected before any model call.

Writes `suites/<name>/cases.jsonl`, `users.jsonl` and `meta.json`. The
suite is frozen once committed: runs record its fingerprint, and runs on
different fingerprints are never compared. Stdlib only.
"""

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from datetime import date, datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import receipts as v1
import tills
from locales import CATEGORY_TITLES, LOCALES, instrument_name
from novel_names import world_for

from spending_agent.session import CONTENT_CHARACTER_LIMIT

SUITE_VERSION = 1
CONTENT_LOCALE = {("en", "GBP"): "en-GB", ("en", "USD"): "en-US", ("zh", "CNY"): "zh", ("ja", "JPY"): "ja",
                  ("fr", "EUR"): "fr", ("es", "EUR"): "es", ("es", "MXN"): "es-MX", ("de", "EUR"): "de"}

COUNTRY = {
    "fr": {"en": "France", "zh": "法国", "ja": "フランス", "fr": "France", "es": "Francia", "de": "Frankreich"},
    "es": {"en": "Spain", "zh": "西班牙", "ja": "スペイン", "fr": "Espagne", "es": "España", "de": "Spanien"},
    "de": {"en": "Germany", "zh": "德国", "ja": "ドイツ", "fr": "Allemagne", "es": "Alemania", "de": "Deutschland"},
    "ja": {"en": "Japan", "zh": "日本", "ja": "日本", "fr": "Japon", "es": "Japón", "de": "Japan"},
    "en-GB": {"en": "UK", "zh": "英国", "ja": "イギリス", "fr": "Royaume-Uni", "es": "Reino Unido", "de": "England"},
    "en-US": {"en": "USA", "zh": "美国", "ja": "アメリカ", "fr": "États-Unis", "es": "EE. UU.", "de": "USA"},
    "es-MX": {"en": "Mexico", "zh": "墨西哥", "ja": "メキシコ", "fr": "Mexique", "es": "México", "de": "Mexiko"},
    "zh": {"en": "China", "zh": "中国", "ja": "中国", "fr": "Chine", "es": "China", "de": "China"},
}
TRIP_NAME = {"en": "{country} Trip", "zh": "{country}旅行", "ja": "{country}旅行", "fr": "Voyage {country}",
             "es": "Viaje a {country}", "de": "{country}-Reise"}
WORK = {"en": "Work", "zh": "工作", "ja": "仕事", "fr": "Travail", "es": "Trabajo", "de": "Arbeit"}
GIFTS = {"en": "Gifts", "zh": "礼物", "ja": "ギフト", "fr": "Cadeaux", "es": "Regalos", "de": "Geschenke"}
WORK_CATEGORY_PREFERENCE = ("transport", "coffee", "dining", "fuel")
GIFT_CATEGORY_PREFERENCE = ("shopping", "clothing", "kids", "home")
SOURCES = (("agent_reviewed", 0.6), ("manual", 0.25), ("agent", 0.15))
VAT_RATE = {"en-GB": 0.2, "fr": 0.2, "es": 0.2, "de": 0.2, "ja": 0.08, "zh": 0.1, "es-MX": 0.16}

BOILERPLATE = {
    "en": {"header": ["VAT No. GB {vat}", "Store {store}  Tel 0{phone}", "Served by: {clerk}"],
           "slip": ["CARDHOLDER COPY", "AID: A0000000{aid}", "TID: {tid}  MID: {mid}", "AUTH CODE: {auth}",
                    "PAN SEQ: 01", "VERIFIED BY DEVICE", "PLEASE RETAIN FOR YOUR RECORDS"],
           "vat": "VAT {rate}%  NET {net}  VAT {tax}",
           "footer": ["You earned {points} points today. Points balance: {balance}",
                      "Returns accepted within 28 days with this receipt.",
                      "Tell us how we did at www.example-feedback.co.uk and enter code {code}",
                      "for a chance to win a £500 voucher. Terms apply."],
           "email": ["Delivery address:", "{name}", "{street}", "Payment summary is shown above.",
                     "Track your parcel in the app using reference {code}.",
                     "You have 30 days to return items. Refunds go to the original payment method.",
                     "This email was sent to you because you placed an order. Manage preferences in your account."],
           "legal": "These terms govern every order placed through our website and app. By placing an order you "
                    "agree that the contract is formed when we dispatch the goods, that prices include VAT at the "
                    "applicable rate, that delivery dates are estimates, and that your statutory rights are not "
                    "affected. Personal data is processed under our privacy notice for fulfilling orders and "
                    "preventing fraud, and kept only as long as the law requires.",
           "names": ["Alex Morgan", "Sam Taylor", "Jordan Lee"], "streets": ["14 Elm Road", "3 Church Lane"]},
    "zh": {"header": ["税号 91310{vat}", "门店 {store}  电话 021-{phone}", "收银员：{clerk}"],
           "slip": ["持卡人存根", "AID: A0000000{aid}", "终端号: {tid}  商户号: {mid}", "授权码: {auth}",
                    "交易参考号: {code}", "请妥善保管"],
           "vat": "税率 {rate}%  不含税 {net}  税额 {tax}",
           "footer": ["本次积分 {points}，累计积分 {balance}", "凭小票七日内可退换货。",
                      "扫码关注公众号，输入 {code} 参与满意度调查。"],
           "email": ["收货地址：", "{name}", "{street}", "订单已付款，正在为您备货。", "物流单号 {code}，可在 App 内查询。",
                     "七天无理由退货，退款原路返回。", "此邮件由系统自动发送，请勿回复。"],
           "legal": "本条款适用于通过本网站及应用下达的全部订单。下单即表示您同意：合同自商品发出时成立；价格已含适用税费；"
                    "配送时间仅供参考；您的法定权利不受影响。我们仅为履行订单和防范欺诈处理您的个人信息，并仅在法律要求的期限内保存。",
           "names": ["张伟", "李娜", "王芳"], "streets": ["上海市浦东新区世纪大道100号", "北京市朝阳区建国路8号"]},
    "ja": {"header": ["登録番号 T{vat}", "店舗 {store}  TEL 03-{phone}", "担当：{clerk}"],
           "slip": ["お客様控え", "AID: A0000000{aid}", "端末番号: {tid}", "承認番号: {auth}", "伝票番号: {code}",
                    "大切に保管してください"],
           "vat": "({rate}%対象 {net} 内税 {tax})",
           "footer": ["今回のポイント {points}P  ポイント残高 {balance}P", "返品・交換はレシートをご持参の上14日以内にお願いします。",
                      "アンケートにご協力ください。受付番号 {code}"],
           "email": ["お届け先：", "{name} 様", "{street}", "ご注文の商品は準備ができ次第発送いたします。",
                     "お問い合わせ番号 {code}", "返品はお届けから30日以内に承ります。", "このメールは送信専用です。"],
           "legal": "本規約は当サイトおよびアプリでのすべてのご注文に適用されます。ご注文により、商品の発送時に契約が成立すること、"
                    "価格には消費税が含まれること、お届け日は目安であることに同意したものとみなします。"
                    "お客様の個人情報はご注文の履行と不正防止のためにのみ利用し、法令で定める期間に限り保管します。",
           "names": ["佐藤 健", "鈴木 花子", "高橋 翔"], "streets": ["東京都渋谷区神南1-2-3", "大阪府大阪市北区梅田2-4-9"]},
    "fr": {"header": ["N° TVA FR{vat}", "Magasin {store}  Tél 01 {phone}", "Caisse : {clerk}"],
           "slip": ["TICKET CLIENT", "AID: A0000000{aid}", "TID: {tid}  MID: {mid}", "AUTO: {auth}",
                    "TRANSACTION ACCEPTÉE", "A CONSERVER"],
           "vat": "TVA {rate}%  HT {net}  TVA {tax}",
           "footer": ["Points gagnés : {points}  Solde : {balance}", "Échange sous 30 jours sur présentation du ticket.",
                      "Donnez votre avis sur www.exemple-avis.fr avec le code {code}"],
           "email": ["Adresse de livraison :", "{name}", "{street}", "Votre commande est en cours de préparation.",
                     "Suivez votre colis avec la référence {code}.", "Retours gratuits sous 30 jours.",
                     "Cet e-mail vous a été envoyé suite à votre commande."],
           "legal": "Les présentes conditions régissent toute commande passée sur notre site et notre application. "
                    "En commandant, vous acceptez que le contrat soit conclu à l'expédition, que les prix incluent la "
                    "TVA applicable et que les délais de livraison soient indicatifs. Vos droits légaux ne sont pas "
                    "affectés. Vos données sont traitées pour exécuter la commande et prévenir la fraude.",
           "names": ["Camille Martin", "Louis Bernard"], "streets": ["12 rue de la Paix, 75002 Paris",
                                                                    "8 avenue Jean Jaurès, 69007 Lyon"]},
    "es": {"header": ["NIF B{vat}", "Tienda {store}  Tel 91 {phone}", "Le atendió: {clerk}"],
           "slip": ["COPIA PARA EL CLIENTE", "AID: A0000000{aid}", "TPV: {tid}  COMERCIO: {mid}", "AUT: {auth}",
                    "OPERACIÓN APROBADA", "CONSERVE ESTE TICKET"],
           "vat": "IVA {rate}%  BASE {net}  CUOTA {tax}",
           "footer": ["Puntos obtenidos: {points}  Saldo: {balance}", "Cambios y devoluciones en 30 días con el ticket.",
                      "Valore su experiencia en www.ejemplo-opinion.es con el código {code}"],
           "email": ["Dirección de envío:", "{name}", "{street}", "Estamos preparando tu pedido.",
                     "Sigue tu envío con la referencia {code}.", "Tienes 30 días para devolver los artículos.",
                     "Has recibido este correo porque realizaste un pedido."],
           "legal": "Estas condiciones rigen todos los pedidos realizados en nuestra web y aplicación. Al realizar un "
                    "pedido aceptas que el contrato se perfecciona con el envío, que los precios incluyen el IVA "
                    "aplicable y que los plazos de entrega son orientativos. Tus derechos legales no se ven afectados. "
                    "Tratamos tus datos para gestionar el pedido y prevenir el fraude.",
           "names": ["Lucía García", "Pablo Fernández"], "streets": ["Calle Mayor 5, 28013 Madrid",
                                                                     "Avenida del Puerto 21, 46021 Valencia"]},
    "de": {"header": ["USt-IdNr. DE{vat}", "Filiale {store}  Tel 030 {phone}", "Es bediente Sie: {clerk}"],
           "slip": ["KUNDENBELEG", "AID: A0000000{aid}", "Terminal-ID: {tid}", "Genehmigungs-Nr.: {auth}",
                    "Zahlung erfolgt", "Bitte Beleg aufbewahren"],
           "vat": "MwSt {rate}%  Netto {net}  MwSt {tax}",
           "footer": ["Gesammelte Punkte: {points}  Punktestand: {balance}",
                      "Umtausch innerhalb von 14 Tagen nur mit Kassenbon.",
                      "Ihre Meinung zählt: www.beispiel-umfrage.de, Code {code}"],
           "email": ["Lieferadresse:", "{name}", "{street}", "Wir bereiten Ihre Bestellung vor.",
                     "Sendungsverfolgung mit der Nummer {code}.", "Sie haben 30 Tage Rückgaberecht.",
                     "Diese E-Mail wurde automatisch versendet."],
           "legal": "Diese Bedingungen gelten für alle Bestellungen über unsere Website und App. Mit Ihrer Bestellung "
                    "stimmen Sie zu, dass der Vertrag mit dem Versand zustande kommt, dass die Preise die gesetzliche "
                    "Mehrwertsteuer enthalten und dass Liefertermine unverbindlich sind. Ihre gesetzlichen Rechte "
                    "bleiben unberührt. Ihre Daten verarbeiten wir zur Abwicklung der Bestellung.",
           "names": ["Anna Schmidt", "Lukas Weber"], "streets": ["Hauptstraße 12, 10115 Berlin",
                                                                 "Bahnhofstraße 3, 80335 München"]},
}
CLERKS = ["Sarah", "Wei", "Yuki", "Marie", "Javier", "Jonas", "Priya"]


def digits(rng, n):
    return "".join(str(rng.randint(0, 9)) for _ in range(n))


def fill(rng, line, extra=None):
    values = {"vat": digits(rng, 9), "store": digits(rng, 4), "phone": digits(rng, 8), "clerk": rng.choice(CLERKS),
              "aid": digits(rng, 7), "tid": digits(rng, 8), "mid": digits(rng, 9), "auth": digits(rng, 6),
              "code": digits(rng, 10), "points": rng.randint(5, 400), "balance": rng.randint(400, 9000)}
    values.update(extra or {})
    return line.format(**values)


def content_locale(case):
    return CONTENT_LOCALE.get((case["language"], case["gold"]["currencyCode"]))


def merchant_entry(locale, name, world=LOCALES):
    return next((m for m in world[locale]["merchants"] if m["name"] == name), None)


def vat_line(rng, locale, text, total):
    loc = LOCALES[locale]
    rate = VAT_RATE.get(locale)
    if not rate:
        return None
    net = round(total / (1 + rate))
    return text["vat"].format(rate=round(rate * 100), net=loc["amount"](rng, net), tax=loc["amount"](rng, total - net))


def lengthen_till(rng, locale, content, total, traits):
    loc = LOCALES[locale]
    text = BOILERPLATE[loc["language"]]
    lines = content.split("\n")
    header = [fill(rng, line) for line in text["header"]]
    lines = lines[:2] + header + lines[2:]
    footer = [""]
    if "cash_change" not in traits:
        footer += [fill(rng, line) for line in text["slip"]]
        if "no_printed_total" not in traits:
            footer.insert(2, loc["amount"](rng, total))
    if not loc.get("tax_added"):
        vat = vat_line(rng, locale, text, total)
        if vat:
            footer += ["", vat]
    footer += [""] + [fill(rng, line) for line in text["footer"]]
    return "\n".join(lines + footer)


def lengthen_email(rng, loc, content):
    text = BOILERPLATE[loc["language"]]
    extra = {"name": rng.choice(text["names"]), "street": rng.choice(text["streets"])}
    return content + "\n\n" + "\n".join(fill(rng, line, extra) for line in text["email"]) + "\n\n" + text["legal"]


def long_receipt(rng, locale, case, gold, world=LOCALES):
    """The same purchase re-rendered as a big shop: 10 to 40 item lines from
    the merchant's pool (repeats allowed, as on real receipts), and the gold
    record rebuilt from what the renderer printed."""
    loc = world[locale]
    merchant = merchant_entry(locale, gold["merchant"], world)
    if merchant is None:
        return None
    pool = merchant["mixed"][1] if "secondary_kind" in case["traits"] and merchant["mixed"] else merchant["items"]
    items = [(name, rng.randint(lo, hi)) for name, lo, hi in (rng.choice(pool) for _ in range(rng.randint(10, 40)))]
    day = date.fromisoformat(gold["purchaseDate"]) if gold["purchaseDate"] else \
        date.fromisoformat(case["today"]) - timedelta(days=rng.randint(0, 10))
    when = datetime(day.year, day.month, day.day, rng.randint(8, 21), rng.randint(0, 59))
    keep = [t for t in case["traits"] if t in ("travel", "mixed_merchant", "secondary_kind", "needs_proposal",
                                               "invented_merchant")]
    traits = list(keep)
    content, total, has_date, has_time, instrument = v1.render_till(rng, loc, merchant, items, when, traits)
    gold.update({
        "paymentInstrument": instrument["gold"] if instrument["mode"] == "exact" else "",
        "paymentInstrumentMode": instrument["mode"], "paymentInstrumentAccepted": instrument["accepted"],
        "totalAmount": v1.gold_amount_string(total, loc["currency"]), "amountMinorUnits": total,
        "purchaseDate": when.date().isoformat() if has_date else "",
        "purchaseTime": f"{when.hour:02d}:{when.minute:02d}" if has_time else "",
        "noteItems": sorted({name for name, _ in items}),
    })
    return content, traits + ["many_items"]


def varied_receipt(rng, locale, case, gold, world=LOCALES):
    loc = world[locale]
    merchant = merchant_entry(locale, gold["merchant"], world)
    if merchant is None:
        return None
    secondary = "secondary_kind" in case["traits"] and merchant["mixed"]
    cat_key, pool = merchant["mixed"] if secondary else (merchant["cat"], merchant["items"])
    day = date.fromisoformat(gold["purchaseDate"]) if gold["purchaseDate"] else \
        date.fromisoformat(case["today"]) - timedelta(days=rng.randint(0, 10))
    when = datetime(day.year, day.month, day.day, rng.randint(8, 21), rng.randint(0, 59))
    traits = [t for t in case["traits"] if t in ("travel", "mixed_merchant", "secondary_kind", "needs_proposal",
                                                 "invented_merchant")]
    content, total, has_date, has_time, instrument, names = tills.render_varied_till(
        rng, loc, locale, merchant, pool, cat_key, when, traits)
    gold.update({
        "paymentInstrument": instrument["gold"] if instrument["mode"] == "exact" else "",
        "paymentInstrumentMode": instrument["mode"], "paymentInstrumentAccepted": instrument["accepted"],
        "totalAmount": v1.gold_amount_string(total, loc["currency"]), "amountMinorUnits": total,
        "purchaseDate": when.date().isoformat() if has_date else "",
        "purchaseTime": f"{when.hour:02d}:{when.minute:02d}" if has_time else "",
        "noteItems": sorted(set(names)),
    })
    return content, traits


def overflow(rng, loc, content, limit):
    legal = BOILERPLATE[loc["language"]]["legal"]
    while len(content) <= limit + 200:
        content += "\n\n" + legal
    return content


def assign_source(rng):
    roll, cumulative = rng.random(), 0.0
    for source, weight in SOURCES:
        cumulative += weight
        if roll < cumulative:
            return source
    return SOURCES[-1][0]


def make_user(rng, profile, cases, world=LOCALES):
    """The generator's profile in the agent's ledger shape, with tags and a memory."""
    home, language = profile["home"], profile["language"]
    titles = CATEGORY_TITLES[language]
    records = []
    for index, entry in enumerate(profile["ledger"]):
        records.append({"id": f"{profile['id']}-r{index:03d}", "date": entry["date"], "time": "",
                        "merchant": entry["merchant"], "payment_instrument": entry["paymentInstrument"],
                        "amount_minor": entry["amountMinorUnits"], "currency": entry["currencyCode"],
                        "category": entry["category"], "tags": [], "note": entry["note"],
                        "source": assign_source(rng), "content": ""})

    by_merchant = {}
    for record in records:
        by_merchant.setdefault(record["merchant"], []).append(record)
    single_category = {name: group for name, group in by_merchant.items()
                       if len({r["category"] for r in group}) == 1 and len(group) >= 2}

    def preferred(names, order):
        ranked = sorted(names, key=lambda n: (next((i for i, key in enumerate(order)
                                                   if titles.get(key) == single_category[n][0]["category"]),
                                                  len(order)), n))
        return ranked[0] if ranked else None

    tags = []
    visits = Counter(c["gold"]["merchant"] for c in cases if c["gold"]["isSpending"] and "travel" not in c["traits"])
    visited = [name for name in single_category if visits[name] >= 2]
    work_merchant = preferred(visited or list(single_category), WORK_CATEGORY_PREFERENCE)
    if work_merchant:
        tags.append(WORK[language])
        for record in single_category[work_merchant]:
            record["tags"].append(WORK[language])

    gift_pool = [r for name, group in by_merchant.items() if name != work_merchant for r in group
                 if any(r["category"] == titles.get(key) for key in GIFT_CATEGORY_PREFERENCE)]
    if len(gift_pool) >= 2:
        tags.append(GIFTS[language])
        for record in rng.sample(gift_pool, 2):
            record["tags"].append(GIFTS[language])

    trips = {}
    travel_cases = [c for c in cases if "travel" in c["traits"] and c["gold"]["isSpending"]]
    for destination in sorted({content_locale(c) for c in travel_cases} - {None}):
        if rng.random() < 0.5:
            continue
        loc = world[destination]
        name = TRIP_NAME[language].format(country=COUNTRY[destination][language])
        start = min(date.fromisoformat(c["today"]) for c in travel_cases if content_locale(c) == destination)
        candidates = [m for m in loc["merchants"] if not m["online"] and m["cat"] in profile["categoryKeys"]]
        if not candidates:
            continue
        tags.append(name)
        trips[destination] = name
        for index in range(rng.randint(3, 4)):
            merchant = rng.choice(candidates)
            picked = v1.pick_items(rng, merchant["items"], rng.randint(1, 3))
            records.append({"id": f"{profile['id']}-t{destination}-{index}",
                            "date": (start - timedelta(days=rng.randint(1, 18))).isoformat(), "time": "",
                            "merchant": merchant["name"], "payment_instrument": rng.choice(["", instrument_name(rng, loc)]),
                            "amount_minor": sum(p for _, p in picked), "currency": loc["currency"],
                            "category": titles[merchant["cat"]], "tags": [name],
                            "note": f"{titles[merchant['cat']]}: " + ", ".join(n for n, _ in picked),
                            "source": assign_source(rng), "content": ""})
    records.sort(key=lambda r: (r["date"], r["id"]))

    memory, wrong_about, helps = make_memory(rng, profile, records, by_merchant, single_category, work_merchant,
                                             tags, visits)
    return {
        "id": profile["id"], "home": home, "language": language, "localCurrency": profile["localCurrency"],
        "preferences": [v1.LOCAL_CURRENCY_PREFERENCE] if profile["prefersLocalCurrencyRule"] else [],
        "categories": profile["categories"], "tags": tags, "records": records, "memory": memory,
        "memoryKind": "empty" if not memory else ("misleading" if wrong_about else "helpful"),
        "memoryWrongAbout": wrong_about, "memoryHelpsWith": sorted(helps),
        "workMerchant": work_merchant, "trips": trips,
    }


def make_memory(rng, profile, records, by_merchant, single_category, work_merchant, tags, visits):
    kind = rng.choices(["empty", "helpful", "misleading"], [0.35, 0.4, 0.25])[0]
    if kind == "empty":
        return "", None, set()
    lines, helps = [], set()
    mixed = [name for name, group in by_merchant.items() if len({r["category"] for r in group}) > 1]
    for name in mixed[:2]:
        parts = []
        for category in sorted({r["category"] for r in by_merchant[name]}):
            items = next(r["note"].split(": ", 1)[-1] for r in by_merchant[name] if r["category"] == category)
            parts.append(f"{items.split(', ')[0]} goes under {category}")
        lines.append(f"{name} is filed two ways by what was bought: " + "; ".join(parts) + ".")
        helps.add(name)
    instruments = sorted({r["payment_instrument"] for r in records if r["payment_instrument"]})
    if instruments:
        lines.append(f"{instruments[0]} is a card the user pays with, never a merchant.")
    if work_merchant and WORK[profile["language"]] in tags:
        lines.append(f"Purchases at {work_merchant} are for work: tag them {WORK[profile['language']]}.")
        helps.add(work_merchant)
    wrong_about = None
    if kind == "misleading":
        candidates = [name for name in single_category if name != work_merchant and name not in helps]
        if candidates:
            most = max(visits[name] for name in candidates)
            wrong_about = rng.choice([name for name in candidates if visits[name] == most])
            right = single_category[wrong_about][0]["category"]
            wrong = rng.choice([c for c in profile["categories"] if c != right])
            lines.insert(rng.randint(0, len(lines)), f"{wrong_about} purchases always go under {wrong}.")
    rng.shuffle(lines)
    memory = "\n".join(lines)
    return memory[:1000], wrong_about, helps


def make_case(rng, case, user, limit, world=LOCALES, varied=None):
    gold = dict(case["gold"])
    traits = list(case["traits"])
    style = case["style"]
    locale = content_locale(case) if gold["isSpending"] else user["home"]
    loc = LOCALES[locale or user["home"]]
    content = case["content"]
    expected = "needs_review" if gold["isSpending"] else "not_spending"

    traits = [t for t in traits if t != "ledger_lookup_useful"]
    roll = rng.random()
    original_gold = dict(gold)
    if style == "till" and 0.03 <= roll < 0.35:
        rendered = long_receipt(rng, locale, case, gold, world)
        if rendered:
            content, traits = rendered
    if style in ("till", "email") and roll < 0.03:
        content = overflow(rng, loc, lengthen_email(rng, loc, content) if style == "email"
                           else lengthen_till(rng, locale, content, gold["amountMinorUnits"], traits), limit)
        expected = "rejected_too_long"
        traits.append("over_limit")
    elif style == "till" and roll < 0.6:
        content = lengthen_till(rng, locale, content, gold["amountMinorUnits"], traits)
        traits.append("long_content")
    elif style == "email" and roll < 0.6:
        content = lengthen_email(rng, loc, content)
        traits.append("long_content")
    if varied is not None and style == "till" and expected != "rejected_too_long" and varied.random() < varied.share:
        varied_gold = dict(original_gold)
        varied_till = varied_receipt(varied, locale, case, varied_gold, world)
        if varied_till:
            content, traits = varied_till
            gold = varied_gold

    gold["categoryAccepted"] = [gold["category"]] if gold["category"] else []
    if style == "casual" and "mixed_merchant" in traits:
        filed = sorted({r["category"] for r in user["records"] if r["merchant"] == gold["merchant"]})
        gold["categoryAccepted"] = filed
        traits.append("category_ambiguous")

    gold["tags"] = []
    gold["proposedTagMode"] = "none"
    if gold["isSpending"]:
        if "travel" in traits:
            if locale in user["trips"]:
                gold["tags"].append(user["trips"][locale])
                traits.append("existing_trip_tag")
                gold["proposedTagMode"] = "none"
            else:
                traits.append("new_trip")
                gold["proposedTagMode"] = "optional"
        elif user["workMerchant"] and gold["merchant"] == user["workMerchant"]:
            gold["tags"].append(WORK[user["language"]])
            traits.append("standing_tag")
        if gold["merchant"] == user["memoryWrongAbout"] and "travel" not in traits:
            traits.append("memory_misleads")
        if gold["merchant"] in user["memoryHelpsWith"] and "travel" not in traits:
            traits.append("memory_helps")

    return {"id": case["id"], "user": user["id"], "language": case["language"], "homeLocale": user["home"],
            "style": style, "traits": sorted(set(traits)), "today": case["today"], "content": content,
            "expectedStatus": expected, "gold": gold}


def interleave(cases):
    """Round-robin by home locale, so any prefix (`run.py --limit`) is balanced."""
    by_locale = {}
    for case in cases:
        by_locale.setdefault(case["homeLocale"], []).append(case)
    out, queues = [], [list(v) for _, v in sorted(by_locale.items())]
    while any(queues):
        for queue in queues:
            if queue:
                out.append(queue.pop(0))
    return out


def fingerprint(cases, users):
    blob = json.dumps({"cases": cases, "users": users}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def varied_stream(seed, case_id, share):
    stream = random.Random(f"variety-{seed}-{case_id}")
    stream.share = share
    return stream


def build(seed, per_locale, users_per_locale, limit=CONTENT_CHARACTER_LIMIT, novel_share=0.0, novel_split="train",
          variety=0.0):
    """`novel_share` is the share of each user's merchants and item names
    that are invented for them (`novel_names`); 0 draws the real list only, as
    the frozen `dev` and `holdout` suites do. `novel_split` is the half of
    the invented names it draws from: `train` for training data, `eval` for
    a suite that must hold only names no training suite can contain."""
    rng = random.Random(seed)
    profiles, v1_cases, worlds = [], [], {}
    for home in v1.HOME_LOCALES:
        users = []
        for index in range(users_per_locale):
            today = date(2026, 1, 15) + timedelta(days=rng.randint(0, 600))
            world = world_for(seed, home, index, novel_share, novel_split)
            profile = v1.make_profile(rng, home, f"{seed}{index}", today, world)
            worlds[profile["id"]] = world
            users.append(profile)
        profiles += users
        for index in range(per_locale):
            profile = users[index % len(users)]
            v1_cases.append(v1.build_sample(rng, profile, index, worlds[profile["id"]]))
    extra = random.Random(seed * 1000 + SUITE_VERSION)
    users = {p["id"]: make_user(extra, p, [c for c in v1_cases if c["profile"] == p["id"]], worlds[p["id"]])
             for p in profiles}
    cases = interleave([make_case(extra, c, users[c["profile"]], limit, worlds[c["profile"]],
                                  varied_stream(seed, c["id"], variety) if variety else None) for c in v1_cases])
    return cases, list(users.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="suite name, e.g. dev or holdout")
    ap.add_argument("--per-locale", type=int, default=40)
    ap.add_argument("--users-per-locale", type=int, default=2)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--novel-share", type=float, default=0.0,
                    help="share of each user's merchants and item names invented for them (0: the real list)")
    ap.add_argument("--novel-split", choices=("train", "eval"), default="train",
                    help="the half of the invented names to draw from: train for training data, eval for a test suite")
    ap.add_argument("--variety", type=float, default=0.0,
                    help="share of till receipts re-rendered in tills.py's varied layouts (0: as the frozen suites)")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "suites"))
    args = ap.parse_args()

    cases, users = build(args.seed, args.per_locale, args.users_per_locale, novel_share=args.novel_share,
                         novel_split=args.novel_split, variety=args.variety)
    folder = os.path.join(args.out_dir, args.name)
    os.makedirs(folder, exist_ok=True)
    for name, rows in (("cases.jsonl", cases), ("users.jsonl", users)):
        with open(os.path.join(folder, name), "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    lengths = sorted(len(c["content"]) for c in cases)
    meta = {"name": args.name, "version": SUITE_VERSION, "seed": args.seed, "perLocale": args.per_locale,
            "usersPerLocale": args.users_per_locale, "novelShare": args.novel_share,
            "novelSplit": args.novel_split if args.novel_share else None, "contentCharacterLimit": CONTENT_CHARACTER_LIMIT,
            "cases": len(cases), "users": len(users), "fingerprint": fingerprint(cases, users),
            "contentLength": {"median": lengths[len(lengths) // 2], "p90": lengths[int(len(lengths) * 0.9)],
                              "max": lengths[-1]},
            "styles": dict(Counter(c["style"] for c in cases)),
            "traits": dict(Counter(t for c in cases for t in c["traits"]).most_common()),
            "memory": dict(Counter(u["memoryKind"] for u in users))}
    if args.variety:
        meta["variety"] = args.variety
    with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
