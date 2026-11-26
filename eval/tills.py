"""Varied till receipts: the layouts real tills print that `receipts.render_till`
never does.

`render_till` prints items, an optional subtotal and tax, a total and a
payment line. Real receipts are messier, and a small model trained only on
the tidy layout reads a messy one wrongly: a "TOTAL DISCOUNT 0.20" line
taken for the grand total, a negative price cut added instead of subtracted,
a per-rate VAT table summed into the total. This renderer prints the same
purchase in the shapes real tills use, so the teacher's sessions (and the
student trained on them) cover them:

- quantity lines ("2 x 1.15") with the line total beside them, product
  codes inside item names, VAT codes after prices, a lone currency heading;
- negative lines that are money taken off: price cuts, multibuy savings,
  member prices, coupons, bottle returns, printed "-0.20", "0,20-" or "△20";
- small positive lines that are money paid: carrier bags, bottle deposits;
- lines that say "total" or carry an amount and are not the total: total
  discount, total savings, item counts, loyalty points;
- a per-rate VAT/tax table (sales and tax by rate), in columns or cell by cell;
- a card slip with masked numbers, printed before or after the VAT table,
  whose amount is printed, blank or masked;
- the grand total printed, printed only on the slip, or not printed at all,
  so the record has to be added up from the lines (as a folded or cut-off
  receipt, or a photo with the amount hidden, reads);
- a logo the OCR misread, so the merchant must come from the user's records.

The gold total is always what a careful person adds up from the paper: the
item lines, minus every reduction, plus every charge (plus tax where tax is
added at the till). Traits name each shape, so a run's score breaks down by
it. Stdlib only. `make_suite.py --variety` turns it on; the frozen suites
never use it.
"""

from locales import grouped
from receipts import cash_tendered

TEXT = {
    "en": {
        "qty": ["{q} x {unit}", "{q} @ {unit}", "{q} X {unit}"],
        "reductions": ["Price Cut", "PRICE REDUCTION", "Multibuy Saving", "Member Price", "Reduced to clear",
                       "Discount", "Promo Saving"],
        "coupons": ["COUPON", "Voucher", "Staff Discount", "Money off coupon"],
        "savings": ["TOTAL DISCOUNT", "TOTAL SAVINGS", "YOU SAVED", "Savings today", "TOTAL PROMOTIONS"],
        "count": ["TOTAL ITEMS {n}", "ITEMS SOLD {n}", "NO. OF ITEMS {n}"],
        "points": ["Points earned {p}", "Points balance {b}", "Clubcard points this visit {p}"],
        "vat_header": ["VAT RATE", "SALES {sym}", "VAT {sym}"],
        "slip_head": ["*CUSTOMER COPY* - PLEASE RETAIN RECEIPT", "CARDHOLDER COPY", "CUSTOMER COPY"],
        "slip_mid": ["MID: ********{d5}", "TID: ****{d4}", "TRNS NO: UK{d18}", "A0000000{d6}"],
        "slip_kind": ["Contactless", "SALE", "Chip & PIN"],
        "slip_amount": ["Amount", "AMOUNT", "SALE AMOUNT"],
        "slip_tail": ["Verification Not Required", "APPROVED", "AUTH CODE {auth}",
                      "PLEASE DEBIT ACCOUNT WITH TOTAL SHOWN"],
        "date": "Date: {date}", "time": "Time: {time}",
        "schemes": ["DEBIT MASTERCARD", "VISA DEBIT", "VISA", "MASTERCARD", "AMEX"],
    },
    "zh": {
        "qty": ["{q}×{unit}", "{q} x {unit}", "数量{q} 单价{unit}"],
        "reductions": ["优惠", "会员价优惠", "促销折扣", "满减", "第二件半价"],
        "coupons": ["优惠券", "代金券", "整单优惠"],
        "savings": ["共节省", "优惠合计", "本单优惠", "折扣合计"],
        "count": ["商品总数 {n}", "共{n}件"],
        "points": ["本次积分 {p}", "累计积分 {b}"],
        "vat_header": None,
        "slip_head": ["持卡人存根", "签购单 持卡人存根"],
        "slip_mid": ["商户号: ********{d5}", "终端号: ****{d4}", "交易参考号: {d12}"],
        "slip_kind": ["交易类型: 消费", "闪付"],
        "slip_amount": ["金额", "交易金额"],
        "slip_tail": ["授权码: {auth}", "本人确认以上交易，同意将其记入本卡账户"],
        "date": "日期: {date}", "time": "时间: {time}",
        "schemes": ["银联", "VISA", "MASTERCARD", "银联借记卡"],
    },
    "ja": {
        "qty": ["{q}点 @{unit}", "@{unit} x {q}", "{q}個 × {unit}"],
        "reductions": ["値引", "割引", "会員割引", "見切り値引", "まとめ割"],
        "coupons": ["クーポン値引", "ポイント値引"],
        "savings": ["お値引合計", "割引合計", "本日のお値引"],
        "count": ["お買上点数 {n}点", "{n}点"],
        "points": ["今回ポイント {p}P", "ポイント残高 {b}P"],
        "vat_header": None,
        "slip_head": ["お客様控え", "クレジット売上票 お客様控え"],
        "slip_mid": ["端末番号: ****{d4}", "伝票番号: {d5}", "承認番号: {auth}"],
        "slip_kind": ["一括", "タッチ決済"],
        "slip_amount": ["金額", "ご利用金額"],
        "slip_tail": ["上記正に領収いたしました"],
        "date": "{date}", "time": "{time}",
        "schemes": ["VISA", "MASTERCARD", "JCB", "AMEX"],
    },
    "fr": {
        "qty": ["{q} x {unit}", "{q} X {unit}"],
        "reductions": ["Remise", "REMISE IMMEDIATE", "Promo", "Réduction fidélité", "Prix choc"],
        "coupons": ["Bon de réduction", "Coupon", "Remise ticket"],
        "savings": ["TOTAL REMISES", "Vous avez économisé", "ECONOMIES", "TOTAL AVANTAGES"],
        "count": ["NOMBRE D'ARTICLES {n}", "{n} ARTICLES"],
        "points": ["Points fidélité {p}", "Cagnotte {b}"],
        "vat_header": ["Taux", "TTC", "TVA"],
        "slip_head": ["TICKET CLIENT", "CARTE BANCAIRE"],
        "slip_mid": ["MID: ****{d5}", "TID: ****{d4}", "A0000000{d6}"],
        "slip_kind": ["SANS CONTACT", "DEBIT"],
        "slip_amount": ["MONTANT", "Montant"],
        "slip_tail": ["TRANSACTION ACCEPTEE", "AUTO: {auth}", "A CONSERVER"],
        "date": "Le {date}", "time": "à {time}",
        "schemes": ["CB", "VISA", "MASTERCARD", "CB VISA"],
    },
    "es": {
        "qty": ["{q} x {unit}", "{q} UD x {unit}"],
        "reductions": ["Descuento", "DTO.", "Promoción", "Oferta 2ª unidad", "Ahorro socio"],
        "coupons": ["Cupón", "Vale descuento"],
        "savings": ["TOTAL AHORRO", "Has ahorrado", "AHORRO TOTAL", "TOTAL DESCUENTOS"],
        "count": ["Nº ARTÍCULOS {n}", "{n} ARTÍCULOS"],
        "points": ["Puntos obtenidos {p}", "Saldo puntos {b}"],
        "vat_header": ["TIPO", "IMPORTE", "CUOTA"],
        "slip_head": ["COPIA PARA EL CLIENTE", "COPIA CLIENTE"],
        "slip_mid": ["COMERCIO: ****{d5}", "TPV: ****{d4}", "A0000000{d6}"],
        "slip_kind": ["VENTA", "CONTACTLESS"],
        "slip_amount": ["IMPORTE", "Importe"],
        "slip_tail": ["OPERACIÓN APROBADA", "AUT: {auth}", "CONSERVE ESTE TICKET"],
        "date": "Fecha: {date}", "time": "Hora: {time}",
        "schemes": ["VISA", "MASTERCARD", "VISA DÉBITO"],
    },
    "de": {
        "qty": ["{q} Stk x {unit}", "{q} x {unit}"],
        "reductions": ["Rabatt", "Preisvorteil", "Aktion", "Sofortrabatt", "Mitgliederrabatt"],
        "coupons": ["Coupon", "Gutschein", "Rabattcoupon"],
        "savings": ["Gesamtersparnis", "Sie haben gespart", "RABATT GESAMT", "Ihr Preisvorteil"],
        "count": ["Anzahl Artikel {n}", "{n} Artikel"],
        "points": ["Punkte heute {p}", "Punktestand {b}"],
        "vat_header": ["MwSt", "Brutto", "MwSt"],
        "slip_head": ["Kundenbeleg", "KUNDENBELEG"],
        "slip_mid": ["Terminal-ID: ****{d4}", "TA-Nr: {d5}", "A0000000{d6}"],
        "slip_kind": ["Kartenzahlung", "kontaktlos"],
        "slip_amount": ["Betrag", "BETRAG"],
        "slip_tail": ["Zahlung erfolgt", "Genehmigungs-Nr.: {auth}", "Bitte Beleg aufbewahren"],
        "date": "Datum: {date}", "time": "Uhrzeit: {time}",
        "schemes": ["girocard", "VISA", "MASTERCARD", "DEBIT MASTERCARD"],
    },
}

LOCALE_TEXT = {
    "en-US": {
        "reductions": ["Club Price", "YOU SAVED", "Instant Savings", "Mfr Coupon", "Sale Price Savings"],
        "coupons": ["COUPON", "Store Coupon", "Digital Coupon"],
        "savings": ["TOTAL SAVINGS", "YOUR SAVINGS TODAY", "YOU SAVED", "TOTAL DISCOUNTS"],
        "points": ["Rewards earned {p}", "Rewards balance {b}"],
        "slip_head": ["CUSTOMER COPY", "*** CUSTOMER COPY ***"],
        "slip_mid": ["MID: ********{d5}", "TID: ****{d4}", "REF #: {d12}", "AID: A0000000{d6}"],
        "slip_kind": ["CONTACTLESS", "PURCHASE", "CHIP READ"],
        "slip_tail": ["APPROVED", "AUTH CODE {auth}", "NO SIGNATURE REQUIRED"],
    },
}
COUNTABLE = ("groceries", "coffee", "dining", "health", "home", "kids", "pets", "shopping")
BIG_SHOP = ("groceries", "home", "health", "kids", "pets", "shopping")

CHARGES = {
    "en-GB": [("Carrier Bag", 10, 10), ("Bag for Life", 50, 70)],
    "en-US": [("Bottle Deposit", 5, 10), ("Paper Bag Fee", 10, 10)],
    "zh": [("购物袋", 30, 50)],
    "ja": [("レジ袋", 3, 5)],
    "fr": [("Sac cabas", 10, 50)],
    "es": [("Bolsa", 5, 15)],
    "es-MX": [("Bolsa", 100, 300)],
    "de": [("Pfand", 25, 25), ("Tragetasche", 20, 30)],
}
RETURNS = {"de": "Leergut", "en-GB": "Bottle Return", "fr": "Retour consigne"}

VAT_RATES = {
    "en-GB": [("A", 0.0), ("B", 0.2)], "de": [("B", 0.07), ("A", 0.19)], "fr": [("A", 0.055), ("B", 0.2)],
    "es": [("A", 0.10), ("B", 0.21)], "es-MX": [("A", 0.0), ("B", 0.16)], "ja": [("※", 0.08), ("", 0.10)],
}
FOOD = ("groceries", "coffee", "dining")
CURRENCY_HEADINGS = {"GBP": ["£", "GBP"], "USD": ["USD", "US$"], "CNY": ["¥", "RMB", "元"], "JPY": ["円", "¥"],
                     "EUR": ["€", "EUR"], "MXN": ["MXN", "$ MXN"]}
SYMBOL = {"GBP": "£", "USD": "$", "CNY": "¥", "JPY": "¥", "EUR": "€", "MXN": "$"}


def plain(loc, minor):
    """An amount with no currency, written the locale's way."""
    if loc["zero_decimal"]:
        return grouped(minor, True)
    if loc["currency"] == "EUR":
        return grouped(minor, False, thousands=" " if loc["language"] == "fr" else ".", decimal=",")
    return grouped(minor, False)


def percent(loc, rate):
    text = f"{rate * 100:.1f}".rstrip("0").rstrip(".")
    return text.replace(".", ",") if loc["currency"] == "EUR" else text


def digits(rng, n):
    return "".join(str(rng.randint(0, 9)) for _ in range(n))


def misread(rng, name):
    """The logo as OCR reads it: one letter swapped for a look-alike, so the
    printed name is near the merchant's, never equal to it."""
    swaps = {"i": "a", "I": "A", "o": "a", "O": "D", "e": "c", "E": "F", "l": "I", "L": "I", "a": "o",
             "n": "m", "S": "5", "B": "8", "c": "e"}
    spots = [i for i, ch in enumerate(name) if ch in swaps]
    if len(name) < 4 or not spots or not name.isascii():
        return None
    i = rng.choice(spots[1:] or spots)
    return name[:i] + swaps[name[i]] + name[i + 1:]


def reduction_amount(rng, loc, line_total):
    cut = max(1, round(line_total * rng.uniform(0.05, 0.4)))
    if loc["zero_decimal"]:
        cut = max(10, cut // 10 * 10)
    return min(cut, line_total - 1) if line_total > 1 else 0


def render_varied_till(rng, loc, locale, merchant, items_pool, cat_key, when, traits):
    """One till receipt for a purchase at `merchant`. Returns (content, total
    in minor units, has_date, has_time, instrument, item names)."""
    T = dict(TEXT[loc["language"]], **LOCALE_TEXT.get(locale, {}))
    L = loc["labels"]
    cur = loc["currency"]
    money = lambda m: loc["amount"](rng, m)
    split = rng.random() < 0.4
    with_symbol = rng.random() < 0.4
    show = (lambda m: money(m)) if with_symbol else (lambda m: plain(loc, m))
    traits.append("varied_till")
    if split:
        traits.append("column_split")

    rates = VAT_RATES.get(locale)
    vat_codes = rates is not None and loc["language"] != "ja" and rng.random() < 0.5
    minus = rng.choice(["-{a}", "-{a}"] + (["{a}-"] if loc["language"] == "de" else [])
                       + (["△{a}", "▲{a}"] if loc["language"] == "ja" else []))

    def negative(minor):
        return minus.format(a=plain(loc, minor))

    lines = []

    def row(label, value):
        lines.extend([label, value] if split else [f"{label}  {value}"])

    # header
    printed = rng.choice(merchant["printed"])
    if rng.random() < 0.12:
        wrong = misread(rng, printed)
        if wrong:
            printed = wrong
            traits.append("logo_misread")
    lines.append(printed)
    if rng.random() < 0.8:
        lines.append(rng.choice(L["addresses"]))
    if rates and rng.random() < 0.5 and loc["language"] == "en":
        lines.append(f"VAT NO. GB{digits(rng, 9)}")
    if not with_symbol and rng.random() < 0.7:
        lines.append(rng.choice(CURRENCY_HEADINGS[cur]))
    has_date = rng.random() < 0.9
    has_time = has_date and rng.random() < 0.8
    stamp_in_slip = has_date and rng.random() < 0.5
    stamp_text = loc["date"](rng, when) if has_date else ""
    time_text = loc["time"](when.hour, when.minute) if has_time else ""
    if has_time and rng.random() < 0.4:
        time_text += f":{rng.randint(0, 59):02d}"
    if has_date and not stamp_in_slip:
        lines.append(f"{stamp_text} {time_text}".strip())

    # items
    many = cat_key in BIG_SHOP and rng.random() < 0.15
    count = rng.randint(8, 20) if many else rng.randint(2, 6)
    picked = [rng.choice(items_pool) for _ in range(count)] if many else \
        rng.sample(items_pool, min(len(items_pool), count))
    if many:
        traits.append("many_items")
    food_weight = 0.75 if cat_key in FOOD else 0.05
    by_rate, prices = {}, {}
    total, savings, names, item_count = 0, 0, [], 0
    for name, lo, hi in picked:
        unit = prices.setdefault(name, rng.randint(lo, hi))
        qty = rng.choice([2, 2, 3, 4]) if cat_key in COUNTABLE and rng.random() < 0.18 else 1
        line_total = unit * qty
        item_count += qty
        names.append(name)
        rate = None
        if rates:
            rate = rates[0] if rng.random() < food_weight else rates[1]
        label = name
        if rng.random() < 0.12:
            label = f"{name} {digits(rng, 7)}"
            traits.append("product_code")
        if rate and loc["language"] == "ja" and rate[0]:
            label = f"{rate[0]}{label}"
        code = f" {rate[0]}" if vat_codes and rate else ""
        if qty > 1:
            traits.append("quantity_line")
            qty_text = rng.choice(T["qty"]).format(q=qty, unit=show(unit))
            if rng.random() < 0.5:
                row(f"{label} {qty_text}", show(line_total) + code)
            else:
                lines.append(label)
                row(f"  {qty_text}", show(line_total) + code)
        else:
            row(label, show(line_total) + code)
        total += line_total
        if rate:
            by_rate[rate] = by_rate.get(rate, 0) + line_total
        if rng.random() < 0.22:
            cut = reduction_amount(rng, loc, line_total)
            if cut:
                row("    " + rng.choice(T["reductions"] + [L["reduction"]]), negative(cut))
                total -= cut
                savings += cut
                if rate:
                    by_rate[rate] -= cut
                traits.append("reduction_line")

    if cat_key in BIG_SHOP and rng.random() < 0.15:
        name, lo, hi = rng.choice(CHARGES[locale] if locale in CHARGES else CHARGES["en-GB"])
        qty = rng.randint(1, 3) if name in ("Pfand", "Bottle Deposit") else 1
        charge = rng.randint(lo, hi) * qty
        row(name if qty == 1 else f"{name} {qty} x {plain(loc, charge // qty)}", show(charge))
        total += charge
        if rates:
            by_rate[rates[-1]] = by_rate.get(rates[-1], 0) + charge
        traits.append("charge_line")
    if locale in RETURNS and cat_key == "groceries" and rng.random() < 0.08:
        bottles = rng.randint(1, 6)
        back = bottles * 25
        room = by_rate.get(rates[-1], 0) if rates else total
        if back < min(total, room):
            row(f"{RETURNS[locale]} {bottles} x {plain(loc, 25)}", negative(back))
            total -= back
            if rates:
                by_rate[rates[-1]] = by_rate.get(rates[-1], 0) - back
            traits.append("reduction_line")

    coupon = 0
    if rng.random() < 0.12 and not rates:
        coupon = reduction_amount(rng, loc, total)
    subtotal = total
    totals = []
    if coupon or (rng.random() < 0.3):
        totals.append((rng.choice(L["subtotal"]), show(subtotal)))
    if coupon:
        totals.append((rng.choice(T["coupons"]), negative(coupon)))
        total -= coupon
        savings += coupon
        traits.append("coupon_line")
    if loc["tax_added"]:
        tax = round(total * rng.choice([0.0725, 0.08875, 0.06]))
        totals.append((rng.choice(L["tax"]), show(tax)))
        total += tax
        traits.append("tax_added")

    # how the grand total shows, if it does
    total_mode = rng.choices(["printed", "slip_only", "none"], [0.55, 0.2, 0.25])[0]
    if total_mode == "printed":
        totals.append((rng.choice(L["total"]), money(total)))
    else:
        traits.append("no_printed_total")
    for label, value in totals:
        row(label, value)

    # payment
    blocks = []
    roll = rng.random()
    if total_mode == "printed" and roll < 0.25:
        tendered = cash_tendered(total, cur)
        row(L["tendered"], show(tendered))
        row(L["change"], show(tendered - total))
        traits.append("cash_change")
        instrument = {"mode": "either", "gold": L["cash_name"], "accepted": [L["cash_name"], "Cash"]}
        if stamp_in_slip:
            lines.append(f"{stamp_text} {time_text}".strip())
    else:
        scheme = rng.choice(T["schemes"])
        instrument = {"mode": "either", "gold": scheme.title(), "accepted": [scheme, scheme.title()]}
        if total_mode == "printed" and rng.random() < 0.5:
            row(scheme if rng.random() < 0.5 else L["card"], money(total))
        if total_mode != "printed" or rng.random() < 0.6:
            blocks.append(slip(rng, T, scheme, total_mode, money(total), stamp_text if stamp_in_slip else "",
                               time_text if stamp_in_slip else "", split))
            traits.append("card_slip")
        elif stamp_in_slip:
            lines.append(f"{stamp_text} {time_text}".strip())

    # lines that carry a number and are not the total
    extras = []
    if savings and rng.random() < 0.6:
        extras.append(pair(rng.choice(T["savings"]), plain(loc, savings), split))
        traits.append("savings_line")
    if rng.random() < 0.25:
        extras.append([rng.choice(T["count"]).format(n=item_count)])
        traits.append("count_line")
    if rng.random() < 0.2:
        extras.append([rng.choice(T["points"]).format(p=rng.randint(5, 300), b=rng.randint(300, 9000))])
    if extras:
        blocks.append([line for block in extras for line in block])
    if rates and by_rate and rng.random() < 0.45:
        blocks.append(vat_table(rng, loc, T, by_rate, split))
        traits.append("vat_table")
    rng.shuffle(blocks)
    for block in blocks:
        lines.append("")
        lines.extend(block)

    if rng.random() < 0.5:
        footer = f"{digits(rng, 4)}   {digits(rng, 6)}/{digits(rng, 2)}"
        if has_date and rng.random() < 0.5:
            footer += f"   {when.strftime('%d.%m.%y')}"
            if has_time:
                footer += f"   {loc['time'](when.hour, when.minute)}"
        lines.append("")
        lines.append(footer)
    if rng.random() < 0.4:
        lines.append(rng.choice(L["thanks"]))

    assert total > 0
    content = "\n".join(lines).replace("\n\n\n", "\n\n").strip()
    return content, total, has_date, has_time, instrument, names


def pair(label, value, split):
    return [label, value] if split else [f"{label}  {value}"]


def slip(rng, T, scheme, total_mode, amount, stamp, time_text, split):
    lines = [rng.choice(T["slip_head"])]
    if stamp:
        lines.append(T["date"].format(date=stamp) + (("   " + T["time"].format(time=time_text)) if time_text else ""))
    fill = {"d4": digits(rng, 4), "d5": digits(rng, 5), "d6": digits(rng, 6), "d12": digits(rng, 12),
            "d18": digits(rng, 18), "auth": digits(rng, 6)}
    for line in T["slip_mid"]:
        if rng.random() < 0.8:
            lines.append(line.format(**fill))
    lines.append(scheme)
    lines.append("*" * 12 + digits(rng, 4))
    for line in T["slip_kind"]:
        if rng.random() < 0.6:
            lines.append(line)
    label = rng.choice(T["slip_amount"])
    if total_mode == "none":
        hidden = rng.choice(["", "****", "▇▇▇▇"])
        lines.append(f"{label} {hidden}".strip())
    else:
        lines.extend(pair(label, amount, split))
    for line in T["slip_tail"]:
        lines.append(line.format(**fill))
    return lines


def vat_table(rng, loc, T, by_rate, split):
    sym = SYMBOL[loc["currency"]]
    rows = []
    for (code, rate), gross in sorted(by_rate.items(), key=lambda kv: kv[0][1]):
        tax = round(gross * rate / (1 + rate))
        if loc["language"] == "ja":
            rows.append([f"({percent(loc, rate)}%対象 {loc['amount'](rng, gross)}  内消費税 {loc['amount'](rng, tax)})"])
        else:
            rows.append([code, f"{percent(loc, rate)} %", plain(loc, gross), plain(loc, tax)])
    if loc["language"] == "ja":
        return [r[0] for r in rows]
    header = [cell.format(sym=sym) for cell in T["vat_header"]]
    if split:
        return header + [cell for r in rows for cell in r]
    return ["  ".join(header)] + ["  ".join(r) for r in rows]
