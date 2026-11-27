from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

ZERO_DECIMAL_CURRENCIES = frozenset({"JPY", "KRW", "VND", "CLP", "ISK", "UGX", "RWF", "XOF", "XAF"})


def minor_units(raw, currency):
    cleaned = "".join(c for c in str(raw) if c.isdigit() or c in ".,-")
    if not cleaned:
        return None
    if "." in cleaned and "," in cleaned:
        if cleaned.rindex(".") > cleaned.rindex(","):
            cleaned = cleaned.replace(",", "")
        else:
            cleaned = cleaned.replace(".", "").replace(",", ".")
    elif "," in cleaned:
        parts = [part for part in cleaned.split(",") if part]
        cleaned = cleaned.replace(",", "." if len(parts) == 2 and len(parts[1]) == 2 else "")
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None
    if value < 0:
        return None
    scale = 0 if currency.upper() in ZERO_DECIMAL_CURRENCIES else 2
    try:
        return int((value * (Decimal(10) ** scale)).quantize(Decimal(1), rounding=ROUND_HALF_UP))
    except InvalidOperation:
        return None


def decimal_string(minor, currency):
    if currency.upper() in ZERO_DECIMAL_CURRENCIES:
        return str(minor)
    sign = "-" if minor < 0 else ""
    minor = abs(minor)
    return f"{sign}{minor // 100}.{minor % 100:02d}"


def money(minor, currency):
    return f"{decimal_string(minor, currency)} {currency}"
