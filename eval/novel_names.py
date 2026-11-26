import random
import re
import zlib

from locales import INVENTED_CODAS, INVENTED_ONSETS, LOCALES

EXTRA_ONSETS = ["Bram", "Cal", "Dun", "Ev", "Gar", "Hol", "Ing", "Kel", "Lor", "Mel", "Nev", "Orm", "Quen",
                "Ros", "Stav", "Tam", "Ul", "Vor", "Wyn", "Yar", "Zel", "Ash", "Brec", "Dar"]
EXTRA_CODAS = ["ley", "wyn", "ham", "worth", "brook", "croft", "hurst", "more", "lin", "rick", "vale", "well",
               "combe", "shaw", "ridge", "ton", "ard", "el", "ova", "ino"]
LINKS = ["", "", "", "a", "e", "i", "o"]

LATIN_KINDS = {
    "en": {"groceries": ["Market", "Foods", "Grocers", "Fresh"], "coffee": ["Coffee", "Café", "Roasters"],
           "dining": ["Kitchen", "Grill", "Diner", "Bistro"], "health": ["Pharmacy", "Chemist"],
           "fuel": ["Fuel", "Petroleum", "Energy"], "transport": ["Rail", "Lines", "Transit"],
           "shopping": ["Store", "Direct", "Goods"], "clothing": ["Apparel", "Clothing", "Outfitters"],
           "pets": ["Pet Supplies", "Pets"], "kids": ["Toys", "Kids"], "home": ["Interiors", "Home"],
           "entertainment": ["Cinemas", "Picturehouse"]},
    "fr": {"groceries": ["Marché", "Épicerie", "Primeur"], "coffee": ["Café", "Torréfacteur"],
           "dining": ["Brasserie", "Bistrot", "Restaurant"], "health": ["Pharmacie"],
           "fuel": ["Station", "Carburants"], "transport": ["Transports", "Rail"], "shopping": ["Bazar", "Magasin"],
           "clothing": ["Boutique", "Mode"], "pets": ["Animalerie"], "kids": ["Jouets"], "home": ["Maison", "Déco"],
           "entertainment": ["Cinéma"]},
    "es": {"groceries": ["Mercado", "Supermercado", "Ultramarinos"], "coffee": ["Café", "Cafetería"],
           "dining": ["Taberna", "Restaurante", "Mesón"], "health": ["Farmacia"],
           "fuel": ["Gasolinera", "Combustibles"], "transport": ["Transportes", "Viajes"],
           "shopping": ["Bazar", "Almacenes"], "clothing": ["Moda", "Boutique"], "pets": ["Mascotas"],
           "kids": ["Juguetes"], "home": ["Hogar", "Muebles"], "entertainment": ["Cines"]},
    "de": {"groceries": ["Markt", "Frischmarkt", "Lebensmittel"], "coffee": ["Café", "Kaffeerösterei"],
           "dining": ["Gasthaus", "Restaurant", "Grill"], "health": ["Apotheke"], "fuel": ["Tankstelle"],
           "transport": ["Reisen", "Verkehr"], "shopping": ["Kaufhaus", "Warenhaus"],
           "clothing": ["Mode", "Textil"], "pets": ["Tierbedarf"], "kids": ["Spielwaren"],
           "home": ["Wohnen", "Einrichtung"], "entertainment": ["Kino", "Filmtheater"]},
}
LATIN_PATTERNS = {
    "en": ["{stem} {kind}", "{stem} {kind}", "{stem}'s", "The {stem} {kind}", "{stem} & {stem2}", "{stem}"],
    "fr": ["{kind} {stem}", "{kind} {stem}", "{stem}", "{kind} du {stem}", "{stem} & Fils"],
    "es": ["{kind} {stem}", "{kind} {stem}", "Casa {stem}", "{stem}", "{kind} El {stem}"],
    "de": ["{kind} {stem}", "{kind} {stem}", "{stem}", "{stem} {kind}", "{kind} am {stem}"],
}
LEGAL = {"en-GB": ["LTD", "Ltd"], "en-US": ["INC", "Co."], "fr": ["SARL", "SAS"], "es": ["S.L.", "S.A."],
         "es-MX": ["S.A. DE C.V."], "de": ["GmbH", "KG"]}
LOOKS_LIKE_PAYMENT = {"en": ["Pay", "Capital", "Trust"], "fr": ["Crédit", "Pay"], "es": ["Caja", "Pay"],
                      "de": ["Kasse", "Pay"]}

ZH_STEM = list("鑫华盛佳福源嘉润宏美乐康禾悦林安恒顺达星汇晨丰泰和德隆祥瑞明")
ZH_KINDS = {"groceries": ["超市", "生鲜", "便利店"], "coffee": ["咖啡"], "dining": ["餐厅", "小馆", "面馆"],
            "health": ["大药房", "药店"], "fuel": ["加油站", "石油"], "transport": ["出行", "客运"],
            "shopping": ["百货", "商城"], "clothing": ["服饰"], "pets": ["宠物"], "kids": ["玩具"],
            "home": ["家居"], "entertainment": ["影城"]}
ZH_BRANCH = ["朝阳", "徐汇", "天河", "南山", "西湖", "江汉", "锦江"]
JA_MORAE = ["ア", "カ", "サ", "タ", "ナ", "ハ", "マ", "ヤ", "ラ", "ワ", "キ", "シ", "チ", "ミ", "リ", "ク", "ス", "ツ", "ム",
            "ル", "ケ", "セ", "テ", "メ", "レ", "コ", "ソ", "ト", "モ", "ロ", "ノ", "ホ"]
JA_FAMILY = ["山本", "松井", "高田", "大森", "小川", "石田", "森下", "川口", "野村", "青木"]
JA_KINDS = {"groceries": ["スーパー", "ストア", "マート"], "coffee": ["珈琲", "カフェ"], "dining": ["食堂", "亭", "屋"],
            "health": ["薬局", "ドラッグ"], "fuel": ["石油", "SS"], "transport": ["鉄道", "交通"],
            "shopping": ["百貨店", "商店"], "clothing": ["衣料", "ファッション"], "pets": ["ペット"], "kids": ["トイ"],
            "home": ["家具", "インテリア"], "entertainment": ["シネマ"]}
JA_BRANCH = ["渋谷", "新宿", "梅田", "栄", "天神", "横浜", "札幌"]

SIZE = re.compile(r"(?<![0-9.])([0-9]+(?:[.,][0-9]+)?)(\s?)(kg|g|ml|cl|L|l|oz|lb|pk|tabs|St\.|個|枚|本|入|x)(?![A-Za-z])")
SIZES = {"kg": ["0.5", "1", "1.5", "2", "2.5", "5"], "g": ["80", "125", "200", "250", "300", "400", "500", "750"],
         "ml": ["250", "330", "500", "600", "750", "1000"], "cl": ["25", "33", "50", "75"],
         "L": ["0.5", "1", "1.5", "2", "4"], "l": ["0.5", "1", "1.5", "2"], "oz": ["8", "12", "16", "24"],
         "lb": ["1", "2", "3", "5"]}


def latin_stem(rng):
    if rng.random() < 0.5:
        stem = rng.choice(INVENTED_ONSETS + EXTRA_ONSETS) + rng.choice(INVENTED_CODAS + EXTRA_CODAS)
    else:
        stem = rng.choice(INVENTED_ONSETS + EXTRA_ONSETS) + rng.choice(LINKS) + rng.choice(EXTRA_CODAS + INVENTED_CODAS)
    return stem[0].upper() + stem[1:]


def ja_stem(rng):
    if rng.random() < 0.3:
        return rng.choice(JA_FAMILY)
    return "".join(rng.choice(JA_MORAE) for _ in range(rng.randint(2, 3)))


def kind_word(rng, kinds, cat):
    if rng.random() < 0.05:
        cat = rng.choice(sorted(kinds))
    options = kinds.get(cat)
    return rng.choice(options) if options else ""


def merchant_name(rng, locale, cat):
    language = LOCALES[locale]["language"]
    if language == "zh":
        stem = "".join(rng.sample(ZH_STEM, 2))
        return stem if rng.random() < 0.2 else stem + kind_word(rng, ZH_KINDS, cat)
    if language == "ja":
        stem = ja_stem(rng)
        return stem if rng.random() < 0.2 else stem + kind_word(rng, JA_KINDS, cat)
    stem = latin_stem(rng)
    if rng.random() < 0.08:
        return f"{stem} {rng.choice(LOOKS_LIKE_PAYMENT[language])}"
    kind = kind_word(rng, LATIN_KINDS[language], cat)
    pattern = rng.choice(LATIN_PATTERNS[language])
    if not kind and "{kind}" in pattern:
        pattern = "{stem}"
    return elided(pattern.format(stem=stem, stem2=latin_stem(rng), kind=kind))


def elided(name):
    return re.sub(r"\bdu ([AEIOUYaeiouy])", r"d'\1", name)


def printed_forms(rng, locale, name):
    language = LOCALES[locale]["language"]
    if language == "zh":
        return [name, f"{name}（{rng.choice(ZH_BRANCH)}店）"]
    if language == "ja":
        return [name, f"{name} {rng.choice(JA_BRANCH)}店"]
    forms = [name, name.upper(), f"{name.upper()} {rng.choice(LEGAL[locale])}"]
    return rng.sample(forms, rng.randint(1, 3))


def resized(rng, text):
    def swap(match):
        number, space, unit = match.groups()
        if unit in SIZES:
            choices = [s for s in SIZES[unit] if s != number] or SIZES[unit]
            value = rng.choice(choices)
            if "," in number:
                value = value.replace(".", ",")
            return f"{value}{space}{unit}"
        count = int(re.sub(r"\D", "", number) or 1)
        return f"{max(2, count + rng.choice([-4, -2, 2, 4, 6]))}{space}{unit}"
    return SIZE.sub(swap, text)


def abbreviated(word):
    return word[:5] + "." if len(word) > 7 and word.isalpha() else word


def item_name(rng, locale, name):
    language = LOCALES[locale]["language"]
    original = name
    if rng.random() < 0.6:
        name = resized(rng, name)
    if rng.random() < 0.4 or name.casefold() == original.casefold():
        if language == "zh":
            name = "".join(rng.sample(ZH_STEM, 2)) + name
        elif language == "ja":
            name = f"{ja_stem(rng)} {name}"
        else:
            name = f"{latin_stem(rng)} {name}"
    if language not in ("zh", "ja"):
        if rng.random() < 0.25:
            name = " ".join(abbreviated(w) for w in name.split())
        if rng.random() < 0.3:
            name = name.upper()
    return name


SPLITS = ("train", "eval")
ATTEMPTS = 500


def in_split(text, split):
    return zlib.crc32(text.casefold().encode("utf-8")) % len(SPLITS) == SPLITS.index(split)


def real_names(locales):
    names = set()
    for loc in locales.values():
        for merchant in loc["merchants"]:
            names |= {n.casefold() for n in [merchant["name"], *merchant["printed"], *merchant["aliases"]]}
            pools = [merchant["items"]] + ([merchant["mixed"][1]] if merchant["mixed"] else [])
            names |= {name.casefold() for pool in pools for name, _, _ in pool}
    return names


def drawn(make, split, avoid):
    for _ in range(ATTEMPTS):
        text = make()
        if in_split(text, split) and text.casefold() not in avoid:
            return text
    raise RuntimeError("no invented name in the split after many attempts")


def rewritten_items(rng, locale, items, share, split, taken):
    out = []
    for name, low, high in items:
        new = name
        if rng.random() < share:
            new = drawn(lambda: item_name(rng, locale, name), split, taken)
        taken.add(new.casefold())
        out.append((new, low, high))
    return out


def world(rng, share, split="train", locales=LOCALES):
    out, real = {}, real_names(locales)
    for code, loc in locales.items():
        used = set(real)
        merchants = []
        for merchant in loc["merchants"]:
            taken = set(real)
            renamed = dict(merchant)
            renamed["items"] = rewritten_items(rng, code, merchant["items"], share, split, taken)
            if merchant["mixed"]:
                renamed["mixed"] = (merchant["mixed"][0],
                                    rewritten_items(rng, code, merchant["mixed"][1], share, split, taken))
            if rng.random() < share:
                name = drawn(lambda: merchant_name(rng, code, merchant["cat"]), split, used)
                used.add(name.casefold())
                renamed.update(name=name, printed=printed_forms(rng, code, name), aliases=[], invented=True)
            merchants.append(renamed)
        out[code] = {**loc, "merchants": merchants}
    return out


def world_for(seed, home, index, share, split="train"):
    if share <= 0:
        return LOCALES
    return world(random.Random(f"names-{seed}-{home}-{index}"), share, split)
