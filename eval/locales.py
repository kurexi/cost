"""Per-locale content for the multilingual receipt set.

Everything a renderer needs to write a believable receipt, notification,
wallet screen, order email or note in one of six languages: merchants with their items and price ranges, the user's category
titles as a user of that language would name them, receipt labels, amount and
date formats, and generated payment-instrument names.

Instrument names are generated, never drawn from a fixed pool alone: a
share of every locale's names is invented from syllables, so a model can
only learn where an instrument sits, not which brands are instruments.

Prices are (low, high) in minor units of the locale's currency; JPY is
zero-decimal, so its minor unit is one yen.
"""

import random

CATEGORY_KEYS = [
    "groceries", "coffee", "dining", "transport", "fuel", "shopping",
    "clothing", "home", "health", "entertainment", "pets", "kids",
]

CATEGORY_TITLES = {
    "en": {"groceries": "Groceries", "coffee": "Coffee", "dining": "Dining", "transport": "Transport",
           "fuel": "Fuel", "shopping": "Shopping", "clothing": "Clothing", "home": "Home",
           "health": "Health", "entertainment": "Entertainment", "pets": "Pets", "kids": "Kids"},
    "zh": {"groceries": "超市", "coffee": "咖啡", "dining": "餐饮", "transport": "交通",
           "fuel": "加油", "shopping": "购物", "clothing": "服饰", "home": "家居",
           "health": "医疗", "entertainment": "娱乐", "pets": "宠物", "kids": "孩子"},
    "ja": {"groceries": "食料品", "coffee": "カフェ", "dining": "外食", "transport": "交通費",
           "fuel": "ガソリン", "shopping": "買い物", "clothing": "衣類", "home": "住まい",
           "health": "医療", "entertainment": "娯楽", "pets": "ペット", "kids": "子ども"},
    "fr": {"groceries": "Courses", "coffee": "Café", "dining": "Restaurants", "transport": "Transports",
           "fuel": "Carburant", "shopping": "Shopping", "clothing": "Vêtements", "home": "Maison",
           "health": "Santé", "entertainment": "Loisirs", "pets": "Animaux", "kids": "Enfants"},
    "es": {"groceries": "Supermercado", "coffee": "Café", "dining": "Restaurantes", "transport": "Transporte",
           "fuel": "Gasolina", "shopping": "Compras", "clothing": "Ropa", "home": "Hogar",
           "health": "Salud", "entertainment": "Ocio", "pets": "Mascotas", "kids": "Niños"},
    "de": {"groceries": "Lebensmittel", "coffee": "Kaffee", "dining": "Essen gehen", "transport": "Verkehr",
           "fuel": "Tanken", "shopping": "Einkäufe", "clothing": "Kleidung", "home": "Haushalt",
           "health": "Gesundheit", "entertainment": "Freizeit", "pets": "Haustiere", "kids": "Kinder"},
}

PROPOSAL_SYNONYMS = {
    "pets": ["Pets", "Pet", "Pet Supplies", "Pet Care", "宠物", "宠物用品", "ペット", "ペット用品",
             "Animaux", "Animalerie", "Mascotas", "Haustiere", "Tierbedarf"],
    "kids": ["Kids", "Children", "Baby", "Toys", "孩子", "儿童", "母婴", "子ども", "子供", "ベビー",
             "Enfants", "Bébé", "Jouets", "Niños", "Bebé", "Juguetes", "Kinder", "Baby", "Spielzeug"],
    "entertainment": ["Entertainment", "Leisure", "Movies", "Cinema", "娱乐", "电影", "娯楽", "映画",
                      "Loisirs", "Cinéma", "Ocio", "Cine", "Freizeit", "Kino", "Unterhaltung"],
    "health": ["Health", "Pharmacy", "Medical", "医疗", "药品", "医療", "薬", "ドラッグストア",
               "Santé", "Pharmacie", "Salud", "Farmacia", "Gesundheit", "Apotheke", "Drogerie"],
    "home": ["Home", "Household", "Furniture", "家居", "家具", "住まい", "日用品", "家具",
             "Maison", "Bricolage", "Hogar", "Haushalt", "Wohnen", "Möbel"],
    "clothing": ["Clothing", "Clothes", "Fashion", "服饰", "服装", "衣類", "衣服", "ファッション",
                 "Vêtements", "Ropa", "Moda", "Kleidung", "Mode"],
    "fuel": ["Fuel", "Petrol", "Gas", "加油", "油费", "ガソリン", "Carburant", "Essence",
             "Gasolina", "Combustible", "Tanken", "Kraftstoff", "Benzin"],
    "transport": ["Transport", "Travel", "交通", "出行", "交通費", "Transports", "Transporte", "Verkehr", "Mobilität"],
}

INVENTED_ONSETS = ["Vel", "Tor", "Mar", "Ken", "Bri", "Sol", "Lan", "Ard", "Fen",
                   "Cor", "Hal", "Nor", "Wes", "Ald", "Pen", "Rav", "Ost", "Elm"]
INVENTED_CODAS = ["mar", "dis", "ton", "by", "ford", "wick", "stone", "field",
                  "gate", "holm", "wood", "bridge", "mere", "dale", "bank"]


def invented_latin(rng):
    return rng.choice(INVENTED_ONSETS) + rng.choice(INVENTED_CODAS)


def plain(minor, zero_decimal):
    """The gold form: '12.34', or '1280' for a zero-decimal currency."""
    if zero_decimal:
        return str(minor)
    return f"{minor // 100}.{minor % 100:02d}"


def grouped(minor, zero_decimal, thousands=",", decimal="."):
    if zero_decimal:
        whole, frac = minor, None
    else:
        whole, frac = minor // 100, minor % 100
    digits = f"{whole:,}".replace(",", thousands)
    return digits if frac is None else f"{digits}{decimal}{frac:02d}"


def M(name, cat, items, printed=None, mixed=None, online=False, aliases=None):
    """One merchant. `items` is [(item, low, high)] in minor units.

    `printed` is how a till prints the name (several spellings: the gold
    merchant is `name`, any of `aliases` is also accepted). `mixed` is a second
    (category, items) the user also files purchases here under."""
    return {
        "name": name, "cat": cat, "items": items,
        "printed": printed or [name], "mixed": mixed, "online": online,
        "aliases": aliases or [],
    }


LOCALES = {}

def _en_gb_amount(rng, minor):
    p = grouped(minor, False)
    return rng.choice([f"£{p}", f"£{p}", p, f"GBP {p}"])


LOCALES["en-GB"] = {
    "language": "en", "currency": "GBP", "zero_decimal": False, "tax_added": False,
    "amount": _en_gb_amount,
    "date": lambda rng, d: rng.choice([d.strftime("%d/%m/%Y"), d.strftime("%d %b %Y"), d.strftime("%d/%m/%y")]),
    "date_no_year": lambda d: d.strftime("%d %b"),
    "time": lambda h, m: f"{h:02d}:{m:02d}",
    "labels": {
        "total": ["TOTAL", "BALANCE DUE", "TOTAL TO PAY", "Total"],
        "subtotal": ["SUBTOTAL"], "items": "{n} ITEMS",
        "tax_included": "VAT INCLUDED {amt}",
        "tendered": "CASH", "change": "CHANGE DUE",
        "was": "WAS {amt}", "reduction": "PRICE REDUCTION",
        "card": "CARD", "cash_name": "Cash",
        "thanks": ["Thank you for shopping with us", "THANK YOU", "Please retain your receipt"],
        "addresses": ["High Street, Southampton SO14 7DU", "12 Market Square, Leeds LS1 6DT",
                      "Oxford Road, Manchester M1 5QA", "Station Rd, Reading RG1 1LG"],
        "receipt_no": "Receipt {n}",
    },
    "merchants": [
        M("Tesco", "groceries", [("Semi Skimmed Milk 2L", 135, 165), ("Hovis Wholemeal", 120, 175), ("Free Range Eggs x6", 175, 260), ("Bananas loose", 70, 130), ("Penne 500g", 75, 110), ("Cheddar 400g", 300, 425)],
          printed=["TESCO", "TESCO EXPRESS", "Tesco Stores Ltd"], mixed=("coffee", [("Latte", 250, 330), ("Flat White", 260, 340), ("Pain au Chocolat", 95, 140)])),
        M("Sainsbury's", "groceries", [("Chicken Breast 300g", 300, 420), ("Basmati Rice 1kg", 180, 260), ("Greek Yoghurt", 120, 200), ("Baby Spinach", 100, 160)],
          printed=["SAINSBURY'S", "Sainsbury’s Local", "SAINSBURYS"]),
        M("Pret A Manger", "coffee", [("Americano", 210, 290), ("Chicken Caesar Wrap", 450, 560), ("Filter Coffee", 150, 220)], printed=["PRET A MANGER", "Pret"]),
        M("Costa Coffee", "coffee", [("Flat White", 320, 390), ("Medio Latte", 330, 400), ("Almond Croissant", 260, 320)], printed=["COSTA COFFEE", "Costa"]),
        M("Boots", "health", [("Paracetamol 16", 50, 95), ("Vitamin D 90 tabs", 400, 700), ("Plasters", 200, 350)], printed=["BOOTS", "Boots UK"]),
        M("Shell", "fuel", [("Unleaded", 3500, 7500)], printed=["SHELL", "Shell Petrol Station"],
          mixed=("groceries", [("Meal Deal", 350, 450), ("Still Water 1.5L", 100, 160), ("Crisps", 90, 150)])),
        M("Nando's", "dining", [("1/2 Chicken", 1095, 1295), ("Peri Chips", 395, 495), ("Bottomless Drink", 325, 395)], printed=["NANDO'S", "Nandos"]),
        M("Trainline", "transport", [("Off-Peak Return", 1500, 6500), ("Booking fee", 50, 150)], online=True),
        M("Argos", "shopping", [("Kettle", 1500, 3500), ("HDMI Cable", 500, 1200), ("Desk Lamp", 1200, 2500)], online=True),
        M("Primark", "clothing", [("Jeans", 1200, 1800), ("Hoodie", 1000, 1500), ("Socks 5pk", 350, 500)], printed=["PRIMARK"]),
        M("Pets at Home", "pets", [("Dry Cat Food 2kg", 900, 1400), ("Cat Litter 10L", 500, 800), ("Dog Chews", 300, 600)], printed=["PETS AT HOME"]),
        M("Smyths Toys", "kids", [("LEGO Classic", 1500, 3500), ("Colouring Set", 500, 1000)], printed=["SMYTHS TOYS"]),
        M("Dunelm", "home", [("Cushion Cover", 600, 1200), ("Bath Towel", 800, 1500), ("Storage Box", 500, 1000)], online=True),
        M("Odeon", "entertainment", [("Adult Ticket", 1100, 1500), ("Large Popcorn", 550, 750)], printed=["ODEON CINEMAS", "Odeon"]),
    ],
    "instruments": ["Monzo", "Starling", "Barclays", "Lloyds", "HSBC", "NatWest", "Nationwide", "Halifax", "Amex", "Revolut", "Santander"],
    "instrument_suffixes": ["", " Debit", " Credit Card", " Current Account", " Card"],
    "notification": [
        "{instrument}: You spent {amount} at {merchant} on {date_no_year} at {time}.",
        "{instrument}\nPayment to {merchant}\n{amount} · {date_no_year}, {time}",
    ],
    "wallet": "{instrument}\n{merchant}\n-{amount}\n{date}, {time}\nStatus: Approved\nCard Number •••• {last4}",
    "email": "Your {merchant} order confirmation\nOrder number: {order}\nOrder placed: {date}\n\n{items}\nDelivery: {shipping}\nOrder total: {total}\n\nThanks for shopping with {merchant}.",
    "casual": ["{amount} at {merchant} {day}", "{merchant} {amount} {day} lunch", "spent {amount} at {merchant} {day}"],
    "yesterday": "yesterday", "today_word": "today",
    "non_spending": [
        "Your one-time passcode is 482913. It expires in 10 minutes. Never share this code.",
        "Refund processed: £23.40 has been returned to your card ending 4412. It may take 3–5 days to appear.",
        "You received £25.00 from Sam Patel — 'thanks for dinner!'",
        "Your parcel from Argos will arrive today between 14:00 and 16:00.",
        "Reminder: dentist appointment tomorrow at 09:30.",
    ],
}

def _en_us_amount(rng, minor):
    p = grouped(minor, False)
    return rng.choice([f"${p}", f"${p}", p, f"USD {p}"])


LOCALES["en-US"] = {
    "language": "en", "currency": "USD", "zero_decimal": False, "tax_added": True,
    "amount": _en_us_amount,
    "date": lambda rng, d: rng.choice([d.strftime("%m/%d/%Y"), d.strftime("%b %d, %Y"), d.strftime("%m/%d/%y")]),
    "date_no_year": lambda d: d.strftime("%b %d"),
    "time": lambda h, m: f"{(h - 1) % 12 + 1}:{m:02d} {'PM' if h >= 12 else 'AM'}",
    "labels": {
        "total": ["TOTAL", "Total", "AMOUNT DUE"],
        "subtotal": ["SUBTOTAL", "Subtotal"], "tax": ["TAX", "Sales Tax 8.875%", "TAX 7.25%"],
        "tip": "Tip", "items": "ITEMS {n}",
        "tendered": "CASH TEND", "change": "CHANGE",
        "was": "REG {amt}", "reduction": "YOU SAVED",
        "card": "CARD", "cash_name": "Cash",
        "thanks": ["THANK YOU FOR SHOPPING", "Have a great day!", "Thanks for visiting"],
        "addresses": ["1440 Broadway, New York, NY 10018", "2201 Market St, San Francisco, CA 94114",
                      "500 N Michigan Ave, Chicago, IL 60611"],
        "receipt_no": "Trans #{n}",
    },
    "merchants": [
        M("Whole Foods Market", "groceries", [("Organic Avocados", 199, 399), ("Almond Milk", 299, 449), ("Granola", 499, 699), ("Sourdough Loaf", 499, 699)],
          printed=["WHOLE FOODS MARKET", "Whole Foods"], mixed=("coffee", [("Cold Brew", 399, 549), ("Blueberry Muffin", 299, 399)])),
        M("Trader Joe's", "groceries", [("Mandarin Chicken", 499, 549), ("Bananas", 25, 35), ("Dark Chocolate Bar", 199, 249)], printed=["TRADER JOE'S"]),
        M("Starbucks", "coffee", [("Grande Latte", 495, 595), ("Cold Brew", 445, 545), ("Banana Bread", 345, 395)], printed=["STARBUCKS", "Starbucks Coffee"]),
        M("Chipotle", "dining", [("Burrito Bowl", 1095, 1395), ("Chips & Guac", 445, 545), ("Fountain Drink", 295, 345)], printed=["CHIPOTLE MEXICAN GRILL"]),
        M("CVS Pharmacy", "health", [("Ibuprofen 100ct", 899, 1299), ("Band-Aids", 499, 699), ("Cough Drops", 299, 449)], printed=["CVS/pharmacy", "CVS PHARMACY"]),
        M("Chevron", "fuel", [("Regular Unleaded", 3500, 8000)], printed=["CHEVRON"]),
        M("Target", "shopping", [("Phone Charger", 1499, 2499), ("Notebook 3pk", 499, 899), ("Batteries AA 8pk", 899, 1299)], printed=["TARGET"],
          mixed=("home", [("Bath Towels", 1299, 2499), ("Shower Curtain", 1499, 2299)])),
        M("Petco", "pets", [("Dog Food 15lb", 2999, 4499), ("Chew Toy", 599, 1299)], printed=["PETCO"]),
        M("Old Navy", "clothing", [("Graphic Tee", 1299, 1999), ("Jeans", 2999, 4499)], online=True),
        M("AMC Theatres", "entertainment", [("Adult Ticket", 1399, 1899), ("Popcorn", 899, 1099)], printed=["AMC THEATRES"]),
        M("Uber", "transport", [("Trip", 1200, 4500)], online=True),
    ],
    "instruments": ["Chase", "Capital One", "Citi", "Amex", "Wells Fargo", "Bank of America", "Discover", "Apple Card"],
    "instrument_suffixes": ["", " Sapphire", " Freedom", " Debit", " Visa"],
    "notification": [
        "{instrument}: A ${plain} transaction at {merchant} was approved on {date_no_year} at {time}.",
        "{instrument} alert\nYou made a purchase of {amount} at {merchant}.",
    ],
    "wallet": "{instrument}\n{merchant}\n{amount}\n{date}, {time}\nStatus: Approved\nCard Number •••• {last4}",
    "email": "Thanks for your order!\nOrder #{order} · {date}\n\n{items}\nShipping: {shipping}\nOrder Total: {total}\n\n— {merchant}",
    "casual": ["{amount} at {merchant} {day}", "{merchant} {amount} {day}"],
    "yesterday": "yesterday", "today_word": "today",
    "non_spending": [
        "Your Uber is arriving in 3 minutes. Look for a grey Toyota Camry, plate 8KDL203.",
        "Venmo: Alex paid you $40.00 for 'concert tickets'.",
        "Your refund of $18.99 from Target has been issued to your original payment method.",
        "Flight UA 512 SFO → JFK, gate F12, boarding 7:45 AM.",
    ],
}

def _zh_amount(rng, minor):
    p = grouped(minor, False)
    q = plain(minor, False)
    return rng.choice([f"¥{p}", f"￥{q}", f"{q}元", q, f"RMB {q}"])


def _zh_date(rng, d):
    return rng.choice([f"{d.year}年{d.month}月{d.day}日", d.strftime("%Y-%m-%d"), d.strftime("%Y/%m/%d")])


LOCALES["zh"] = {
    "language": "zh", "currency": "CNY", "zero_decimal": False, "tax_added": False,
    "amount": _zh_amount,
    "date": _zh_date,
    "date_no_year": lambda d: f"{d.month}月{d.day}日",
    "time": lambda h, m: f"{h:02d}:{m:02d}",
    "labels": {
        "total": ["合计", "应付金额", "实付金额", "总计"],
        "subtotal": ["小计"], "items": "共{n}件",
        "tax_included": None,
        "tendered": "实收", "change": "找零",
        "was": "原价 {amt}", "reduction": "优惠",
        "card": "银行卡", "cash_name": "现金",
        "thanks": ["谢谢惠顾，欢迎再次光临", "请妥善保管小票", "谢谢光临"],
        "addresses": ["上海市浦东新区世纪大道100号", "北京市朝阳区建国路88号", "广州市天河区天河路208号", "深圳市南山区科技园路1号"],
        "receipt_no": "流水号: {n}",
    },
    "merchants": [
        M("盒马鲜生", "groceries", [("鲜牛奶", 1290, 1890), ("全麦面包", 990, 1590), ("鸡蛋10枚", 1490, 1990), ("香蕉", 590, 990), ("挂面", 590, 890)],
          printed=["盒马鲜生", "盒马"], aliases=["Freshippo", "Hema"]),
        M("永辉超市", "groceries", [("大米5kg", 3990, 5990), ("食用油", 5990, 7990), ("酸奶", 690, 1290), ("青菜", 290, 590)], printed=["永辉超市"]),
        M("瑞幸咖啡", "coffee", [("生椰拿铁", 990, 1990), ("美式咖啡", 990, 1590), ("厚乳拿铁", 1290, 1990)], printed=["瑞幸咖啡", "luckin coffee"], aliases=["Luckin Coffee", "luckin"]),
        M("星巴克", "coffee", [("大杯拿铁", 3300, 3800), ("美式咖啡", 2700, 3200), ("可颂", 1800, 2400)], printed=["星巴克", "STARBUCKS 星巴克"], aliases=["Starbucks"]),
        M("海底捞", "dining", [("番茄锅底", 5800, 7800), ("肥牛", 4800, 6800), ("虾滑", 3800, 5800), ("米饭", 300, 500)], printed=["海底捞火锅", "海底捞"]),
        M("肯德基", "dining", [("香辣鸡腿堡套餐", 3500, 4500), ("薯条", 1200, 1600), ("可乐", 800, 1100)], printed=["肯德基", "KFC"], aliases=["KFC"]),
        M("中国石化", "fuel", [("92号汽油", 25000, 45000)], printed=["中国石化加油站", "中国石化"], aliases=["Sinopec"],
          mixed=("groceries", [("矿泉水", 200, 400), ("方便面", 450, 800), ("口香糖", 800, 1200)])),
        M("屈臣氏", "health", [("维生素C", 5900, 8900), ("创可贴", 990, 1590), ("润喉糖", 1290, 1990)], printed=["屈臣氏"], aliases=["Watsons"]),
        M("优衣库", "clothing", [("圆领T恤", 7900, 9900), ("牛仔裤", 19900, 29900), ("袜子3双", 3900, 4900)], printed=["优衣库", "UNIQLO 优衣库"], aliases=["Uniqlo"]),
        M("宜家家居", "home", [("靠垫套", 2999, 4999), ("收纳盒", 1999, 3999), ("台灯", 9900, 14900)], printed=["宜家家居", "IKEA 宜家"], aliases=["IKEA", "宜家"],
          mixed=("dining", [("瑞典肉丸", 2900, 3900), ("肉桂卷", 600, 900)])),
        M("万达影城", "entertainment", [("电影票", 3900, 6900), ("爆米花套餐", 3500, 4500)], printed=["万达影城"]),
        M("毛孩子宠物店", "pets", [("猫粮2kg", 9900, 15900), ("猫砂", 3900, 5900)], printed=["毛孩子宠物店"]),
        M("滴滴出行", "transport", [("快车", 1500, 6500)], online=True, aliases=["DiDi", "滴滴"]),
        M("京东", "shopping", [("数据线", 2900, 5900), ("蓝牙耳机", 19900, 39900), ("充电宝", 9900, 15900)], online=True, aliases=["JD", "JD.com", "京东商城"]),
    ],
    "instruments": ["招商银行", "工商银行", "建设银行", "中国银行", "农业银行", "交通银行", "支付宝", "微信支付"],
    "invented": lambda rng: rng.choice(["华", "兴", "民", "泰", "瑞", "嘉", "恒", "安", "宁", "丰"])
    + rng.choice(["通", "信", "和", "盛", "达", "源", "融"])
    + rng.choice(["银行", "银行信用卡", "银行储蓄卡", "支付"]),
    "instrument_suffixes": ["", "信用卡", "储蓄卡"],
    "notification": [
        "【{instrument}】您尾号{last4}的卡于{date_no_year}{time}在{merchant}消费人民币{plain}元。",
        "【{instrument}】交易提醒：{merchant} 消费 {amount}，时间 {date_no_year} {time}。",
    ],
    "wallet": "{instrument}\n{merchant}\n-{plain}\n{date} {time}\n交易状态 支付成功\n卡号 •••• {last4}",
    "email": "【{merchant}】订单确认\n订单编号：{order}\n下单时间：{date}\n\n{items}\n运费：{shipping}\n实付款：{total}\n\n感谢您的购买。",
    "casual": ["{day}在{merchant}花了{plain}块", "{merchant} {plain}元 {day}", "{day}{merchant}午饭{plain}"],
    "yesterday": "昨天", "today_word": "今天",
    "non_spending": [
        "【顺丰速运】您的快递已到达小区驿站，取件码 3-2-1045，请尽快领取。",
        "您的验证码为 482913，10分钟内有效，请勿泄露给他人。",
        "【招商银行】您尾号4412的账户收到转账人民币500.00元，付款方：张伟。",
        "明天下午三点开会，记得带上季度报告。",
        "退款成功：¥59.00 已原路退回至您的支付账户。",
    ],
}

def _ja_amount(rng, minor):
    g = grouped(minor, True)
    return rng.choice([f"¥{g}", f"{g}円", f"￥{g}", f"¥{g}"])


def _ja_date(rng, d):
    wd = "月火水木金土日"[d.weekday()]
    return rng.choice([f"{d.year}年{d.month}月{d.day}日({wd})", d.strftime("%Y/%m/%d"), f"{d.year}年{d.month:02d}月{d.day:02d}日"])


LOCALES["ja"] = {
    "language": "ja", "currency": "JPY", "zero_decimal": True, "tax_added": False,
    "amount": _ja_amount,
    "date": _ja_date,
    "date_no_year": lambda d: f"{d.month}/{d.day}",
    "time": lambda h, m: f"{h:02d}:{m:02d}",
    "labels": {
        "total": ["合計", "お会計", "合計金額"],
        "subtotal": ["小計"], "items": "{n}点",
        "tax_included": "(内消費税等 {amt})",
        "tendered": "お預り", "change": "お釣り",
        "was": "通常価格 {amt}", "reduction": "値引",
        "card": "クレジット", "cash_name": "現金",
        "thanks": ["毎度ありがとうございます", "またのご来店をお待ちしております", "ありがとうございました"],
        "addresses": ["東京都渋谷区道玄坂2-1", "大阪府大阪市北区梅田3-1-1", "神奈川県横浜市西区みなとみらい2-2", "京都府京都市下京区烏丸通"],
        "receipt_no": "レシートNo.{n}",
    },
    "merchants": [
        M("セブン-イレブン", "groceries", [("おにぎり 鮭", 140, 190), ("牛乳 1000ml", 230, 280), ("食パン", 160, 220), ("サラダ", 300, 420)],
          printed=["セブン-イレブン", "セブンイレブン 渋谷店"], aliases=["7-Eleven", "Seven-Eleven"],
          mixed=("coffee", [("セブンカフェ ホットR", 120, 140), ("カフェラテ M", 200, 260), ("ドーナツ", 120, 180)])),
        M("イオン", "groceries", [("豚こま肉", 398, 598), ("キャベツ", 158, 258), ("卵 10個", 248, 298), ("納豆 3P", 88, 128)], printed=["イオン", "イオンスタイル"], aliases=["AEON"]),
        M("スターバックス", "coffee", [("スターバックスラテ Tall", 455, 495), ("ドリップコーヒー", 350, 400), ("スコーン", 300, 340)], printed=["スターバックス コーヒー", "STARBUCKS"], aliases=["Starbucks"]),
        M("ドトールコーヒー", "coffee", [("ブレンドコーヒー", 250, 300), ("ミラノサンド", 420, 480)], printed=["ドトールコーヒーショップ"], aliases=["Doutor"]),
        M("一蘭", "dining", [("天然とんこつラーメン", 980, 1180), ("替玉", 190, 240), ("半熟塩ゆで玉子", 140, 180)], printed=["一蘭"], aliases=["Ichiran"]),
        M("吉野家", "dining", [("牛丼 並盛", 468, 498), ("みそ汁", 72, 90), ("生卵", 72, 90)], printed=["吉野家"], aliases=["Yoshinoya"]),
        M("JR東日本", "transport", [("乗車券", 170, 3000), ("特急券", 1000, 3000)], printed=["JR東日本"], aliases=["JR East"]),
        M("ENEOS", "fuel", [("レギュラー", 4000, 9000)], printed=["ENEOS", "ENEOS 渋谷SS"]),
        M("マツモトキヨシ", "health", [("風邪薬", 980, 1580), ("目薬", 498, 798), ("ばんそうこう", 298, 498)], printed=["マツモトキヨシ"], aliases=["Matsukiyo"]),
        M("ユニクロ", "clothing", [("エアリズムT", 1500, 1990), ("チノパン", 3990, 4990)], printed=["ユニクロ", "UNIQLO"], aliases=["Uniqlo"]),
        M("ニトリ", "home", [("クッションカバー", 699, 1290), ("収納ボックス", 799, 1490), ("バスタオル", 999, 1490)], printed=["ニトリ"], aliases=["Nitori"]),
        M("TOHOシネマズ", "entertainment", [("一般", 2000, 2000), ("ポップコーン", 500, 700)], printed=["TOHOシネマズ"], aliases=["TOHO Cinemas"]),
        M("ペットの店わんにゃん", "pets", [("キャットフード", 1280, 2580), ("猫砂", 680, 980)], printed=["ペットの店わんにゃん"]),
        M("Amazon.co.jp", "shopping", [("USBケーブル", 980, 1980), ("モバイルバッテリー", 2980, 4980)], online=True, aliases=["Amazon", "アマゾン"]),
    ],
    "instruments": ["三井住友カード", "楽天カード", "PayPay", "イオンカード", "JCBカード", "みずほ銀行", "Suica"],
    "invented": lambda rng: rng.choice(["アオ", "ミナ", "ハル", "ソラ", "ユメ", "カエ", "ヒカ"])
    + rng.choice(["デ", "リ", "ノ", "タ", "ル"]) + rng.choice(["カード", "銀行", "ペイ", " Pay"]),
    "instrument_suffixes": ["", " ゴールド", " デビット"],
    "notification": [
        "【{instrument}】カードご利用のお知らせ\nご利用日時：{date_no_year} {time}\nご利用先：{merchant}\nご利用金額：{amount}",
        "{instrument}：{merchant}で{amount}のお支払いがありました（{date_no_year} {time}）",
    ],
    "wallet": "{instrument}\n{merchant}\n-{amount}\n{date} {time}\nステータス 完了\nカード番号 •••• {last4}",
    "email": "【{merchant}】ご注文の確認\n注文番号：{order}\n注文日：{date}\n\n{items}\n配送料：{shipping}\nご請求額：{total}\n\nご利用ありがとうございます。",
    "casual": ["{day}{merchant}で{amount}", "{merchant} ランチ {amount} {day}"],
    "yesterday": "昨日", "today_word": "今日",
    "non_spending": [
        "【ヤマト運輸】お荷物のお届け予定：本日 14時〜16時",
        "認証コードは 482913 です。10分間有効です。",
        "明日は10時から定例ミーティングです。資料の準備をお願いします。",
        "【楽天銀行】振込入金のお知らせ：30,000円 ヤマダ タロウ様より",
    ],
}

def _fr_amount(rng, minor):
    g = grouped(minor, False, thousands=" ", decimal=",")
    return rng.choice([f"{g} €", f"{g}€", f"{g} EUR", g, f"€{plain(minor, False)}"])


LOCALES["fr"] = {
    "language": "fr", "currency": "EUR", "zero_decimal": False, "tax_added": False,
    "amount": _fr_amount,
    "date": lambda rng, d: rng.choice([d.strftime("%d/%m/%Y"), d.strftime("%d/%m/%y"), f"{d.day} {['janv.', 'févr.', 'mars', 'avr.', 'mai', 'juin', 'juil.', 'août', 'sept.', 'oct.', 'nov.', 'déc.'][d.month - 1]} {d.year}"]),
    "date_no_year": lambda d: d.strftime("%d/%m"),
    "time": lambda h, m: f"{h:02d}h{m:02d}",
    "labels": {
        "total": ["TOTAL TTC", "TOTAL", "Montant à payer", "NET A PAYER"],
        "subtotal": ["SOUS-TOTAL"], "items": "{n} ARTICLES",
        "tax_included": "dont TVA {amt}",
        "tendered": "ESPECES", "change": "RENDU",
        "was": "PRIX INITIAL {amt}", "reduction": "REMISE",
        "card": "CB", "cash_name": "Espèces",
        "thanks": ["Merci de votre visite", "A bientôt", "Conservez votre ticket"],
        "addresses": ["12 rue de Rivoli, 75004 Paris", "Place Bellecour, 69002 Lyon", "35 cours Mirabeau, 13100 Aix-en-Provence"],
        "receipt_no": "Ticket n° {n}",
    },
    "merchants": [
        M("Carrefour", "groceries", [("Lait demi-écrémé 1L", 95, 125), ("Baguette", 95, 130), ("Oeufs x12", 320, 420), ("Pâtes 500g", 95, 160), ("Comté 250g", 450, 650)],
          printed=["CARREFOUR", "Carrefour City", "CARREFOUR MARKET"], mixed=("home", [("Ampoules LED x3", 800, 1200), ("Boîte de rangement", 600, 1200)])),
        M("Monoprix", "groceries", [("Yaourt nature x4", 180, 260), ("Pommes 1kg", 250, 350), ("Jambon blanc", 280, 380)], printed=["MONOPRIX"]),
        M("Paul", "coffee", [("Café allongé", 210, 280), ("Croissant", 120, 160), ("Pain au chocolat", 130, 170)], printed=["PAUL", "Boulangerie PAUL"]),
        M("Columbus Café", "coffee", [("Cappuccino", 380, 450), ("Muffin", 290, 350)], printed=["COLUMBUS CAFE & CO"]),
        M("Brasserie Le Central", "dining", [("Plat du jour", 1450, 1850), ("Café gourmand", 750, 950), ("Verre de vin", 550, 750)], printed=["BRASSERIE LE CENTRAL"]),
        M("McDonald's", "dining", [("Menu Big Mac", 1050, 1250), ("Frites", 350, 450)], printed=["McDONALD'S", "MCDONALDS"]),
        M("SNCF", "transport", [("Billet TGV", 2500, 9500), ("Billet TER", 800, 2500)], online=True, aliases=["SNCF Connect"]),
        M("TotalEnergies", "fuel", [("SP95-E10", 4000, 8000), ("Gazole", 4000, 8000)], printed=["TOTALENERGIES", "Station TotalEnergies"], aliases=["Total"],
          mixed=("groceries", [("Sandwich jambon", 450, 650), ("Bouteille d'eau", 150, 250)])),
        M("Pharmacie du Marché", "health", [("Doliprane 1000mg", 220, 350), ("Pansements", 350, 550)], printed=["PHARMACIE DU MARCHE"]),
        M("Decathlon", "shopping", [("Gourde", 500, 900), ("Ballon", 700, 1500), ("Chaussettes x3", 600, 900)], printed=["DECATHLON"]),
        M("Zara", "clothing", [("T-shirt", 1295, 1995), ("Jean", 2995, 3995)], printed=["ZARA"]),
        M("Leroy Merlin", "home", [("Perceuse", 3990, 7990), ("Peinture 2,5L", 2490, 3990)], printed=["LEROY MERLIN"]),
        M("UGC Ciné Cité", "entertainment", [("Place plein tarif", 1190, 1490), ("Pop-corn", 600, 800)], printed=["UGC CINE CITE"]),
        M("Maxi Zoo", "pets", [("Croquettes chat 2kg", 1490, 2490), ("Litière", 790, 1290)], printed=["MAXI ZOO"]),
    ],
    "instruments": ["Crédit Agricole", "BNP Paribas", "Société Générale", "La Banque Postale", "Boursorama", "LCL", "Lydia", "Revolut"],
    "invented": lambda rng: rng.choice(["Banque ", "Crédit ", ""]) + invented_latin(rng),
    "instrument_suffixes": ["", " Visa", " Carte Gold"],
    "notification": [
        "{instrument} : paiement par carte de {amount} chez {merchant} le {date_no_year} à {time}.",
        "{instrument}\nNouvelle opération : {merchant}\n-{amount} · {date_no_year} {time}",
    ],
    "wallet": "{instrument}\n{merchant}\n-{amount}\n{date}, {time}\nStatut : Effectué\nNuméro de carte •••• {last4}",
    "email": "Confirmation de votre commande {merchant}\nCommande n° {order} du {date}\n\n{items}\nLivraison : {shipping}\nTotal de la commande : {total}\n\nMerci pour votre achat.",
    "casual": ["{amount} chez {merchant} {day}", "{merchant} {amount} déjeuner {day}"],
    "yesterday": "hier", "today_word": "aujourd'hui",
    "non_spending": [
        "Votre code de vérification est 482913. Il expire dans 10 minutes.",
        "Remboursement effectué : 23,40 € ont été recrédités sur votre carte.",
        "Votre colis Colissimo sera livré demain entre 9h et 13h.",
        "Rappel : rendez-vous chez le dentiste jeudi à 10h30.",
    ],
}

def _es_amount(rng, minor):
    g = grouped(minor, False, thousands=".", decimal=",")
    return rng.choice([f"{g} €", f"{g}€", f"{g} EUR", g])


LOCALES["es"] = {
    "language": "es", "currency": "EUR", "zero_decimal": False, "tax_added": False,
    "amount": _es_amount,
    "date": lambda rng, d: rng.choice([d.strftime("%d/%m/%Y"), d.strftime("%d-%m-%Y"), d.strftime("%d/%m/%y")]),
    "date_no_year": lambda d: d.strftime("%d/%m"),
    "time": lambda h, m: f"{h:02d}:{m:02d}",
    "labels": {
        "total": ["TOTAL", "TOTAL A PAGAR", "IMPORTE TOTAL"],
        "subtotal": ["SUBTOTAL", "BASE IMPONIBLE"], "items": "{n} ARTÍCULOS",
        "tax_included": "IVA incluido {amt}",
        "tendered": "ENTREGADO", "change": "CAMBIO",
        "was": "ANTES {amt}", "reduction": "DESCUENTO",
        "card": "TARJETA", "cash_name": "Efectivo",
        "thanks": ["Gracias por su visita", "GRACIAS Y HASTA PRONTO", "Conserve su ticket"],
        "addresses": ["Calle Mayor 15, 28013 Madrid", "Passeig de Gràcia 45, 08007 Barcelona", "Avenida de la Constitución 8, 41004 Sevilla"],
        "receipt_no": "Factura simplificada {n}",
    },
    "merchants": [
        M("Mercadona", "groceries", [("Leche entera 1L", 89, 115), ("Pan de molde", 125, 175), ("Huevos L x12", 240, 320), ("Aceite de oliva 1L", 650, 950), ("Tomates", 150, 290)], printed=["MERCADONA, S.A.", "MERCADONA"]),
        M("Día", "groceries", [("Arroz 1kg", 99, 149), ("Yogur natural x4", 99, 159), ("Galletas", 89, 149)], printed=["DIA %", "DIA"], aliases=["Dia"]),
        M("Café de la Plaza", "coffee", [("Café con leche", 140, 190), ("Cortado", 120, 160), ("Tostada con tomate", 220, 320)], printed=["CAFE DE LA PLAZA"]),
        M("Taberna El Puerto", "dining", [("Menú del día", 1200, 1500), ("Caña", 200, 300), ("Patatas bravas", 450, 650)], printed=["TABERNA EL PUERTO"]),
        M("Telepizza", "dining", [("Pizza mediana", 1095, 1495), ("Refresco", 250, 350)], printed=["TELEPIZZA"]),
        M("Renfe", "transport", [("Billete AVE", 2500, 8500), ("Cercanías", 170, 400)], online=True),
        M("Repsol", "fuel", [("Gasolina 95", 4000, 8000), ("Diésel", 4000, 8000)], printed=["REPSOL", "E.S. REPSOL"],
          mixed=("groceries", [("Agua 1,5L", 90, 150), ("Bocadillo", 350, 550), ("Chicles", 120, 200)])),
        M("Farmacia Central", "health", [("Ibuprofeno 600mg", 250, 400), ("Tiritas", 300, 500)], printed=["FARMACIA CENTRAL"]),
        M("El Corte Inglés", "shopping", [("Auriculares", 2990, 5990), ("Libro", 1500, 2500), ("Perfume", 4500, 8500)], printed=["EL CORTE INGLES"],
          mixed=("clothing", [("Camisa", 2995, 4995), ("Pantalón", 3495, 5995)])),
        M("Zara", "clothing", [("Camiseta", 1295, 1995), ("Vaqueros", 2995, 3995)], printed=["ZARA"]),
        M("Cinesa", "entertainment", [("Entrada", 850, 1100), ("Palomitas", 550, 750)], printed=["CINESA"]),
        M("Tiendanimal", "pets", [("Pienso perro 3kg", 1490, 2490), ("Juguete", 390, 790)], online=True),
    ],
    "instruments": ["BBVA", "CaixaBank", "Santander", "Bankinter", "Sabadell", "ING", "Bizum", "Revolut"],
    "invented": lambda rng: rng.choice(["Banco ", "Caja ", ""]) + invented_latin(rng),
    "instrument_suffixes": ["", " Visa", " Débito", " Crédito"],
    "notification": [
        "{instrument}: Compra con tarjeta ****{last4} por {amount} en {merchant} el {date_no_year} a las {time}.",
        "{instrument}\nHas pagado {amount} en {merchant}\n{date_no_year} · {time}",
    ],
    "wallet": "{instrument}\n{merchant}\n-{amount}\n{date}, {time}\nEstado: Completada\nNúmero de tarjeta •••• {last4}",
    "email": "Confirmación de tu pedido en {merchant}\nPedido n.º {order} · {date}\n\n{items}\nEnvío: {shipping}\nTotal del pedido: {total}\n\n¡Gracias por tu compra!",
    "casual": ["{amount} en {merchant} {day}", "{merchant} {amount} comida {day}"],
    "yesterday": "ayer", "today_word": "hoy",
    "non_spending": [
        "Tu código de verificación es 482913. Caduca en 10 minutos.",
        "Bizum recibido: Laura te ha enviado 15,00 € — 'cena'.",
        "Tu pedido de Correos llegará mañana entre las 9:00 y las 14:00.",
        "Reunión de vecinos el jueves a las 19:00 en el portal.",
    ],
}

def _es_mx_amount(rng, minor):
    p = grouped(minor, False)
    return rng.choice([f"${p}", f"${p} MXN", f"MXN {p}", f"${p}"])


LOCALES["es-MX"] = dict(LOCALES["es"])
LOCALES["es-MX"].update({
    "currency": "MXN", "amount": _es_mx_amount,
    "labels": dict(LOCALES["es"]["labels"], **{
        "tax_included": "IVA 16% incluido {amt}",
        "addresses": ["Av. Insurgentes Sur 1235, Ciudad de México, CDMX", "Av. Juárez 50, Guadalajara, Jal.", "Calle 60 No. 491, Mérida, Yuc."],
        "receipt_no": "Ticket {n}",
    }),
    "merchants": [
        M("OXXO", "groceries", [("Coca-Cola 600ml", 1800, 2200), ("Sabritas", 1700, 2300), ("Pan Bimbo", 4500, 5500)], printed=["OXXO", "OXXO Insurgentes"]),
        M("Sanborns", "dining", [("Enchiladas suizas", 18900, 22900), ("Café americano", 4500, 5500)], printed=["SANBORNS"]),
        M("Pemex", "fuel", [("Magna", 50000, 90000)], printed=["PEMEX", "Gasolinera PEMEX"]),
        M("Farmacias Guadalajara", "health", [("Paracetamol", 3500, 6500), ("Curitas", 2500, 4500)], printed=["FARMACIAS GUADALAJARA"]),
        M("Liverpool", "clothing", [("Playera", 29900, 49900), ("Jeans", 59900, 89900)], printed=["LIVERPOOL"]),
        M("Cinépolis", "entertainment", [("Boleto", 7500, 12000), ("Palomitas", 8500, 11000)], printed=["CINEPOLIS"]),
    ],
    "instruments": ["BBVA México", "Banorte", "Santander", "Citibanamex", "Nu", "Mercado Pago"],
})

def _de_amount(rng, minor):
    g = grouped(minor, False, thousands=".", decimal=",")
    return rng.choice([f"{g} €", f"{g} EUR", g, f"EUR {g}"])


LOCALES["de"] = {
    "language": "de", "currency": "EUR", "zero_decimal": False, "tax_added": False,
    "amount": _de_amount,
    "date": lambda rng, d: rng.choice([d.strftime("%d.%m.%Y"), d.strftime("%d.%m.%y")]),
    "date_no_year": lambda d: d.strftime("%d.%m."),
    "time": lambda h, m: f"{h:02d}:{m:02d}",
    "labels": {
        "total": ["SUMME", "Summe EUR", "ZU ZAHLEN", "Gesamtbetrag"],
        "subtotal": ["Zwischensumme"], "items": "{n} Artikel",
        "tax_included": "enth. MwSt 19% {amt}",
        "tendered": "Gegeben BAR", "change": "Rückgeld",
        "was": "statt {amt}", "reduction": "Rabatt",
        "card": "girocard", "cash_name": "Bar",
        "thanks": ["Vielen Dank für Ihren Einkauf", "Bis zum nächsten Mal", "Danke und auf Wiedersehen"],
        "addresses": ["Kaufingerstraße 12, 80331 München", "Schildergasse 70, 50667 Köln", "Friedrichstraße 100, 10117 Berlin"],
        "receipt_no": "Bon-Nr. {n}",
    },
    "merchants": [
        M("REWE", "groceries", [("Vollmilch 1L", 109, 139), ("Roggenbrot", 189, 289), ("Eier 10 St.", 249, 329), ("Bananen", 129, 199), ("Gouda", 199, 299)], printed=["REWE", "REWE Markt GmbH"]),
        M("EDEKA", "groceries", [("Butter 250g", 199, 249), ("Äpfel 1kg", 229, 299), ("Joghurt", 49, 89)], printed=["EDEKA", "EDEKA Center"]),
        M("Lidl", "groceries", [("Haferflocken", 59, 89), ("Nudeln 500g", 79, 119), ("Tomaten", 149, 229)], printed=["LIDL", "Lidl Dienstleistung"]),
        M("Bäckerei Schmidt", "coffee", [("Cappuccino", 290, 360), ("Laugenbrezel", 85, 120), ("Franzbrötchen", 180, 240)], printed=["BÄCKEREI SCHMIDT", "Baeckerei Schmidt"]),
        M("Tchibo", "coffee", [("Caffè Crema", 220, 280), ("Latte Macchiato", 320, 380)], printed=["TCHIBO"],
          mixed=("shopping", [("Thermobecher", 999, 1499), ("Regenschirm", 1299, 1999)])),
        M("Vapiano", "dining", [("Pasta Carbonara", 1250, 1490), ("Pizza Margherita", 990, 1190), ("Apfelschorle", 390, 490)], printed=["VAPIANO"]),
        M("Deutsche Bahn", "transport", [("ICE Fahrkarte", 1990, 8990), ("Sitzplatzreservierung", 490, 590)], online=True, aliases=["DB", "DB Fernverkehr"]),
        M("Aral", "fuel", [("Super E10", 4000, 8500), ("Diesel", 4000, 8500)], printed=["ARAL", "Aral Tankstelle"],
          mixed=("groceries", [("Belegtes Brötchen", 349, 499), ("Wasser 0,5L", 149, 229)])),
        M("dm-drogerie markt", "health", [("Ibuprofen 400", 299, 499), ("Pflaster", 195, 345), ("Vitamin C", 245, 395)], printed=["dm-drogerie markt", "dm"], aliases=["dm"]),
        M("H&M", "clothing", [("T-Shirt", 799, 1499), ("Jeans", 2499, 3999)], printed=["H&M"]),
        M("IKEA", "home", [("Kissenbezug", 499, 999), ("Aufbewahrungsbox", 599, 1299), ("Lampe", 1999, 3999)], printed=["IKEA Deutschland", "IKEA"],
          mixed=("dining", [("Köttbullar", 599, 799), ("Zimtschnecke", 100, 150)])),
        M("CineStar", "entertainment", [("Kinokarte", 1050, 1450), ("Popcorn", 550, 750)], printed=["CINESTAR"]),
        M("Fressnapf", "pets", [("Katzenfutter 2kg", 1299, 1999), ("Katzenstreu", 699, 1099)], printed=["FRESSNAPF"]),
    ],
    "instruments": ["Sparkasse", "Volksbank", "Commerzbank", "Deutsche Bank", "DKB", "N26", "Comdirect", "PayPal"],
    "invented": lambda rng: invented_latin(rng) + rng.choice(["bank", " Bank", "kasse", ""]),
    "instrument_suffixes": ["", " Visa", " Mastercard", " Girokonto"],
    "notification": [
        "{instrument}: Kartenzahlung über {amount} bei {merchant} am {date_no_year} um {time} Uhr.",
        "{instrument}\nZahlung an {merchant}\n-{amount} · {date_no_year} {time}",
    ],
    "wallet": "{instrument}\n{merchant}\n-{amount}\n{date}, {time}\nStatus: Abgeschlossen\nKartennummer •••• {last4}",
    "email": "Bestellbestätigung – {merchant}\nBestellnummer: {order}\nBestelldatum: {date}\n\n{items}\nVersand: {shipping}\nGesamtsumme: {total}\n\nVielen Dank für Ihre Bestellung.",
    "casual": ["{amount} bei {merchant} {day}", "{merchant} {amount} Mittagessen {day}"],
    "yesterday": "gestern", "today_word": "heute",
    "non_spending": [
        "Ihr Bestätigungscode lautet 482913. Er ist 10 Minuten gültig.",
        "Gutschrift: 120,00 € von Max Mustermann – 'Miete Anteil'.",
        "Ihre DHL-Sendung wird heute zwischen 14:00 und 16:00 Uhr zugestellt.",
        "Erinnerung: Elternabend am Donnerstag um 19 Uhr.",
    ],
}


def instrument_name(rng, loc):
    """A card, bank or wallet name for the locale: real, invented, or with a
    kind suffix — so the name itself is never what identifies it."""
    roll = rng.random()
    if roll < 0.45 or "invented" not in loc:
        if roll >= 0.45:
            stem = invented_latin(rng) + rng.choice(["", " Bank", "Pay", " Card"])
        else:
            stem = rng.choice(loc["instruments"])
    else:
        stem = loc["invented"](rng)
    if rng.random() < 0.35:
        stem += rng.choice(loc.get("instrument_suffixes", [""]))
    return stem.strip()
