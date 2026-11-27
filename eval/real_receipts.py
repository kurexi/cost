#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import date, timedelta
from difflib import SequenceMatcher

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from locales import CATEGORY_TITLES
from make_suite import fingerprint

from spending_agent.money import decimal_string, minor_units
from spending_agent.session import CONTENT_CHARACTER_LIMIT

SUITE_VERSION = 1
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
THREE_DECIMAL_CURRENCIES = frozenset({"BHD", "IQD", "JOD", "KWD", "LYD", "OMR", "TND"})
FIELD_KEYS = {
    "merchant": ("company", "company_name", "merchant", "merchant_name", "store", "store_name", "shop", "shop_name",
                 "vendor", "vendor_name", "seller", "business", "business_name", "supplier", "nom_magasin", "magasin",
                 "enseigne", "societe"),
    "total": ("total", "total_amount", "grand_total", "totalamount", "total_price", "amount", "total_ttc", "net_total",
              "montant", "montant_total", "total_a_payer", "amount_due", "balance_due", "totalprice"),
    "date": ("date", "purchase_date", "transaction_date", "invoice_date", "receipt_date", "date_achat"),
    "time": ("time", "purchase_time", "transaction_time", "heure"),
}
CURRENCY_MARKERS = (
    ("MYR", (r"\bRM\s?\d", r"\(RM\)", r"\bRM\b", r"\bMYR\b", r"\bRINGGIT\b")), ("SGD", (r"\bS\$", r"\bSGD\b")),
    ("TND", (r"\bTND\b", r"\d\s?DT\b", r"\bDT\s?\d", r"\bDINARS?\b")), ("IDR", (r"\bRp\.?\s?\d", r"\bIDR\b")),
    ("INR", (r"₹", r"\bINR\b", r"\bRs\.?\s?\d")), ("VND", (r"₫", r"\bVND\b")), ("GBP", (r"£", r"\bGBP\b")),
    ("EUR", (r"\bEUR\b", r"\bEUROS?\b")), ("CNY", (r"\bRMB\b", r"\bCNY\b", r"元")),
    ("JPY", (r"円", r"\bJPY\b", r"¥")), ("CAD", (r"\bCAD\b", r"\bC\$")), ("AUD", (r"\bAUD\b", r"\bA\$")),
    ("USD", (r"\bUSD\b", r"\bUS\$")),
)
CATEGORY_WORDS = {
    "fuel": ("petrol", "petronas", "shell", "caltex", "petron", "bhpetrol", "esso", "texaco", "chevron", "fuel",
             "station service", "stesen minyak", "petrokiosk", "carburant", "essence", "gasoil", "agil", "total energies", "totalenergies", "oilibya"),
    "health": ("pharmacy", "pharmacie", "farmasi", "clinic", "klinik", "clinique", "guardian", "watsons", "medical",
               "dental", "hospital", "hopital", "caring", "parapharmacie", "optique", "optical", "cvs", "walgreens"),
    "coffee": ("coffee", "kopi", "cafe", "café", "starbucks", "tealive", "boba", "bubble tea", "salon de the",
               "coffeeshop", "espresso"),
    "dining": ("restaurant", "restoran", "resto", "food", "kitchen", "bistro", "grill", "noodle", "noodles", "mee",
               "nasi", "kedai makan", "burger", "pizza", "pizzeria", "sushi", "bbq", "steamboat", "catering", "eatery",
               "dining", "canteen", "kantin", "seafood", "dim sum", "mcdonald", "mcdonalds", "kfc", "domino", "subway",
               "kopitiam", "chicken rice", "brasserie", "snack", "fast food", "patisserie", "boulangerie", "bakery",
               "bakeri", "cuisine", "tacos", "shawarma", "creperie", "rotisserie", "f b", "rice", "restaurants",
               "confectionery", "cake", "cakes", "pastry", "steakhouse", "burgers", "snacks"),
    "groceries": ("supermarket", "supermarche", "supermarché", "hypermarket", "hypermarche", "mart", "minimart",
                  "mini market", "grocer", "grocery", "groceries", "pasar", "pasaraya", "provision", "sundry",
                  "kedai runcit", "runcit", "cash carry", "borong",
                  "speedmart", "familymart", "7 eleven", "7eleven", "aeon", "giant", "tesco", "lotus", "mydin",
                  "econsave", "carrefour", "monoprix", "geant", "magasin general", "aziza", "epicerie", "épicerie",
                  "walmart", "kroger", "aldi", "lidl", "costco", "fresh", "market", "marché", "marche", "fruits"),
    "home": ("hardware", "perkakasan", "furniture", "furnishing", "ikea", "mr diy", "mrdiy", "home", "electrical",
             "building", "papan", "besi", "simen", "cement", "welding", "steel", "light", "lights", "garden",
             "household", "machinery",
             "quincaillerie", "bricolage", "plumbing", "paint", "lighting", "maison", "kitchenware"),
    "transport": ("parking", "taxi", "grab", "toll", "transit", "railway", "rail", "bus", "ticket", "tiket", "uber",
                  "metro", "airline", "airport", "peage", "péage", "louage"),
    "entertainment": ("cinema", "cinéma", "gsc", "tgv cinemas", "karaoke", "bowling", "arcade", "theatre", "theater",
                      "museum", "musee", "concert"),
    "clothing": ("fashion", "apparel", "boutique", "uniqlo", "shoes", "footwear", "tailor", "textile", "vetements",
                 "vêtements", "chaussures", "zara"),
    "pets": ("pet shop", "petshop", "veterinary", "vet clinic", "animalerie"),
    "kids": ("toys", "toy", "baby", "jouets"),
    "shopping": ("stationery", "stationary", "book", "books", "bookstore", "bookcentre", "buku", "librairie", "papeterie", "popular",
                 "gift", "printing", "electronics", "electronic", "computer", "mobile", "phone", "department store",
                 "mall", "boutique cadeaux"),
}
CORE_KINDS = ("groceries", "dining", "coffee")
LEGAL_SUFFIXES = re.compile(
    r"\b(sdn\.?\s*bhd\.?|bhd\.?|s/b|\(m\)|pte\.?\s*ltd\.?|ltd\.?|limited|inc\.?|llc|l\.l\.c\.|corp\.?|corporation|"
    r"gmbh|plc|s\.a\.r\.l\.?|sarl|s\.a\.?|suarl|co\.\s*,?\s*ltd\.?)(?=\W|$)", re.I)
TRADING_NAMES = {
    "unihakka": ("Bar Wang Rice",), "gerbangalaf": ("McDonald's",), "goldenarches": ("McDonald's",),
    "ikanohandel": ("IKEA",), "urbanidea": ("Subway",), "cheflee": ("Old Asia",), "elitetrax": ("Harvey Norman",),
    "tkdivisionkitchen": ("Kiku Japanese Restaurant",), "monsieur": ("Home's Harmony",),
    "mizumentai": ("Sushi Mentai",), "kmffoodicious": ("Sushi Mentai",), "mentaiinitial": ("Sushi Mentai",),
    "subanghealthcare": ("AA Pharmacy",), "ogngroup": ("Jimart",), "oldtownkopit": ("OldTown White Coffee",),
    "exquisitegarden": ("OldTown White Coffee",), "grandcompanions": ("Tony Roma's",),
    "anekaintertrade": ("Wellmart",), "trishaas": ("Mydin",), "alibabainternational": ("Salon Du Chocolat",),
    "sayangyoustation": ("Juwita Weddings",), "dewinahost": ("Burger King",), "birupetroleum": ("Caltex",),
    "platinumracking": ("Plat Mart",), "freedomoptimum": ("The GeoVenture",), "maxincome": ("FamilyMart",),
    "haxincone": ("FamilyMart",), "securityoatrading": ("Biz Link",), "pacificasianfood": ("Win Soon Cafe",),
    "kecharavegetarian": ("Kechara Oasis",), "globalfoodempire": ("Lemon Tree Restaurant",),
    "pelitasamudra": ("Nasi Kandar Pelita",), "koreandine": ("Nam San Seoul Korean BBQ",),
    "castleblue": ("Brewery Tap",), "tasteoftheworld": ("Geoventure",), "timelesskitchenette": ("Morganfield's",),
    "tasleisure": ("Chili's",), "zawiajaya": ("Al Mendis",), "goldenscoop": ("Baskin Robbins",),
    "littlecravings": ("Nyonya Colors",), "dommalfood": ("Domino's Pizza",), "baifu": ("Daiso",),
    "trendymax": ("Sarang Hae Yo",), "ghdistributor": ("Ghee Hiang",), "rcventure": ("City Milk",),
    "fourquarters": ("A Pie Thing",), "edenimpression": ("Sushi Q",), "ftofnoodle": ("Face To Face Noodle House",),
    "tplusfb": ("Amphawa Thai Tea",), "doveholdings": ("Sizzling Stonegrill",), "rapidrail": ("rapidKL", "MRT"),
    "esjayfuel": ("BHPetrol",), "pagohrest": ("myNEWS",), "beconenterprise": ("Becon Stationer",),
    "anzeladvertising": ("EXPrint",),
}
LABEL_FIXES = {("URBAN IDEA SDN BHD", "27/02/2018", "RM22.65"): {"total": "RM24.00"}}
CREDIT_NOTE = re.compile(r"\bcredit\s*note\b", re.I)
REGISTRATION = re.compile(r"[\(\[]\s*(co\.?\s*(reg\.?|no\.?)?\s*:?)?\s*[A-Z]{0,3}\d{3,}[\w\-]*\s*[\)\]]", re.I)
BOX_LINE = re.compile(r"^\s*-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?){7}\s*,(.*)$")
TIME = re.compile(r"(?<![\d:])([01]?\d|2[0-3]):([0-5]\d)(?::[0-5]\d)?(?:\s*([AaPp])\.?\s*[Mm]\.?)?(?![\d:])")
NUMBER = re.compile(r"(?<![\d.,])\d{1,3}(?:[.,\s]\d{3})*(?:[.,]\d{1,3})?(?![\d])|(?<![\d.,])\d+(?:[.,]\d{1,3})?(?![\d])")
MONTHS = {name: index for index, names in enumerate((
    ("jan", "january", "janvier", "januari"), ("feb", "february", "fev", "fevrier", "février", "februari"),
    ("mar", "march", "mars", "mac"), ("apr", "april", "avr", "avril"), ("may", "mai", "mei"),
    ("jun", "june", "juin"), ("jul", "july", "juil", "juillet", "julai"), ("aug", "august", "aout", "août", "ogos"),
    ("sep", "sept", "september", "septembre"), ("oct", "october", "octobre", "okt", "oktober"),
    ("nov", "november", "novembre"), ("dec", "december", "decembre", "décembre", "dis", "disember")), 1)
    for name in names}
FRENCH_WORDS = ("merci", "total ttc", "tva", "espèces", "especes", "monnaie", "à payer", "a payer", "prix", "qté",
                "quantité", "rendu", "ticket", "facture", "magasin", "caisse")


def norm_key(key):
    return re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")


EXCLUDED_KEY_WORDS = ("sub", "tax", "tva", "vat", "discount", "remise", "change", "rendu", "cash", "rounding", "tip",
                     "service", "item", "unit", "qty", "quantity", "address", "count", "paid", "tender")


def field_of(key):
    key = norm_key(key)
    tokens = key.split("_")
    if any(t.startswith(w) for t in tokens for w in EXCLUDED_KEY_WORDS) or \
            any(w in key for w in ("subtotal", "tax", "discount", "change", "rounding")):
        return None
    for field, keys in FIELD_KEYS.items():
        if key in keys:
            return field
    if key.endswith("_total") or key.startswith("total"):
        return "total"
    if key.endswith("date"):
        return "date"
    return None


def labels_from_mapping(mapping):
    labels = {}
    for key, value in mapping.items():
        field = field_of(key)
        if field and field not in labels and isinstance(value, (str, int, float)) and str(value).strip():
            labels[field] = str(value).strip()
    return labels


def read_text(path):
    with open(path, "rb") as f:
        raw = f.read()
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def stem_of(path):
    return os.path.splitext(os.path.basename(path))[0].lower()


def text_from_lines(lines):
    out = [line.rstrip() for line in lines]
    while out and not out[-1].strip():
        out.pop()
    while out and not out[0].strip():
        out.pop(0)
    return "\n".join(out)


def parse_text_file(path):
    body = read_text(path).strip()
    if not body:
        return None
    if body.startswith("{"):
        try:
            data = json.loads(body)
        except ValueError:
            data = None
        if isinstance(data, dict):
            return parse_json_object(data)
    lines = body.splitlines()
    boxed = [BOX_LINE.match(line) for line in lines if line.strip()]
    if boxed and sum(1 for m in boxed if m) >= 0.8 * len(boxed):
        return {"text": text_from_lines(m.group(1) for m in boxed if m)}
    return {"text": text_from_lines(lines)}


def cord_text(lines):
    rows = defaultdict(list)
    for line in lines:
        for word in line.get("words", []):
            quad = word.get("quad") or {}
            rows[(line.get("group_id", 0), line.get("line_id", 0))].append((quad.get("x1", 0), word.get("text", "")))
    return "\n".join(" ".join(t for _, t in sorted(words)) for _, words in sorted(rows.items()))


def positioned_words(data):
    out = []
    if isinstance(data, dict):
        where = data.get("Coordinates") or data.get("coordinates")
        text = data.get("Text", data.get("text"))
        if isinstance(where, dict) and isinstance(text, (str, int, float)) and str(text).strip():
            try:
                box = tuple(float(where[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
            except (KeyError, TypeError, ValueError):
                box = None
            if box:
                return [(*box, str(text).strip())]
        for value in data.values():
            out.extend(positioned_words(value))
    elif isinstance(data, list):
        for value in data:
            out.extend(positioned_words(value))
    return out


def lines_from_words(words):
    rows = []
    for x0, y0, x1, y1, text in sorted(words, key=lambda w: (w[1] + w[3]) / 2):
        middle = (y0 + y1) / 2
        if rows and abs(middle - rows[-1]["middle"]) <= max(4.0, 0.5 * rows[-1]["height"]):
            rows[-1]["words"].append((x0, text))
        else:
            rows.append({"middle": middle, "height": max(1.0, y1 - y0), "words": [(x0, text)]})
    return "\n".join(" ".join(t for _, t in sorted(row["words"])) for row in rows)


def parse_important_words(data):
    labels, ignored = {}, []
    for group in data["important_words"]:
        if not isinstance(group, dict):
            continue
        name, words = str(group.get("label") or ""), positioned_words(group.get("words", []))
        field = field_of(name)
        if not field:
            ignored.append(name)
            continue
        if field in labels or not words:
            continue
        value = " ".join(lines_from_words(words).splitlines())
        if field == "total":
            amounts = NUMBER.findall(value)
            value = amounts[-1] if amounts else ""
        if value.strip():
            labels[field] = value.strip()
    found = {"labels": labels} if labels else {}
    if ignored:
        found["ignored_labels"] = ignored
    return found or None


def parse_json_object(data):
    if isinstance(data, list):
        words = positioned_words(data)
        return {"text": lines_from_words(words)} if words else None
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("important_words"), list):
        return parse_important_words(data)
    if "images" in data and "annotations" in data:
        return {"coco": data}
    if "ground_truth" in data and isinstance(data["ground_truth"], str):
        try:
            data = {**data, **json.loads(data["ground_truth"])}
        except ValueError:
            pass
    found = {}
    parse = data.get("gt_parse")
    if isinstance(parse, dict):
        flat = {}
        for section, value in parse.items():
            if isinstance(value, dict):
                flat.update({f"{section}_{k}" if section != "total" else k: v for k, v in value.items()})
            else:
                flat[section] = value
        found["labels"] = labels_from_mapping(flat)
    if isinstance(data.get("valid_line"), list):
        found["text"] = cord_text(data["valid_line"])
    for key in ("text", "ocr", "ocr_text", "content", "raw_text", "full_text", "transcription"):
        if isinstance(data.get(key), str) and data[key].strip() and "text" not in found:
            found["text"] = data[key].strip()
    if isinstance(data.get("lines"), list) and "text" not in found:
        lines = [line if isinstance(line, str) else (line.get("text") or line.get("transcription") or "")
                 for line in data["lines"]]
        found["text"] = text_from_lines(lines)
    if "text" not in found:
        words = positioned_words(data)
        if words:
            found["text"] = lines_from_words(words)
    labels = labels_from_mapping(data)
    for key in ("labels", "fields", "entities", "annotation", "annotations", "key_info", "kie"):
        if isinstance(data.get(key), dict):
            labels = {**labels_from_mapping(data[key]), **labels}
    if labels:
        found["labels"] = {**labels, **found.get("labels", {})}
    if isinstance(data.get("entities"), list) or isinstance(data.get("annotations"), list):
        boxes = []
        for item in data.get("entities") or data.get("annotations"):
            if isinstance(item, dict):
                label = item.get("label") or item.get("class") or item.get("name") or item.get("category")
                boxes.append({"label": str(label or ""), "text": str(item.get("text") or item.get("value") or ""),
                              "box": box_of(item)})
        if boxes:
            found["boxes"] = boxes
    return found or None


def box_of(item):
    if all(k in item for k in ("xmin", "ymin", "xmax", "ymax")):
        return tuple(float(item[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
    for key in ("bbox", "box"):
        value = item.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 4:
            x, y, w, h = (float(v) for v in value)
            return (x, y, x + w, y + h) if key == "bbox" else (x, y, w, h)
        if isinstance(value, (list, tuple)) and len(value) == 8:
            xs, ys = [float(v) for v in value[0::2]], [float(v) for v in value[1::2]]
            return (min(xs), min(ys), max(xs), max(ys))
    return None


def parse_voc(path):
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return None
    if root.tag != "annotation":
        return None
    boxes = []
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip()
        text = obj.findtext("text") or obj.findtext("transcription") or obj.findtext("value") or ""
        for attribute in obj.iter("attribute"):
            if (attribute.findtext("name") or "").strip().lower() in ("text", "value", "transcription"):
                text = text or attribute.findtext("value") or ""
        bnd = obj.find("bndbox")
        box = None
        if bnd is not None:
            try:
                box = tuple(float(bnd.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            except (TypeError, ValueError):
                box = None
        boxes.append({"label": name, "text": text.strip(), "box": box})
    return {"boxes": boxes, "image_name": (root.findtext("filename") or "").strip()}


def csv_columns(header):
    lowered = [norm_key(h) for h in header]

    def pick(*names):
        return next((header[i] for i, h in enumerate(lowered) if h in names), None)

    return {"file": pick("filename", "file_name", "file", "image", "image_name", "image_id", "img", "path", "image_path"),
            "label": pick("class", "label", "name", "category", "class_name", "field", "key"),
            "text": pick("text", "transcription", "value", "content", "ocr", "word", "words"),
            "box": [pick(n) for n in ("xmin", "ymin", "xmax", "ymax")]}


def parse_csv(path):
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    header = list(rows[0].keys())
    columns = csv_columns(header)
    out = {}
    if columns["file"] and columns["label"]:
        for row in rows:
            box = None
            if all(columns["box"]):
                try:
                    box = tuple(float(row[c]) for c in columns["box"])
                except (TypeError, ValueError):
                    box = None
            entry = out.setdefault(stem_of(row[columns["file"]]), {"boxes": [], "image_name": row[columns["file"]]})
            entry["boxes"].append({"label": row[columns["label"]] or "", "box": box,
                                   "text": (row.get(columns["text"]) or "").strip() if columns["text"] else ""})
        return out
    if columns["file"]:
        for row in rows:
            labels = labels_from_mapping(row)
            text = (row.get(columns["text"]) or "").strip() if columns["text"] else ""
            if labels or text:
                out[stem_of(row[columns["file"]])] = {"labels": labels, **({"text": text} if text else {})}
    return out


class TesseractOCR:
    def __init__(self, languages="eng", workers=None):
        import pytesseract
        from PIL import Image, ImageOps
        self.tesseract, self.image, self.ops, self.languages = pytesseract, Image, ImageOps, languages
        self.workers = workers or os.cpu_count() or 1
        os.environ.setdefault("OMP_THREAD_LIMIT", "1")

    def open(self, path):
        return self.ops.exif_transpose(self.image.open(path)).convert("L")

    def read(self, path, crops, full):
        try:
            image = self.open(path)
            fields = {}
            for field, boxes in crops.items():
                fields[field] = next((v for v in (self.crop(image, box) for box in boxes) if v), "")
            return fields, self.full(image) if full else "", None
        except Exception as error:
            return {}, "", f"{type(error).__name__}: {error}"

    def read_all(self, jobs):
        from concurrent.futures import ThreadPoolExecutor
        started, out = time.time(), []
        with ThreadPoolExecutor(self.workers) as pool:
            for done, result in enumerate(pool.map(lambda job: self.read(*job), jobs), 1):
                out.append(result)
                if done % 25 == 0 or done == len(jobs):
                    elapsed = time.time() - started
                    print(f"ocr {done}/{len(jobs)} receipts, {elapsed:.0f} s, about "
                          f"{elapsed / done * (len(jobs) - done):.0f} s to go ({self.workers} workers)",
                          file=sys.stderr, flush=True)
        return out

    def full(self, image):
        return self.tesseract.image_to_string(image, lang=self.languages, config="--psm 4")

    def crop(self, image, box):
        x0, y0, x1, y1 = box
        if max(box) <= 1.0:
            x0, x1, y0, y1 = x0 * image.width, x1 * image.width, y0 * image.height, y1 * image.height
        pad = 4
        region = image.crop((max(0, x0 - pad), max(0, y0 - pad), min(image.width, x1 + pad), min(image.height, y1 + pad)))
        if region.height < 40:
            scale = 40 / max(1, region.height)
            region = region.resize((max(1, int(region.width * scale)), 40))
        return self.tesseract.image_to_string(region, lang=self.languages, config="--psm 7").strip()


def scan(root):
    found = defaultdict(dict)
    images, coco = {}, []
    for folder, dirs, names in os.walk(root):
        dirs.sort()
        for name in sorted(names):
            path = os.path.join(folder, name)
            ext = os.path.splitext(name)[1].lower()
            stem = stem_of(name)
            if ext in IMAGE_EXTENSIONS:
                images.setdefault(stem, path)
                continue
            if ext == ".xml":
                parsed = parse_voc(path)
                if parsed:
                    found[stem].setdefault("boxes", []).extend(parsed["boxes"])
                    if parsed["image_name"]:
                        found[stem]["image_name"] = parsed["image_name"]
                    found[stem].setdefault("formats", set()).add("voc")
                continue
            if ext == ".csv":
                for key, entry in parse_csv(path).items():
                    target = found[key]
                    target.setdefault("boxes", []).extend(entry.get("boxes", []))
                    target.setdefault("labels", {}).update(entry.get("labels", {}))
                    for field in ("text", "image_name"):
                        if entry.get(field):
                            target[field] = entry[field]
                    target.setdefault("formats", set()).add("csv")
                continue
            if ext == ".jsonl":
                with open(path, encoding="utf-8", errors="replace") as f:
                    for index, line in enumerate(f):
                        try:
                            data = json.loads(line)
                        except ValueError:
                            continue
                        if isinstance(data, dict):
                            key = stem_of(str(data.get("file_name") or data.get("filename") or data.get("image")
                                              or data.get("id") or f"{stem}-{index}"))
                            merge(found[key], parse_json_object(data), "jsonl")
                continue
            if ext not in (".txt", ".json"):
                continue
            try:
                parsed = parse_text_file(path) if ext == ".txt" else parse_json_object(json.loads(read_text(path)))
            except ValueError:
                parsed = None
            if parsed and "coco" in parsed:
                coco.append(parsed["coco"])
                continue
            merge(found[stem], parsed, ext[1:])
    for data in coco:
        by_id = {img["id"]: img for img in data.get("images", [])}
        names = {c["id"]: c.get("name", "") for c in data.get("categories", [])}
        for ann in data.get("annotations", []):
            image = by_id.get(ann.get("image_id"))
            if not image:
                continue
            key = stem_of(image.get("file_name", str(image["id"])))
            entry = found[key]
            entry.setdefault("boxes", []).append({"label": names.get(ann.get("category_id"), ""), "box": box_of(ann),
                                                  "text": str(ann.get("text") or ann.get("transcription") or "")})
            entry["image_name"] = image.get("file_name", "")
            entry.setdefault("formats", set()).add("coco")
    for key, entry in found.items():
        name = entry.get("image_name")
        entry["image"] = images.get(stem_of(name)) if name else None
        entry["image"] = entry["image"] or images.get(key)
    for key, path in images.items():
        if key not in found:
            found[key] = {"image": path, "formats": {"image_only"}}
    return found


def merge(target, parsed, fmt):
    if not parsed:
        return
    target.setdefault("formats", set()).add(fmt)
    if parsed.get("labels"):
        target.setdefault("labels", {}).update({k: v for k, v in parsed["labels"].items()
                                                if k not in target.get("labels", {})})
    if parsed.get("text") and len(parsed["text"]) > len(target.get("text", "")):
        target["text"] = parsed["text"]
    if parsed.get("boxes"):
        target.setdefault("boxes", []).extend(parsed["boxes"])
    if parsed.get("ignored_labels"):
        target.setdefault("ignored_labels", []).extend(parsed["ignored_labels"])


def receipts_from(root, ocr=None, stats=None, text_dir=None):
    stats = stats if stats is not None else Counter()
    pending = []
    for key, entry in sorted(scan(root).items()):
        stats["files:" + "+".join(sorted(entry.get("formats", ())))] += 1
        for name in set(entry.get("ignored_labels", ())):
            stats[f"label_not_used:{name}"] += 1
        labels, crops = dict(entry.get("labels", {})), {}
        for box in entry.get("boxes", []):
            field = field_of(box["label"])
            if not field or field in labels:
                continue
            if box.get("text"):
                labels[field] = box["text"]
                crops.pop(field, None)
            elif ocr and entry.get("image") and box.get("box"):
                crops.setdefault(field, []).append(box["box"])
        text = entry.get("text", "")
        if not text and entry.get("boxes") and all(b.get("text") for b in entry["boxes"]) \
                and not all(field_of(b["label"]) for b in entry["boxes"]):
            ordered = sorted((b for b in entry["boxes"] if b.get("box")), key=lambda b: (b["box"][1], b["box"][0]))
            text = text_from_lines(b["text"] for b in ordered)
        read_elsewhere = False
        if not text and text_dir and entry.get("image"):
            path = os.path.join(text_dir, stem_of(entry["image"]) + ".txt")
            if os.path.exists(path):
                text, read_elsewhere = read_text(path), True
                stats["text_from_text_dir"] += 1
        full = bool(not text and ocr and entry.get("image"))
        if crops or full:
            missing = [f for f in ("total", "merchant") if not labels.get(f) and f not in crops]
            if missing:
                stats["dropped:no_" + missing[0]] += 1
                continue
        pending.append((key, entry, labels, crops, text, full, read_elsewhere))
    jobs = [i for i, p in enumerate(pending) if p[3] or p[5]]
    read = dict(zip(jobs, ocr.read_all([(pending[i][1]["image"], pending[i][3], pending[i][5]) for i in jobs])
                    if jobs else []))
    out = []
    for index, (key, entry, labels, crops, text, full, read_elsewhere) in enumerate(pending):
        used_ocr = index in read or read_elsewhere
        if index in read:
            fields, full_text, error = read[index]
            if error:
                stats["ocr_failed"] += 1
                print(f"ocr failed on {entry['image']}: {error}", file=sys.stderr, flush=True)
            labels.update({k: v for k, v in fields.items() if v})
            text = text or full_text
        text = text_from_lines((text or "").splitlines())
        if not text:
            stats["dropped:no_text"] += 1
            continue
        if not labels.get("total") or not labels.get("merchant"):
            stats["dropped:no_" + ("total" if not labels.get("total") else "merchant")] += 1
            continue
        out.append({"key": key, "text": text, "labels": labels, "ocr": used_ocr})
    return out


EURO_SURE = re.compile(r"€\s?\d+[.,]\d{2}(?!\d)|(?<![\d.,])\d+,\d{2}\s?€(?!\w)")
EURO_AFTER_DOT = re.compile(r"(?<![\d.,])\d+\.\d{2}\s?€(?!\w)")


def detect_currency(text, labels, default):
    blob = labels.get("total", "") + "\n" + text
    for code, patterns in CURRENCY_MARKERS:
        if any(re.search(p, blob, re.I if code not in ("JPY", "CNY") else 0) for p in patterns):
            return code
        if code == "EUR" and (EURO_SURE.search(blob) or len(EURO_AFTER_DOT.findall(blob)) >= 2):
            return code
    return default


def detect_language(receipts):
    french = sum(1 for r in receipts if sum(w in r["text"].lower() for w in FRENCH_WORDS) >= 2)
    return "fr" if french > len(receipts) / 2 else "en"


def parse_date(raw, day_first=True):
    raw = (raw or "").strip().lower()
    if not raw:
        return None
    match = re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", raw)
    if match:
        y, m, d = (int(g) for g in match.groups())
        return safe_date(y, m, d)
    match = re.search(r"(\d{1,2})[-/. ](\d{1,2})[-/. ](\d{2,4})(?!\d)", raw)
    if match:
        a, b, y = (int(g) for g in match.groups())
        y = y + 2000 if y < 100 else y
        d, m = (a, b) if day_first else (b, a)
        if m > 12 and d <= 12:
            d, m = m, d
        return safe_date(y, m, d)
    match = re.search(r"(\d{1,2})(?:st|nd|rd|th|er)?[-/ .,]*([a-zéû]{3,9})\.?[-/ .,]*(\d{2,4})", raw)
    if match and match.group(2) in MONTHS:
        y = int(match.group(3))
        return safe_date(y + 2000 if y < 100 else y, MONTHS[match.group(2)], int(match.group(1)))
    match = re.search(r"([a-zéû]{3,9})\.?[-/ .]*(\d{1,2})(?:st|nd|rd|th)?[-/ .,]+(\d{2,4})", raw)
    if match and match.group(1) in MONTHS:
        y = int(match.group(3))
        return safe_date(y + 2000 if y < 100 else y, MONTHS[match.group(1)], int(match.group(2)))
    return None


def safe_date(y, m, d):
    try:
        day = date(y, m, d)
    except ValueError:
        return None
    return day if 1990 <= day.year <= 2035 else None


def times_in(text):
    found = set()
    for hour, minute, half in TIME.findall(text):
        hour = int(hour)
        if half:
            if hour > 12:
                continue
            hour = hour % 12 + (12 if half.lower() == "p" else 0)
        found.add(f"{hour:02d}:{minute}")
    return found


SPLIT_DECIMALS = re.compile(r"(\d)([.,]) (\d{2})(?!\d)")


def amounts_in(text, currency):
    values = set()
    for match in NUMBER.finditer(text + "\n" + SPLIT_DECIMALS.sub(r"\1\2\3", text)):
        value = minor_units(match.group(0).replace(" ", ""), currency)
        if value is not None:
            values.add(value)
    return values


def clean_merchant(raw):
    name = re.sub(r"\s+", " ", raw).strip(" .,-:;")
    name = REGISTRATION.sub(" ", name)
    name = LEGAL_SUFFIXES.sub(" ", name)
    name = re.sub(r"\s+", " ", name).strip(" .,-:;&")
    name = re.sub(r"\(\s*\)", "", name).strip()
    if name.count("(") > name.count(")"):
        name += ")"
    if len(re.sub(r"\W", "", name)) < 3:
        name = re.sub(r"\s+", " ", raw).strip()
    if name.isupper():
        name = re.sub(r"[A-Z][A-Z']*", lambda m: m.group(0)[0] + m.group(0)[1:].lower(), name)
    return name


def accepted_names(merchant, raw, spellings=()):
    folded = re.sub(r"[^a-z0-9]", "", raw.lower())
    names = {merchant, raw, re.sub(r"\s+", " ", re.sub(r"\([^)]*\)", " ", merchant)).strip(), *spellings}
    for fragment, trading in TRADING_NAMES.items():
        if fragment in folded:
            names.update(trading)
    return sorted(n for n in names if n)


def words_in(text):
    text = re.sub(r"\b(?:[a-z]\.){2,}", lambda m: m.group(0).replace(".", ""), text.lower())
    return " " + " ".join(re.findall(r"[a-z0-9éèêàâûôîç]+", text)) + " "


def category_kind(merchant, text):
    name, body = words_in(merchant), words_in(text)
    for kind, words in CATEGORY_WORDS.items():
        if any(f" {w} " in name or (w in ("mart",) and re.search(r"[a-z]mart ", name)) for w in words):
            return kind, True
    hits = Counter(kind for kind, words in CATEGORY_WORDS.items() for w in words if f" {w} " in body)
    if hits:
        (kind, count), *rest = hits.most_common()
        if count >= 2 and (not rest or rest[0][1] < count):
            return kind, True
        return kind, False
    return "shopping", False


def unify_merchants(receipts, stats):
    names = sorted({re.sub(r"[^a-z0-9]", "", words_in(r["merchant"])) for r in receipts})
    same = {name: name for name in names}
    for index, name in enumerate(names):
        for other in names[:index]:
            if len(name) >= 8 and SequenceMatcher(None, name, other).ratio() >= 0.9:
                same[name] = same[other]
                break
    groups = defaultdict(list)
    for receipt in receipts:
        groups[same[re.sub(r"[^a-z0-9]", "", words_in(receipt["merchant"]))]].append(receipt)
    for group in groups.values():
        voters = [r for r in group if r["kind_sure"]] or group
        votes = Counter(r["kind"] for r in voters)
        kind = min(votes, key=lambda k: (-votes[k], k))
        sure = any(r["kind_sure"] for r in group)
        spellings = sorted({r["merchant"] for r in group})
        for receipt in group:
            if receipt["kind"] != kind:
                stats["category_unified"] += 1
            receipt["kind"], receipt["kind_sure"], receipt["spellings"] = kind, sure, spellings
    return receipts


def prepare(receipts, currency, day_first, stats):
    out, seen = [], set()
    for receipt in receipts:
        labels, text = receipt["labels"], receipt["text"]
        fix = LABEL_FIXES.get((labels["merchant"].strip(), labels.get("date", "").strip(), labels["total"].strip()))
        if fix:
            labels = {**labels, **fix}
            stats["label_fixed"] += 1
        digest = hashlib.sha256(re.sub(r"\s+", " ", text.lower()).encode("utf-8")).hexdigest()
        if digest in seen:
            stats["dropped:duplicate"] += 1
            continue
        seen.add(digest)
        code = detect_currency(text, labels, currency)
        if code in THREE_DECIMAL_CURRENCIES:
            stats[f"dropped:three_decimal_currency_{code}"] += 1
            continue
        amount = minor_units(labels["total"], code)
        if not amount:
            stats["dropped:unreadable_total"] += 1
            continue
        total_printed = amount in amounts_in(text, code)
        day = parse_date(labels.get("date", ""), day_first)
        if day is None:
            stats["dropped:no_date" if not labels.get("date") else "dropped:unreadable_date"] += 1
            continue
        labelled_time = times_in(labels.get("time", ""))
        printed = times_in(text)
        if labelled_time:
            when = sorted(labelled_time)[0]
        elif len(printed) <= 1:
            when = next(iter(printed), "")
        else:
            stats["dropped:several_times_printed"] += 1
            continue
        merchant = clean_merchant(labels["merchant"])
        kind, sure = category_kind(merchant, text)
        traits = ["real_receipt"] + (["ocr_from_image"] if receipt["ocr"] else [])
        if day not in {parse_date(line, day_first) for line in text.splitlines()}:
            traits.append("date_not_matched_in_text")
        if not total_printed:
            traits.append("total_not_in_text")
            stats["total_not_in_text"] += 1
        if len(text) > CONTENT_CHARACTER_LIMIT:
            traits.append("over_limit")
        spending = not CREDIT_NOTE.search(text)
        if not spending:
            traits.append("credit_note")
            stats["credit_note"] += 1
        out.append({"key": receipt["key"], "digest": digest, "text": text, "merchant": merchant,
                    "merchant_raw": re.sub(r"\s+", " ", labels["merchant"]).strip(), "currency": code,
                    "amount": amount, "date": day, "time": when, "kind": kind, "kind_sure": sure, "traits": traits,
                    "spending": spending})
    return out


def split_of(receipt, eval_share, seed):
    roll = int(hashlib.sha256(f"{seed}-{receipt['digest']}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "eval" if roll < eval_share else "train"


def build_split(receipts, name, seed, language, copies, cases_per_user, known_share):
    rng = random.Random(f"{seed}-{name}")
    titles = CATEGORY_TITLES[language]
    by_currency = defaultdict(list)
    for receipt in receipts:
        by_currency[receipt["currency"]].append(receipt)
    users, cases = [], []
    for currency, pool in sorted(by_currency.items()):
        groups = []
        for _ in range(copies):
            order = list(pool)
            rng.shuffle(order)
            groups += [order[i:i + cases_per_user] for i in range(0, len(order), cases_per_user)]
        for group in groups:
            user, user_cases = make_user(rng, f"{name}-u{len(users):04d}", group, pool, currency, language, titles,
                                         known_share)
            users.append(user)
            cases += user_cases
    rng.shuffle(cases)
    return cases, users


def past_record(rng, user_id, index, merchant, kind, titles, amount, currency, day):
    return {"id": f"{user_id}-r{index:03d}", "date": day.isoformat(), "time": "", "merchant": merchant,
            "payment_instrument": "", "amount_minor": max(1, int(amount * rng.uniform(0.4, 1.6))), "currency": currency,
            "category": titles[kind], "tags": [], "note": titles[kind],
            "source": rng.choices(["agent_reviewed", "manual", "agent"], [0.6, 0.25, 0.15])[0], "content": ""}


def make_user(rng, user_id, receipts, pool, currency, language, titles, known_share):
    records, kinds = [], set(CORE_KINDS)
    case_merchants = {r["merchant"].casefold() for r in receipts}
    plan = []
    for receipt in receipts:
        known = (not receipt["kind_sure"]) or rng.random() < known_share
        plan.append((receipt, known))
        kinds.add(receipt["kind"])
    for receipt, known in plan:
        if not known:
            continue
        for _ in range(rng.randint(1, 4)):
            day = receipt["date"] - timedelta(days=rng.randint(2, 150))
            records.append(past_record(rng, user_id, len(records), receipt["merchant"], receipt["kind"], titles,
                                       receipt["amount"], currency, day))
    others = [r for r in pool if r["merchant"].casefold() not in case_merchants]
    first = min(r["date"] for r in receipts)
    last = max(r["date"] for r in receipts)
    for other in rng.sample(others, min(len(others), rng.randint(10, 30))):
        day = first - timedelta(days=rng.randint(1, 180)) + timedelta(days=rng.randint(0, max(0, (last - first).days)))
        records.append(past_record(rng, user_id, len(records), other["merchant"], other["kind"], titles,
                                   other["amount"], currency, day))
        kinds.add(other["kind"])
    records.sort(key=lambda r: (r["date"], r["id"]))
    for index, record in enumerate(records):
        record["id"] = f"{user_id}-r{index:03d}"
    order = list(titles)
    categories = [titles[k] for k in order if k in kinds]
    user = {"id": user_id, "home": "real", "language": language, "localCurrency": currency, "preferences": [],
            "categories": categories, "tags": [], "records": records, "memory": "", "memoryKind": "empty",
            "memoryWrongAbout": None, "memoryHelpsWith": [], "workMerchant": None, "trips": {}}
    cases = []
    for index, (receipt, known) in enumerate(plan):
        title = titles[receipt["kind"]]
        accepted = accepted_names(receipt["merchant"], receipt["merchant_raw"], receipt.get("spellings", ()))
        today = receipt["date"] + timedelta(days=rng.choice([0, 0, 1, 2, 3, 5, 9]))
        over = "over_limit" in receipt["traits"]
        traits = receipt["traits"] + ["known_merchant" if known else "new_merchant", f"category_{receipt['kind']}"]
        cases.append({
            "id": f"{user_id}-{index:04d}", "user": user_id, "language": language, "homeLocale": "real",
            "style": "till", "traits": sorted(set(traits)), "today": today.isoformat(), "content": receipt["text"],
            "expectedStatus": "rejected_too_long" if over else "needs_review", "source": receipt["key"],
            "gold": {"isSpending": receipt["spending"], "merchant": receipt["merchant"], "merchantAccepted": accepted,
                     "paymentInstrument": "", "paymentInstrumentMode": "either", "paymentInstrumentAccepted": [],
                     "totalAmount": decimal_string(receipt["amount"], currency),
                     "amountMinorUnits": receipt["amount"], "currencyCode": currency,
                     "purchaseDate": receipt["date"].isoformat(), "purchaseTime": receipt["time"],
                     "category": title, "proposedCategory": "", "proposalAccepted": [], "noteItems": [],
                     "categoryAccepted": [title], "tags": [], "proposedTagMode": "none"}})
    return user, cases


def write_suite(folder, name, cases, users, extra):
    os.makedirs(folder, exist_ok=True)
    for file_name, rows in (("cases.jsonl", cases), ("users.jsonl", users)):
        with open(os.path.join(folder, file_name), "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    lengths = sorted(len(c["content"]) for c in cases) or [0]
    meta = {"name": name, "version": SUITE_VERSION, "contentCharacterLimit": CONTENT_CHARACTER_LIMIT,
            "cases": len(cases), "users": len(users), "fingerprint": fingerprint(cases, users),
            "contentLength": {"median": lengths[len(lengths) // 2], "p90": lengths[int(len(lengths) * 0.9)],
                              "max": lengths[-1]},
            "styles": dict(Counter(c["style"] for c in cases)),
            "traits": dict(Counter(t for c in cases for t in c["traits"]).most_common()),
            "currencies": dict(Counter(c["gold"]["currencyCode"] for c in cases)),
            "memory": dict(Counter(u["memoryKind"] for u in users)), **extra}
    with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    return meta


def convert(root, seed, eval_share, currency, day_first, language, ocr=None, text_dir=None):
    stats = Counter()
    raw = receipts_from(root, ocr, stats, text_dir)
    stats["parsed"] = len(raw)
    receipts = unify_merchants(prepare(raw, currency, day_first, stats), stats)
    stats["kept"] = len(receipts)
    language = detect_language(receipts) if language == "auto" else language
    train = [r for r in receipts if split_of(r, eval_share, seed) == "train"]
    held = [r for r in receipts if split_of(r, eval_share, seed) == "eval"]
    return stats, language, train, held


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="folder holding the downloaded dataset")
    ap.add_argument("--dataset", default="", help="where the data came from, recorded in meta.json")
    ap.add_argument("--train-name", default="train-real", help="must start with train, so only it is trained on")
    ap.add_argument("--eval-name", default="dev-real")
    ap.add_argument("--seed", type=int, default=51)
    ap.add_argument("--eval-share", type=float, default=0.15)
    ap.add_argument("--currency", default="USD", help="currency when a receipt shows none")
    ap.add_argument("--month-first", action="store_true", help="read 03/04/2019 as March 4 (default: 3 April)")
    ap.add_argument("--language", default="auto", choices=("auto", *CATEGORY_TITLES))
    ap.add_argument("--train-copies", type=int, default=2, help="users each train receipt is given to")
    ap.add_argument("--cases-per-user", type=int, default=8)
    ap.add_argument("--known-share", type=float, default=0.7,
                    help="share of receipts whose merchant the user has filed before")
    ap.add_argument("--ocr", action="store_true", help="read text from images with tesseract where none is given")
    ap.add_argument("--ocr-languages", default="eng")
    ap.add_argument("--text-dir", help="<image stem>.txt files read from the photos elsewhere, for example by "
                    "ocr_macos.py: used where the dataset gives no text, before --ocr")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "suites"))
    args = ap.parse_args()
    if not args.train_name.startswith("train"):
        raise SystemExit("--train-name must start with train")
    if args.eval_name.startswith("train"):
        raise SystemExit("--eval-name must not start with train: the trainers would accept it")

    ocr = TesseractOCR(args.ocr_languages) if args.ocr else None
    stats, language, train, held = convert(args.source, args.seed, args.eval_share, args.currency, not args.month_first,
                                           args.language, ocr, args.text_dir)
    print(json.dumps(dict(sorted(stats.items())), indent=1))
    if not train and not held:
        raise SystemExit("no usable receipts: look at the counts above (files: shows the formats found)")
    common = {"seed": args.seed, "dataset": args.dataset, "evalShare": args.eval_share, "dayFirst": not args.month_first,
              "defaultCurrency": args.currency, "languageSetting": args.language, "ocr": args.ocr,
              "ocrLanguages": args.ocr_languages if args.ocr else None, "textDir": bool(args.text_dir), "trainCopies": args.train_copies,
              "knownShare": args.known_share, "casesPerUser": args.cases_per_user, "conversion": dict(stats)}
    for name, receipts, copies in ((args.train_name, train, args.train_copies), (args.eval_name, held, 1)):
        if not receipts:
            print(f"{name}: no receipts")
            continue
        cases, users = build_split(receipts, name, args.seed, language, copies, args.cases_per_user, args.known_share)
        meta = write_suite(os.path.join(args.out_dir, name), name, cases, users,
                           {**common, "receipts": len(receipts), "copies": copies, "language": language})
        print(name, json.dumps({k: meta[k] for k in ("cases", "users", "receipts", "fingerprint", "currencies")}))


if __name__ == "__main__":
    main()
