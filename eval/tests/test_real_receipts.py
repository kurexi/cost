import json
import os
import shutil
import sys
import tempfile
import unittest
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import real_receipts as rr
from grade import score_case
from suite import load_suite, materialize

SROIE_TEXT = """72,25,326,25,326,64,72,64,TAN WOON YANN
50,82,440,82,440,121,50,121,BOOK TA .K (TAMAN DAYA) SDN BHD
205,121,285,121,285,139,205,139,789417-W
110,144,383,144,383,163,110,163,NO.53 55,57 & 59, JALAN SAGU 18,
26,365,111,365,111,384,26,384,DATE : 25/12/2018 8:13:39 PM
26,600,170,600,170,618,26,618,KF MODELLING CLAY KIDDY FISH
30,700,200,700,200,720,30,720,TOTAL: RM 9.00
30,720,200,720,200,740,30,740,CASH 10.00
30,740,200,740,200,760,30,760,CHANGE 1.00"""
SROIE_KEY = {"company": "BOOK TA .K (TAMAN DAYA) SDN BHD", "date": "25/12/2018",
             "address": "NO.53 55,57 & 59, JALAN SAGU 18, TAMAN DAYA, 81100 JOHOR BAHRU, JOHOR.", "total": "9.00"}


class StubOCR:
    def __init__(self, full, crops):
        self.full_text, self.crops = full, crops

    def full(self, path):
        return self.full_text

    def crop(self, path, box):
        return self.crops[box]


def oracle_result(case):
    gold = case["gold"]
    return {"status": "needs_review", "draft": {
        "is_spending": True, "amount_minor": gold["amountMinorUnits"], "currency": gold["currencyCode"],
        "merchant": gold["merchant"], "payment_instrument": "", "category": gold["category"], "proposed_category": "",
        "date": gold["purchaseDate"], "time": gold["purchaseTime"], "tags": [], "proposed_tag": "", "note": "",
        "confidence": 0.9}}


class Fixture(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)

    def write(self, relative, body):
        path = os.path.join(self.root, relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(body if isinstance(body, str) else json.dumps(body))
        return path


class Parsing(Fixture):
    def test_sroie_box_and_key_files_pair_by_name(self):
        self.write("box/X51005200931.txt", SROIE_TEXT)
        self.write("key/X51005200931.txt", SROIE_KEY)
        receipts = rr.receipts_from(self.root)
        self.assertEqual(len(receipts), 1)
        self.assertTrue(receipts[0]["text"].startswith("TAN WOON YANN\nBOOK TA .K"))
        self.assertEqual(receipts[0]["labels"]["total"], "9.00")
        self.assertNotIn("address", receipts[0]["labels"])

    def test_csv_boxes_with_text_give_labels_and_text(self):
        rows = ["filename,class,xmin,ymin,xmax,ymax,text",
                "r1.jpg,shop_name,0,0,10,10,CAFE LUNA",
                "r1.jpg,other,0,20,10,30,Latte 4.50",
                "r1.jpg,date,0,40,10,50,03/02/2021",
                "r1.jpg,total,0,60,10,70,4.50"]
        self.write("labels.csv", "\n".join(rows))
        receipts = rr.receipts_from(self.root)
        self.assertEqual(receipts[0]["labels"], {"merchant": "CAFE LUNA", "date": "03/02/2021", "total": "4.50"})
        self.assertEqual(receipts[0]["text"], "CAFE LUNA\nLatte 4.50\n03/02/2021\n4.50")

    def test_voc_boxes_without_text_are_read_by_ocr(self):
        self.write("images/a.jpg", "not really an image")
        self.write("annotations/a.xml", """<annotation><filename>a.jpg</filename>
<object><name>Total</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>3</xmax><ymax>4</ymax></bndbox></object>
<object><name>Shop</name><bndbox><xmin>5</xmin><ymin>6</ymin><xmax>7</xmax><ymax>8</ymax></bndbox></object>
<object><name>Date</name><bndbox><xmin>9</xmin><ymin>9</ymin><xmax>9</xmax><ymax>9</ymax></bndbox></object>
</annotation>""")
        ocr = StubOCR("PIZZERIA ROMA\n12/05/2022 19:04\nTOTAL 23.40 EUR",
                      {(1.0, 2.0, 3.0, 4.0): "23.40", (5.0, 6.0, 7.0, 8.0): "PIZZERIA ROMA", (9.0,) * 4: "12/05/2022"})
        receipts = rr.receipts_from(self.root, ocr)
        self.assertEqual(receipts[0]["labels"], {"total": "23.40", "merchant": "PIZZERIA ROMA", "date": "12/05/2022"})
        self.assertTrue(receipts[0]["ocr"])

    def test_receipts_without_a_merchant_or_total_are_counted_not_kept(self):
        self.write("box/a.txt", SROIE_TEXT)
        self.write("key/a.txt", {"date": "25/12/2018", "total": "9.00"})
        stats = {}
        from collections import Counter
        stats = Counter()
        self.assertEqual(rr.receipts_from(self.root, None, stats), [])
        self.assertEqual(stats["dropped:no_merchant"], 1)

    def test_subtotals_and_tax_are_not_the_total(self):
        self.assertIsNone(rr.field_of("Sub Total"))
        self.assertIsNone(rr.field_of("total_tax"))
        self.assertIsNone(rr.field_of("TotalDiscount"))
        self.assertEqual(rr.field_of("Total Amount"), "total")
        self.assertEqual(rr.field_of("company"), "merchant")


class Fields(unittest.TestCase):
    def test_dates(self):
        self.assertEqual(rr.parse_date("25/12/2018"), date(2018, 12, 25))
        self.assertEqual(rr.parse_date("03/04/19"), date(2019, 4, 3))
        self.assertEqual(rr.parse_date("03/04/19", day_first=False), date(2019, 3, 4))
        self.assertEqual(rr.parse_date("2018-03-17"), date(2018, 3, 17))
        self.assertEqual(rr.parse_date("17 MAR 2018"), date(2018, 3, 17))
        self.assertEqual(rr.parse_date("Mar 17, 2018"), date(2018, 3, 17))
        self.assertIsNone(rr.parse_date("total 9.00"))

    def test_times(self):
        self.assertEqual(rr.times_in("DATE : 25/12/2018 8:13:39 PM"), {"20:13"})
        self.assertEqual(rr.times_in("12:05 AM"), {"00:05"})
        self.assertEqual(rr.times_in("TOTAL 12.30 TEL 03:12345"), set())

    def test_currency_and_merchant(self):
        self.assertEqual(rr.detect_currency("TOTAL: RM 9.00", {}, "USD"), "MYR")
        self.assertEqual(rr.detect_currency("TOTAL 9.00", {}, "USD"), "USD")
        self.assertEqual(rr.detect_currency("Total 4,50 €", {}, "USD"), "EUR")
        self.assertEqual(rr.clean_merchant("BOOK TA .K (TAMAN DAYA) SDN BHD"), "Book Ta .K (Taman Daya)")
        self.assertEqual(rr.clean_merchant("SANYU STATIONERY SHOP (002247584-X)"), "Sanyu Stationery Shop")

    def test_categories(self):
        self.assertEqual(rr.category_kind("Restoran Wan Sheng", ""), ("dining", True))
        self.assertEqual(rr.category_kind("99 Speedmart", ""), ("groceries", True))
        self.assertEqual(rr.category_kind("Syarikat Perniagaan Gin Kee", "CEMENT 50KG\nHARDWARE"), ("home", False))
        self.assertEqual(rr.category_kind("ABC Trading", "")[1], False)


class Suites(Fixture):
    def make_dataset(self, count=40):
        kinds = ["RESTORAN", "HARDWARE", "MINI MARKET", "PHARMACY", "BOOKSTORE"]
        for index in range(count):
            name = f"{kinds[index % len(kinds)]} NUMBER{index} SDN BHD"
            day = f"{1 + index % 27:02d}/0{1 + index % 9}/2019"
            text = f"{name}\nNO {index}, JALAN BUKIT\n{day} 1{index % 10}:2{index % 10}\nITEM A {index}.50\n" \
                   f"TOTAL RM {index + 1}.50\nCASH {index + 5}.00"
            self.write(f"box/r{index}.txt", text)
            self.write(f"key/r{index}.txt", {"company": name, "date": day, "total": f"{index + 1}.50"})

    def test_build_writes_graded_suites_with_disjoint_receipts(self):
        self.make_dataset()
        stats, language, train, held = rr.convert(self.root, 7, 0.25, "USD", True, "auto")
        self.assertEqual(stats["kept"], 40)
        self.assertEqual(language, "en")
        self.assertFalse({r["digest"] for r in train} & {r["digest"] for r in held})
        out = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, out)
        for name, receipts, copies in (("train-x", train, 2), ("dev-x", held, 1)):
            cases, users = rr.build_split(receipts, name, 7, language, copies, 8, 0.7)
            meta = rr.write_suite(os.path.join(out, name), name, cases, users, {})
            self.assertEqual(meta["cases"], len(receipts) * copies)
            _, loaded, by_id = load_suite(os.path.join(out, name))
            for case in loaded:
                user = by_id[case["user"]]
                gold = case["gold"]
                self.assertEqual(gold["currencyCode"], "MYR")
                self.assertIn(gold["category"], user["categories"])
                self.assertLessEqual(gold["purchaseDate"], case["today"])
                visible = [r for r in user["records"] if r["date"] <= case["today"]]
                known = any(r["merchant"] == gold["merchant"] for r in visible)
                self.assertEqual(known, "known_merchant" in case["traits"])
                scores = score_case(case, oracle_result(case), user)
                self.assertEqual(scores["record_all"], 1.0, case["id"])
                folder = tempfile.mkdtemp()
                self.addCleanup(shutil.rmtree, folder)
                materialize(user, case["today"], folder)

    def test_build_is_deterministic(self):
        self.make_dataset(20)
        first = rr.convert(self.root, 3, 0.2, "USD", True, "en")
        second = rr.convert(self.root, 3, 0.2, "USD", True, "en")
        self.assertEqual(rr.build_split(first[2], "train-y", 3, "en", 2, 8, 0.7),
                         rr.build_split(second[2], "train-y", 3, "en", 2, 8, 0.7))

    def test_totals_missing_from_the_text_are_dropped(self):
        self.write("box/a.txt", "SHOP SDN BHD\n01/01/2019\nTOTAL 5.00")
        self.write("key/a.txt", {"company": "SHOP SDN BHD", "date": "01/01/2019", "total": "7.00"})
        stats, _, train, held = rr.convert(self.root, 1, 0.0, "USD", True, "en")
        self.assertEqual(stats["dropped:total_not_in_text"], 1)
        self.assertEqual(train + held, [])


if __name__ == "__main__":
    unittest.main()
