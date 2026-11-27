import json
import os
import shutil
import sys
import tempfile
import unittest
from collections import Counter
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
        self.full_text, self.crops, self.read = full, crops, []

    def read_all(self, jobs):
        self.read.extend(path for path, _, _ in jobs)
        return [({field: next((self.crops[b] for b in boxes if self.crops[b]), "") for field, boxes in crops.items()},
                 self.full_text if full else "", None) for _, crops, full in jobs]


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

    def test_important_words_and_positioned_words_need_no_ocr(self):
        def word(text, x, y):
            return {"Coordinates": {"xmin": x, "ymin": y, "xmax": x + 60, "ymax": y + 30}, "Text": text}

        self.write("fullDataset/images/7.jpg", "not really an image")
        self.write("fullDataset/info_data/7.json", {"image_path": "/content/Dataset/train/images/7.jpg",
                                                    "important_words": [
            {"label": "date", "words": [word("01/01/18", 400, 900)]},
            {"label": "subtotal", "words": [word("SUBTOTAL", 100, 700), word("20.00", 500, 702)]},
            {"label": "total", "words": [word("TOTAL", 100, 800), word("22.44", 500, 798)]},
            {"label": "company", "words": [word("CORNER", 100, 50), word("CAFE", 170, 52)]},
            {"label": "logo", "words": []}]})
        self.write("fullDataset/gdt/7.json", {"words": [
            word("CAFE", 170, 52), word("CORNER", 100, 50), word("SUBTOTAL", 100, 700), word("20.00", 500, 702),
            word("TOTAL", 100, 800), word("22.44", 500, 798), word("01/01/18", 400, 900)]})
        ocr, stats = StubOCR("", {}), Counter()
        receipts = rr.receipts_from(self.root, ocr, stats)
        self.assertEqual(ocr.read, [])
        self.assertEqual(receipts[0]["labels"], {"date": "01/01/18", "total": "22.44", "merchant": "CORNER CAFE"})
        self.assertEqual(receipts[0]["text"], "CORNER CAFE\nSUBTOTAL 20.00\nTOTAL 22.44\n01/01/18")
        self.assertFalse(receipts[0]["ocr"])
        self.assertEqual(stats["label_not_used:subtotal"], 1)

    def test_folders_are_read_in_sorted_order_so_gdt_labels_win(self):
        def word(text, x, y):
            return {"Coordinates": {"xmin": x, "ymin": y, "xmax": x + 60, "ymax": y + 30}, "Text": text}

        self.write("fullDataset/images/7.jpg", "not really an image")
        self.write("fullDataset/info_data/7.json", {"important_words": [
            {"label": "total", "words": [word("TOTAL", 100, 800), word("RMO3.90", 500, 798)]}]})
        self.write("fullDataset/gdt/7.json", {"company": "CORNER CAFE", "date": "01/01/18", "total": "63.90"})
        listed = os.walk
        self.addCleanup(setattr, rr.os, "walk", listed)

        def info_data_first(top, **kwargs):
            for folder, dirs, names in listed(top, **kwargs):
                dirs[:] = sorted(dirs, reverse=True)
                yield folder, dirs, names

        rr.os.walk = info_data_first
        entry = rr.scan(self.root)["7"]
        self.assertEqual(entry["labels"]["total"], "63.90")

    def test_a_bare_dollar_is_the_datasets_own_currency(self):
        text = "UNIHAKKA INTERNATIONAL SDN BHD\nJohor Bahru\nTotal Amount: $8.20"
        self.assertEqual(rr.detect_currency(text, {"total": "8.20"}, "MYR"), "MYR")
        self.assertEqual(rr.detect_currency(text, {"total": "8.20"}, "USD"), "USD")
        self.assertEqual(rr.detect_currency("TOTAL US$ 8.20", {"total": "8.20"}, "MYR"), "USD")
        self.assertEqual(rr.detect_currency("TOTAL S$8.20", {"total": "8.20"}, "MYR"), "SGD")

    def test_ocr_noise_read_as_euro_is_not_euro(self):
        self.assertEqual(rr.detect_currency("BESI 33.9\n136.00 €R\n63.35 €", {"total": "63.35"}, "MYR"), "MYR")
        self.assertEqual(rr.detect_currency("Amount (RM)\n8.4€", {"total": "8.40"}, "MYR"), "MYR")
        self.assertEqual(rr.detect_currency("CAFE 2,50 €\nTOTAL 12,50 €", {"total": "12,50"}, "MYR"), "EUR")
        self.assertEqual(rr.detect_currency("TOTAL €12.50", {"total": "12.50"}, "MYR"), "EUR")
        self.assertEqual(rr.detect_currency("CAFE 2.50 €\nTOTAL 12.50 €", {"total": "12.50"}, "MYR"), "EUR")
        self.assertEqual(rr.detect_currency("TOTAL EUR 12.50", {"total": "12.50"}, "MYR"), "EUR")

    def test_malay_shop_words_and_dotted_names_find_their_category(self):
        for merchant, text, kind in (("Mr. D.I.Y. (M)", "", "home"), ("Pasaraya Borong Pintar", "", "groceries"),
                                     ("De Jelita Stesen Minyak", "", "fuel"), ("The Toast F&B", "", "dining"),
                                     ("Kedai Papan Yew Chuan", "", "home"),
                                     ("Unihakka International", "TAX INVOICE\n" * 30 + "BAR WANG RICE\nRICE SET", "dining")):
            self.assertEqual(rr.category_kind(merchant, text)[0], kind, merchant)

    def test_a_merchant_keeps_one_category_across_its_receipts(self):
        def receipt(merchant, kind, sure):
            return {"merchant": merchant, "kind": kind, "kind_sure": sure}

        receipts = [receipt("Mr. D.I.Y. (M)", "home", True), receipt("Mr. D.I.Y. (M)", "groceries", False),
                    receipt("Mr. D.I.Y. (M)", "health", True), receipt("Mr. D.I.Y. (M)", "home", True),
                    receipt("Unihakka International", "dining", True),
                    receipt("Unihakka Interantional", "shopping", False)]
        stats = Counter()
        rr.one_kind_per_merchant(receipts, stats)
        self.assertEqual([r["kind"] for r in receipts], ["home"] * 4 + ["dining"] * 2)
        self.assertTrue(all(r["kind_sure"] for r in receipts))
        self.assertEqual(stats["category_unified"], 3)

    def test_text_read_elsewhere_fills_in_for_the_dataset_and_tesseract(self):
        self.write("images/7.jpg", "not really an image")
        self.write("labels/7.json", {"total": "22.44", "company": "CORNER CAFE", "date": "01/01/18"})
        text_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, text_dir)
        with open(os.path.join(text_dir, "7.txt"), "w", encoding="utf-8") as f:
            f.write("CORNER CAFE\nTOTAL 22.44\n01/01/18\n")
        ocr, stats = StubOCR("tesseract text", {}), Counter()
        receipts = rr.receipts_from(self.root, ocr, stats, text_dir)
        self.assertEqual(ocr.read, [])
        self.assertEqual(receipts[0]["text"], "CORNER CAFE\nTOTAL 22.44\n01/01/18")
        self.assertTrue(receipts[0]["ocr"])
        self.assertEqual(stats["text_from_text_dir"], 1)

    def test_macos_ocr_boxes_become_lines(self):
        import ocr_macos
        boxes = [(0.6, 0.502, 0.8, 0.52, "22.44"), (0.1, 0.5, 0.3, 0.52, "TOTAL"), (0.1, 0.1, 0.5, 0.12, "CORNER CAFE")]
        self.assertEqual(ocr_macos.lines_from_boxes(boxes), "CORNER CAFE\nTOTAL 22.44")

    def test_images_without_a_total_or_merchant_label_are_not_read(self):
        self.write("images/a.jpg", "not really an image")
        self.write("labels/a.json", {"date": "12/05/2022"})
        from collections import Counter
        ocr, stats = StubOCR("PIZZERIA ROMA\nTOTAL 23.40", {}), Counter()
        self.assertEqual(rr.receipts_from(self.root, ocr, stats), [])
        self.assertEqual(ocr.read, [])
        self.assertEqual(stats["dropped:no_total"], 1)

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
        self.assertEqual(rr.category_kind("Syarikat Perniagaan Gin Kee", "CEMENT 50KG\nHARDWARE"), ("home", True))
        self.assertEqual(rr.category_kind("Syarikat Perniagaan Gin Kee", "PAINT 1LT\nROLLER"), ("home", False))
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

    def test_totals_missing_from_the_text_are_kept_and_marked(self):
        self.write("box/a.txt", "SHOP SDN BHD\n01/01/2019\nCOFFEE 3.00\nCAKE 4.00\nTOTAL 5.OO")
        self.write("key/a.txt", {"company": "SHOP SDN BHD", "date": "01/01/2019", "total": "7.00"})
        stats, _, train, held = rr.convert(self.root, 1, 0.0, "USD", True, "en")
        self.assertEqual(stats["total_not_in_text"], 1)
        self.assertIn("total_not_in_text", (train + held)[0]["traits"])
        self.assertEqual((train + held)[0]["amount"], 700)

    def test_a_total_split_by_ocr_after_the_decimal_point_counts_as_printed(self):
        self.assertIn(504, rr.amounts_in("Total RM Incl. of GST 5. 04", "MYR"))
        self.assertIn(41292, rr.amounts_in("Total 412. 92", "MYR"))


if __name__ == "__main__":
    unittest.main()
