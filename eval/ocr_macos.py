import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic")


def lines_from_boxes(boxes):
    rows = []
    for x0, y0, x1, y1, text in sorted(boxes, key=lambda b: (b[1] + b[3]) / 2):
        middle = (y0 + y1) / 2
        if rows and abs(middle - rows[-1]["middle"]) <= 0.5 * rows[-1]["height"]:
            rows[-1]["boxes"].append((x0, text))
        else:
            rows.append({"middle": middle, "height": max(1e-4, y1 - y0), "boxes": [(x0, text)]})
    return "\n".join(" ".join(t for _, t in sorted(row["boxes"])) for row in rows)


def recognise(path, languages, fast, layout):
    import objc
    import Vision
    from Foundation import NSURL

    with objc.autorelease_pool():
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelFast if fast
                                     else Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(not fast)
        if languages:
            request.setRecognitionLanguages_(languages)
        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(NSURL.fileURLWithPath_(path), None)
        ok, error = handler.performRequests_error_([request], None)
        if not ok:
            raise RuntimeError(str(error))
        boxes = []
        for observation in request.results() or []:
            candidates = observation.topCandidates_(1)
            if not candidates:
                continue
            box = observation.boundingBox()
            x0, top = box.origin.x, 1 - box.origin.y - box.size.height
            boxes.append((x0, top, x0 + box.size.width, top + box.size.height, str(candidates[0].string())))
        if layout == "observations":
            return "\n".join(text for *_, text in boxes)
        return lines_from_boxes(boxes)


def main():
    ap = argparse.ArgumentParser(description="Reads receipt photos into text with macOS Vision OCR, for real_receipts.py --text-dir")
    ap.add_argument("--images", required=True, help="folder of receipt photos (searched recursively)")
    ap.add_argument("--out", required=True, help="folder for the .txt files")
    ap.add_argument("--languages", default="en-US,fr-FR", help="Vision recognition languages, most likely first")
    ap.add_argument("--fast", action="store_true", help="Vision's fast level: quicker, less accurate")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--layout", choices=("observations", "lines"), default="observations",
                    help="observations: one line per Vision observation, as the app sends it (default); "
                         "lines: observations side by side joined into one line")
    args = ap.parse_args()
    if sys.platform != "darwin":
        raise SystemExit("ocr_macos.py runs on macOS only")
    try:
        import Vision
    except ImportError:
        raise SystemExit("needs PyObjC's Vision bindings: uv run --with pyobjc-framework-Vision python ocr_macos.py ...")

    images = {}
    for folder, _, names in os.walk(os.path.expanduser(args.images)):
        for name in sorted(names):
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                images.setdefault(os.path.splitext(name)[0].lower(), os.path.join(folder, name))
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    todo = [(stem, path) for stem, path in sorted(images.items()) if not os.path.exists(f"{out}/{stem}.txt")]
    print(f"{len(images)} photos, {len(images) - len(todo)} already read, {len(todo)} to read", flush=True)
    languages = [l.strip() for l in args.languages.split(",") if l.strip()]

    def one(item):
        stem, path = item
        try:
            text = recognise(path, languages, args.fast, args.layout)
        except Exception as error:
            return stem, f"{type(error).__name__}: {error}"
        with open(f"{out}/{stem}.txt.part", "w", encoding="utf-8") as f:
            f.write(text + "\n")
        os.replace(f"{out}/{stem}.txt.part", f"{out}/{stem}.txt")
        return stem, None

    started, failed = time.time(), 0
    with ThreadPoolExecutor(args.workers) as pool:
        for done, (stem, error) in enumerate(pool.map(one, todo), 1):
            if error:
                failed += 1
                print(f"{stem}: {error}", file=sys.stderr, flush=True)
            if done % 25 == 0 or done == len(todo):
                elapsed = time.time() - started
                print(f"{done}/{len(todo)}, {elapsed:.0f} s, about {elapsed / done * (len(todo) - done):.0f} s to go",
                      flush=True)
    print(f"done: {len(todo) - failed} read, {failed} failed, text in {out}")


if __name__ == "__main__":
    main()
