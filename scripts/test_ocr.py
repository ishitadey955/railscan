#!/usr/bin/env python3
"""Run the RailScan OCR component on one or more sign images.

Examples (from the project root):
    python scripts/test_ocr.py data/sign_images/Hindi/board1.jpg
    python scripts/test_ocr.py sign.jpg --script Hindi            # skip the CNN
    python scripts/test_ocr.py sign.jpg --lang eng+kan --attempts # force language, show every try
    python scripts/test_ocr.py data/sign_images/Telugu --json     # a whole folder
    python scripts/test_ocr.py --check                            # verify EasyOCR is installed

The CNN script classifier is used unless --script/--lang is given. If the CNN
model is not available the engine says so in "warnings" and tries every
supported script instead.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ocr import OCRConfig, OCREngine, STATUS_OK  # noqa: E402
from src.ocr.preprocessing import STRATEGIES  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect(paths):
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS)
        else:
            files.append(p)
    return files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*", help="image file(s) or folder(s)")
    ap.add_argument("--script", help="Hindi | Kannada | Marathi | Telugu | English (skips the CNN)")
    ap.add_argument("--lang", help="explicit language string, e.g. eng+hin (skips CNN and mapping)")
    ap.add_argument("--mode", default="bilingual", choices=["bilingual", "native", "both"],
                    help="bilingual: eng+<lang> (default); native: <lang> only; both: try both")
    ap.add_argument("--deskew", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--strategies", help="comma-separated preprocessing strategies to try, in order; "
                                         f"choices: {', '.join(STRATEGIES)}")
    ap.add_argument("--attempts", action="store_true", help="include every (language, preprocessing) attempt")
    ap.add_argument("--json", action="store_true", help="print JSON only")
    ap.add_argument("--check", action="store_true", help="check that EasyOCR is installed")
    args = ap.parse_args(argv)

    cfg = OCRConfig(
        languages_mode=args.mode,
        deskew={"auto": "auto", "on": True, "off": False}[args.deskew],
        strategies=args.strategies.split(",") if args.strategies else None,
    )
    engine = OCREngine(cfg)

    if args.check:
        langs = sorted(engine.available_languages())
        print("EasyOCR languages:", ", ".join(langs) or "(none - pip install -r requirements.txt)")
        print("Models (~100 MB per script) download on first use to ~/.EasyOCR/model.")
        return 0 if langs else 1

    if not args.images:
        ap.error("give at least one image (or use --check)")

    files = collect(args.images)
    if not files:
        print("no images found", file=sys.stderr)
        return 2
    results, failures = [], 0
    for f in files:
        r = engine.recognize(f, script=args.script, languages=args.lang, return_attempts=args.attempts)
        r["file"] = str(f)
        failures += r["status"] != STATUS_OK
        results.append(r)
        if not args.json:
            print(f"\n=== {f}")
            print(f"status      : {r['status']}")
            print(f"script      : {r['script']} ({r['script_source']}"
                  + (f", p={r['script_confidence']:.2f}" if isinstance(r["script_confidence"], float) else "") + ")")
            print(f"language    : {r['ocr_language']}   preprocessing: {r['preprocessing_method']}")
            print(f"confidence  : {r['ocr_confidence']}" + ("  (LOW)" if r["low_confidence"] else ""))
            print(f"clean_text  : {r['clean_text']!r}")
            print(f"station_text: {r['station_text']!r}")
            print(f"train_numbers: {r['train_numbers']}")
            for w in r["warnings"]:
                print("warning     :", w)
            if r["error"]:
                print("error       :", r["error"])
            for a in r.get("attempts", []):
                print("  attempt   :", a)
    if args.json:
        print(json.dumps(results if len(results) > 1 else results[0], ensure_ascii=False, indent=2))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
