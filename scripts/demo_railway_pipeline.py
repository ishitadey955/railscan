#!/usr/bin/env python3
"""End-to-end demo:  OCR text -> matched station -> station code -> railway data.

    # text you already have (or the OCR output of a sign)
    python scripts/demo_railway_pipeline.py --text "12301 नई दिल्ली" --journey-date 2025-03-08

    # a sign photo: runs the OCR component first (needs EasyOCR, see src/ocr/README.md)
    python scripts/demo_railway_pipeline.py --image sign.jpg --journey-date 2025-03-08

    # real data (default): data/processed/RailScan_Final_Dataset.csv  or  --dataset path.csv
    # no dataset yet? run the code path on the SYNTHETIC test fixture (clearly labelled, NOT real data):
    python scripts/demo_railway_pipeline.py --text "12301 नई दिल्ली" --journey-date 2025-03-08 --fixture
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.railway import RailwayPipeline  # noqa: E402


def fmt(v):
    return "n/a" if v is None else v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="OCR text, e.g. 'नई दिल्ली' or '12301 NEW DELHI'")
    src.add_argument("--image", help="sign image; the OCR component reads it")
    ap.add_argument("--script", help="skip the CNN: Hindi | Kannada | Marathi | Telugu | English")
    ap.add_argument("--journey-date", help="YYYY-MM-DD (needed unless the data holds one date for the train+station)")
    ap.add_argument("--train-no", help="train number if it is not on the sign")
    ap.add_argument("--dataset", help="processed dataset CSV (default data/processed/RailScan_Final_Dataset.csv)")
    ap.add_argument("--fixture", action="store_true", help="use the SYNTHETIC test fixture instead of the real dataset")
    ap.add_argument("--accept-possible", action="store_true", help="also use 75-89 'possible' station matches")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    a = ap.parse_args(argv)

    data = a.dataset
    if a.fixture:
        sys.path.insert(0, str(ROOT / "tests"))
        from railway_fixture import FIXTURE_BANNER, build_fixture
        data = build_fixture()
        print(f"*** {FIXTURE_BANNER} ***\n")

    if a.image:
        from src.ocr import OCREngine
        ocr = OCREngine().recognize(a.image, script=a.script)
        print(f"OCR (script {ocr['script']}, lang {ocr['ocr_language']}, conf {ocr['ocr_confidence']}):")
        print(f"  {ocr['clean_text']!r}   status={ocr['status']}")
    else:
        ocr = a.text
        print(f"OCR text: {a.text!r}")

    pipe = RailwayPipeline(data=data, accept_possible_station=a.accept_possible)
    r = pipe.process(ocr, journey_date=a.journey_date, train_no=a.train_no)
    if a.json:
        print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
        return 0

    s, t, st = r["station"], r["train"], r["train_state"]
    print("   |\n   v")
    if s and s["station_code"]:
        print(f"Matched station : {s['station_name']}   [{s['match_method']}, score {s['match_score']}, {s['confidence_level']}]")
        print("   |\n   v")
        print(f"Station code    : {s['station_code']}")
    else:
        print(f"Matched station : {s['message'] if s else r['message']}")
    print("   |\n   v")
    print(f"Pipeline status : {r['status']}  -  {r['message']}")
    if t and t.get("train_no"):
        print(f"Train           : {t['train_no']}  {fmt(t.get('train_name'))}  ({fmt(t.get('train_type'))})")
    if st and st["status"] in ("ok", "insufficient_history"):
        h = st["historical"]
        print(f"Railway data    : journey {st['journey_date']} at {st['station_name']} ({st['station_code']})")
        print(f"  current delay          : {fmt(st['current_delay_min'])} min")
        print(f"  scheduled arr / dep    : {fmt(st['scheduled_arrival'])} / {fmt(st['scheduled_departure'])}")
        print(f"  actual arr / dep (derived): {fmt(st['actual_arrival'])} / {fmt(st['actual_departure'])}")
        print(f"  next station           : {fmt(st['next_station_name'])} ({fmt(st['next_station_code'])}), "
              f"scheduled arrival {fmt(st['next_scheduled_arrival'])}")
        print(f"  historical avg delay   : {fmt(h['avg_delay_min'])} min  (std {fmt(h['delay_std_min'])}, "
              f"n={h['observation_count']}, sufficient={h['sufficient']})")
        print(f"  historical on-time rate: {fmt(h['on_time_rate_pct'])} %  (journeys < 15 min late at next station)")
        if st["missing_fields"]:
            print(f"  missing in dataset     : {', '.join(st['missing_fields'])}")
    for w in r["warnings"]:
        print("warning:", w)
    return 0 if r["status"] in ("ok", "insufficient_history") else 1


if __name__ == "__main__":
    sys.exit(main())
