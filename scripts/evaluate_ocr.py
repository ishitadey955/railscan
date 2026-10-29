#!/usr/bin/env python3
"""
RailScan - evaluate OCR on held-out signboard images against their transcriptions.

Uses the TEST split of data/processed/RailScan_CNN_Metadata.csv. The true
language is given to the OCR engine, so this measures text recognition alone
(script detection is evaluated separately by train_cnn.py).

Metrics (after Unicode NFC normalisation and whitespace collapsing):
  CER  = Levenshtein(characters) / len(reference)      (lower is better, can exceed 1)
  word recall = share of reference words found in the OCR output
Images where OCR returns nothing count with CER = 1.

Usage:  python scripts/evaluate_ocr.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ocr import OCRConfig, OCREngine  # noqa: E402


def norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", str(text or "")).split())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate OCR on held-out signboards.")
    ap.add_argument("--limit", type=int, default=0, help="max images per language (0 = all)")
    args = ap.parse_args(argv)

    meta = pd.read_csv(ROOT / "data" / "processed" / "RailScan_CNN_Metadata.csv")
    test = meta[meta.split == "test"]
    if args.limit:
        test = test.groupby("language_label", group_keys=False).head(args.limit)
    engine = OCREngine(OCRConfig(languages_mode="native"))
    if not engine.available_languages():
        raise SystemExit("EasyOCR is not installed: pip install -r requirements.txt")
    print(f"Backend: {engine.backend}; images: {len(test)}")

    rows, t0 = [], time.time()
    for i, r in enumerate(test.itertuples(), 1):
        res = engine.recognize(ROOT / r.image_path, script=r.language_label)
        ref, hyp = norm(r.transcription), norm(res.get("clean_text") if res["status"] == "ok" else "")
        cer = Levenshtein.distance(ref, hyp) / max(len(ref), 1) if hyp else 1.0
        ref_words = ref.split()
        recall = sum(w in hyp.split() for w in ref_words) / max(len(ref_words), 1)
        rows.append({"image_path": r.image_path, "language": r.language_label, "status": res["status"],
                     "cer": cer, "word_recall": recall, "confidence": res.get("ocr_confidence"),
                     "reference": ref, "ocr_text": hyp})
        if i % 20 == 0:
            print(f"  {i}/{len(test)}  ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    by_lang = df.groupby("language").agg(images=("cer", "size"), mean_cer=("cer", "mean"),
                                         median_cer=("cer", "median"), word_recall=("word_recall", "mean"),
                                         ocr_failed=("status", lambda s: int((s != "ok").sum())))
    print(by_lang.round(3).to_string())
    print(f"Overall: mean CER {df.cer.mean():.3f}, median CER {df.cer.median():.3f}, "
          f"word recall {df.word_recall.mean():.3f}")

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    report = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "backend": engine.backend, "images": len(df), "seconds": round(time.time() - t0, 1),
              "split": "test (RailScan_CNN_Metadata.csv)", "language_given": True,
              "overall": {"mean_cer": df.cer.mean(), "median_cer": df.cer.median(),
                          "word_recall": df.word_recall.mean(), "ocr_failed": int((df.status != "ok").sum())},
              "by_language": by_lang.reset_index().to_dict(orient="records"),
              "examples": df.sort_values("cer").iloc[[0, len(df) // 2, -1]][
                  ["language", "cer", "reference", "ocr_text"]].to_dict(orient="records")}
    (out / "ocr_evaluation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    df.to_csv(out / "ocr_evaluation_rows.csv", index=False, encoding="utf-8")
    print("Saved reports/ocr_evaluation.json and reports/ocr_evaluation_rows.csv")


if __name__ == "__main__":
    main()
