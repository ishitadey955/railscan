#!/usr/bin/env python3
"""
RailScan - Google OCR (Cloud Vision or Gemini) vs the app's EasyOCR pipeline on the held-out signboards.

Uses the TEST split of data/processed/RailScan_CNN_Metadata.csv and its transcriptions. Both
engines get only the photo (as in the app): EasyOCR through the app's own procedure
(src/script_evidence.identify_script, the reading of the chosen script), Google through
src/ocr/google_vision. Same metrics as scripts/evaluate_ocr.py:
  CER         = Levenshtein(characters) / len(reference)  (lower is better; no text = 1)
  word recall = share of reference words found in the output
  word precision = share of output words that are in the reference (low = invented or watermark text)
  script      = script with the largest share of the output's letters (Google), or the app's
                combined decision (EasyOCR), compared with the true script

Needs GOOGLE_VISION_API_KEY (--engine vision; one unit per photo, 1,000 free a month) or
GEMINI_API_KEY (--engine gemini; free tier, rate-limited: --pause slows the requests down).

Usage:  python scripts/evaluate_google_ocr.py --engine gemini [--limit N] [--pause 4]
Writes: reports/<engine>_ocr_evaluation.json, reports/<engine>_ocr_rows.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_ocr import norm  # noqa: E402
from src.image.prediction import predict_script  # noqa: E402
from src.ocr import OCREngine, gemini_ocr, google_vision  # noqa: E402
from src.script_evidence import SCRIPTS, identify_script, native_share  # noqa: E402


def scores(ref: str, hyp: str) -> dict:
    ref, hyp = norm(ref), norm(hyp)
    words, out = ref.split(), hyp.split()
    return {"cer": Levenshtein.distance(ref, hyp) / max(len(ref), 1) if hyp else 1.0,
            "word_recall": sum(w in out for w in words) / max(len(words), 1),
            "word_precision": sum(w in words for w in out) / len(out) if out else None}


def text_script(text: str):
    shares = {s: native_share(text, s) for s in SCRIPTS}
    best = max(shares, key=shares.get)
    return best if shares[best] > 0 else None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["vision", "gemini"], default="gemini")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pause", type=float, default=0.0, help="seconds between requests (free-tier rate limits)")
    args = ap.parse_args(argv)
    mod, err = (google_vision, google_vision.GoogleOCRError) if args.engine == "vision" else (gemini_ocr, gemini_ocr.GeminiOCRError)
    if not mod.available():
        raise SystemExit(f"Set {mod.ENV_KEY} first (see docs/runbooks/deploy.md, Google OCR).")
    meta = pd.read_csv(ROOT / "data" / "processed" / "RailScan_CNN_Metadata.csv")
    test = meta[meta.split == "test"]
    if args.limit:
        test = test.groupby("script_label", group_keys=False).head(args.limit)
    engine = OCREngine(script_predictor=predict_script)
    rows, t0 = [], time.time()
    for i, r in enumerate(test.itertuples(), 1):
        data = (ROOT / r.image_path).read_bytes()
        easy = identify_script(data, predict_script, engine)
        e_text = easy["ocr"].get("clean_text") or "" if easy["ocr"]["status"] == "ok" else ""
        try:
            g = mod.recognize(data)
            g_text, g_error = (g["clean_text"] if g["status"] == "ok" else ""), None
        except err as exc:
            g_text, g_error = "", str(exc)
        time.sleep(args.pause)
        rows.append({"image_path": r.image_path, "script": r.script_label, "reference": norm(r.transcription),
                     "easyocr_text": norm(e_text), "google_text": norm(g_text), "google_error": g_error,
                     **{f"easyocr_{k}": v for k, v in scores(r.transcription, e_text).items()},
                     **{f"google_{k}": v for k, v in scores(r.transcription, g_text).items()},
                     "easyocr_script_right": easy["combined"]["script"] == r.script_label,
                     "google_script_right": text_script(g_text) == r.script_label})
        if i % 20 == 0:
            print(f"  {i}/{len(test)} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "reports" / f"{args.engine}_ocr_rows.csv", index=False)

    def summary(d: pd.DataFrame, e: str) -> dict:
        return {"mean_cer": round(d[f"{e}_cer"].mean(), 4), "word_recall": round(d[f"{e}_word_recall"].mean(), 4),
                "word_precision": round(d[f"{e}_word_precision"].dropna().mean(), 4),
                "no_text": int((d[f"{e}_text"] == "").sum()), "script_right": round(d[f"{e}_script_right"].mean(), 4)}
    res = {"engine": args.engine, "model": gemini_ocr.model_name() if args.engine == "gemini" else "TEXT_DETECTION",
           "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "images": len(df),
           "google_errors": int(df.google_error.notna().sum()),
           "overall": {e: summary(df, e) for e in ("easyocr", "google")},
           "by_script": {s: {e: summary(g, e) for e in ("easyocr", "google")} for s, g in df.groupby("script")}}
    (ROOT / "reports" / f"{args.engine}_ocr_evaluation.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res["overall"], indent=2))


if __name__ == "__main__":
    sys.exit(main())
