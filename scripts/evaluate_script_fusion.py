#!/usr/bin/env python3
"""
RailScan - can OCR evidence make the script decision more reliable than the CNN alone?

For every VALIDATION and TEST image (RailScan_CNN_Metadata.csv) this records:
  * the CNN's class probabilities (models/language_classifier.keras)
  * for each script (Devanagari, Kannada, Telugu): EasyOCR read with that script's
    recogniser (+ English), the best usable attempt's confidence and score, and the
    share of recognised letters that belong to that script's Unicode block.

Then it compares decision rules. Every rule parameter is chosen on VALIDATION and
reported on TEST once, so the test accuracy is not tuned.

Outputs: reports/script_fusion_rows.csv, reports/script_fusion_evaluation.json
Usage:   python scripts/evaluate_script_fusion.py [--limit N]   (~30-60 min on CPU)
         python scripts/evaluate_script_fusion.py --analyse-only  (re-use the rows CSV)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCRIPTS = ["Devanagari", "Kannada", "Telugu"]
ROWS = ROOT / "reports" / "script_fusion_rows.csv"


def collect(limit: int) -> pd.DataFrame:
    from src.image.prediction import predict_script
    from src.ocr import OCREngine, language_candidates
    from src.script_evidence import native_share

    meta = pd.read_csv(ROOT / "data" / "processed" / "RailScan_CNN_Metadata.csv")
    meta = meta[meta.split.isin(["validation", "test"])]
    if limit:
        meta = meta.groupby(["split", "script_label"], group_keys=False).head(limit)
    eng = OCREngine()
    rows, t0 = [], time.time()
    for i, r in enumerate(meta.itertuples(), 1):
        path = ROOT / r.image_path
        cnn = predict_script(path)
        row = {"image_path": r.image_path, "split": r.split, "true": r.script_label,
               **{f"cnn_{s}": cnn["class_probabilities"][s] for s in SCRIPTS}}
        for s in SCRIPTS:
            lang = language_candidates(s)[0]
            res = eng.recognize(path, languages=lang, return_attempts=True)
            usable = res["status"] == "ok"
            row[f"ocr_conf_{s}"] = res["ocr_confidence"] if usable else 0.0
            row[f"ocr_share_{s}"] = native_share(res["clean_text"], s) if usable else 0.0
        rows.append(row)
        if i % 20 == 0:
            print(f"  {i}/{len(meta)} images ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROWS, index=False)
    return df


def decide(df: pd.DataFrame, rule: str, params: dict) -> pd.Series:
    from src.script_evidence import fuse
    out = []
    for _, r in df.iterrows():
        cnn = {s: r[f"cnn_{s}"] for s in SCRIPTS}
        ocr = {s: {"confidence": r[f"ocr_conf_{s}"], "native_share": r[f"ocr_share_{s}"]} for s in SCRIPTS}
        out.append(fuse(cnn, ocr if rule != "cnn" else {}, **params)["script"])
    return pd.Series(out, index=df.index)


def analyse(df: pd.DataFrame) -> dict:
    val, test = df[df.split == "validation"], df[df.split == "test"]
    acc = lambda d, pred: float((pred == d.true).mean())
    results = {"cnn_only": {"validation": acc(val, decide(val, "cnn", {})),
                            "test": acc(test, decide(test, "cnn", {}))}}
    # grid over the fusion weight (how much OCR evidence counts vs the CNN), chosen on validation
    grid = []
    for w in (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 35.0, 60.0):
        grid.append((acc(val, decide(val, "fused", {"ocr_weight": w})), w))
    best_val, best_w = max(grid)
    pred_test = decide(test, "fused", {"ocr_weight": best_w})
    results["cnn_plus_ocr_evidence"] = {"ocr_weight_chosen_on_validation": best_w,
                                        "validation_grid": {str(w): a for a, w in grid},
                                        "validation": best_val, "test": acc(test, pred_test)}
    # does the fused confidence mean something? accuracy by confidence band on test
    from src.script_evidence import fuse
    confs = []
    for _, r in test.iterrows():
        cnn = {s: r[f"cnn_{s}"] for s in SCRIPTS}
        ocr = {s: {"confidence": r[f"ocr_conf_{s}"], "native_share": r[f"ocr_share_{s}"]} for s in SCRIPTS}
        confs.append(fuse(cnn, ocr, ocr_weight=best_w)["confidence"])
    t = test.assign(pred=pred_test.values, conf=confs)
    bands = []
    for lo, hi in ((0, .6), (.6, .8), (.8, 1.01)):
        part = t[(t.conf >= lo) & (t.conf < hi)]
        bands.append({"confidence": f"{lo:.1f}-{min(hi, 1):.1f}", "images": int(len(part)),
                      "accuracy": float((part.pred == part.true).mean()) if len(part) else None})
    results["test_accuracy_by_fused_confidence"] = bands
    # threshold for "confident" chosen on VALIDATION: lowest cut where accuracy above it is >= 90%
    vconf = []
    for _, r in val.iterrows():
        cnn = {s: r[f"cnn_{s}"] for s in SCRIPTS}
        ocr = {s: {"confidence": r[f"ocr_conf_{s}"], "native_share": r[f"ocr_share_{s}"]} for s in SCRIPTS}
        vconf.append(fuse(cnn, ocr, ocr_weight=best_w)["confidence"])
    v = val.assign(pred=decide(val, "fused", {"ocr_weight": best_w}).values, conf=vconf)
    cut = next((c for c in (0.5, 0.6, 0.7, 0.8, 0.9)
                if len(v[v.conf >= c]) and (v[v.conf >= c].pred == v[v.conf >= c].true).mean() >= 0.9), 0.9)
    above = t[t.conf >= cut]
    results["confident_threshold_chosen_on_validation"] = cut
    results["test_at_threshold"] = {"share_confident": float((t.conf >= cut).mean()),
                                    "accuracy_when_confident": float((above.pred == above.true).mean()),
                                    "accuracy_when_not": float((t[t.conf < cut].pred == t[t.conf < cut].true).mean())}
    cnn_conf = t[[f"cnn_{s}" for s in SCRIPTS]].max(axis=1)
    results["share_of_test_images_confident_ge_0.6"] = {"cnn_only": float((cnn_conf >= .6).mean()),
                                                        "fused": float((t.conf >= .6).mean())}
    return results


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="images per split and script (0 = all)")
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args(argv)
    df = pd.read_csv(ROWS) if args.analyse_only else collect(args.limit)
    res = analyse(df)
    res.update(created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
               images={"validation": int((df.split == "validation").sum()), "test": int((df.split == "test").sum())})
    (ROOT / "reports" / "script_fusion_evaluation.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
