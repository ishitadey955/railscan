"""Combine the CNN's script probabilities with OCR evidence.

The CNN looks at the whole photo and is right about 49% of the time on the
held-out signboards. OCR adds independent evidence: EasyOCR reads with one
recogniser per script, and a correct reading of real text usually has a much
higher confidence than forcing, say, Kannada characters out of Devanagari text.

    evidence_s = (OCR confidence with script s / 100) x (share of letters in s's Unicode block)
    combined_s = softmax( log P_cnn(s) + ocr_weight x evidence_s )

A reading the OCR rejected as noise still counts through its best attempt (reading_evidence).
``ocr_weight`` was chosen on the validation images and the resulting accuracy
is reported on the test images (scripts/evaluate_script_fusion.py, reports/script_fusion_evaluation.json):
test accuracy 49.2% (CNN alone) -> 86.3% (combined). The combined confidence is
a heuristic score, not a calibrated probability; with the threshold chosen on
validation (0.9), 80% of test images were "confident" and 94.3% of those were right.
"""
from __future__ import annotations

import json
import math
import unicodedata
from pathlib import Path
from typing import Dict, Mapping, Optional

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "script_fusion_evaluation.json"

# Unicode blocks of the scripts the CNN predicts
BLOCKS = {
    "Devanagari": [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
    "Kannada": [(0x0C80, 0x0CFF)],
    "Telugu": [(0x0C00, 0x0C7F)],
    "Bengali": [(0x0980, 0x09FF)],          # Bengali-Assamese script
}
DEFAULT_OCR_WEIGHT = 35.0         # validated value (reports/script_fusion_evaluation.json) if the report is missing
DEFAULT_CONFIDENT = 0.70          # likewise: "confident" threshold chosen on validation
SCRIPTS = ("Devanagari", "Kannada", "Telugu", "Bengali")


def _in(ch: str, script: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in BLOCKS[script])


def native_share(text: str, script: str) -> float:
    """Share of letters/marks in ``text`` that belong to ``script`` (0 when no letters)."""
    letters = [c for c in text or "" if unicodedata.category(c)[0] in "LM"]
    if not letters:
        return 0.0
    return sum(_in(c, script) for c in letters) / len(letters)


def validated_ocr_weight() -> float:
    try:
        return float(json.loads(REPORT.read_text())["cnn_plus_ocr_evidence"]["ocr_weight_chosen_on_validation"])
    except (OSError, KeyError, ValueError):
        return DEFAULT_OCR_WEIGHT


def validated_threshold() -> float:
    try:
        return float(json.loads(REPORT.read_text())["confident_threshold_chosen_on_validation"])
    except (OSError, KeyError, ValueError):
        return DEFAULT_CONFIDENT


def fuse(cnn_probs: Mapping[str, float], ocr: Optional[Mapping[str, Mapping[str, float]]] = None,
         ocr_weight: Optional[float] = None) -> Dict:
    """Combine CNN probabilities with per-script OCR evidence.

    ``ocr``: {script: {"confidence": 0-100, "native_share": 0-1}}; scripts missing
    from it contribute no evidence. Without any OCR evidence the CNN result is
    returned unchanged.
    """
    w = validated_ocr_weight() if ocr_weight is None else ocr_weight
    scripts = list(cnn_probs)
    logits, evidence = {}, {}
    for s in scripts:
        e = 0.0
        if ocr and s in ocr:
            e = max(0.0, float(ocr[s].get("confidence") or 0.0)) / 100.0 * float(ocr[s].get("native_share") or 0.0)
        evidence[s] = e
        logits[s] = math.log(max(float(cnn_probs[s]), 1e-6)) + (w * e if ocr else 0.0)
    m = max(logits.values())
    exp = {s: math.exp(v - m) for s, v in logits.items()}
    z = sum(exp.values())
    probs = {s: v / z for s, v in exp.items()}
    best = max(probs, key=probs.get)
    return {"script": best, "confidence": probs[best], "probabilities": probs, "ocr_evidence": evidence,
            "used_ocr": bool(ocr), "low_confidence": probs[best] < validated_threshold(), "ocr_weight": w}


def reading_evidence(reading: Mapping, script: str) -> Dict[str, float]:
    """OCR evidence for ``script`` from one reading: {"confidence", "native_share"}.

    A reading the OCR rejected as noise (best attempt below its confidence cutoff of 30) still
    counts through its best attempt, in proportion to that attempt's confidence. Before, it counted
    as 0, so a reading at 30.4 counted fully and one at 29.9 not at all; tiny numeric differences
    between machines then flipped the decision (Devanagari on one computer, Kannada on the server).
    Measured on the held-out images (scripts/evaluate_script_fusion.py): test accuracy 79.2% -> 86.3%.
    """
    if reading.get("status") == "ok":
        return {"confidence": float(reading.get("ocr_confidence") or 0.0),
                "native_share": native_share(reading.get("clean_text") or "", script)}
    tried = [a for a in reading.get("attempts") or [] if a.get("ocr_confidence") is not None and a.get("text")]
    best = max(tried, key=lambda a: a["ocr_confidence"], default=None)
    if best is None:
        return {"confidence": 0.0, "native_share": 0.0}
    return {"confidence": float(best["ocr_confidence"]), "native_share": native_share(best["text"], script)}


def identify_script(image, predict, engine) -> Dict:
    """The evaluated procedure: CNN probabilities + an OCR reading with EACH script, combined.

    ``predict``: the CNN predictor (src.image.prediction.predict_script); ``engine``: an OCREngine.
    Returns the CNN result, the combined decision, every script's OCR result, and the OCR
    result of the chosen script (what the app shows and matches stations against).
    """
    from src.ocr import language_candidates
    cnn = predict(image)
    readings, evidence = {}, {}
    for s in SCRIPTS:
        try:      # the attempts keep faint readings for station suggestions (other_readings)
            r = engine.recognize(image, languages=language_candidates(s)[0], return_attempts=True)
        except TypeError:  # an engine without that option
            r = engine.recognize(image, languages=language_candidates(s)[0])
        readings[s] = r
        evidence[s] = reading_evidence(r, s)
    combined = fuse(cnn["class_probabilities"], evidence)
    chosen = readings[combined["script"]]
    chosen = {**chosen, "script": combined["script"], "script_confidence": combined["confidence"],
              "script_source": "cnn+ocr", "script_low_confidence": combined["low_confidence"]}
    return {"cnn": cnn, "combined": combined, "readings": readings, "ocr": chosen}


def other_readings(result: Dict) -> list:
    """Every text the OCR saw in this photo besides the chosen reading: the readings with the other
    scripts, and the attempts it rejected as too uncertain (for station suggestions only)."""
    chosen = (result["ocr"].get("clean_text") or "").strip()
    texts = []
    for r in result["readings"].values():
        for t in [r.get("clean_text") or ""] + [a.get("text") or "" for a in r.get("attempts") or []]:
            t = t.strip()
            if t and t != chosen and t not in texts:
                texts.append(t)
    return texts
