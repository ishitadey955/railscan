"""Photo checks for the sign assistant: when should RailScan ask for another photo?

Measured on the 145 held-out signboard test images (reports/ocr_evaluation_rows.csv):
  * blur, brightness and contrast did NOT predict OCR failure on these photos,
    so they are only flagged at extremes beyond anything in that set;
  * images with a short side under ~300 px had 15% word recall vs 34-39% for larger ones;
  * OCR confidence <= 50 went with 39% word recall vs 57% above 70.
The advice therefore relies mostly on what OCR and matching actually produced.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np

MIN_SHORT_SIDE = 300          # measured: smaller images read clearly worse
LOW_OCR_CONFIDENCE = 50       # measured: lower confidence went with lower word recall
# extremes only (outside the range of the evaluated photos; not validated thresholds)
MIN_BRIGHTNESS, MAX_BRIGHTNESS = 35, 225
MIN_SHARPNESS = 15


def measure(rgb: np.ndarray) -> Dict[str, float]:
    """Size, brightness (mean grey), contrast (grey std) and sharpness (variance of the Laplacian)."""
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    scale = 1000 / max(h, w)
    if scale < 1:                       # sharpness is compared at a common size
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return {"width": int(w), "height": int(h), "brightness": float(gray.mean()),
            "contrast": float(gray.std()), "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var())}


def photo_issues(m: Dict[str, float]) -> List[str]:
    issues = []
    if min(m["width"], m["height"]) < MIN_SHORT_SIDE:
        issues.append(f"Low resolution ({m['width']} x {m['height']} px); move closer or use a larger photo.")
    if m["brightness"] < MIN_BRIGHTNESS:
        issues.append("The photo is very dark.")
    if m["brightness"] > MAX_BRIGHTNESS:
        issues.append("The photo is very bright or washed out.")
    if m["sharpness"] < MIN_SHARPNESS:
        issues.append("The photo is very blurry; hold the camera still.")
    return issues


def retake_advice(m: Dict[str, float], ocr: Dict, station: Optional[Dict]) -> Dict:
    """Combine photo checks with what OCR and station matching produced.

    Returns {"retake": bool, "reasons": [...]} - reasons are shown to the user.
    """
    reasons = photo_issues(m)
    if ocr.get("status") != "ok":
        reasons.append("No text could be read from this photo.")
    elif ocr.get("ocr_confidence") is not None and ocr["ocr_confidence"] <= LOW_OCR_CONFIDENCE:
        # engines without a confidence score (Gemini) get no low-confidence remark
        reasons.append(f"The text was read with low confidence ({ocr['ocr_confidence']:.0f}/100).")
    if ocr.get("status") == "ok" and (not station or station.get("status") in ("not_identified", "no_text")):
        reasons.append("No station name could be matched to the text.")
    return {"retake": bool(reasons), "reasons": reasons}
