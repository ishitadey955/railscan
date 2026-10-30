"""Google Cloud Vision OCR for sign photos (optional; EasyOCR is the fallback).

Used when GOOGLE_VISION_API_KEY is set (Streamlit: Manage app -> Settings -> Secrets). The photo is
sent to Google's Vision API (TEXT_DETECTION, made for text in photos) with hints for the scripts
RailScan reads; the answer is returned in the same shape as an EasyOCR result (src.ocr.ocr_engine),
so the rest of the app does not care which engine read the sign.

    from src.ocr import google_vision
    if google_vision.available():
        result = google_vision.recognize(image_bytes)     # raises GoogleOCRError on any failure

Python's standard library makes the request (no extra dependency). The key travels in the
x-goog-api-key header, never in the URL, so it cannot end up in an error message or a log.
"""
from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .text_cleaning import clean_text
from .train_numbers import split_train_numbers

ENV_KEY = "GOOGLE_VISION_API_KEY"
ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
LANGUAGE_HINTS = ["hi", "mr", "bn", "kn", "te", "en"]      # the scripts RailScan reads, plus English
MAX_SIDE = 2000          # px; larger photos are sent smaller (text on a sign stays readable)
TIMEOUT_S = 15.0


class GoogleOCRError(RuntimeError):
    """The Vision API could not read the photo (network, key, quota, bad answer). Message is safe to log."""


def available() -> bool:
    return bool(os.environ.get(ENV_KEY, "").strip())


def _jpeg(image_bytes: bytes) -> bytes:
    """The photo as JPEG, at most MAX_SIDE px on its longest side (keeps requests small)."""
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    out = io.BytesIO()
    img.save(out, "JPEG", quality=92)
    return out.getvalue()


def _post(body: Dict[str, Any], key: str, timeout: float) -> Dict[str, Any]:
    req = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:          # 400 bad key / 403 not enabled / 429 quota
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("error", {}).get("message", "")
        except Exception:
            detail = ""
        raise GoogleOCRError(f"Vision API HTTP {exc.code}: {detail[:200]}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise GoogleOCRError(f"Vision API unreachable: {type(exc).__name__}") from None


def _confidence(annotation: Dict[str, Any]) -> Optional[float]:
    """Mean word confidence (0-100) from fullTextAnnotation, or None when Google gives none."""
    scores: List[float] = []
    for page in annotation.get("pages", []):
        for block in page.get("blocks", []):
            for para in block.get("paragraphs", []):
                scores += [w["confidence"] for w in para.get("words", []) if w.get("confidence")]
    return round(100 * sum(scores) / len(scores), 1) if scores else None


def parse(response: Dict[str, Any]) -> Dict[str, Any]:
    """One Vision API answer -> an OCR result dict like src.ocr.ocr_engine's."""
    r = (response.get("responses") or [{}])[0]
    if r.get("error"):
        raise GoogleOCRError(f"Vision API error: {r['error'].get('message', '')[:200]}")
    full = r.get("fullTextAnnotation") or {}
    raw = full.get("text") or ((r.get("textAnnotations") or [{}])[0].get("description") or "")
    locales = sorted({lang.get("languageCode") for page in full.get("pages", [])
                      for lang in page.get("property", {}).get("detectedLanguages", []) if lang.get("languageCode")})
    clean = clean_text(raw)
    numbers, station_text = split_train_numbers(clean)
    ok = bool(clean.strip())
    conf = _confidence(full)
    return {
        "status": "ok" if ok else "OCR failed", "text": clean, "raw_text": raw, "clean_text": clean,
        "ocr_confidence": conf if ok else None, "low_confidence": bool(ok and conf is not None and conf < 60),
        "ocr_language": "google:" + "+".join(locales) if locales else "google",
        "ocr_backend": "google", "preprocessing_method": None, "train_numbers": numbers if ok else [],
        "station_text": station_text if ok else "", "error": None if ok else "Google found no text in the photo.",
        "warnings": [], "detected_languages": locales,
    }


def recognize(image_bytes: bytes, key: Optional[str] = None, timeout: float = TIMEOUT_S) -> Dict[str, Any]:
    """Read the text of a photo with Google Cloud Vision. Raises GoogleOCRError on any failure."""
    key = (key or os.environ.get(ENV_KEY, "")).strip()
    if not key:
        raise GoogleOCRError("no Google Vision API key configured")
    try:
        content = base64.b64encode(_jpeg(image_bytes)).decode("ascii")
    except Exception as exc:
        raise GoogleOCRError(f"photo could not be prepared: {type(exc).__name__}") from None
    body = {"requests": [{"image": {"content": content}, "features": [{"type": "TEXT_DETECTION"}],
                          "imageContext": {"languageHints": LANGUAGE_HINTS}}]}
    return parse(_post(body, key, timeout))
