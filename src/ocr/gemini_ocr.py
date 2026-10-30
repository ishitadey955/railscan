"""Gemini OCR for sign photos (optional; EasyOCR is the fallback). No Google Cloud billing needed.

Used when GEMINI_API_KEY is set (a free key from Google AI Studio; Streamlit: Manage app -> Settings ->
Secrets). The photo goes to the Gemini API with an instruction to copy the text exactly, line by line,
in the original scripts, without guessing, translating or including stock-photo watermarks. Gemini
answers in a fixed JSON shape; it is returned in the same shape as an EasyOCR result
(src.ocr.ocr_engine), so the rest of the app does not care which engine read the sign.

    from src.ocr import gemini_ocr
    if gemini_ocr.available():
        result = gemini_ocr.recognize(image_bytes)        # raises GeminiOCRError on any failure

Notes
    * Gemini gives no confidence score, so ``ocr_confidence`` is None.
    * A language model can "read" a name it expects rather than what is printed; the app therefore
      never accepts a station found only in Gemini's text automatically (the user confirms it).
    * Free tier: Google states that content is used to improve its products (Gemini API pricing page,
      30 Oct 2026); the app's privacy note says so. Free-tier rate limits apply (HTTP 429 -> EasyOCR).
    * The key travels in the x-goog-api-key header, never in the URL or in an error message.
"""
from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from .text_cleaning import clean_text
from .train_numbers import split_train_numbers

ENV_KEY = "GEMINI_API_KEY"
ENV_MODEL = "GEMINI_OCR_MODEL"
DEFAULT_MODEL = "gemini-3.6-flash"     # stable, multimodal, free tier (ai.google.dev, 30 Oct 2026)
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
MAX_SIDE = 1600
TIMEOUT_S = 30.0

PROMPT = """You are an OCR engine for photos of Indian railway station signs.
Copy every piece of text that is printed on the sign, exactly as written, one line of the sign per item,
in its original script (Devanagari, Bengali, Kannada, Telugu, Latin, digits). Rules:
- Do not translate, transliterate, correct spelling or complete words.
- Do not guess. If a word is not readable, leave it out.
- Leave out watermarks and photo credits (for example "Adobe Stock", "Shutterstock", "Alamy", image numbers).
- If there is no readable text, return an empty list."""

SCHEMA = {"type": "OBJECT",
          "properties": {"lines": {"type": "ARRAY", "items": {"type": "STRING"}},
                         "scripts": {"type": "ARRAY", "items": {"type": "STRING"}}},
          "required": ["lines"]}


class GeminiOCRError(RuntimeError):
    """Gemini could not read the photo (network, key, rate limit, blocked or bad answer). Safe to log."""


def available() -> bool:
    return bool(os.environ.get(ENV_KEY, "").strip())


def model_name() -> str:
    return os.environ.get(ENV_MODEL, "").strip() or DEFAULT_MODEL


def _jpeg(image_bytes: bytes) -> bytes:
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    out = io.BytesIO()
    img.save(out, "JPEG", quality=92)
    return out.getvalue()


def _post(url: str, body: Dict[str, Any], key: str, timeout: float) -> Dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:          # 400 bad key / 403 / 404 model / 429 rate limit
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("error", {}).get("message", "")
        except Exception:
            detail = ""
        raise GeminiOCRError(f"Gemini API HTTP {exc.code}: {detail[:200]}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise GeminiOCRError(f"Gemini API unreachable: {type(exc).__name__}") from None


def parse(response: Dict[str, Any], model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """One generateContent answer -> an OCR result dict like src.ocr.ocr_engine's."""
    block = (response.get("promptFeedback") or {}).get("blockReason")
    if block:
        raise GeminiOCRError(f"Gemini refused the photo ({block})")
    cands = response.get("candidates") or []
    if not cands:
        raise GeminiOCRError("Gemini returned no answer")
    parts = (cands[0].get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts).strip()
    if not text:
        raise GeminiOCRError(f"Gemini returned no text (finish reason {cands[0].get('finishReason')})")
    try:
        data = json.loads(text)
        lines = [str(x).strip() for x in data.get("lines", []) if str(x).strip()]
    except (ValueError, AttributeError):
        raise GeminiOCRError("Gemini's answer was not in the expected format") from None
    raw = "\n".join(lines)
    clean = clean_text(raw)
    numbers, station_text = split_train_numbers(clean)
    ok = bool(clean.strip())
    scripts = [str(s) for s in data.get("scripts", []) if str(s).strip()] if isinstance(data, dict) else []
    return {
        "status": "ok" if ok else "OCR failed", "text": clean, "raw_text": raw, "clean_text": clean,
        "ocr_confidence": None, "low_confidence": None, "ocr_language": "gemini",
        "ocr_backend": "gemini", "ocr_model": model, "preprocessing_method": None,
        "train_numbers": numbers if ok else [], "station_text": station_text if ok else "",
        "error": None if ok else "Gemini found no readable text in the photo.", "warnings": [],
        "detected_languages": scripts,
    }


def recognize(image_bytes: bytes, key: Optional[str] = None, model: Optional[str] = None,
              timeout: float = TIMEOUT_S) -> Dict[str, Any]:
    """Read the text of a photo with Gemini. Raises GeminiOCRError on any failure."""
    key = (key or os.environ.get(ENV_KEY, "")).strip()
    if not key:
        raise GeminiOCRError("no Gemini API key configured")
    model = model or model_name()
    try:
        content = base64.b64encode(_jpeg(image_bytes)).decode("ascii")
    except Exception as exc:
        raise GeminiOCRError(f"photo could not be prepared: {type(exc).__name__}") from None
    body = {"contents": [{"parts": [{"text": PROMPT},
                                    {"inline_data": {"mime_type": "image/jpeg", "data": content}}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                                 "responseSchema": SCHEMA}}
    return parse(_post(ENDPOINT.format(model=model), body, key, timeout), model)
