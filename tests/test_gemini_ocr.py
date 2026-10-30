"""Gemini OCR: request shape, answer parsing and failures, without calling Google."""
import base64
import io
import json
import urllib.error

import pytest
from PIL import Image

from src.ocr import gemini_ocr as gm


def _photo(w=2400, h=1800):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, "JPEG")
    return buf.getvalue()


def _answer(payload):
    return {"candidates": [{"content": {"parts": [{"text": json.dumps(payload, ensure_ascii=False)}]},
                            "finishReason": "STOP"}]}


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_request_shape_and_parsed_reading(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout):
        seen["url"], seen["headers"], seen["body"] = req.full_url, dict(req.header_items()), json.loads(req.data)
        return _Resp(json.dumps(_answer({"lines": ["रत्नागिरी", "RATNAGIRI", "12229"],
                                         "scripts": ["Devanagari", "Latin"]})).encode())
    monkeypatch.setattr(gm.urllib.request, "urlopen", fake_urlopen)
    r = gm.recognize(_photo(), key="SECRET-KEY", model="gemini-test")
    assert seen["url"].endswith("/models/gemini-test:generateContent") and "SECRET-KEY" not in seen["url"]
    assert seen["headers"]["X-goog-api-key"] == "SECRET-KEY"
    cfg = seen["body"]["generationConfig"]
    assert cfg["temperature"] == 0 and cfg["responseMimeType"] == "application/json"
    parts = seen["body"]["contents"][0]["parts"]
    assert "Do not guess" in parts[0]["text"]
    sent = Image.open(io.BytesIO(base64.b64decode(parts[1]["inline_data"]["data"])))
    assert max(sent.size) <= gm.MAX_SIDE
    assert r["status"] == "ok" and r["ocr_backend"] == "gemini" and r["ocr_confidence"] is None
    assert r["clean_text"].splitlines()[:2] == ["रत्नागिरी", "RATNAGIRI"] and r["train_numbers"] == ["12229"]


def test_no_text_is_a_failed_reading():
    r = gm.parse(_answer({"lines": []}))
    assert r["status"] == "OCR failed" and r["clean_text"] == ""


def test_bad_answers_and_refusals_raise():
    with pytest.raises(gm.GeminiOCRError, match="refused"):
        gm.parse({"promptFeedback": {"blockReason": "SAFETY"}})
    with pytest.raises(gm.GeminiOCRError, match="no answer"):
        gm.parse({"candidates": []})
    with pytest.raises(gm.GeminiOCRError, match="expected format"):
        gm.parse({"candidates": [{"content": {"parts": [{"text": "not json"}]}}]})


def test_rate_limit_raises_without_the_key(monkeypatch):
    def limited(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {},
                                     io.BytesIO(b'{"error": {"message": "Resource has been exhausted"}}'))
    monkeypatch.setattr(gm.urllib.request, "urlopen", limited)
    with pytest.raises(gm.GeminiOCRError) as e:
        gm.recognize(_photo(400, 300), key="SECRET-KEY")
    assert "429" in str(e.value) and "SECRET-KEY" not in str(e.value)


def test_model_can_be_changed_by_setting(monkeypatch):
    monkeypatch.setenv(gm.ENV_MODEL, "gemini-other")
    assert gm.model_name() == "gemini-other"
    monkeypatch.delenv(gm.ENV_MODEL)
    assert gm.model_name() == gm.DEFAULT_MODEL
