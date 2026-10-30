"""Google Cloud Vision OCR: request shape, answer parsing and failures, without calling Google."""
import io
import json
import urllib.error

import pytest
from PIL import Image

from src.ocr import google_vision as gv


def _photo(w=3000, h=2000):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, "JPEG")
    return buf.getvalue()


ANSWER = {"responses": [{"textAnnotations": [{"description": "रत्नागिरी\nRATNAGIRI 12229\n"}],
                         "fullTextAnnotation": {
                             "text": "रत्नागिरी\nRATNAGIRI 12229\n",
                             "pages": [{"property": {"detectedLanguages": [{"languageCode": "hi"},
                                                                         {"languageCode": "en"}]},
                                        "blocks": [{"paragraphs": [{"words": [{"confidence": 0.98},
                                                                              {"confidence": 0.94}]}]}]}]}}]}


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_request_uses_header_key_hints_and_a_resized_photo(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout):
        seen["url"], seen["headers"], seen["body"] = req.full_url, dict(req.header_items()), json.loads(req.data)
        return _Resp(json.dumps(ANSWER).encode())
    monkeypatch.setattr(gv.urllib.request, "urlopen", fake_urlopen)
    r = gv.recognize(_photo(), key="SECRET-KEY")
    assert "SECRET-KEY" not in seen["url"] and seen["headers"]["X-goog-api-key"] == "SECRET-KEY"
    req = seen["body"]["requests"][0]
    assert req["features"] == [{"type": "TEXT_DETECTION"}] and "hi" in req["imageContext"]["languageHints"]
    sent = Image.open(io.BytesIO(__import__("base64").b64decode(req["image"]["content"])))
    assert max(sent.size) <= gv.MAX_SIDE
    assert r["status"] == "ok" and r["ocr_backend"] == "google"
    assert "रत्नागिरी" in r["clean_text"] and r["train_numbers"] == ["12229"]
    assert r["ocr_confidence"] == 96.0 and r["detected_languages"] == ["en", "hi"]


def test_empty_answer_is_a_failed_reading():
    r = gv.parse({"responses": [{}]})
    assert r["status"] == "OCR failed" and r["ocr_confidence"] is None and r["clean_text"] == ""


def test_api_errors_raise_without_leaking_the_key(monkeypatch):
    def denied(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {},
                                     io.BytesIO(b'{"error": {"message": "Cloud Vision API has not been used"}}'))
    monkeypatch.setattr(gv.urllib.request, "urlopen", denied)
    with pytest.raises(gv.GoogleOCRError) as e:
        gv.recognize(_photo(400, 300), key="SECRET-KEY")
    assert "403" in str(e.value) and "SECRET-KEY" not in str(e.value)
    with pytest.raises(gv.GoogleOCRError, match="Vision API error"):
        gv.parse({"responses": [{"error": {"message": "bad image"}}]})


def test_no_key_means_not_available(monkeypatch):
    monkeypatch.delenv(gv.ENV_KEY, raising=False)
    assert not gv.available()
    with pytest.raises(gv.GoogleOCRError, match="no Google Vision API key"):
        gv.recognize(_photo(400, 300))
