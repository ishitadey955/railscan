"""CNN model file / label consistency on real held-out images, and EasyOCR result handling."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.ocr.ocr_engine as oe
from src.ocr import OCREngine, STATUS_FAILED, STATUS_OK

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "language_classifier.keras"
LABELS = ROOT / "models" / "language_labels.json"
META = ROOT / "data" / "processed" / "RailScan_CNN_Metadata.csv"


@pytest.mark.skipif(not MODEL.exists(), reason="CNN not trained")
def test_model_input_shape_matches_label_file():
    pytest.importorskip("keras")
    from src.image.prediction import load_model
    model, classes, size = load_model()
    assert model.input_shape == (None, size, size, 3)
    assert model.output_shape == (None, len(classes))
    assert classes == json.loads(LABELS.read_text())["classes"]


@pytest.mark.skipif(not MODEL.exists(), reason="CNN not trained")
def test_predict_script_on_real_test_images():
    pytest.importorskip("keras")
    meta = pd.read_csv(META)
    paths = meta[meta.split == "test"].groupby("script_label").head(1)
    if paths.empty or not (ROOT / paths.image_path.iat[0]).exists():
        pytest.skip("signboard images not downloaded (python scripts/download_datasets.py)")
    from src.image.prediction import predict_script
    for p in paths.image_path:
        out = predict_script(ROOT / p)
        assert out["predicted_script"] in out["class_probabilities"]
        assert sum(out["class_probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
        assert out["low_confidence"] == (out["confidence"] < 0.60)


def test_cnn_metadata_splits_are_disjoint_and_stratified():
    meta = pd.read_csv(META)
    assert meta.sha1.is_unique                      # duplicates removed before splitting
    assert set(meta.split) == {"train", "validation", "test"}
    for lang, g in meta.groupby("language_label"):
        assert (g.split == "test").sum() > 0 and (g.split == "validation").sum() > 0, lang


# ------------------------------------------------------------ EasyOCR handling (stub reader)
class _Reader:
    def __init__(self, boxes):
        self.boxes = boxes

    def detect(self, image, reformat=True):
        return [[]], [[]]

    def recognize(self, grey, horizontal, free, detail=1, paragraph=False, reformat=True):
        return self.boxes


def _box(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _engine(monkeypatch, boxes):
    monkeypatch.setattr(oe, "easyocr", object())          # pretend the package is installed
    monkeypatch.setattr(oe, "_easyocr_reader", lambda langs: _Reader(boxes))
    return OCREngine()


def test_easyocr_boxes_grouped_into_lines(monkeypatch):
    img = np.full((120, 400, 3), 255, np.uint8)
    boxes = [(_box(200, 10, 300, 40), "DELHI", 0.9), (_box(10, 12, 150, 42), "NEW", 0.8),
             (_box(10, 70, 120, 100), "12301", 0.7)]
    r = _engine(monkeypatch, boxes).recognize(img, script="Hindi")
    assert r["status"] == STATUS_OK and r["ocr_backend"] == "easyocr"
    assert r["raw_text"] == "NEW DELHI\n12301"
    assert r["ocr_confidence"] == pytest.approx(80.0)
    assert r["train_numbers"] == ["12301"]


def test_easyocr_empty_result_is_a_clean_failure(monkeypatch):
    img = np.full((120, 400, 3), 255, np.uint8)
    r = _engine(monkeypatch, []).recognize(img, script="Kannada")
    assert r["status"] == STATUS_FAILED and r["text"] == "" and r["train_numbers"] == []


def test_easyocr_missing_package_reported(monkeypatch):
    monkeypatch.setattr(oe, "easyocr", None)
    eng = OCREngine()
    r = eng.recognize(np.full((60, 200, 3), 255, np.uint8), script="Telugu")
    assert r["status"] == STATUS_FAILED and "EasyOCR is not installed" in r["error"]


# ------------------------------------------------------------ CNN + OCR evidence (validated combination)
def test_ocr_evidence_overrules_an_unsure_cnn():
    from src.script_evidence import fuse
    cnn = {"Devanagari": 0.42, "Kannada": 0.47, "Telugu": 0.11}
    out = fuse(cnn, {"Devanagari": {"confidence": 44, "native_share": 0.5},
                     "Kannada": {"confidence": 5, "native_share": 1.0},
                     "Telugu": {"confidence": 0, "native_share": 0}}, ocr_weight=35)
    assert out["script"] == "Devanagari" and out["confidence"] > 0.9 and not out["low_confidence"]
    assert abs(sum(out["probabilities"].values()) - 1) < 1e-9


def test_without_ocr_evidence_the_cnn_result_is_unchanged():
    from src.script_evidence import fuse
    cnn = {"Devanagari": 0.2, "Kannada": 0.7, "Telugu": 0.1}
    out = fuse(cnn, None)
    assert out["script"] == "Kannada" and abs(out["confidence"] - 0.7) < 1e-9


def test_native_share_counts_only_the_scripts_letters():
    from src.script_evidence import native_share
    assert native_share("नई दिल्ली", "Devanagari") == 1.0
    assert native_share("NEW DELHI", "Devanagari") == 0.0
    assert native_share("", "Kannada") == 0.0
    assert 0 < native_share("नई NEW", "Devanagari") < 1


def test_identify_script_reads_every_script_and_returns_the_chosen_reading():
    from src.script_evidence import SCRIPTS, identify_script

    class Engine:
        def __init__(self):
            self.langs = []

        def recognize(self, image, languages=None):
            self.langs.append(languages)
            text = {"eng+hin+mar": "नई दिल्ली"}.get(languages, "")
            return {"status": "ok" if text else "OCR failed", "clean_text": text,
                    "ocr_confidence": 60.0 if text else None, "ocr_language": languages}

    eng = Engine()
    predict = lambda img: {"class_probabilities": {"Devanagari": 0.3, "Kannada": 0.6, "Telugu": 0.1}}
    r = identify_script(b"img", predict, eng)
    assert len(eng.langs) == len(SCRIPTS)                     # one reading per script
    assert r["combined"]["script"] == "Devanagari"
    assert r["ocr"]["clean_text"] == "नई दिल्ली" and r["ocr"]["script_source"] == "cnn+ocr"
