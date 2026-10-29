"""Tests for the RailScan OCR component (src/ocr).

Run from the project root:   python -m pytest tests/test_ocr.py -v

Three kinds of tests
* pure unit tests (cleaning, train numbers, preprocessing, language mapping)
  - always run
* mocked-OCR tests (engine control flow, EasyOCR call replaced) - always run
* real OCR tests - render known text to an image with a system font and run
  the real EasyOCR. They are SKIPPED, with a reason, when easyocr or a font
  for the script is not installed. The first run downloads EasyOCR's models. Set RAILSCAN_FONT_DIR to a folder with e.g. NotoSansDevanagari.ttf,
  NotoSansKannada.ttf, NotoSansTelugu.ttf to point the tests at your fonts.

The real-OCR tests compare against the exact string that was rendered (so the
ground truth is known) and check text similarity. They do not assert or
publish any accuracy figure: synthetic images say nothing about real photos
of railway boards.
"""
from __future__ import annotations

import difflib
import glob
import os
import sys
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ocr import OCRConfig, OCREngine, STATUS_FAILED, STATUS_OK, language_candidates  # noqa: E402
from src.ocr import ocr_engine as oe  # noqa: E402
from src.ocr import preprocessing as pp  # noqa: E402
from src.ocr.text_cleaning import (  # noqa: E402
    clean_text, cleanup_punctuation, normalize_unicode, normalize_whitespace, remove_duplicate_lines,
)
from src.ocr.train_numbers import extract_train_numbers, find_train_numbers, split_train_numbers  # noqa: E402

RESULT_KEYS = {
    "status", "text", "raw_text", "clean_text", "ocr_confidence", "ocr_language",
    "preprocessing_method", "train_numbers", "station_text", "warnings", "error",
}
ALL_LANGS = {"eng", "hin", "kan", "mar", "tel"}


# --------------------------------------------------------------------------
# Helpers: fonts, rendering, similarity
# --------------------------------------------------------------------------
_FONT_PATTERNS = {
    "latin": ["DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "arialbd.ttf", "arial.ttf", "Arial.ttf", "LiberationSans-Bold.ttf"],
    "devanagari": ["NotoSansDevanagari*.ttf", "NotoSansDevanagari*.otf", "Nirmala*.ttf", "Nirmala*.ttc", "mangal*.ttf",
                   "Lohit-Devanagari*.ttf", "Devanagari Sangam MN*.ttc"],
    "kannada": ["NotoSansKannada*.ttf", "NotoSansKannada*.otf", "Nirmala*.ttf", "Nirmala*.ttc", "tunga*.ttf",
                "Lohit-Kannada*.ttf", "Kannada Sangam MN*.ttc"],
    "telugu": ["NotoSansTelugu*.ttf", "NotoSansTelugu*.otf", "Nirmala*.ttf", "Nirmala*.ttc", "gautami*.ttf",
               "Lohit-Telugu*.ttf", "Telugu Sangam MN*.ttc"],
}
_FONT_DIRS = ["/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts"),
              os.path.expanduser("~/.local/share/fonts"), "C:/Windows/Fonts",
              "/System/Library/Fonts", "/System/Library/Fonts/Supplemental", "/Library/Fonts"]


def find_font(kind: str):
    """Path of a usable font for 'latin'|'devanagari'|'kannada'|'telugu', else None."""
    dirs = ([os.environ["RAILSCAN_FONT_DIR"]] if os.environ.get("RAILSCAN_FONT_DIR") else []) + _FONT_DIRS
    for pattern in _FONT_PATTERNS[kind]:
        for d in dirs:
            hits = sorted(glob.glob(os.path.join(d, "**", pattern), recursive=True))
            if hits:
                return hits[0]
    return None


def render(lines, size=72, fg=(0, 0, 0), bg=(255, 255, 255), pad=30):
    """Render [(text, font_kind), ...] as an RGB PIL image (needs libraqm for Indic shaping)."""
    from PIL import Image, ImageDraw, ImageFont
    fonts = []
    for _, kind in lines:
        path = find_font(kind)
        fonts.append(ImageFont.truetype(path, size) if path else ImageFont.load_default(size=size))
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    width = max(int(probe.textlength(t, font=f)) for (t, _), f in zip(lines, fonts)) + 2 * pad
    line_h = int(size * 1.6)
    img = Image.new("RGB", (width, line_h * len(lines) + 2 * pad), bg)
    draw = ImageDraw.Draw(img)
    for i, ((text, _), font) in enumerate(zip(lines, fonts)):
        draw.text((pad, pad + i * line_h), text, font=font, fill=fg)
    return img


def similarity(a: str, b: str) -> float:
    norm = lambda s: unicodedata.normalize("NFC", "".join(s.split())).casefold()
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def assert_valid_result(r: dict):
    assert RESULT_KEYS <= set(r), f"missing keys: {RESULT_KEYS - set(r)}"
    assert r["status"] in (STATUS_OK, STATUS_FAILED)
    assert r["text"] == r["clean_text"]
    assert isinstance(r["train_numbers"], list)
    if r["status"] == STATUS_FAILED:
        assert r["text"] == ""
        assert r["ocr_confidence"] is None  # never a made-up number
    else:
        assert r["text"].strip()
        assert 0.0 <= r["ocr_confidence"] <= 100.0
        assert r["ocr_language"] and r["preprocessing_method"] in pp.STRATEGIES


@pytest.fixture(scope="module")
def real_engine():
    engine = OCREngine()
    if not engine.available_languages():
        pytest.skip("easyocr is not installed (pip install -r requirements.txt)")
    return engine


def require(engine, *langs):
    missing = [l for l in langs if l not in engine.available_languages()]
    if missing:
        pytest.skip(f"OCR languages not available: {', '.join(missing)}")


def require_font(*kinds):
    from PIL import features
    if not features.check("raqm"):
        pytest.skip("Pillow was built without libraqm: Indic text cannot be shaped for the test images")
    for k in kinds:
        if find_font(k) is None:
            pytest.skip(f"no {k} font found (set RAILSCAN_FONT_DIR)")


def mocked_engine(responses=None, predictor=None, langs=ALL_LANGS, config=None):
    """Engine whose EasyOCR call is replaced. ``responses``: callable(lang, _, call_no) -> (text, conf)."""
    engine = OCREngine(config, script_predictor=predictor)
    engine._langs_cache = set(langs)
    calls = []

    def fake(image, lang):
        calls.append(lang)
        return responses(lang, None, len(calls)) if responses else ("", None)

    engine._run_easyocr = fake
    engine.calls = calls
    return engine


@pytest.fixture
def sign_image():
    return render([("NEW DELHI", "latin")], size=60)


# --------------------------------------------------------------------------
# Real OCR: Hindi / Marathi / Kannada / Telugu / English+bilingual
# --------------------------------------------------------------------------
@pytest.mark.parametrize("script,font,text,lang_pack", [
    ("Hindi", "devanagari", "नई दिल्ली", "hin"),
    ("Marathi", "devanagari", "पुणे जंक्शन", "mar"),
    ("Kannada", "kannada", "ಬೆಂಗಳೂರು", "kan"),
    ("Telugu", "telugu", "హైదరాబాద్", "tel"),
], ids=["hindi", "marathi", "kannada", "telugu"])
def test_indic_script_ocr(real_engine, script, font, text, lang_pack):
    require(real_engine, "eng", lang_pack)
    require_font(font)
    r = real_engine.recognize(render([(text, font)]), script=script)
    assert_valid_result(r)
    assert r["status"] == STATUS_OK, r
    assert lang_pack in r["ocr_language"].split("+")
    assert r["script"] == script and r["script_source"] == "override"
    assert similarity(r["clean_text"], text) >= 0.8, (r["clean_text"], text)


def test_english_ocr(real_engine):
    require(real_engine, "eng")
    r = real_engine.recognize(render([("HOWRAH JUNCTION", "latin")]), script="English")
    assert_valid_result(r)
    assert r["ocr_language"] == "eng"
    assert similarity(r["clean_text"], "HOWRAH JUNCTION") >= 0.85


def test_bilingual_english_hindi_sign(real_engine):
    require(real_engine, "eng", "hin")
    require_font("devanagari")
    r = real_engine.recognize(render([("NEW DELHI", "latin"), ("नई दिल्ली", "devanagari")]), script="Hindi")
    assert_valid_result(r)
    assert r["ocr_language"] == "eng+hin"
    assert "NEW DELHI" in r["clean_text"].upper()
    assert similarity(r["clean_text"], "NEW DELHI\nनई दिल्ली") >= 0.8


def test_train_number_in_real_ocr_is_separated_from_station_text(real_engine):
    require(real_engine, "eng")
    r = real_engine.recognize(render([("12301 HOWRAH RAJDHANI", "latin")]), script="English")
    assert r["status"] == STATUS_OK
    assert r["train_numbers"] == ["12301"]
    assert "12301" not in r["station_text"] and "HOWRAH" in r["station_text"].upper()


def test_light_text_on_dark_board_is_handled(real_engine):
    require(real_engine, "eng", "hin")
    require_font("devanagari")
    img = render([("बेंगलुरु जंक्शन", "devanagari")], fg=(255, 255, 255), bg=(10, 60, 140))
    r = real_engine.recognize(img, script="Hindi")
    assert r["status"] == STATUS_OK
    assert similarity(r["clean_text"], "बेंगलुरु जंक्शन") >= 0.8


# --------------------------------------------------------------------------
# Empty OCR / poor image / unreadable input: valid result, no crash
# --------------------------------------------------------------------------
def test_empty_ocr_real_blank_image(real_engine):
    require(real_engine, "eng", "hin")
    r = real_engine.recognize(np.full((200, 600, 3), 255, np.uint8), script="Hindi")
    assert_valid_result(r)
    assert r["status"] == "OCR failed" and r["text"] == "" and r["ocr_confidence"] is None
    assert r["train_numbers"] == [] and r["station_text"] == ""


def test_empty_ocr_mocked_returns_failed_not_exception(sign_image):
    eng = mocked_engine(lambda l, p, n: ("", None))
    r = eng.recognize(sign_image, script="Hindi")
    assert_valid_result(r)
    assert r["status"] == "OCR failed" and r["text"] == ""


def test_whitespace_and_symbol_only_output_is_a_failure(sign_image):
    eng = mocked_engine(lambda l, p, n: (" | ~~ \n -- ", 80.0))
    r = eng.recognize(sign_image, script="Hindi")
    assert r["status"] == "OCR failed" and r["text"] == ""


def test_poor_image_never_crashes_and_never_fakes_confidence(real_engine):
    """Heavy blur + noise + low contrast. Either a valid failure or an
    honest (possibly low-confidence) result; never an exception or a
    made-up confidence."""
    require(real_engine, "eng")
    base = np.array(render([("PUNE JN", "latin")], size=40))[:, :, ::-1]
    rng = np.random.default_rng(1)
    poor = cv2.GaussianBlur(base, (0, 0), 9)
    poor = np.clip(poor.astype(float) * 0.25 + 90 + rng.normal(0, 40, poor.shape), 0, 255).astype(np.uint8)
    r = real_engine.recognize(poor, script="English", return_attempts=True)
    assert_valid_result(r)
    assert isinstance(r["attempts"], list)


@pytest.mark.parametrize("bad", [None, "/no/such/file.jpg", np.zeros((0, 0, 3), np.uint8), b"not an image", 12345],
                         ids=["none", "missing-file", "empty-array", "bytes", "int"])
def test_unreadable_input_returns_failed_result(bad):
    eng = mocked_engine(lambda l, p, n: ("x", 99.0))
    r = eng.recognize(bad, script="Hindi")
    assert_valid_result(r)
    assert r["status"] == STATUS_FAILED and r["error"]
    assert eng.calls == []  # OCR never invoked


def test_corrupt_image_file(tmp_path):
    f = tmp_path / "broken.jpg"
    f.write_bytes(b"\xff\xd8\xff garbage")
    r = mocked_engine().recognize(f, script="Hindi")
    assert r["status"] == STATUS_FAILED and r["error"]


def test_easyocr_unavailable_is_reported_not_raised(sign_image, monkeypatch):
    eng = OCREngine()
    eng._langs_cache = set(ALL_LANGS)

    def boom(*a, **k):
        raise oe.OCRBackendUnavailable("easyocr models not found")

    monkeypatch.setattr(eng, "_run_easyocr", boom)
    r = eng.recognize(sign_image, script="Hindi")
    assert r["status"] == STATUS_FAILED and "not found" in r["error"]


def test_no_language_data_installed_is_reported(sign_image):
    eng = mocked_engine(lambda l, p, n: ("x", 99.0), langs={"eng"})
    r = eng.recognize(sign_image, script="Kannada")
    assert r["status"] == STATUS_FAILED and "kan" in r["error"]
    assert eng.calls == []


def test_ocr_runtime_error_in_one_attempt_does_not_stop_others(sign_image):
    def responses(lang, psm, n):
        if n == 1:
            raise RuntimeError("OCR process timeout")
        return ("नई दिल्ली", 91.0)

    eng = mocked_engine(responses)
    r = eng.recognize(sign_image, script="Hindi")
    assert r["status"] == STATUS_OK and r["ocr_confidence"] == 91.0
    assert any("timeout" in w for w in r["warnings"])


# --------------------------------------------------------------------------
# Script -> language selection, CNN integration (stubbed predictor)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("script,expected", [
    ("Hindi", "eng+hin"), ("Kannada", "eng+kan"), ("Marathi", "eng+mar"), ("Telugu", "eng+tel"), ("English", "eng"),
])
def test_language_mapping_bilingual(script, expected):
    assert language_candidates(script) == [expected]


def test_language_mapping_modes_and_codes():
    assert language_candidates("Hindi", "native") == ["hin"]
    assert language_candidates("kannada", "native") == ["kan"]
    assert language_candidates("mar", "both") == ["eng+mar", "mar"]
    assert oe.SCRIPT_TO_LANG == {"English": "eng", "Hindi": "hin", "Kannada": "kan", "Marathi": "mar",
                                 "Telugu": "tel", "Devanagari": "hin+mar"}
    # the CNN predicts the Devanagari script; OCR then tries both Devanagari languages
    assert language_candidates("Devanagari") == ["eng+hin+mar"]
    assert oe.easyocr_languages("eng+hin+mar") == ["hi", "mr", "en"]
    assert oe.easyocr_languages("eng+kan") == ["kn", "en"]
    with pytest.raises(ValueError):
        language_candidates("Klingon")


@pytest.mark.parametrize("script,lang", [("Hindi", "eng+hin"), ("Kannada", "eng+kan"),
                                         ("Marathi", "eng+mar"), ("Telugu", "eng+tel")])
def test_cnn_script_selects_ocr_language(sign_image, script, lang):
    def predictor(image, threshold=0.6):
        return {"script": script, "predicted_script": script, "confidence": 0.97,
                "class_probabilities": {script: 0.97}, "low_confidence": False}

    eng = mocked_engine(lambda l, p, n: ("स्टेशन", 90.0), predictor)
    r = eng.recognize(sign_image)
    assert r["script"] == script and r["script_source"] == "cnn" and r["script_confidence"] == 0.97
    assert r["ocr_language"] == lang
    assert set(eng.calls) == {lang}


def test_explicit_language_override_skips_cnn(sign_image):
    def predictor(*a, **k):
        raise AssertionError("CNN must not be called")

    eng = mocked_engine(lambda l, p, n: ("ABC", 90.0), predictor)
    r = eng.recognize(sign_image, languages="eng+tel")
    assert r["ocr_language"] == "eng+tel" and r["script_source"] == "language_override"


def test_ambiguous_cnn_tries_runner_up_and_picks_best_scoring(sign_image):
    """Hindi/Marathi look alike: both are tried when the CNN is unsure."""
    def predictor(image, threshold=0.6):
        return {"predicted_script": "Hindi", "script": None, "confidence": 0.52, "low_confidence": True,
                "class_probabilities": {"Hindi": 0.52, "Marathi": 0.41, "Kannada": 0.05, "Telugu": 0.02}}

    def responses(lang, psm, n):
        return ("पुणे जंक्शन", 93.0) if lang == "eng+mar" else ("पुण जक्शन", 70.0)

    eng = mocked_engine(responses, predictor)
    r = eng.recognize(sign_image)
    assert r["script"] == "Hindi" and r["script_low_confidence"] is True
    assert {"eng+hin", "eng+mar"} <= set(eng.calls)
    assert "eng+kan" not in eng.calls
    assert r["ocr_language"] == "eng+mar" and r["clean_text"] == "पुणे जंक्शन"


def test_cnn_unavailable_falls_back_and_warns(sign_image):
    def broken(*a, **k):
        raise ImportError("No module named 'src.image.prediction'")

    eng = mocked_engine(lambda l, p, n: ("ABC", 90.0) if l == "eng" else ("", None), broken)
    r = eng.recognize(sign_image)
    assert r["status"] == STATUS_OK and r["script_source"] == "unavailable"
    assert any("script detection unavailable" in w for w in r["warnings"])


def test_missing_language_pack_only_drops_that_candidate(sign_image):
    def predictor(image, threshold=0.6):
        return {"predicted_script": "Hindi", "confidence": 0.5, "low_confidence": True,
                "class_probabilities": {"Hindi": 0.5, "Kannada": 0.4}}

    eng = mocked_engine(lambda l, p, n: ("स्टेशन", 88.0), predictor, langs={"eng", "hin"})
    r = eng.recognize(sign_image)
    assert r["ocr_language"] == "eng+hin"
    assert any("kan" in w for w in r["warnings"])


# --------------------------------------------------------------------------
# Multiple preprocessing strategies: none is assumed to always work
# --------------------------------------------------------------------------
def test_engine_tries_next_strategy_when_first_is_weak(sign_image):
    seen = []

    def responses(lang, psm, n):
        seen.append(n)
        return ("स्टेशन", 40.0) if n < 3 else ("स्टेशन", 92.0)

    eng = mocked_engine(responses, config=OCRConfig(max_strategies=3))
    r = eng.recognize(sign_image, script="Hindi", return_attempts=True)
    assert len(r["attempts"]) == 3
    assert r["ocr_confidence"] == 92.0
    assert r["preprocessing_method"] == r["attempts"][2]["preprocessing_method"]
    assert len({a["preprocessing_method"] for a in r["attempts"]}) == 3


def test_default_tries_at_most_two_preprocessing_variants_per_language(sign_image):
    """EasyOCR takes seconds per call on CPU, so the default caps the variants."""
    eng = mocked_engine(lambda l, p, n: ("स्टेशन", 40.0))
    r = eng.recognize(sign_image, script="Hindi", return_attempts=True)
    assert len(r["attempts"]) == 2 and r["ocr_confidence"] == 40.0


def test_engine_keeps_best_when_none_reach_early_stop(sign_image):
    confs = [55.0, 78.0, 61.0]
    eng = mocked_engine(lambda l, p, n: ("स्टेशन", confs[(n - 1) % 3]), config=OCRConfig(max_strategies=3))
    r = eng.recognize(sign_image, script="Hindi", return_attempts=True)
    assert r["ocr_confidence"] == 78.0
    assert r["low_confidence"] is False
    assert len(r["attempts"]) >= 3


def test_low_confidence_is_flagged_not_hidden(sign_image):
    eng = mocked_engine(lambda l, p, n: ("स्टेशन", 45.0))
    r = eng.recognize(sign_image, script="Hindi")
    assert r["status"] == STATUS_OK and r["ocr_confidence"] == 45.0 and r["low_confidence"] is True


def test_confidence_below_noise_floor_is_a_failure(sign_image):
    eng = mocked_engine(lambda l, p, n: ("xq zv", 12.0))
    r = eng.recognize(sign_image, script="Hindi")
    assert r["status"] == STATUS_FAILED and r["ocr_confidence"] is None


def test_forced_strategy_list_is_respected(sign_image):
    eng = mocked_engine(lambda l, p, n: ("ABC", 10.0), config=OCRConfig(strategies=["otsu_threshold", "upscale"]))
    r = eng.recognize(sign_image, script="English", return_attempts=True)
    assert [a["preprocessing_method"] for a in r["attempts"]][:2] == ["otsu_threshold", "upscale"]


def test_all_strategies_produce_valid_images():
    img = pp.load_image(np.array(render([("PUNE", "latin")], size=60))[:, :, ::-1])
    assert set(pp.STRATEGIES) >= {"original", "grayscale", "denoise", "contrast", "adaptive_threshold",
                                  "otsu_threshold", "upscale", "deskew"}
    for name in pp.STRATEGIES:
        out = pp.apply_strategy(img, name)
        assert out.dtype == np.uint8 and out.ndim in (2, 3) and out.size > 0, name
    assert pp.apply_strategy(img, "original").shape == img.shape
    assert pp.apply_strategy(img, "grayscale").ndim == 2
    assert pp.apply_strategy(img, "upscale").shape[0] > img.shape[0]
    assert set(np.unique(pp.apply_strategy(img, "otsu_threshold"))) <= {0, 255}
    assert set(np.unique(pp.apply_strategy(img, "adaptive_threshold"))) <= {0, 255}
    with pytest.raises(KeyError):
        pp.apply_strategy(img, "magic")


def test_threshold_strategies_make_text_dark_on_light_for_dark_boards():
    img = np.array(render([("PUNE", "latin")], size=60, fg=(255, 255, 255), bg=(10, 40, 120)))[:, :, ::-1]
    for name in ("otsu_threshold", "adaptive_threshold", "inverted"):
        out = pp.apply_strategy(img, name)
        assert out.mean() > 127, f"{name}: background should end up light"


def test_strategy_order_adapts_to_image():
    dark = pp.analyze_image(np.array(render([("PUNE", "latin")], fg=(255, 255, 255), bg=(10, 40, 120)))[:, :, ::-1])
    assert dark.dark_background
    order = pp.select_strategies(dark, deskew_mode=False)
    assert order[0] == "inverted" and "deskew" not in order
    tiny = pp.analyze_image(np.full((40, 120, 3), 255, np.uint8))
    assert pp.select_strategies(tiny, False)[0] == "upscale"
    assert pp.select_strategies(tiny, True)[0] == "deskew"          # forced on
    assert pp.select_strategies(tiny, "auto")[0] != "deskew"        # level image: not needed
    assert pp.select_strategies(tiny, False, allowed=["contrast"]) == ["contrast"]


def test_optional_deskew_levels_a_tilted_image():
    img = np.array(render([("HOWRAH JUNCTION STATION", "latin")], size=60))[:, :, ::-1]
    tilted = pp.rotate_image(img, 6.0)
    angle = pp.estimate_skew_angle(tilted)
    assert abs(abs(angle) - 6.0) <= 1.0, angle
    fixed = pp.apply_strategy(tilted, "deskew")
    assert abs(pp.estimate_skew_angle(fixed)) <= 1.0
    assert pp.estimate_skew_angle(img) == 0.0


def test_load_image_accepts_path_pil_and_array(tmp_path):
    pil = render([("PUNE", "latin")], size=40)
    path = tmp_path / "sign.png"
    pil.save(path)
    a, b, c = pp.load_image(path), pp.load_image(pil), pp.load_image(np.array(pil)[:, :, ::-1])
    assert a.shape == b.shape == c.shape and a.dtype == np.uint8
    big = pp.load_image(np.zeros((4000, 3000, 3), np.uint8), max_side=1000)
    assert max(big.shape[:2]) == 1000
    assert pp.load_image(np.zeros((10, 10), np.uint8)).shape == (10, 10, 3)


# --------------------------------------------------------------------------
# Train number detection
# --------------------------------------------------------------------------
def test_train_number_basic():
    assert extract_train_numbers("12301") == ["12301"]
    assert extract_train_numbers("Train 12301 Howrah Rajdhani") == ["12301"]


def test_train_number_multiple_and_unique():
    assert extract_train_numbers("12301/12302 and 12951, 12301") == ["12301", "12302", "12951"]
    assert extract_train_numbers("12301 12301", unique=False) == ["12301", "12301"]


def test_train_number_rejects_wrong_length():
    assert extract_train_numbers("1230 123456 1234567 PIN 700001") == []
    assert extract_train_numbers("Platform 4, 10:45") == []


def test_train_number_in_native_digits_is_returned_as_ascii():
    assert extract_train_numbers("गाड़ी १२३०१ नई दिल्ली") == ["12301"]      # Devanagari
    assert extract_train_numbers("ರೈಲು ೧೨೯೫೧") == ["12951"]                # Kannada
    assert extract_train_numbers("రైలు ౧౨౩౦౧") == ["12301"]                # Telugu


def test_train_number_inside_text_without_spaces_and_offsets():
    m = find_train_numbers("Tr.12301-Howrah")
    assert [(x.number, x.start, x.end) for x in m] == [("12301", 3, 8)]
    assert find_train_numbers("") == [] and extract_train_numbers(None) == []


def test_split_keeps_station_text_separate():
    nums, station = split_train_numbers("12301 HOWRAH RAJDHANI\nनई दिल्ली")
    assert nums == ["12301"]
    assert station == "HOWRAH RAJDHANI\nनई दिल्ली"
    nums, station = split_train_numbers("12301/12302 हावड़ा")
    assert nums == ["12301", "12302"] and station == "हावड़ा"
    assert split_train_numbers("नई दिल्ली") == ([], "नई दिल्ली")
    assert split_train_numbers("") == ([], "")


def test_engine_result_separates_train_numbers(sign_image):
    eng = mocked_engine(lambda l, p, n: ("12301\nहावड़ा जंक्शन", 90.0))
    r = eng.recognize(sign_image, script="Hindi")
    assert r["train_numbers"] == ["12301"]
    assert r["station_text"] == "हावड़ा जंक्शन"
    assert "12301" in r["clean_text"]  # clean_text itself is untouched


# --------------------------------------------------------------------------
# Text cleaning
# --------------------------------------------------------------------------
def test_unicode_normalization_nfc():
    decomposed = "क\u093c"  # ka + nukta
    assert normalize_unicode("\u0958") == decomposed
    assert unicodedata.is_normalized("NFC", normalize_unicode("e\u0301"))
    assert normalize_unicode("a\u200bb\ufeffc\u00ad") == "abc"
    assert normalize_unicode("A\u2013B \u2019") == "A-B '"


def test_whitespace_normalization():
    assert normalize_whitespace("  नई    दिल्ली \t \n\n  \n New   Delhi  ") == "नई दिल्ली\nNew Delhi"
    assert clean_text("a\u00a0\u00a0b\r\nc\x0c d") == "a b\nc\nd"


def test_duplicate_line_removal():
    assert remove_duplicate_lines("PUNE\npune\nPUNE \nMUMBAI\nPUNE") == "PUNE\nMUMBAI"
    assert clean_text("नई दिल्ली\nनई  दिल्ली\nNEW DELHI\nनई दिल्ली") == "नई दिल्ली\nNEW DELHI"


def test_punctuation_cleanup_removes_noise_only():
    assert cleanup_punctuation("| PUNE JN. ~") == "PUNE JN."
    assert cleanup_punctuation("--- ... ,,, ***") == ""
    assert cleanup_punctuation("[Pune]  {Jn}  <10>") == "Pune Jn 10"
    assert cleanup_punctuation("Delhi - Agra") == "Delhi Agra"          # lone dash is a separator
    assert cleanup_punctuation("Delhi-Agra, (Exp)") == "Delhi-Agra (Exp)"   # trailing comma trimmed, inner hyphen kept
    assert cleanup_punctuation("Delhi & Agra") == "Delhi & Agra"
    assert cleanup_punctuation("PUNE))") == "PUNE"


@pytest.mark.parametrize("word", [
    "नई दिल्ली", "बेंगलुरु जंक्शन", "क्षत्रपति शिवाजी", "ज्ञान", "श्री", "कृष्ण", "ड़ढ़",
    "ಬೆಂಗಳೂರು", "ಕೆಂಪೇಗೌಡ", "ಮೈಸೂರು ಜಂಕ್ಷನ್", "హైదరాబాద్", "విజయవాడ", "శ్రీకాకుళం", "చెన్నై",
    "पुणे जंक्शन ०१२३४", "೧೨೩೪೫", "౧౨౩౪౫",
])
def test_indian_characters_survive_cleaning(word):
    """No letter, vowel sign, virama, nukta, anusvara or native digit may be lost."""
    out = clean_text(word)
    keep = lambda s: [c for c in unicodedata.normalize("NFC", s) if unicodedata.category(c)[0] in "LMN"]
    assert keep(out) == keep(word)
    assert out == unicodedata.normalize("NFC", word)


def test_noise_around_indian_text_is_removed_but_text_kept():
    assert clean_text("| नई दिल्ली ~\n॥ ") == "नई दिल्ली"
    assert clean_text("ಬೆಂಗಳೂರು |||") == "ಬೆಂಗಳೂರು"
    assert clean_text("  ") == "" and clean_text("") == "" and clean_text(None) == ""


def test_zero_width_joiners_stripped_from_clean_text_but_kept_in_raw(sign_image):
    raw = "नई दिल्\u200dली"
    assert clean_text(raw) == "नई दिल्ली"
    assert "\u200d" in clean_text(raw, keep_joiners=True)
    eng = mocked_engine(lambda l, p, n: (raw, 90.0))
    r = eng.recognize(sign_image, script="Hindi")
    assert r["raw_text"] == raw and r["clean_text"] == "नई दिल्ली"


# --------------------------------------------------------------------------
# Result contract
# --------------------------------------------------------------------------
def test_result_contract_on_success(sign_image):
    eng = mocked_engine(lambda l, p, n: ("  नई  दिल्ली |", 87.4))
    r = eng.recognize(sign_image, script="Hindi")
    assert_valid_result(r)
    assert r["raw_text"] == "  नई  दिल्ली |"
    assert r["clean_text"] == "नई दिल्ली"
    assert r["ocr_confidence"] == 87.4 and r["ocr_language"] == "eng+hin"
    assert r["status"] == "ok"


def test_result_is_json_serialisable(sign_image):
    import json
    eng = mocked_engine(lambda l, p, n: ("नई दिल्ली", 87.4))
    json.dumps(eng.recognize(sign_image, script="Hindi", return_attempts=True), ensure_ascii=False)
    json.dumps(eng.recognize(None), ensure_ascii=False)
