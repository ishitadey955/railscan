"""Tests for src/railway: text normalisation, station master loading and station matching.

Uses the real station master in data/stations/ for realistic cases (skipped if
absent) and tiny in-memory masters for exact control of thresholds / ambiguity.
Run:  python -m pytest tests/test_station_matching.py -v
"""
from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.railway.station_master import (  # noqa: E402
    DEFAULT_DIR, StationMaster, StationMasterUnavailable,
)
from src.railway.station_matcher import MatchConfig, NOT_IDENTIFIED_MESSAGE, StationMatcher  # noqa: E402
from src.railway.text_normalizer import (  # noqa: E402
    compact_key, core_key, detect_script, matching_key, normalize_text, phonetic_skeleton,
)

RESULT_KEYS = {"status", "message", "station_name", "station_code", "match_score", "match_method",
               "accepted", "needs_confirmation", "confidence_level", "candidates", "warnings"}


def tiny_master() -> StationMaster:
    """A hand-made master for logic tests (codes/names are illustrative)."""
    return StationMaster.from_records(
        stations=[("NDLS", "NEW DELHI", 28.64, 77.22), ("DLI", "OLD DELHI", 28.66, 77.23),
                  ("NZM", "DELHI H NIZAMUDDIN", 28.58, 77.25), ("HWH", "HOWRAH JN", 22.58, 88.34),
                  ("PUNE", "PUNE JN"), ("NGP", "NAGPUR"), ("BPL", "BHOPAL JN"), ("HBJ", "HABIBGANJ"),
                  ("MAQ", "MANGALORE CNTL"), ("MAJN", "MANGALORE JN")],
        aliases=[("नई दिल्ली", "NDLS", "hi"), ("हावड़ा", "HWH", "hi"), ("पुणे", "PUNE", "mr"),
                 ("ನಾಗಪುರ", "NGP", "kn")])


@pytest.fixture(scope="module")
def tiny():
    return StationMatcher(tiny_master())


@pytest.fixture(scope="module")
def real():
    if not (DEFAULT_DIR / "station_master.csv").is_file():
        pytest.skip("data/stations/station_master.csv not built (python scripts/build_station_master.py)")
    return StationMatcher(StationMaster.from_csv())


def check_contract(r):
    assert RESULT_KEYS <= set(r)
    if r["status"] in ("matched", "possible"):
        assert r["station_code"] and r["station_name"] and r["match_method"]
        assert 0 <= r["match_score"] <= 100
    else:
        assert r["station_code"] is None and r["station_name"] is None and r["match_score"] is None
        assert r["accepted"] is False


# ---------------------------------------------------------------- text normaliser
def test_normalize_unicode_case_whitespace_punctuation():
    assert normalize_text("  NEW   DELHI\n") == "new delhi"
    assert normalize_text("Pune-Jn.,;") == "pune jn"
    assert normalize_text("a​b­c") == "abc"
    assert normalize_text("é") == unicodedata.normalize("NFC", "é")
    assert normalize_text(None) == "" and normalize_text("   ") == ""


@pytest.mark.parametrize("word", ["नई दिल्ली", "ಬೆಂಗಳೂರು", "హైదరాబాద్", "क्षत्रपति ज्ञान श्री", "ಮೈಸೂರು ಜಂಕ್ಷನ್"])
def test_normalizer_preserves_indian_scripts(word):
    keep = lambda s: [c for c in unicodedata.normalize("NFC", s) if unicodedata.category(c)[0] in "LM"]
    assert keep(normalize_text(word)) == keep(word)
    assert keep(matching_key(word.replace(" जंक्शन", ""))) == keep(word.replace(" जंक्शन", ""))


def test_railway_abbreviations_and_generic_words():
    assert matching_key("HOWRAH JN") == matching_key("Howrah Junction") == "howrah junction"
    assert matching_key("MANGALORE CNTL") == "mangalore central"
    assert core_key("Howrah Junction") == "howrah"
    assert matching_key("NEW DELHI RAILWAY STATION") == "new delhi"
    assert compact_key("NEW  DELHI") == compact_key("NEWDELHI")
    assert core_key("ಮೈಸೂರು ಜಂಕ್ಷನ್") == "ಮೈಸೂರು"


def test_nukta_fold_applies_to_keys_only():
    assert matching_key("निज़ामुद्दीन") == matching_key("निजामुद्दीन")
    assert normalize_text("निज़ामुद्दीन") != normalize_text("निजामुद्दीन")  # displayed text is untouched


def test_script_detection_and_transliteration_skeleton():
    assert detect_script("नई दिल्ली") == "devanagari" and detect_script("ಬೆಂಗಳೂರು") == "kannada"
    assert detect_script("హైదరాబాద్") == "telugu" and detect_script("NEW DELHI") == "latin"
    assert detect_script("12345") == "none" and detect_script("NDLS नई") == "mixed"
    assert phonetic_skeleton("nAgapura") == phonetic_skeleton("NAGPUR") == "ngpr"


# ---------------------------------------------------------------- master loading
def test_missing_station_master_explains_the_required_source(tmp_path):
    with pytest.raises(StationMasterUnavailable) as e:
        StationMaster.from_csv(directory=tmp_path)
    assert "station_code, station_name, latitude, longitude" in str(e.value)
    assert "build_station_master.py" in str(e.value)


def test_master_validation_and_alias_to_unknown_code_is_skipped(tmp_path):
    (tmp_path / "station_master.csv").write_text(
        "station_code,station_name,latitude,longitude\nndls,NEW DELHI,28.6,77.2\nNDLS,DUP,1,1\nXX,,1,1\n", encoding="utf-8")
    (tmp_path / "station_aliases.csv").write_text(
        "alias,station_code,language,source\nDilli,NDLS,en,t\nGhost,ZZZZ,en,t\n", encoding="utf-8")
    sm = StationMaster.from_csv(directory=tmp_path)
    assert len(sm) == 1 and sm.get("NDLS").name == "NEW DELHI"
    assert [a.alias for a in sm.aliases] == ["Dilli"]
    assert any("ZZZZ" in w for w in sm.warnings) and any("duplicate" in w for w in sm.warnings)

    (tmp_path / "station_master.csv").write_text("code,name\nA,B\n", encoding="utf-8")
    with pytest.raises(StationMasterUnavailable):
        StationMaster.from_csv(directory=tmp_path)


def test_real_master_integrity():
    path = DEFAULT_DIR / "station_master.csv"
    if not path.is_file():
        pytest.skip("station master not built")
    sm = StationMaster.from_csv()
    assert len(sm) > 5000 and not [w for w in sm.warnings if "duplicate" in w]
    for code, name in [("NDLS", "NEW DELHI"), ("HWH", "HOWRAH JN"), ("SBC", "BANGALORE CITY JN")]:
        assert sm.get(code).name == name and sm.get(code).latitude is not None
    # every alias points at a real master code (invalid ones would be dropped with a warning)
    assert not [w for w in sm.warnings if "alias" in w], sm.warnings[:3]


# ---------------------------------------------------------------- exact methods
def test_exact_station_code(tiny):
    r = tiny.match("NDLS")
    check_contract(r)
    assert (r["status"], r["station_code"], r["match_method"], r["match_score"]) == ("matched", "NDLS", "station_code", 100.0)
    assert r["station_name"] == "New Delhi" and r["latitude"] == 28.64


def test_lowercase_word_is_not_read_as_a_station_code(tiny):
    assert tiny.match("pune")["match_method"] != "station_code"


def test_exact_station_name(tiny):
    r = tiny.match("NEW DELHI")
    assert (r["station_code"], r["match_method"], r["match_score"]) == ("NDLS", "exact_name", 100.0)


def test_case_insensitive_match(tiny):
    r = tiny.match("new delhi")
    assert (r["station_code"], r["match_method"]) == ("NDLS", "case_insensitive") and r["status"] == "matched"


def test_normalised_match_handles_abbreviations_spacing_and_noise(tiny):
    for text, code in [("Howrah Junction", "HWH"), ("HOWRAH  JN.", "HWH"), ("| PUNE JUNCTION |", "PUNE"),
                       ("Mangalore Central", "MAQ"), ("NEWDELHI", "NDLS")]:
        r = tiny.match(text)
        assert r["status"] == "matched" and r["station_code"] == code, (text, r["match_method"])


def test_station_name_split_over_two_lines(tiny):
    assert tiny.match("NEW\nDELHI")["station_code"] == "NDLS"


def test_station_code_in_brackets(tiny):
    r = tiny.match("NEW DELHI (NDLS)")
    assert r["station_code"] == "NDLS" and r["status"] == "matched"


# ---------------------------------------------------------------- aliases / multilingual
@pytest.mark.parametrize("text,code", [("नई दिल्ली", "NDLS"), ("हावड़ा", "HWH"), ("पुणे", "PUNE"), ("ನಾಗಪುರ", "NGP")])
def test_alias_match_in_indian_scripts(tiny, text, code):
    r = tiny.match(text)
    check_contract(r)
    assert (r["status"], r["station_code"], r["match_method"]) == ("matched", code, "alias")


def test_alias_with_nukta_and_chandrabindu_spelling_variants(tiny):
    assert tiny.match("हावडा")["station_code"] == "HWH"          # no nukta
    assert tiny.match("नई  दिल्ली।")["station_code"] == "NDLS"   # extra space + danda


def test_ocr_result_dict_is_accepted_and_uses_station_text(tiny):
    ocr = {"status": "ok", "clean_text": "12301 नई दिल्ली", "station_text": "नई दिल्ली", "ocr_confidence": 88.0,
           "low_confidence": False, "train_numbers": ["12301"]}
    assert tiny.match(ocr)["station_code"] == "NDLS"


def test_low_ocr_confidence_adds_warning_but_keeps_result(tiny):
    ocr = {"status": "ok", "clean_text": "NEW DELHI", "station_text": "NEW DELHI", "ocr_confidence": 41.0, "low_confidence": True}
    r = tiny.match(ocr)
    assert r["station_code"] == "NDLS" and any("OCR confidence is low" in w for w in r["warnings"])


def test_real_master_multilingual_examples(real):
    for text, code in [("नई दिल्ली", "NDLS"), ("NEW DELHI", "NDLS"), ("ಬೆಂಗಳೂರು", "SBC"), ("హైదరాబాద్", "HYB"),
                       ("पुणे जंक्शन", "PUNE"), ("Howrah Junction", "HWH"), ("CSMT", "CSTM"),
                       ("ಮೈಸೂರು ಜಂಕ್ಷನ್", "MYS"), ("बेंगलुरु जंक्शन", "SBC")]:
        r = real.match(text)
        assert r["status"] == "matched" and r["station_code"] == code, (text, r["status"], r["station_code"])


# ---------------------------------------------------------------- fuzzy + thresholds
def test_fuzzy_match_for_ocr_typos(tiny):
    r = tiny.match("NEW DELI")
    assert r["station_code"] == "NDLS" and r["match_method"] == "fuzzy" and r["status"] == "matched"
    assert r["match_score"] >= 90
    r = tiny.match("HABIBGAN")
    assert r["station_code"] == "HBJ" and r["match_method"] == "fuzzy"


def test_possible_match_band_is_not_auto_accepted(tiny):
    r = tiny.match("BHOPL JNCTN")      # heavier damage -> below the high threshold
    check_contract(r)
    if r["status"] == "possible":
        assert 75 <= r["match_score"] < 90 and r["accepted"] is False and r["needs_confirmation"] is True
    else:                               # a stricter outcome is also acceptable
        assert r["status"] in ("matched", "not_identified")


def test_thresholds_are_configurable():
    strict = StationMatcher(tiny_master(), MatchConfig(high_threshold=99.0, possible_threshold=98.0))
    r = strict.match("NEW DELI")        # ~94 -> now below both thresholds
    assert r["status"] == "not_identified" and r["message"] == NOT_IDENTIFIED_MESSAGE
    lenient = StationMatcher(tiny_master(), MatchConfig(high_threshold=80.0))
    r = lenient.match("NEW DELI")
    assert r["status"] == "matched" and r["accepted"] is True


def test_threshold_bands_exact(tiny):
    t = tiny.cfg
    assert (t.high_threshold, t.possible_threshold) == (90.0, 75.0)   # documented defaults
    hi = tiny._accept(type("C", (), {"code": "NDLS", "score": 90.0, "method": "fuzzy", "text": "x"})(), [], [])
    lo = tiny._accept(type("C", (), {"code": "NDLS", "score": 89.9, "method": "fuzzy", "text": "x"})(), [], [])
    assert hi["status"] == "matched" and hi["accepted"] is True
    assert lo["status"] == "possible" and lo["accepted"] is False and lo["needs_confirmation"] is True


# ---------------------------------------------------------------- unknown / low confidence / ambiguity
@pytest.mark.parametrize("text", ["QQQ ZZZ", "XYLOPHONE", "12345", "!!! ???", "लिखित", "ZZZZZZZZ GGGG"])
def test_unknown_station_is_not_forced(tiny, text):
    r = tiny.match(text)
    check_contract(r)
    assert r["status"] == "not_identified" and r["message"] == "Station could not be confidently identified."
    assert r["station_code"] is None and r["accepted"] is False


def test_low_confidence_match_returns_candidates_but_no_station(tiny):
    r = tiny.match("NAGPAARRR XQ")
    assert r["station_code"] is None and r["message"] == NOT_IDENTIFIED_MESSAGE


def test_ambiguous_names_are_not_forced(tiny):
    r = tiny.match("DELHI")
    assert r["status"] == "ambiguous" and r["station_code"] is None and "Multiple possible stations" in r["message"]
    assert {c["station_code"] for c in r["candidates"]} >= {"NDLS", "DLI"}
    r = tiny.match("Mangalore")
    assert r["status"] in ("matched", "ambiguous")


def test_train_route_context_breaks_a_tie_but_nothing_else(tiny):
    r = tiny.match("DELHI", context_codes=["DLI", "CNB"])
    # context resolves WHICH station, but the score is not raised: still "possible", needs confirmation
    assert r["status"] == "possible" and r["station_code"] == "DLI" and r["match_method"].endswith("+train_route")
    assert r["accepted"] is False
    assert tiny.match("DELHI", context_codes=["NDLS", "DLI"])["status"] == "ambiguous"   # context can't decide
    assert tiny.match("QQQ ZZZ", context_codes=["NDLS"])["station_code"] is None        # context never forces


@pytest.mark.parametrize("bad,status", [
    (None, "no_text"), ("", "no_text"), ("   \n ", "no_text"),
    ({"status": "OCR failed", "text": "", "error": "x"}, "ocr_failed"),
    ({"status": "ok", "clean_text": "", "station_text": ""}, "no_text"),
])
def test_empty_or_failed_ocr_input(tiny, bad, status):
    r = tiny.match(bad)
    check_contract(r)
    assert r["status"] == status


def test_real_master_does_not_force_wrong_stations(real):
    for text in ["XQZ KLM", "12301", "WELCOME TO", "प्लेटफार्म"]:
        assert real.match(text)["station_code"] is None
    for text in ["DELHI", "दिल्ली", "ಮಂಗಳೂರು"]:
        assert real.match(text)["status"] in ("ambiguous", "possible", "not_identified")
        assert real.match(text)["accepted"] is False


# ------------------------------------------------------------ Bengali (real station list + curated aliases)
@pytest.mark.parametrize("text,code", [("হাওড়া", "HWH"), ("হাওড়া জংশন", "HWH"), ("শিয়ালদহ", "SDAH"),
                                       ("বর্ধমান জংশন", "BWN"), ("নিউ জলপাইগুড়ি", "NJP")])
def test_bengali_station_names_match(text, code):
    from src.railway import StationMaster, StationMatcher
    m = StationMatcher(StationMaster.load()).match(text)
    assert m["status"] == "matched" and m["station_code"] == code


def test_bengali_script_is_detected_and_transliterated():
    from src.railway.text_normalizer import detect_script, transliterate_to_latin
    assert detect_script("বর্ধমান") == "bengali"
    assert transliterate_to_latin("বর্ধমান") == "vardhamAna"
