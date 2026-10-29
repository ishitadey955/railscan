"""Photo-to-railway assistant: photo checks, retake advice, candidates, scheduled services."""
import numpy as np
import pytest

from src.image.quality import LOW_OCR_CONFIDENCE, measure, photo_issues, retake_advice
from src.railway_assistant import match_options, needs_confirmation, station_services, train_summary
from src.timetable import load_timetable


@pytest.fixture(scope="module")
def tt():
    return load_timetable()


def _img(h=600, w=900, value=128, noise=True):
    rng = np.random.default_rng(0)
    img = np.full((h, w, 3), value, np.uint8)
    if noise:
        img = np.clip(img + rng.normal(0, 40, img.shape), 0, 255).astype(np.uint8)
    return img


GOOD_OCR = {"status": "ok", "ocr_confidence": 80.0}
MATCHED = {"status": "matched", "accepted": True, "station_code": "SPN"}


def test_good_photo_and_read_needs_no_retake():
    assert retake_advice(measure(_img()), GOOD_OCR, MATCHED) == {"retake": False, "reasons": []}


def test_small_dark_blurry_photos_are_flagged():
    assert any("Low resolution" in r for r in photo_issues(measure(_img(h=120, w=200))))
    assert any("very dark" in r for r in photo_issues(measure(_img(value=10, noise=False))))
    assert any("blurry" in r for r in photo_issues(measure(_img(noise=False))))   # flat image: no edges


def test_ocr_and_matching_outcomes_drive_the_advice():
    m = measure(_img())
    assert "No text could be read" in retake_advice(m, {"status": "OCR failed"}, None)["reasons"][0]
    low = retake_advice(m, {"status": "ok", "ocr_confidence": LOW_OCR_CONFIDENCE - 5}, MATCHED)
    assert low["retake"] and "low confidence" in low["reasons"][0]
    nomatch = retake_advice(m, GOOD_OCR, {"status": "not_identified"})
    assert nomatch["retake"] and "No station name" in nomatch["reasons"][0]


def test_match_options_are_ranked_and_unique():
    match = {"status": "ambiguous", "candidates": [
        {"station_code": "A", "station_name": "Alpha", "match_score": 88, "match_method": "fuzzy"},
        {"station_code": "A", "station_name": "Alpha", "match_score": 80, "match_method": "alias"},
        {"station_code": "B", "station_name": "Beta", "match_score": 86, "match_method": "fuzzy"}]}
    assert [o["station_code"] for o in match_options(match)] == ["A", "B"]
    assert match_options(None) == []


def test_only_confident_accepted_matches_skip_confirmation():
    assert not needs_confirmation({"status": "matched", "accepted": True})
    for m in ({"status": "possible", "accepted": False}, {"status": "ambiguous"}, None):
        assert needs_confirmation(m)


def test_station_services_for_a_station_on_the_route(tt):
    # 12229 Lucknow Mail calls at Shahjahanpur after midnight; many other trains call there too
    s = station_services(tt, "SPN").set_index("train_no")
    assert "12229" in s.index and len(s) > 10
    assert s.loc["12229", "Arrives"] == "00:34 (+1 day)" and s.loc["12229", "Departs"] == "00:36 (+1 day)"
    assert s.loc["12229", "Next stop"].endswith("(BE)")


def test_station_not_served_gives_empty_table(tt):
    assert station_services(tt, "ZZZZ").empty


def test_origin_and_terminus(tt):
    s = station_services(tt, "NDLS").set_index("train_no")
    assert s.loc["12229", "Next stop"] == "terminates" and s.loc["12229", "Departs"] == "-"
    info = train_summary(tt, "12229")
    assert info["origin"].endswith("(LKO)") and info["destination"].endswith("(NDLS)") and info["stops"] == 9
    assert train_summary(tt, "99999") is None
