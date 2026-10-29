"""Tests for train lookup, train state and the OCR -> station -> train -> state pipeline.

They run on tests/railway_fixture.py: a SYNTHETIC dataset in the schema of
data/processed/RailScan_Final_Dataset.csv. The numbers in it are made up and
only exercise code paths; they say nothing about real trains.
Run:  python -m pytest tests/test_train_state.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from railway_fixture import build_fixture  # noqa: E402
from src.railway.pipeline import RailwayPipeline  # noqa: E402
from src.railway.railway_data import RailwayData, RailwayDataUnavailable, normalize_train_no, parse_date  # noqa: E402
from src.railway.station_master import DEFAULT_DIR, StationMaster  # noqa: E402
from src.railway.train_state import TRAIN_UNAVAILABLE_MESSAGE, TrainStateConfig, TrainStateService  # noqa: E402

TARGET_COLUMNS = ["next_station_arrival_delay_min", "next_station_delayed_15", "next_delay_risk_label",
                  "target_next_actual_arrival"]


@pytest.fixture(scope="module")
def frame():
    return build_fixture()


@pytest.fixture(scope="module")
def master():
    if not (DEFAULT_DIR / "station_master.csv").is_file():
        pytest.skip("data/stations/station_master.csv not built")
    return StationMaster.from_csv()


@pytest.fixture(scope="module")
def svc(frame, master):
    return TrainStateService(frame, master)


@pytest.fixture(scope="module")
def pipe(frame, master):
    return RailwayPipeline(master, frame)


# ---------------------------------------------------------------- train number -> identity
def test_train_number_matching(svc):
    r = svc.lookup_train("12301")
    assert r["status"] == "ok"
    assert (r["train_no"], r["train_name"], r["train_type"]) == ("12301", "FIXTURE RAJDHANI EXPRESS", "Rajdhani")
    assert r["method"] == "unique_train_number"


@pytest.mark.parametrize("value", [12301, " 12301 ", "12301.0", "१२३०१"])
def test_train_number_input_variants(svc, value):
    assert svc.lookup_train(value)["train_no"] == "12301"


def test_unknown_train_returns_unavailable_message(svc):
    r = svc.lookup_train("98765")
    assert r["status"] == "unknown_train" and r["message"] == "Train information unavailable"
    assert r["train_name"] is None and r["train_type"] is None


@pytest.mark.parametrize("bad", [None, "", "abc", "123", "123456", "12 301"])
def test_invalid_train_number(svc, bad):
    assert svc.lookup_train(bad)["status"] == "invalid_input"


def test_multiple_matches_use_date_then_station_context(svc):
    assert svc.lookup_train("22222")["status"] == "ambiguous_train"          # nothing to narrow with
    a = svc.lookup_train("22222", journey_date="2025-03-02")
    b = svc.lookup_train("22222", journey_date="2025-03-05")
    assert (a["train_name"], a["method"]) == ("FIXTURE ALPHA EXPRESS", "journey_date")
    assert b["train_name"] == "FIXTURE BETA EXPRESS"
    # a date with no record does not decide, and says so
    r = svc.lookup_train("22222", journey_date="2030-01-01")
    assert r["status"] == "ambiguous_train" and any("journey date" in w for w in r["warnings"])
    assert len(r["candidates"]) == 2


def test_ambiguous_train_never_picks_one(svc):
    r = svc.lookup_train("22222")
    assert r["train_name"] is None and {c["train_name"] for c in r["candidates"]} == {
        "FIXTURE ALPHA EXPRESS", "FIXTURE BETA EXPRESS"}


def test_leading_zero_train_numbers_match():
    df = build_fixture()
    df["train_no"] = df["train_no"].replace({"12301": "04601"})
    assert TrainStateService(df).lookup_train("04601")["status"] == "ok"
    assert TrainStateService(df).lookup_train("4601")["status"] == "ok"


# ---------------------------------------------------------------- state: the happy path
def test_state_contains_everything_requested(svc):
    s = svc.get_state("12301", "2025-03-08", "CNB")
    assert s["status"] == "ok" and s["message"] == "Train state found."
    assert (s["train_no"], s["station_code"], s["station_name"], s["journey_date"]) == ("12301", "CNB", "Kanpur Central", "2025-03-08")
    for key in ("current_delay_min", "scheduled_arrival", "scheduled_departure", "actual_arrival",
                "actual_departure", "next_station_code", "next_station_name", "next_scheduled_arrival"):
        assert s[key] is not None, key
    assert s["next_station_code"] == "ALD" and s["next_station_name"] == "Allahabad Jn"
    h = s["historical"]
    assert {"avg_delay_min", "on_time_rate_pct", "observation_count", "sufficient"} <= set(h)
    assert h["observation_count"] == 7 and h["sufficient"] is True
    assert 0 <= h["on_time_rate_pct"] <= 100


def test_values_come_straight_from_the_dataset_row(svc, frame):
    row = frame[(frame.train_no == "12301") & (frame.journey_date == "2025-03-06") & (frame.station_code == "ALD")].iloc[0]
    s = svc.get_state("12301", "2025-03-06", "ALD")
    assert s["current_delay_min"] == row.current_observed_delay_min
    assert s["historical"]["avg_delay_min"] == pytest.approx(row.historical_avg_delay_min, abs=1e-3)
    assert s["historical"]["on_time_rate_pct"] == pytest.approx(row.historical_on_time_rate_pct, abs=1e-3)
    assert s["scheduled_departure"] == row.current_scheduled_departure.isoformat(sep=" ")
    assert s["distance_to_next_km"] == row.distance_to_next_km


def test_actual_times_are_scheduled_plus_observed_delay_and_labelled_derived(svc):
    s = svc.get_state("12301", "2025-03-04", "CNB")
    sched_dep, dep_delay = pd.Timestamp(s["scheduled_departure"]), s["departure_delay_min"]
    assert pd.Timestamp(s["actual_departure"]) == sched_dep + pd.Timedelta(minutes=dep_delay)
    assert "derived" in s["field_sources"]["actual_departure"]


def test_target_columns_are_never_returned(svc):
    s = svc.get_state("12301", "2025-03-08", "CNB")
    blob = json.dumps(s)
    for col in TARGET_COLUMNS + ["expected_next_station_arrival_baseline"]:
        assert col not in blob, f"{col} leaked into train state"


def test_result_is_json_serialisable(svc):
    for args in [("12301", "2025-03-08", "CNB"), ("99999", "2025-03-08", "CNB"), ("12301", "x", "CNB")]:
        json.dumps(svc.get_state(*args), ensure_ascii=False)


def test_missing_values_stay_missing_and_are_listed(svc):
    s = svc.get_state("12301", "2025-03-08", "NDLS")           # first stop: no scheduled/actual arrival
    assert s["scheduled_arrival"] is None and s["actual_arrival"] is None
    assert {"scheduled_arrival", "actual_arrival"} <= set(s["missing_fields"])
    assert s["historical"]["avg_actual_stop_min"] is None      # not 0, not guessed


# ---------------------------------------------------------------- structured failures
def test_missing_railway_record_unknown_train(svc):
    s = svc.get_state("98765", "2025-03-08", "NDLS")
    assert s["status"] == "unknown_train" and s["message"] == TRAIN_UNAVAILABLE_MESSAGE
    assert s["current_delay_min"] is None and s["historical"] is None


def test_missing_railway_record_for_the_date(svc):
    s = svc.get_state("12301", "2025-04-01", "NDLS")
    assert s["status"] == "no_record_for_date" and s["train_name"] == "FIXTURE RAJDHANI EXPRESS"
    assert s["available_dates"][-1] == "2025-03-08"


def test_unknown_station_for_train(svc):
    s = svc.get_state("12301", "2025-03-08", "BPL")
    assert s["status"] == "unknown_station_for_train" and "NDLS" in s["stations_in_record"]
    assert s["current_delay_min"] is None


def test_no_next_station_at_last_stop(svc):
    s = svc.get_state("12301", "2025-03-08", "HWH")
    assert s["status"] == "no_next_station" and s["previous_station_code"] == "DHN"
    assert s["next_station_code"] is None and s["current_delay_min"] is None
    assert s["scheduled_arrival"] is not None                  # known from the previous row's next stop


def test_row_without_next_station_reports_no_next_station(frame):
    df = frame.copy()
    df.loc[(df.train_no == "12301") & (df.journey_date == "2025-03-08") & (df.station_code == "DHN"), "next_station_code"] = None
    s = TrainStateService(df).get_state("12301", "2025-03-08", "DHN")
    assert s["status"] == "no_next_station" and s["next_station_code"] is None


def test_insufficient_history_still_returns_the_state(svc):
    s = svc.get_state("33333", "2025-03-04", "CNB")
    assert s["status"] == "insufficient_history"
    assert s["current_delay_min"] is not None and s["next_station_code"] == "ALD"
    assert s["historical"]["sufficient"] is False and s["historical"]["observation_count"] == 0
    assert s["historical"]["avg_delay_min"] is None and s["historical"]["on_time_rate_pct"] is None
    assert "historical.avg_delay_min" in s["missing_fields"]


def test_history_threshold_is_configurable(frame):
    strict = TrainStateService(frame, config=TrainStateConfig(min_history_observations=8))
    assert strict.get_state("12301", "2025-03-08", "CNB")["status"] == "insufficient_history"
    lenient = TrainStateService(frame, config=TrainStateConfig(min_history_observations=1))
    assert lenient.get_state("22222", "2025-03-02", "CNB")["status"] == "ok"


def test_ambiguous_train_and_ambiguous_stop(svc):
    assert svc.get_state("22222", "2030-01-01", "CNB")["status"] in ("ambiguous_train",)
    s = svc.get_state("44444", "2025-03-05", "CNB")
    assert s["status"] == "ambiguous_stop" and [c["seq"] for c in s["candidates"]] == [1, 3]
    assert svc.get_state("44444", "2025-03-05", "CNB", seq=3)["next_station_code"] == "GAYA"


@pytest.mark.parametrize("args", [(None, "2025-03-08", "CNB"), ("12301", None, "CNB"), ("12301", "2025-03-08", None),
                                  ("12301", "not a date", "CNB"), ("12301", "2025-03-08", "  ")])
def test_invalid_input_is_a_status_not_an_exception(svc, args):
    s = svc.get_state(*args)
    assert s["status"] == "invalid_input" and s["message"]


def test_station_code_is_case_and_space_insensitive(svc):
    assert svc.get_state("12301", "2025-03-08", " cnb ")["station_code"] == "CNB"


def test_dates_accept_several_types(svc):
    from datetime import date, datetime
    for d in ["2025-03-08", date(2025, 3, 8), datetime(2025, 3, 8, 14, 30), pd.Timestamp("2025-03-08"), "08 Mar 2025"]:
        assert svc.get_state("12301", d, "CNB")["status"] == "ok", d


def test_dataset_unavailable_is_reported(tmp_path, master):
    svc = TrainStateService(tmp_path / "missing.csv", master)
    assert svc.get_state("12301", "2025-03-08", "CNB")["status"] == "data_unavailable"
    assert svc.lookup_train("12301")["status"] == "data_unavailable"
    assert "build_dataset.py" in svc.load_error


def test_dataset_missing_required_columns(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("a,b\n1,2\n")
    assert TrainStateService(p).get_state("12301", "2025-03-08", "CNB")["status"] == "data_unavailable"
    with pytest.raises(RailwayDataUnavailable):
        RailwayData(pd.DataFrame({"a": [1]}))


def test_csv_round_trip_keeps_leading_zero_ids(tmp_path, frame):
    df = frame.copy()
    df["train_no"] = df["train_no"].replace({"12301": "04601"})
    p = tmp_path / "RailScan_Final_Dataset.csv"
    df.to_csv(p, index=False)
    s = TrainStateService(p).get_state("04601", "2025-03-08", "CNB")
    assert s["status"] == "ok" and s["train_no"] == "04601"


def test_helpers():
    assert normalize_train_no("12301") == "12301" and normalize_train_no("1230") == "1230"
    assert normalize_train_no("123") is None
    assert parse_date("2025-03-08") == pd.Timestamp("2025-03-08") and parse_date("zzz") is None


# ---------------------------------------------------------------- OCR -> station -> train -> state
def ocr(clean, station=None, numbers=(), **kw):
    d = {"status": "ok", "clean_text": clean, "text": clean, "station_text": station if station is not None else clean,
         "train_numbers": list(numbers), "ocr_confidence": 90.0, "low_confidence": False}
    d.update(kw)
    return d


def test_pipeline_end_to_end_hindi_station_with_train_number(pipe):
    r = pipe.process(ocr("12301\nनई दिल्ली", "नई दिल्ली", ["12301"]), journey_date="2025-03-08")
    assert r["status"] == "ok"
    assert (r["station"]["station_name"], r["station"]["station_code"]) == ("New Delhi", "NDLS")
    assert r["station"]["match_method"] == "alias" and r["station"]["match_score"] == 97.0
    assert r["train"]["train_name"] == "FIXTURE RAJDHANI EXPRESS"
    st = r["train_state"]
    assert st["next_station_code"] == "CNB" and st["current_delay_min"] is not None and st["historical"]["sufficient"]


def test_pipeline_plain_string_input(pipe):
    r = pipe.process("12301 NEW DELHI", journey_date="2025-03-08")
    assert r["status"] == "ok" and r["station"]["station_code"] == "NDLS"


def test_pipeline_ocr_failure(pipe):
    for bad in [{"status": "OCR failed", "text": "", "error": "no text"}, None, "", {"status": "ok", "clean_text": ""}]:
        r = pipe.process(bad)
        assert r["status"] == "ocr_failed" and r["station"] is None and r["train_state"] is None


def test_pipeline_unknown_station(pipe):
    r = pipe.process(ocr("12301 QQQ ZZZ", "QQQ ZZZ", ["12301"]), journey_date="2025-03-08")
    assert r["status"] == "station_not_identified" and r["message"] == "Station could not be confidently identified."
    assert r["train_state"] is None


def test_pipeline_ambiguous_station(pipe):
    r = pipe.process(ocr("DELHI", "DELHI"))
    assert r["status"] == "station_ambiguous" and r["station"]["station_code"] is None


def test_pipeline_possible_station_needs_confirmation(pipe, frame, master):
    r = pipe.process(ocr("12301 NEW DEHLI JN", "NEW DEHLI JN", ["12301"]), journey_date="2025-03-08")
    assert r["status"] == "station_needs_confirmation" and r["train_state"] is None
    permissive = RailwayPipeline(master, frame, accept_possible_station=True)
    r = permissive.process(ocr("12301 NEW DEHLI JN", "NEW DEHLI JN", ["12301"]), journey_date="2025-03-08")
    assert r["status"] == "ok" and r["station"]["station_code"] == "NDLS"


def test_pipeline_unknown_train(pipe):
    r = pipe.process(ocr("98765 NEW DELHI", "NEW DELHI", ["98765"]), journey_date="2025-03-08")
    assert r["status"] == "train_unavailable" and r["message"] == "Train information unavailable"
    assert r["station"]["station_code"] == "NDLS"          # the station result is still reported


def test_pipeline_station_only(pipe):
    r = pipe.process(ocr("नई दिल्ली"))
    assert r["status"] == "no_train_number" and r["station"]["station_code"] == "NDLS"


def test_pipeline_needs_journey_date_unless_only_one_exists(pipe):
    r = pipe.process(ocr("12301 NEW DELHI", "NEW DELHI", ["12301"]))
    assert r["status"] == "journey_date_required" and r["available_dates"][-1] == "2025-03-08"
    r = pipe.process(ocr("33333 KANPUR CENTRAL", "KANPUR CENTRAL", ["33333"]))     # one date on record
    assert r["status"] == "insufficient_history" and r["journey_date"] == "2025-03-04"
    assert any("only date on record" in w for w in r["warnings"])


def test_pipeline_last_stop_has_no_next_station(pipe):
    r = pipe.process(ocr("12301 HOWRAH JN", "HOWRAH JN", ["12301"]), journey_date="2025-03-08")
    assert r["status"] == "no_next_station" and r["train_state"]["previous_station_code"] == "DHN"


def test_pipeline_uses_train_route_to_resolve_station(pipe):
    r = pipe.process(ocr("12301 DELHI", "DELHI", ["12301"]), journey_date="2025-03-08")
    # NDLS is on the train's route; DLI etc. are not -> the route picks it, but only as "possible"
    assert r["status"] in ("station_needs_confirmation", "station_ambiguous")
    if r["status"] == "station_needs_confirmation":
        assert r["station"]["station_code"] == "NDLS" and r["station"]["match_method"].endswith("+train_route")


def test_pipeline_without_dataset_reports_it(master, tmp_path):
    p = RailwayPipeline(master, tmp_path / "nope.csv")
    r = p.process(ocr("12301 NEW DELHI", "NEW DELHI", ["12301"]), journey_date="2025-03-08")
    assert r["status"] == "data_unavailable" and r["station"]["station_code"] == "NDLS"


def test_pipeline_result_is_json_serialisable(pipe):
    json.dumps(pipe.process(ocr("12301 नई दिल्ली", "नई दिल्ली", ["12301"]), journey_date="2025-03-08"), ensure_ascii=False)
