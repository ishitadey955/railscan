"""Journey planner for any date: timetable placement, past-only history, estimates, error cases."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from src.delay_prediction import DelayPredictor  # noqa: E402
from src.planner import (feature_row, forecast, history_features, model_coverage,  # noqa: E402
                         selected_methods, typical_delays)
from src.railscan_service import load_master  # noqa: E402
from src.timetable import departure_stations, load_timetable, runs_on, schedule_on  # noqa: E402


# Reference train: 12229 Lucknow Mail, Lucknow 22:00 -> New Delhi 06:55 next day, daily,
# every stop in northern India (timetable built from the recorded journeys).
TRAIN = "12229"


@pytest.fixture(scope="module")
def data():
    return load_master(train_no=TRAIN)          # the app also works one train at a time


@pytest.fixture(scope="module")
def tt():
    return load_timetable()


@pytest.fixture(scope="module")
def predictor():
    return DelayPredictor.load()


# ------------------------------------------------------------------ timetable
def test_schedule_on_any_date_handles_overnight_day_offset(tt):
    s = schedule_on(tt, TRAIN, "2030-01-04").set_index("station_code")
    assert s.loc["LKO", "sched_dep"] == pd.Timestamp("2030-01-04 22:00")
    assert s.loc["SPN", "sched_arr"] == pd.Timestamp("2030-01-05 00:34")        # next day
    assert s.loc["SPN", "halt_min"] == 2
    assert pd.isna(s.loc["LKO", "sched_arr"])                                   # origin: no arrival
    assert pd.isna(s.loc["NDLS", "sched_dep"])                                  # destination: no departure


def test_travel_time_across_midnight_is_positive(tt):
    s = schedule_on(tt, TRAIN, "2026-12-31")
    assert (s.travel_to_next_min.dropna() > 0).all()


def test_unknown_train_gives_empty_schedule(tt):
    assert schedule_on(tt, "99999", "2027-01-01").empty


def test_departure_stations_exclude_destination(tt):
    codes = [s["code"] for s in departure_stations(tt, TRAIN)]
    assert codes[0] == "LKO" and "NDLS" not in codes


def test_running_days_known(tt):
    assert runs_on(tt, TRAIN, "2027-06-15") is True                            # runs daily in recent records


# ------------------------------------------------------------------ history parity and leakage
def test_history_features_reproduce_training_values(data):
    cols = ["historical_observation_count", "historical_avg_delay_min", "historical_on_time_rate_pct",
            "historical_avg_actual_stop_min", "historical_stop_reliability_pct"]
    assert len(data) > 100
    for r in data.iloc[::10].itertuples():
        h = history_features(data, r.train_no, r.station_code, r.next_station_code, r.observation_time)
        for c in cols:
            a, b = h[c], getattr(r, c)
            assert (pd.isna(a) and pd.isna(b)) or np.isclose(a, b), (c, a, b)


def test_history_before_first_record_is_empty(data):
    h = history_features(data, TRAIN, "HRI", "SPN", pd.Timestamp("2020-01-01"))
    assert h["historical_observation_count"] == 0 and np.isnan(h["historical_avg_delay_min"])


def test_calendar_features_come_from_journey_date(data, tt):
    # journey starts Friday 22:00; Shahjahanpur is reached on Saturday but the features say Friday, as in training
    s = schedule_on(tt, TRAIN, "2027-01-01").set_index("station_code")
    stop = s.loc["SPN"].copy()
    stop["station_code"] = "SPN"
    x = feature_row(data, stop, TRAIN, "SUF", "2027-01-01", 10.0, cutoff=pd.Timestamp("2027-01-01"))
    assert x["day_of_week"] == 4 and x["is_weekend"] == 0 and x["month"] == 1


# ------------------------------------------------------------------ forecasts
def test_forecast_for_a_future_date(data, tt, predictor):
    f = forecast(data, tt, predictor, TRAIN, "2028-05-10", "HRI", 20.0)
    assert f["status"] == "ok"
    r = f["route"]
    assert r.station_code.iloc[0] == "SPN" and r.station_code.iloc[-1] == "NDLS"
    assert r.method.iloc[0] in ("model", "no_coverage")
    assert r.scheduled_arrival.iloc[0] == pd.Timestamp("2028-05-11 00:34")
    assert f["reference_time"] == pd.Timestamp("2028-05-10 23:56")          # 23:36 + 20 min
    assert 0 <= f["next"]["probability_delayed_15"] <= 1
    exp = r.scheduled_arrival + pd.to_timedelta(r.predicted_delay_min, unit="m")
    assert ((exp - pd.to_datetime(r.expected_arrival)).abs() < pd.Timedelta(seconds=1)).all()


def test_later_stops_use_the_validated_method(data, tt, predictor):
    sel = selected_methods()
    f = forecast(data, tt, predictor, TRAIN, "2027-02-02", "LKO", 12.0)
    later = f["route"][f["route"].stops_ahead >= 2]
    if sel.get("2-3") == "carry":
        assert (later[later.stops_ahead <= 3].predicted_delay_min == 12.0).all()
    assert later.basis.notna().all()


def test_uncovered_segment_is_labelled_not_extrapolated(data, tt, predictor):
    f = forecast(data, tt, predictor, TRAIN, "2027-02-02", "BE", 30.0, covered=set())
    first = f["route"].iloc[0]
    assert first.method == "no_coverage" and first.predicted_delay_min == 30.0
    assert first.probability_delayed_15 is None


def test_last_stop_and_unknown_station(data, tt, predictor):
    assert forecast(data, tt, predictor, TRAIN, "2027-01-01", "NDLS", 0)["status"] == "no_next_station"
    assert forecast(data, tt, predictor, TRAIN, "2027-01-01", "PUNE", 0)["status"] == "unknown_station"
    assert forecast(data, tt, predictor, "99999", "2027-01-01", "LKO", 0)["status"] == "unknown_train"


def test_coverage_comes_from_training_rows_only():
    cov = model_coverage()
    tr = pd.read_csv(ROOT / "data" / "processed" / "train.csv", dtype=str,
                     usecols=["train_no", "station_code", "next_station_code"])
    assert cov == set(zip(tr.train_no.str.lstrip("0"), tr.station_code, tr.next_station_code))


def test_typical_delays_are_recorded_medians(data, tt):
    t = typical_delays(data, tt, TRAIN, "2027-03-15").set_index("station_code")
    rows = data[(data.train_no == TRAIN) & (data.station_code == "SPN")]
    nxt = data[(data.train_no == TRAIN) & (data.next_station_code == "SPN")]
    arr = pd.concat([nxt[["journey_date", "next_station_arrival_delay_min"]].rename(
        columns={"next_station_arrival_delay_min": "d"}),
        rows[["journey_date", "current_arrival_delay_min"]].rename(columns={"current_arrival_delay_min": "d"})])
    arr = arr.dropna().drop_duplicates("journey_date")
    assert t.loc["SPN", "recorded_arrivals"] == len(arr)
    assert t.loc["SPN", "median_delay_min"] == pytest.approx(arr.d.median())


# ------------------------------------------------------------------ builder: next regular stop
def test_next_stop_is_next_regular_stop_not_seq_plus_one():
    import build_dataset as bd
    bd.REGULAR_STOPS.clear()
    bd.REGULAR_STOPS.update({"01023": [10, 13, 14, 17]})
    cur = pd.DataFrame({"train_no": ["01023"] * 3, "seq": [10, 13, 14], "next_station_seq": [13, 14, 20]})
    ok = bd._next_regular_ok(cur)
    bd.REGULAR_STOPS.clear()
    # 10 -> 13 is consecutive (11, 12 are passed without stopping); 14 -> 20 skips regular stop 17
    assert ok.tolist() == [True, True, False]
