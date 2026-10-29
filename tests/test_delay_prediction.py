"""Delay model serving, feature consistency, leakage guards and the real northern-India dataset."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data" / "processed" / "RailScan_Final_Dataset.csv"
MODEL_DIR = ROOT / "models" / "delay"

from src.delay_prediction import DelayPredictor, dc, risk_label  # noqa: E402

needs_data = pytest.mark.skipif(not FINAL.exists(), reason="processed dataset not built")
needs_models = pytest.mark.skipif(not (MODEL_DIR / "delay_regressor.pkl").exists(), reason="models not trained")


# ------------------------------------------------------------ leakage guards
def test_no_target_or_future_column_is_a_feature():
    for f in dc.FEATURE_COLUMNS:
        assert f not in (dc.REG_TARGET, dc.CLF_TARGET, dc.RISK_TARGET)
        for pat in dc.FORBIDDEN_FEATURE_PATTERNS:
            assert not re.search(pat, f), (f, pat)


def test_saved_feature_list_matches_code():
    spec = json.loads((MODEL_DIR / "feature_columns.json").read_text())
    assert spec["features"] == dc.FEATURE_COLUMNS


def test_missing_departure_delay_falls_back_to_arrival_delay():
    row = {"current_arrival_delay_min": 294.0, "current_departure_delay_min": np.nan,
           "current_scheduled_arrival": "2026-08-12 23:00", "current_scheduled_departure": "2026-08-12 23:05",
           "train_no": "01023", "train_type": "TOD", "station_code": "MRJ", "next_station_code": "JSP"}
    X = DelayPredictor.features(row)
    assert X["current_departure_delay_min"].iat[0] == 294.0
    assert list(X.columns) == dc.FEATURE_COLUMNS


@pytest.mark.parametrize("d,label", [(-3, "Low"), (14.9, "Low"), (15, "Medium"), (59, "Medium"),
                                     (60, "High"), (None, None)])
def test_risk_label_rule(d, label):
    assert risk_label(d) == label


# ------------------------------------------------------------ real dataset
MAX_ROWS = 1_000_000                       # dataset size limit (fewer than a million rows)
KEYS = ["train_no", "journey_date", "seq", "station_code", "next_station_code", "observation_time",
        "target_next_actual_arrival"]


@needs_data
def test_dataset_is_within_the_row_limit_and_has_valid_keys():
    df = pd.read_csv(FINAL, dtype={"train_no": str}, usecols=KEYS)
    manifest = pd.read_csv(ROOT / "data" / "processed" / "dataset_manifest.csv").set_index("file")
    assert 0 < len(df) < MAX_ROWS and len(df) == manifest.loc["RailScan_Final_Dataset.csv", "rows"]
    assert not df.duplicated(["train_no", "journey_date", "seq"]).any()
    assert (df.next_station_code != df.station_code).all()
    assert (pd.to_datetime(df.observation_time) <= pd.to_datetime(df.target_next_actual_arrival)).all()


@needs_data
def test_target_matches_its_definition_on_real_rows():
    df = pd.read_csv(FINAL, usecols=["target_next_actual_arrival", "next_scheduled_arrival",
                                     "next_station_arrival_delay_min", "next_station_delayed_15"])
    d = (pd.to_datetime(df.target_next_actual_arrival) - pd.to_datetime(df.next_scheduled_arrival)
         ).dt.total_seconds() / 60
    assert np.allclose(d, df.next_station_arrival_delay_min)
    assert ((df.next_station_arrival_delay_min >= 15).astype(int) == df.next_station_delayed_15).all()


@needs_data
def test_history_uses_only_earlier_outcomes():
    """Recompute history for some real rows by brute force from strictly earlier rows.

    History is per train, so the rows of a few trains contain everything needed."""
    df = pd.read_csv(FINAL, dtype={"train_no": str})
    df = df[df.train_no.isin(["12229", "12004", "12230"])].reset_index(drop=True)
    assert len(df) > 100
    for c in ("observation_time", "target_next_actual_arrival"):
        df[c] = pd.to_datetime(df[c])
    for i in range(0, len(df), 7):
        r = df.iloc[i]
        past = df[(df.train_no == r.train_no) & (df.next_station_code == r.next_station_code)
                  & (df.target_next_actual_arrival < r.observation_time)]
        assert len(past) == r.historical_observation_count
        stops = df[(df.train_no == r.train_no) & (df.station_code == r.station_code)
                   & (df.observation_time < r.observation_time)
                   & df.actual_stop_min.notna() & df.scheduled_stop_min.notna()]
        assert len(stops) == r.historical_stop_observation_count
        if len(stops):
            rel = ((stops.actual_stop_min - stops.scheduled_stop_min).abs() <= 2).mean() * 100
            assert rel == pytest.approx(r.historical_stop_reliability_pct)


@needs_data
def test_chronological_split_has_no_shared_dates():
    s = {n: pd.read_csv(ROOT / "data" / "processed" / f"{n}.csv", usecols=["journey_date"])
         for n in ("train", "validation", "test")}
    assert s["train"].journey_date.max() < s["validation"].journey_date.min()
    assert s["validation"].journey_date.max() < s["test"].journey_date.min()
    assert sum(len(v) for v in s.values()) == len(pd.read_csv(FINAL, usecols=["seq"]))


# ------------------------------------------------------------ serving
@needs_data
@needs_models
def test_serving_predictions_equal_training_pipeline():
    """App path (DelayPredictor) must reproduce the training-path predictions exactly."""
    import joblib
    df = pd.read_csv(FINAL, dtype={"train_no": str}, nrows=50)
    p = DelayPredictor.load()
    served = [r["predicted_next_station_delay_min"] for r in p.predict(df)]
    train_path = joblib.load(MODEL_DIR / "delay_regressor.pkl").predict(
        dc.make_features(pd.read_csv(FINAL, nrows=50)))
    assert np.allclose(served, train_path)
    assert p.unknown_categories(p.features(df)) == {}


@needs_models
def test_unknown_station_is_reported_not_hidden():
    p = DelayPredictor.load()
    row = {"train_no": "12229", "train_type": "SUF", "station_code": "PUNE", "next_station_code": "GZB",
           "current_arrival_delay_min": 10, "current_departure_delay_min": 12,
           "current_scheduled_arrival": "2025-01-01 10:00", "current_scheduled_departure": "2025-01-01 10:02"}
    out = p.predict(row)[0]
    assert out["unknown_categories"] == {"station_code": ["PUNE"]}       # Pune is outside the northern data
    assert 0.0 <= out["probability_delayed_15"] <= 1.0
