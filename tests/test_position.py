"""Estimated train position (no live data): timeline from schedule + delays, interpolation, map points."""
import numpy as np
import pandas as pd
import pytest

from src.position import estimate_from_forecast, locate, map_points, timeline


def _route():
    # SYNTHETIC 3-stop route: A dep 22:00 -> B arr 23:00 (halt 2) -> C arr 01:00 next day
    return pd.DataFrame({
        "station_code": ["A", "B", "C"], "station_name": ["ALPHA", "BETA", None],
        "halt_min": [np.nan, 2.0, np.nan],
        "latitude": [26.0, 27.0, 28.0], "longitude": [80.0, 79.0, 78.0]})


def _later(delays):
    return pd.DataFrame({"station_code": ["B", "C"],
                         "scheduled_arrival": pd.to_datetime(["2030-01-01 23:00", "2030-01-02 01:00"]),
                         "predicted_delay_min": delays})


def _tl(delays=(10.0, 10.0), start_delay=10):
    dep = pd.Timestamp("2030-01-01 22:00") + pd.Timedelta(minutes=start_delay)
    return timeline(_route(), "A", dep, _later(list(delays)), "predicted_delay_min")


def test_timeline_adds_delay_and_halt():
    tl = _tl().set_index("station_code")
    assert tl.loc["A", "departure"] == pd.Timestamp("2030-01-01 22:10")
    assert tl.loc["B", "arrival"] == pd.Timestamp("2030-01-01 23:10")
    assert tl.loc["B", "departure"] == pd.Timestamp("2030-01-01 23:12")       # + 2 min halt
    assert tl.loc["C", "arrival"] == pd.Timestamp("2030-01-02 01:10")         # across midnight
    assert pd.isna(tl.loc["C", "departure"])                                  # destination


def test_before_departure_and_after_arrival():
    tl = _tl()
    assert locate(tl, "2030-01-01 21:00")["state"] == "not_started"
    end = locate(tl, "2030-01-02 02:00")
    assert end["state"] == "arrived" and end["latitude"] == 28.0


def test_halt_at_intermediate_station():
    p = locate(_tl(), "2030-01-01 23:11")
    assert p["state"] == "at_station" and "BETA (B)" in p["text"] and p["latitude"] == 27.0


def test_interpolates_between_stations_by_time():
    p = locate(_tl(), "2030-01-01 22:40")          # 30 of 60 min between A (22:10) and B (23:10)
    assert p["state"] == "between" and p["from"] == "A" and p["to"] == "B"
    assert p["progress"] == pytest.approx(0.5)
    assert p["latitude"] == pytest.approx(26.5) and p["longitude"] == pytest.approx(79.5)
    assert "30 min" in p["text"]


def test_missing_coordinates_give_text_but_no_point():
    r = _route()
    r.loc[r.station_code == "B", ["latitude", "longitude"]] = np.nan
    tl = timeline(r, "A", pd.Timestamp("2030-01-01 22:10"), _later([10.0, 10.0]), "predicted_delay_min")
    p = locate(tl, "2030-01-01 22:40")
    assert p["state"] == "between" and p["latitude"] is None
    pts = map_points(tl, p)
    assert len(pts) == 2 and "estimated train position" not in set(pts.label)


def test_fallback_coordinates_fill_gaps():
    r = _route()
    r.loc[r.station_code == "B", ["latitude", "longitude"]] = np.nan
    tl = timeline(r, "A", pd.Timestamp("2030-01-01 22:10"), _later([10.0, 10.0]), "predicted_delay_min",
                  fallback_coords={"B": (27.0, 79.0)})
    assert locate(tl, "2030-01-01 22:40")["latitude"] == pytest.approx(26.5)


def test_station_without_name_is_shown_by_code():
    p = locate(_tl(), "2030-01-02 00:30")
    assert p["state"] == "between" and "and C," in p["text"]


def test_map_marks_the_estimate_in_red():
    tl = _tl()
    pts = map_points(tl, locate(tl, "2030-01-01 22:40"))
    assert (pts.label == "estimated train position").sum() == 1
    assert pts.loc[pts.label == "estimated train position", "color"].iat[0] == "#d1342f"


def test_only_successful_forecasts_are_placed():
    assert estimate_from_forecast(_route(), {"status": "unknown_station"}, pd.Timestamp("2030-01-01")) is None
    f = {"status": "ok", "station_code": "A", "reference_time": pd.Timestamp("2030-01-01 22:10"),
         "route": _later([10.0, 10.0])}
    assert estimate_from_forecast(_route(), f, pd.Timestamp("2030-01-01 22:40"))["position"]["state"] == "between"
