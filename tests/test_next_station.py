"""Next-station selection, expected arrival and time remaining."""
import pandas as pd

from src.next_station import describe_remaining, expected_arrival, minutes_until, next_stop


def _journey():
    return pd.DataFrame({
        "station_code": ["JSP", "MRJ", "JSP"], "seq": [6, 7, 9],
        "next_station_code": ["MRJ", "JSP", None], "next_station_seq": [7, 8, None],
        "next_scheduled_arrival": pd.to_datetime(["2023-11-06 00:37", "2023-11-06 01:00", None]),
        "distance_to_next_km": [11.2, 8.0, None], "scheduled_travel_to_next_min": [22, 15, None],
    })


def test_next_stop_found():
    n = next_stop(_journey(), "MRJ")
    assert n["station_code"] == "JSP" and n["seq"] == 8
    assert n["scheduled_arrival"] == pd.Timestamp("2023-11-06 01:00")


def test_next_stop_repeated_station_uses_seq():
    assert next_stop(_journey(), "JSP", seq=6)["station_code"] == "MRJ"
    assert next_stop(_journey(), "JSP", seq=9) is None          # no next stop recorded


def test_unknown_station():
    assert next_stop(_journey(), "NDLS") is None


def test_expected_arrival_crosses_midnight():
    exp = expected_arrival(pd.Timestamp("2025-03-10 23:50"), 25)
    assert exp == pd.Timestamp("2025-03-11 00:15")


def test_expected_arrival_missing_inputs():
    assert expected_arrival(None, 5) is None
    assert expected_arrival(pd.Timestamp("2025-01-01"), None) is None
    assert expected_arrival(pd.NaT, 5) is None


def test_minutes_until_and_negative_remaining():
    ref = pd.Timestamp("2025-03-10 23:40")
    assert minutes_until(pd.Timestamp("2025-03-11 00:15"), ref) == 35
    assert minutes_until(pd.Timestamp("2025-03-10 23:30"), ref) == -10
    assert describe_remaining(-10) == "due 10 min ago"
    assert describe_remaining(0.4) == "due now"
    assert describe_remaining(75) == "1 h 15 min"
    assert describe_remaining(None) == "unknown"
