"""Scheduled halt, actual stop, delay and stop-category rules (incl. midnight crossings)."""
import math

import pandas as pd
import pytest

from src.stop_analysis import (actual_stop_min, delay_min, historical_stop_stats, minutes_between,
                               scheduled_halt_min, stop_category)


def test_scheduled_halt_same_day():
    assert scheduled_halt_min("2025-03-10 14:05", "2025-03-10 14:10") == 5


def test_halt_across_midnight_with_dates():
    assert scheduled_halt_min("2025-03-10 23:58", "2025-03-11 00:03") == 5


def test_halt_across_midnight_clock_times_only():
    # bare clock times: a negative difference means the stop crossed midnight
    assert minutes_between("23:58", "00:03") == 5


def test_actual_stop_and_delay_signs():
    assert actual_stop_min(pd.Timestamp("2025-01-01 21:46"), pd.Timestamp("2025-01-01 21:57")) == 11
    assert delay_min("2025-01-01 21:45", "2025-01-01 22:00") == 15     # late
    assert delay_min("2025-01-01 21:45", "2025-01-01 21:40") == -5     # early


def test_missing_values_return_none():
    assert scheduled_halt_min(None, "2025-01-01 10:00") is None
    assert actual_stop_min(pd.NaT, pd.Timestamp("2025-01-01")) is None


@pytest.mark.parametrize("halt,cat", [(0, "Short"), (4.9, "Short"), (5, "Moderate"), (10, "Moderate"),
                                      (10.5, "Long"), (None, None), (-1, None), (math.nan, None)])
def test_stop_category_boundaries(halt, cat):
    assert stop_category(halt) == cat


def _frame():
    return pd.DataFrame({
        "train_no": ["01023"] * 4 + ["01024"],
        "station_code": ["KRG"] * 5,
        "scheduled_stop_min": [2, 2, 2, 2, 2],
        "actual_stop_min": [3, 6, None, 2, 9],
        "observation_time": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03",
                                            "2025-01-04", "2025-01-02"]),
    })


def test_historical_stop_stats_uses_only_valid_and_earlier_rows():
    s = historical_stop_stats(_frame(), "KRG", "01023", before="2025-01-04")
    # rows on 01-01 (3 min) and 01-02 (6 min); 01-03 has no actual stop; 01-04 is not earlier
    assert s["count"] == 2
    assert s["avg_actual_stop_min"] == pytest.approx(4.5)
    assert s["median_actual_stop_min"] == pytest.approx(4.5)
    assert s["stop_reliability_pct"] == pytest.approx(50.0)      # |3-2|<=2 yes, |6-2|<=2 no


def test_historical_stop_stats_train_filter_accepts_unpadded_number():
    assert historical_stop_stats(_frame(), "KRG", "1024")["count"] == 1


def test_no_observations_gives_none_not_a_number():
    assert historical_stop_stats(_frame(), "KRG", "01023", before="2024-01-01") is None
    assert historical_stop_stats(_frame(), "XXXX") is None
