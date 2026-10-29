"""Scheduled halt and recorded stop-duration analysis.

Scheduled halt (timetable) and actual stop (recorded observation) are kept
apart on purpose: the first is a plan, the second is what happened.

All times are naive local times in Asia/Kolkata (IST), as in the source data.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

STOP_TOLERANCE_MIN = 2      # a stop is "reliable" if |actual - scheduled| <= 2 min
SHORT_MAX = 5               # Short: < 5 min
MODERATE_MAX = 10           # Moderate: 5..10 min inclusive; Long: > 10 min


def minutes_between(start: Any, end: Any) -> Optional[float]:
    """Minutes from ``start`` to ``end``.

    Full datetimes are subtracted directly, so a stop that runs past midnight
    (23:58 -> 00:03 on the next date) gives +5, not -1435. For bare clock
    times ("23:58", "00:03") without dates, a negative difference is read as a
    midnight crossing and one day is added. Returns ``None`` if either is missing.
    """
    if start is None or end is None or pd.isna(start) or pd.isna(end):
        return None
    if isinstance(start, str) and isinstance(end, str) and len(start) <= 8 and len(end) <= 8:
        s, e = pd.to_datetime(start, format="mixed"), pd.to_datetime(end, format="mixed")
        diff = (e - s).total_seconds() / 60
        return diff + 1440 if diff < 0 else diff
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    if pd.isna(s) or pd.isna(e):
        return None
    return (e - s).total_seconds() / 60


def scheduled_halt_min(scheduled_arrival: Any, scheduled_departure: Any) -> Optional[float]:
    """scheduled_departure - scheduled_arrival (midnight-safe)."""
    return minutes_between(scheduled_arrival, scheduled_departure)


def actual_stop_min(actual_arrival: Any, actual_departure: Any) -> Optional[float]:
    """actual_departure - actual_arrival (midnight-safe)."""
    return minutes_between(actual_arrival, actual_departure)


def delay_min(scheduled: Any, actual: Any) -> Optional[float]:
    """actual - scheduled; positive = late, negative = early."""
    return minutes_between(scheduled, actual)


def stop_category(halt_min: Optional[float]) -> Optional[str]:
    """Short (< 5), Moderate (5-10 inclusive), Long (> 10). Based on the SCHEDULED halt."""
    if halt_min is None or pd.isna(halt_min) or halt_min < 0:
        return None
    if halt_min < SHORT_MAX:
        return "Short"
    if halt_min <= MODERATE_MAX:
        return "Moderate"
    return "Long"


def historical_stop_stats(frame: pd.DataFrame, station_code: str, train_no: Optional[str] = None,
                          before: Any = None, tolerance: float = STOP_TOLERANCE_MIN) -> Optional[Dict[str, Any]]:
    """Recorded stop durations at a station, optionally only those completed before ``before``.

    ``frame`` needs ``station_code``, ``actual_stop_min``, ``scheduled_stop_min``,
    ``observation_time`` (and ``train_no`` when filtering by train).
    Returns ``None`` if there is no valid observed stop - never a made-up value.
    """
    if frame is None or frame.empty:
        return None
    rows = frame[(frame["station_code"] == station_code)
                 & frame["actual_stop_min"].notna() & frame["scheduled_stop_min"].notna()]
    if train_no is not None:
        rows = rows[rows["train_no"].astype(str).str.lstrip("0") == str(train_no).lstrip("0")]
    if before is not None:
        rows = rows[pd.to_datetime(rows["observation_time"]) < pd.Timestamp(before)]
    if rows.empty:
        return None
    a = rows["actual_stop_min"].astype(float)
    within = (a - rows["scheduled_stop_min"].astype(float)).abs() <= tolerance
    return {
        "count": int(len(a)),
        "avg_actual_stop_min": float(a.mean()),
        "median_actual_stop_min": float(a.median()),
        "std_actual_stop_min": float(a.std(ddof=1)) if len(a) > 1 else None,
        "stop_reliability_pct": float(within.mean() * 100),
        "tolerance_min": tolerance,
        "first_date": str(pd.to_datetime(rows["observation_time"]).min().date()),
        "last_date": str(pd.to_datetime(rows["observation_time"]).max().date()),
    }
