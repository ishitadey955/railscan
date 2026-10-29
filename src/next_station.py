"""Next-station selection, expected arrival and time remaining.

    expected_next_station_arrival = next_station_scheduled_arrival + predicted_delay
    minutes_until_next_station    = expected_next_station_arrival - reference_time

The reference time is the moment the prediction is made. For a recorded
(historical) observation that is its ``observation_time``; using today's
clock with a 2025 record would give meaningless numbers.
Times are naive IST datetimes. Scheduled datetimes in the dataset already
carry the correct calendar date (journey day changes and midnight crossings
were resolved when the dataset was built), so plain subtraction is safe.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd


def next_stop(journey: pd.DataFrame, station_code: str, seq: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """The stop after ``station_code`` on one journey (rows of one train and journey date).

    Returns ``None`` for an unknown station or when the dataset has no next stop.
    If the train calls at the station twice, ``seq`` selects the call.
    """
    rows = journey[journey["station_code"] == station_code]
    if seq is not None:
        rows = rows[rows["seq"] == seq]
    if rows.empty:
        return None
    r = rows.sort_values("seq").iloc[0]
    if pd.isna(r.get("next_station_code")):
        return None
    return {"station_code": r["next_station_code"], "seq": r.get("next_station_seq"),
            "scheduled_arrival": pd.Timestamp(r["next_scheduled_arrival"])
            if pd.notna(r.get("next_scheduled_arrival")) else None,
            "distance_km": r.get("distance_to_next_km"),
            "scheduled_travel_min": r.get("scheduled_travel_to_next_min")}


def expected_arrival(scheduled_arrival: Any, predicted_delay_min: Optional[float]) -> Optional[pd.Timestamp]:
    """Scheduled arrival + predicted delay; ``None`` if either is missing."""
    if scheduled_arrival is None or predicted_delay_min is None:
        return None
    if pd.isna(scheduled_arrival) or pd.isna(predicted_delay_min):
        return None
    return pd.Timestamp(scheduled_arrival) + pd.Timedelta(minutes=float(predicted_delay_min))


def minutes_until(expected: Any, reference: Any) -> Optional[float]:
    """Minutes from ``reference`` to ``expected`` (negative means the time has passed)."""
    if expected is None or reference is None or pd.isna(expected) or pd.isna(reference):
        return None
    return (pd.Timestamp(expected) - pd.Timestamp(reference)).total_seconds() / 60


def describe_remaining(minutes: Optional[float]) -> str:
    """Readable text for :func:`minutes_until`, including the 'already due' case."""
    if minutes is None:
        return "unknown"
    if minutes < 0:
        return f"due {abs(minutes):.0f} min ago"
    if minutes < 1:
        return "due now"
    h, m = divmod(round(minutes), 60)
    return f"{h} h {m} min" if h else f"{m} min"
