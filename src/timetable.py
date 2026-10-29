"""Current timetable of each train, placed on any calendar date.

Built by scripts/build_timetable.py from the most recent recorded journeys
(data/processed/RailScan_Timetable.csv). Times are naive IST. A stop's day
offset says on which day after the journey date the train reaches it, so a
journey starting 21:40 reaches Satara at 00:22 on the next date.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TIMETABLE_CSV = ROOT / "data" / "processed" / "RailScan_Timetable.csv"
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


class TimetableUnavailable(RuntimeError):
    pass


def load_timetable(path: Path = TIMETABLE_CSV, source: str = "auto") -> pd.DataFrame:
    """``source``: "auto" (SQLite timetable table if filled, else CSV), "sqlite" or "csv"."""
    if source in ("auto", "sqlite"):
        from src.database.app_data import data_ready, load_timetable_db
        if data_ready():
            tt = load_timetable_db()
            tt.attrs["source"] = "SQLite"
            return tt.sort_values(["train_no", "seq"]).reset_index(drop=True)
        if source == "sqlite":
            raise TimetableUnavailable("The SQLite database has no timetable. Run: python scripts/setup_database.py all")
    if not Path(path).exists():
        raise TimetableUnavailable(f"Timetable not found: {path}. Run: python scripts/build_timetable.py")
    tt = pd.read_csv(path, dtype={"train_no": str, "sch_arrival": str, "sch_departure": str})
    tt["train_no"] = tt["train_no"].str.zfill(5)
    tt = tt.sort_values(["train_no", "seq"]).reset_index(drop=True)
    tt.attrs["source"] = "CSV"
    return tt


def route(tt: pd.DataFrame, train_no: str) -> pd.DataFrame:
    return tt[tt.train_no == str(train_no).zfill(5)].sort_values("seq").reset_index(drop=True)


def runs_on(tt: pd.DataFrame, train_no: str, date) -> Optional[bool]:
    """True/False if the recorded running days are known, None if not."""
    r = route(tt, train_no)
    if r.empty or pd.isna(r.runs_on.iat[0]):
        return None
    return DAY_NAMES[pd.Timestamp(date).day_of_week] in str(r.runs_on.iat[0]).split(",")


def _at(date, hhmm, offset) -> Optional[pd.Timestamp]:
    if hhmm is None or pd.isna(hhmm) or offset is None or pd.isna(offset):
        return None
    return pd.Timestamp(date).normalize() + pd.Timedelta(days=int(offset)) + pd.to_timedelta(f"{hhmm}:00")


def schedule_on(tt: pd.DataFrame, train_no: str, journey_date) -> pd.DataFrame:
    """The train's route with scheduled datetimes for a journey starting on ``journey_date``.

    Adds: sched_arr, sched_dep (Timestamps or None), halt_min, next_station_code,
    next_seq, next_sched_arr, travel_to_next_min, distance_to_next_km.
    """
    r = route(tt, train_no).copy()
    if r.empty:
        return r
    r["sched_arr"] = [_at(journey_date, t, o) for t, o in zip(r.sch_arrival, r.arrival_day_offset)]
    r["sched_dep"] = [_at(journey_date, t, o) for t, o in zip(r.sch_departure, r.departure_day_offset)]
    mins = lambda a, b: None if pd.isna(a) or pd.isna(b) else (b - a).total_seconds() / 60
    r["halt_min"] = [mins(a, d) for a, d in zip(r.sched_arr, r.sched_dep)]
    r["next_station_code"] = r.station_code.shift(-1)
    r["next_station_name"] = r.station_name.shift(-1)
    r["next_seq"] = r.seq.shift(-1)
    r["next_sched_arr"] = r.sched_arr.shift(-1)
    r["travel_to_next_min"] = [mins(d, a) for d, a in zip(r.sched_dep, r.next_sched_arr)]
    # per-journey median leg length (build_timetable.py); falls back to the difference of positions
    legs = r["distance_to_next_km"] if "distance_to_next_km" in r else None
    r["distance_to_next_km"] = (r.distance_km.shift(-1) - r.distance_km) if legs is None else legs.fillna(r.distance_km.shift(-1) - r.distance_km)
    return r


def departure_stations(tt: pd.DataFrame, train_no: str) -> List[dict]:
    """Stops the train departs from (every stop except the destination)."""
    r = route(tt, train_no)
    r = r[r.sch_departure.notna()]
    return [{"code": x.station_code, "name": x.station_name or x.station_code, "seq": int(x.seq),
             "label": f"{x.station_name or x.station_code} ({x.station_code}) - dep. {x.sch_departure}"}
            for x in r.itertuples()]
