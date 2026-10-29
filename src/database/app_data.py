"""Read the app's railway data from SQLite, in the same shape as the processed CSVs.

    master    = load_master_db()     # train_runs + station names (== RailScan_Master_Dataset.csv)
    timetable = load_timetable_db()  # timetable table (== RailScan_Timetable.csv)

The CSVs remain the source the database is built from (scripts/setup_database.py all);
``tests/test_app_data.py`` checks that both give identical data.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .connection import PathLike, connect, database_ready
from .schema import DB_TO_DATASET


def data_ready(db_path: Optional[PathLike] = None) -> bool:
    """True when the database holds both the observations and the timetable."""
    return database_ready(db_path, "train_runs") and database_ready(db_path, "timetable")


def load_master_db(db_path: Optional[PathLike] = None) -> pd.DataFrame:
    """train_runs with dataset column names, station names, and the master CSV's derived columns."""
    from src.stop_analysis import stop_category
    with connect(db_path, read_only=True) as conn:
        df = pd.read_sql_query(
            "SELECT r.*, s.station_name AS station_name, n.station_name AS next_station_name "
            "FROM train_runs r "
            "LEFT JOIN stations s ON s.station_code = r.station_code "
            "LEFT JOIN stations n ON n.station_code = r.next_station_code "
            "ORDER BY r.journey_date, r.observation_time, r.train_no, r.seq", conn)
    df = df.rename(columns=DB_TO_DATASET).drop(columns=["id", "train_key", "actual_time_basis"], errors="ignore")
    df["train_no"] = df["train_no"].astype(str).str.zfill(5)
    for c in ("observation_time", "current_scheduled_arrival", "current_scheduled_departure",
              "next_scheduled_arrival", "expected_next_station_arrival_baseline", "target_next_actual_arrival",
              "actual_arrival", "actual_departure"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    df["journey_date"] = pd.to_datetime(df["journey_date"]).dt.date
    # derived columns that the master CSV carries (same definitions as scripts/prepare_dataset.py)
    df["scheduled_halt_min"] = df["scheduled_stop_min"]
    df["stop_category"] = df["scheduled_halt_min"].map(stop_category)
    df["arrival_delay_min"] = df["current_arrival_delay_min"]
    df["departure_delay_min"] = df["current_departure_delay_min"]
    df["current_delay_min"] = df["current_observed_delay_min"]
    df["observation_date"] = df["observation_time"].dt.date.astype(str)
    return df


def load_timetable_db(db_path: Optional[PathLike] = None) -> pd.DataFrame:
    with connect(db_path, read_only=True) as conn:
        tt = pd.read_sql_query("SELECT * FROM timetable ORDER BY train_no, seq", conn)
    tt["train_no"] = tt["train_no"].astype(str).str.zfill(5)
    return tt
