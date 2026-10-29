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


def load_master_db(db_path: Optional[PathLike] = None, train_no: Optional[str] = None) -> pd.DataFrame:
    """train_runs with dataset column names, station names, and the master CSV's derived columns.

    ``train_no``: only that train's rows (the app loads one train at a time; the
    whole table is close to a million rows).
    """
    from src.stop_analysis import stop_category
    where, params = "", ()
    if train_no is not None:
        where, params = "WHERE r.train_no = ? ", (str(train_no).zfill(5),)
    with connect(db_path, read_only=True) as conn:
        df = pd.read_sql_query(
            "SELECT r.*, s.station_name AS station_name, n.station_name AS next_station_name "
            "FROM train_runs r "
            "LEFT JOIN stations s ON s.station_code = r.station_code "
            "LEFT JOIN stations n ON n.station_code = r.next_station_code "
            + where +
            "ORDER BY r.journey_date, r.observation_time, r.train_no, r.seq", conn, params=params)
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


def train_list_db(db_path: Optional[PathLike] = None) -> pd.DataFrame:
    """One row per train: train_no, train_name, journeys, rows (same as railscan_service.list_trains)."""
    with connect(db_path, read_only=True) as conn:
        df = pd.read_sql_query(
            "SELECT train_no, MAX(train_name) AS train_name, COUNT(DISTINCT journey_date) AS journeys, "
            "COUNT(*) AS rows FROM train_runs GROUP BY train_no ORDER BY train_no", conn)
    df["train_no"] = df["train_no"].astype(str).str.zfill(5)
    return df


def overview_db(db_path: Optional[PathLike] = None) -> dict:
    """Dataset-wide figures for the Data page, computed in SQL (no full load)."""
    with connect(db_path, read_only=True) as conn:
        one = lambda sql: conn.execute(sql).fetchone()[0]
        cols = [r[1] for r in conn.execute("PRAGMA table_info(train_runs)") if r[1] not in ("id", "train_key")]
        nulls = conn.execute("SELECT COUNT(*), " + ", ".join(f'SUM("{c}" IS NULL)' for c in cols)
                             + " FROM train_runs").fetchone()
        rows = nulls[0]
        delays = pd.read_sql_query("SELECT next_station_arrival_delay_min AS d FROM train_runs", conn)["d"]
        risk = dict(conn.execute("SELECT next_delay_risk_label, COUNT(*) FROM train_runs GROUP BY 1").fetchall())
        overview = {
            "rows": rows, "columns": len(cols),
            "trains": one("SELECT COUNT(DISTINCT train_no) FROM train_runs"),
            "journeys": one("SELECT COUNT(*) FROM (SELECT DISTINCT train_no, journey_date FROM train_runs)"),
            "stations": one("SELECT COUNT(*) FROM (SELECT station_code FROM train_runs UNION "
                            "SELECT next_station_code FROM train_runs WHERE next_station_code IS NOT NULL)"),
            "date_min": pd.Timestamp(one("SELECT MIN(journey_date) FROM train_runs")).date(),
            "date_max": pd.Timestamp(one("SELECT MAX(journey_date) FROM train_runs")).date(),
        }
    missing = pd.DataFrame({"column": [DB_TO_DATASET.get(c, c) for c in cols],
                            "missing %": [round(n / rows * 100, 2) if rows else 0.0 for n in nulls[1:]]})
    missing = missing[missing["missing %"] > 0].sort_values("missing %", ascending=False).reset_index(drop=True)
    return {"overview": overview, "missing": missing, "delays": delays, "risk": risk}


def load_timetable_db(db_path: Optional[PathLike] = None) -> pd.DataFrame:
    with connect(db_path, read_only=True) as conn:
        tt = pd.read_sql_query("SELECT * FROM timetable ORDER BY train_no, seq", conn)
    tt["train_no"] = tt["train_no"].astype(str).str.zfill(5)
    return tt
