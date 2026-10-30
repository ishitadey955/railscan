"""Read the app's railway data from SQLite, in the same shape as the processed CSVs.

    master    = load_master_db()     # train_runs + station names (== RailScan_Master_Dataset.csv)
    timetable = load_timetable_db()  # timetable table (== RailScan_Timetable.csv)

The CSVs remain the source the database is built from (scripts/setup_database.py all);
``tests/test_app_data.py`` checks that both give identical data.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any, Callable, Optional

import pandas as pd

from .connection import PathLike, connect, database_ready, table_exists, transaction
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


# Summaries of the ~1M-row train_runs table take 5-40 s to compute, so they are stored in
# this small table and recomputed only when train_runs changes. Every import adds a row to
# database_metadata, so (last import, last train_runs id) identifies the table contents.
CACHE_TABLE = "app_cache"


def _fingerprint(conn: sqlite3.Connection) -> str:
    meta = conn.execute("SELECT MAX(id) FROM database_metadata").fetchone()[0]         if table_exists(conn, "database_metadata") else None
    return f"{meta}:{conn.execute('SELECT MAX(id) FROM train_runs').fetchone()[0]}"


def _cached(db_path: Optional[PathLike], key: str, compute: Callable[[sqlite3.Connection], Any]) -> Any:
    """``compute(conn)`` (JSON-serialisable), read from CACHE_TABLE when still valid."""
    with connect(db_path, read_only=True) as conn:
        fp = _fingerprint(conn)
        if table_exists(conn, CACHE_TABLE):
            row = conn.execute(f"SELECT value FROM {CACHE_TABLE} WHERE key = ? AND fingerprint = ?",
                               (key, fp)).fetchone()
            if row:
                return json.loads(row[0])
        value = compute(conn)
    try:
        with connect(db_path) as conn, transaction(conn):
            conn.execute(f"CREATE TABLE IF NOT EXISTS {CACHE_TABLE} "
                         "(key TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, value TEXT NOT NULL)")
            conn.execute(f"INSERT OR REPLACE INTO {CACHE_TABLE} VALUES (?, ?, ?)", (key, fp, json.dumps(value)))
    except sqlite3.Error:      # read-only database: the result is still right, only not stored
        pass
    return value


def train_list_db(db_path: Optional[PathLike] = None) -> pd.DataFrame:
    """One row per train: train_no, train_name, journeys, rows (same as railscan_service.list_trains)."""
    def compute(conn):
        return pd.read_sql_query(
            "SELECT train_no, MAX(train_name) AS train_name, COUNT(DISTINCT journey_date) AS journeys, "
            "COUNT(*) AS rows FROM train_runs GROUP BY train_no ORDER BY train_no", conn).to_dict("list")
    df = pd.DataFrame(_cached(db_path, "train_list", compute))
    df["train_no"] = df["train_no"].astype(str).str.zfill(5)
    return df


def train_list_parquet(path: Optional[PathLike] = None) -> pd.DataFrame:
    """train_list_db() from the committed Parquet copy of the dataset (the file the database is
    built from), for use while the database is not built yet. Same result, under a second."""
    import pyarrow.parquet as pq
    from .import_data import DEFAULT_PARQUET
    d = pq.read_table(path or DEFAULT_PARQUET, columns=["train_no", "train_name", "journey_date"]).to_pandas()
    d["train_no"] = d["train_no"].astype(str).str.zfill(5)
    return (d.groupby("train_no").agg(train_name=("train_name", "max"), journeys=("journey_date", "nunique"),
                                      rows=("train_no", "size")).reset_index())


def overview_db(db_path: Optional[PathLike] = None) -> dict:
    """Dataset-wide figures for the Data page, computed in SQL (no full load)."""
    def compute(conn):
        one = lambda sql: conn.execute(sql).fetchone()[0]
        cols = [r[1] for r in conn.execute("PRAGMA table_info(train_runs)") if r[1] not in ("id", "train_key")]
        nulls = conn.execute("SELECT COUNT(*), " + ", ".join(f'SUM("{c}" IS NULL)' for c in cols)
                             + " FROM train_runs").fetchone()
        return {
            "cols": cols, "nulls": list(tuple(nulls)),
            "risk": [list(r) for r in conn.execute("SELECT next_delay_risk_label, COUNT(*) FROM train_runs GROUP BY 1")],
            "trains": one("SELECT COUNT(DISTINCT train_no) FROM train_runs"),
            "journeys": one("SELECT COUNT(*) FROM (SELECT DISTINCT train_no, journey_date FROM train_runs)"),
            "stations": one("SELECT COUNT(*) FROM (SELECT station_code FROM train_runs UNION "
                            "SELECT next_station_code FROM train_runs WHERE next_station_code IS NOT NULL)"),
            "date_min": one("SELECT MIN(journey_date) FROM train_runs"),
            "date_max": one("SELECT MAX(journey_date) FROM train_runs"),
        }
    c = _cached(db_path, "overview", compute)
    with connect(db_path, read_only=True) as conn:
        delays = pd.read_sql_query("SELECT next_station_arrival_delay_min AS d FROM train_runs", conn)["d"]
    cols, nulls = c["cols"], c["nulls"]
    rows = nulls[0]
    overview = {"rows": rows, "columns": len(cols), "trains": c["trains"], "journeys": c["journeys"],
                "stations": c["stations"], "date_min": pd.Timestamp(c["date_min"]).date(),
                "date_max": pd.Timestamp(c["date_max"]).date()}
    missing = pd.DataFrame({"column": [DB_TO_DATASET.get(col, col) for col in cols],
                            "missing %": [round(n / rows * 100, 2) if rows else 0.0 for n in nulls[1:]]})
    missing = missing[missing["missing %"] > 0].sort_values("missing %", ascending=False).reset_index(drop=True)
    return {"overview": overview, "missing": missing, "delays": delays, "risk": dict(map(tuple, c["risk"]))}


def load_timetable_db(db_path: Optional[PathLike] = None) -> pd.DataFrame:
    with connect(db_path, read_only=True) as conn:
        tt = pd.read_sql_query("SELECT * FROM timetable ORDER BY train_no, seq", conn)
    tt["train_no"] = tt["train_no"].astype(str).str.zfill(5)
    return tt
