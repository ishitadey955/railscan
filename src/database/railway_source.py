"""SQLite-backed replacement for ``src.railway.railway_data.RailwayData``.

``TrainStateService`` and ``RailwayPipeline`` only call a handful of lookup
methods. This class answers them with indexed SQL queries, so a lookup loads
one journey (or a GROUP BY summary) instead of the whole CSV.

Frames are built through ``RailwayData(...)`` itself, so column names, dtypes
and normalisation are exactly those of the CSV path. Target columns are not
loaded at all: the train-state layer never returns them.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd

from src.railway.railway_data import RailwayData, RailwayDataUnavailable, parse_date

from .connection import DatabaseUnavailable, PathLike, connect, resolve_db_path, table_count
from .repository import STATE_COLUMNS, normalize_station_code, train_key
from .schema import DB_TO_DATASET

_SELECT = ", ".join(f"{c} AS {DB_TO_DATASET.get(c, c)}" for c in STATE_COLUMNS)


class SQLiteRailwayData:
    """Same lookup interface as :class:`RailwayData`, served from ``train_runs``."""

    backend = "sqlite"

    def __init__(self, db_path: Optional[PathLike] = None):
        self.db_path = resolve_db_path(db_path)
        try:
            with connect(self.db_path, read_only=True) as conn:
                self._n = table_count(conn, "train_runs")
        except (DatabaseUnavailable, sqlite3.Error) as exc:
            raise RailwayDataUnavailable(str(exc)) from exc
        if self._n == 0:
            raise RailwayDataUnavailable(
                f"train_runs in {self.db_path} is empty. Run `python scripts/setup_database.py import-data`.")

    # ------------------------------------------------------------------ plumbing
    def _query(self, where: str, params: List[Any]) -> pd.DataFrame:
        with connect(self.db_path, read_only=True) as conn:
            cur = conn.execute(f"SELECT {_SELECT} FROM train_runs WHERE {where} ORDER BY journey_date, seq", params)
            cols = [d[0] for d in cur.description]
            frame = pd.DataFrame(cur.fetchall(), columns=cols)
        if frame.empty:
            frame = pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
        return RailwayData(frame).df

    def __len__(self) -> int:
        return self._n

    @property
    def columns(self) -> List[str]:
        return [DB_TO_DATASET.get(c, c) for c in STATE_COLUMNS] + ["_tn"]

    # ------------------------------------------------------------------ lookups
    def train_rows(self, train_no: Any) -> pd.DataFrame:
        """All rows of one train (kept for compatibility; prefer the narrower methods)."""
        key = train_key(train_no)
        return self._query("train_key = ?", [key]) if key else self._query("0", [])

    def train_day_rows(self, train_no: Any, journey_date: Any) -> pd.DataFrame:
        key, jd = train_key(train_no), parse_date(journey_date)
        if key is None or jd is None:
            return self._query("0", [])
        return self._query("train_key = ? AND journey_date = ?", [key, jd.date().isoformat()])

    def has_train(self, train_no: Any) -> bool:
        key = train_key(train_no)
        if key is None:
            return False
        with connect(self.db_path, read_only=True) as conn:
            return conn.execute("SELECT 1 FROM train_runs WHERE train_key = ? LIMIT 1", (key,)).fetchone() is not None

    def name_type_counts(self, train_no: Any, journey_date: Any = None,
                         station_code: Optional[str] = None) -> pd.DataFrame:
        """``train_name, train_type, records`` groups (same ordering as the pandas path)."""
        key = train_key(train_no)
        empty = pd.DataFrame({"train_name": [], "train_type": [], "records": []})
        if key is None:
            return empty
        sql = "SELECT train_name, train_type, COUNT(*) AS records FROM train_runs WHERE train_key = ?"
        params: List[Any] = [key]
        if journey_date is not None:
            jd = parse_date(journey_date)
            if jd is None:
                return empty
            sql += " AND journey_date = ?"
            params.append(jd.date().isoformat())
        if station_code:
            sc = normalize_station_code(station_code)
            sql += " AND (station_code = ? OR next_station_code = ?)"
            params += [sc, sc]
        # pandas groupby(dropna=False) sorts values ascending with missing values last
        sql += (" GROUP BY train_name, train_type ORDER BY train_name IS NULL, train_name, "
                "train_type IS NULL, train_type")
        with connect(self.db_path, read_only=True) as conn:
            rows = [dict(r) for r in conn.execute(sql, params)]
        return pd.DataFrame(rows, columns=["train_name", "train_type", "records"]) if rows else empty

    def route_codes(self, train_no: Any) -> List[str]:
        key = train_key(train_no)
        if key is None:
            return []
        with connect(self.db_path, read_only=True) as conn:
            return [r[0] for r in conn.execute(
                "SELECT station_code FROM train_runs WHERE train_key = ? UNION "
                "SELECT next_station_code FROM train_runs WHERE train_key = ? AND next_station_code IS NOT NULL "
                "ORDER BY 1", (key, key))]

    def available_dates(self, train_no: Any, station_code: Optional[str] = None) -> List[pd.Timestamp]:
        key = train_key(train_no)
        if key is None:
            return []
        sql, params = "SELECT DISTINCT journey_date FROM train_runs WHERE train_key = ?", [key]
        if station_code:
            sc = normalize_station_code(station_code)
            sql += " AND (station_code = ? OR next_station_code = ?)"
            params += [sc, sc]
        with connect(self.db_path, read_only=True) as conn:
            return [pd.Timestamp(r[0]) for r in conn.execute(sql + " ORDER BY journey_date", params)]


def is_sqlite_path(p: Any) -> bool:
    return isinstance(p, (str, Path)) and Path(p).suffix.lower() in (".db", ".sqlite", ".sqlite3")
