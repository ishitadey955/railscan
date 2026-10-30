"""One train at a time from the committed Parquet file, so a fresh server needs no full build.

Building the whole ``train_runs`` table (~925k rows) takes 4-5 minutes of full CPU on a small
host such as Streamlit Community Cloud, and it would be repeated after every restart. The app
only ever needs one train's rows at a time, so on such a server each train is imported when it
is first asked for:

    rows of that train (Parquet, filtered on train_no: ~0.2 s)
      -> import_dataset() into a small SQLite file    (the same code as the full build)
      -> load_master_db(train_no=...)                  (the same query as the full database)

so the result is the same as reading the fully built database (tests/test_on_demand.py).
"""
from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from typing import Optional

import pandas as pd

from .connection import PathLike, connect, table_exists

DEFAULT_PATH = Path(tempfile.gettempdir()) / "railscan" / "railscan_trains.db"
_LOCK = threading.Lock()          # one import at a time (SQLite allows one writer)
_STATIONS_READY: set = set()


def _has_train(db: Path, train_no: str) -> bool:
    if not db.exists():
        return False
    with connect(db, read_only=True) as conn:
        return table_exists(conn, "train_runs") and conn.execute(
            "SELECT EXISTS (SELECT 1 FROM train_runs WHERE train_no = ?)", (train_no,)).fetchone()[0] == 1


def load_train(train_no: str, db_path: Optional[PathLike] = None,
               parquet_path: Optional[PathLike] = None) -> pd.DataFrame:
    """That train's observations, exactly as ``load_master_db(train_no=...)`` on the full database.

    Empty frame (with the usual columns) when the dataset has no rows for the train.
    """
    import pyarrow.parquet as pq

    from .app_data import load_master_db
    from .import_data import DEFAULT_PARQUET, import_dataset, import_stations
    from .schema import initialize_database

    db = Path(db_path or DEFAULT_PATH)
    train_no = str(train_no).zfill(5)
    with _LOCK:
        db.parent.mkdir(parents=True, exist_ok=True)
        if str(db) not in _STATIONS_READY:
            initialize_database(db)
            report = import_stations(db_path=db, replace=True)        # station names for the join
            if report.status != "success":
                raise RuntimeError(f"station import failed: {report.error}")
            _STATIONS_READY.add(str(db))
        if not _has_train(db, train_no):
            rows = pq.read_table(parquet_path or DEFAULT_PARQUET, filters=[("train_no", "=", train_no)])
            if rows.num_rows == 0:
                return load_master_db(db, train_no=train_no)            # no rows: empty frame, right columns
            with tempfile.TemporaryDirectory() as tmp:
                part = Path(tmp) / f"train_{train_no}.parquet"
                pq.write_table(rows, part)
                report = import_dataset(part, db_path=db)
            if report.status != "success":
                raise RuntimeError(f"import of train {train_no} failed: {report.error}")
    return load_master_db(db, train_no=train_no)
