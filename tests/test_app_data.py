"""The app's data read from SQLite must equal the processed CSVs it was built from."""
import numpy as np
import pandas as pd
import pytest

from src.database.app_data import data_ready
from src.railscan_service import DataUnavailable, load_master
from src.timetable import load_timetable

needs_db = pytest.mark.skipif(not data_ready(), reason="SQLite database not built (setup_database.py all)")
KEY = ["train_no", "journey_date", "seq"]


def _same(a: pd.Series, b: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        return bool(np.allclose(a.astype(float), b.astype(float), equal_nan=True))
    na, nb = a.isna(), b.isna()
    return bool((na == nb).all() and (a[~na].astype(str).values == b[~nb].astype(str).values).all())


@needs_db
def test_sqlite_holds_every_csv_row():
    from src.database.app_data import overview_db, train_list_db
    from src.railscan_service import MASTER_CSV
    csv = pd.read_csv(MASTER_CSV, dtype={"train_no": str}, usecols=["train_no"])
    assert overview_db()["overview"]["rows"] == len(csv)
    trains = train_list_db()
    assert trains.rows.sum() == len(csv) and set(trains.train_no) == set(csv.train_no.str.zfill(5))


@needs_db
@pytest.mark.parametrize("train_no", ["12229", "12004", "14164"])
def test_master_from_sqlite_equals_csv_on_every_column(train_no):
    """Per train (the app's access pattern): every column identical to the CSV."""
    db = load_master(source="sqlite", train_no=train_no).sort_values(KEY).reset_index(drop=True)
    csv = load_master(source="csv", train_no=train_no).sort_values(KEY).reset_index(drop=True)
    assert db.attrs["source"] == "SQLite" and csv.attrs["source"] == "CSV"
    assert len(db) == len(csv) > 0
    missing = sorted(set(csv.columns) - set(db.columns))
    assert missing == [], f"columns only in the CSV: {missing}"
    bad = [c for c in csv.columns if not _same(db[c], csv[c])]
    assert bad == [], f"columns that differ: {bad}"


@needs_db
def test_timetable_from_sqlite_equals_csv():
    db, csv = load_timetable(source="sqlite"), load_timetable(source="csv")
    assert set(db.columns) == set(csv.columns) and len(db) == len(csv) > 0
    bad = [c for c in csv.columns if not _same(db[c], csv[c])]
    assert bad == []


@needs_db
def test_auto_prefers_sqlite():
    assert load_master(train_no="12229").attrs["source"] == "SQLite"
    assert load_timetable().attrs["source"] == "SQLite"


def test_sqlite_requested_but_empty_database_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("RAILSCAN_DB_PATH", str(tmp_path / "empty.db"))
    with pytest.raises(DataUnavailable, match="setup_database.py all"):
        load_master(source="sqlite")
    assert load_master(train_no="12229").attrs["source"] == "CSV"     # auto falls back to the CSV
