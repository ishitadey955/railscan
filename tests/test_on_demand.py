"""Without the full database (fresh server): trains from the Parquet file, Data page figures from
the exported overview. Both must give what the full database gives."""
import numpy as np
import pandas as pd
import pytest

from src.database.app_data import (OVERVIEW_FILE, data_ready, load_master_db, overview_db, overview_files,
                                   train_list_db, train_list_parquet)
from src.database.import_data import DEFAULT_PARQUET, file_sha256
from src.database.on_demand import load_train

full_db = pytest.mark.skipif(not data_ready(), reason="full database not built (scripts/setup_database.py all)")


def test_committed_overview_belongs_to_committed_parquet():
    import json
    assert json.loads(OVERVIEW_FILE.read_text())["parquet_sha256"] == file_sha256(DEFAULT_PARQUET), \
        "dataset changed: run python scripts/setup_database.py all (it exports the overview again)"


@full_db
@pytest.mark.parametrize("train_no", ["12229", "12301", "99999"])
def test_on_demand_train_equals_full_database(tmp_path, train_no):
    got = load_train(train_no, db_path=tmp_path / "trains.db").drop(columns="id", errors="ignore")
    want = load_master_db(train_no=train_no).drop(columns="id", errors="ignore")
    pd.testing.assert_frame_equal(got.reset_index(drop=True), want.reset_index(drop=True))


def test_on_demand_reuses_imported_trains(tmp_path):
    db = tmp_path / "trains.db"
    first = load_train("12229", db_path=db)
    assert len(first) > 0 and first.train_no.eq("12229").all()
    pd.testing.assert_frame_equal(load_train("12229", db_path=db), first)


@full_db
def test_train_list_from_parquet_equals_database():
    pd.testing.assert_frame_equal(train_list_parquet(), train_list_db())


@full_db
def test_overview_from_files_equals_database():
    a, b = overview_files(), overview_db()
    assert not a["stale"]
    assert a["overview"] == b["overview"] and a["risk"] == b["risk"]
    pd.testing.assert_frame_equal(a["missing"], b["missing"])
    assert np.array_equal(np.sort(a["delays"].to_numpy(float)), np.sort(b["delays"].to_numpy(float)), equal_nan=True)
