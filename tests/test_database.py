"""Tests for the SQLite layer (src/database) and its integration with src/railway.

Every test uses a temporary database under pytest's tmp_path; the real
data/database/railscan.db is never opened for writing.

Most tests import tests/railway_fixture.py: a SYNTHETIC dataset in the schema
of RailScan_Final_Dataset.csv (values are made up; they only exercise code
paths). Tests marked "real data" use the actual processed CSV / station master /
saved models and are skipped if those files are absent.

Run:  python -m pytest tests/test_database.py -v
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from railway_fixture import build_fixture  # noqa: E402
from src.database import import_data as imp  # noqa: E402
from src.database import repository as repo  # noqa: E402
from src.database.connection import DatabaseUnavailable, connect, database_ready, open_connection  # noqa: E402
from src.database.import_data import import_dataset, import_stations  # noqa: E402
from src.database.schema import INDEXES, SCHEMA_VERSION, TABLE_ORDER, TARGET_COLUMNS, initialize_database  # noqa: E402
from src.railway.pipeline import RailwayPipeline  # noqa: E402
from src.railway.station_master import StationMaster, StationMasterUnavailable  # noqa: E402
from src.railway.train_state import TrainStateService  # noqa: E402

REAL_DATASET = ROOT / "data" / "processed" / "RailScan_Final_Dataset.csv"
REAL_MASTER = ROOT / "data" / "stations" / "station_master.csv"
REAL_MODELS = ROOT / "models" / "delay"

FIXTURE_STATIONS = [("NDLS", "NEW DELHI", 28.6427, 77.2197), ("CNB", "KANPUR CENTRAL", 26.4537, 80.3510),
                    ("ALD", "ALLAHABAD JN", 25.4462, 81.8253), ("GAYA", "GAYA JN", 24.8016, 85.0091),
                    ("DHN", "DHANBAD JN", 23.7915, 86.4310), ("HWH", "HOWRAH JN", 22.5839, 88.3428),
                    ("DLI", "DELHI", 28.6611, 77.2276), ("DEE", "DELHI SARAI ROHILLA", 28.6634, 77.1858)]


# ============================================================================ fixtures
def _write_stations(directory: Path, rows=FIXTURE_STATIONS, extra: str = "") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / "station_master.csv"
    lines = ["station_code,station_name,latitude,longitude"] + [f"{c},{n},{la},{lo}" for c, n, la, lo in rows]
    p.write_text("\n".join(lines) + "\n" + extra, encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def frame():
    return build_fixture()


@pytest.fixture
def fixture_csv(tmp_path, frame) -> Path:
    p = tmp_path / "RailScan_Final_Dataset.csv"
    frame.to_csv(p, index=False)
    return p


@pytest.fixture
def db(tmp_path, fixture_csv) -> Path:
    """Temp DB with the fixture dataset and a small station table."""
    path = tmp_path / "test.db"
    master = _write_stations(tmp_path / "stations")
    assert import_stations(master, db_path=path).status == "success"
    rep = import_dataset(fixture_csv, path)
    assert rep.status == "success", rep.error
    return path


def _count(path: Path, table: str) -> int:
    with connect(path) as c:
        return c.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def _small_csv(tmp_path: Path, frame: pd.DataFrame, name="RailScan_Final_Dataset.csv", **edits) -> Path:
    df = frame.copy()
    for col, fn in edits.items():
        df = fn(df)
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


# ============================================================================ initialisation
def test_initialize_creates_file_tables_and_indexes(tmp_path):
    path = tmp_path / "sub" / "dir" / "railscan.db"            # directory created automatically
    res = initialize_database(path)
    assert path.is_file()
    assert res["created_tables"] == TABLE_ORDER and res["schema_problems"] == []
    with connect(path) as c:
        names = {r[0] for r in c.execute("SELECT name FROM sqlite_master")}
        assert set(TABLE_ORDER) <= names
        for ddl in INDEXES:
            assert ddl.split("EXISTS ")[1].split(" ")[0] in names
        assert c.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert c.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_repeated_initialization_is_safe_and_keeps_data(db):
    before = {t: _count(db, t) for t in TABLE_ORDER}
    for _ in range(3):
        res = initialize_database(db)
        assert res["created_tables"] == [] and res["schema_problems"] == []
    assert {t: _count(db, t) for t in TABLE_ORDER} == before


def test_connection_helpers(tmp_path, db):
    with pytest.raises(DatabaseUnavailable):
        open_connection(tmp_path / "nope.db", create=False)
    assert not (tmp_path / "nope.db").exists()                  # no empty file created by a read
    assert database_ready(db) and not database_ready(tmp_path / "nope.db")
    with connect(db, read_only=True) as c:
        assert c.execute("SELECT station_name FROM stations WHERE station_code = ?", ("NDLS",)).fetchone()[
            "station_name"] == "NEW DELHI"                      # rows readable by column name
        with pytest.raises(sqlite3.OperationalError):
            c.execute("DELETE FROM stations")                   # read-only really is read-only


def test_foreign_keys_are_enforced(db):
    with connect(db) as c, pytest.raises(sqlite3.IntegrityError):
        c.execute("INSERT INTO train_runs (journey_date, train_no, train_key, station_code, seq) "
                  "VALUES ('2025-01-01', '77777', '77777', 'NDLS', 1)")


# ============================================================================ dataset import
def test_csv_import_counts_metadata_and_audit_row(db, frame):
    assert _count(db, "train_runs") == len(frame)
    assert _count(db, "train_metadata") == frame["train_no"].nunique()
    with connect(db) as c:
        audit = c.execute("SELECT * FROM database_metadata WHERE target_table = 'train_runs'").fetchone()
        assert audit["status"] == "success" and audit["rows_imported"] == len(frame)
        assert len(audit["source_sha256"]) == 64
        assert c.execute("PRAGMA foreign_key_check").fetchall() == []


def test_reimport_is_idempotent(db, fixture_csv, frame):
    rep = import_dataset(fixture_csv, db)
    assert rep.status == "success"
    assert (rep.rows_inserted, rep.rows_updated) == (0, len(frame))
    assert _count(db, "train_runs") == len(frame)


def test_upsert_keeps_rows_missing_from_csv_and_replace_removes_them(db, tmp_path, frame):
    smaller = _small_csv(tmp_path, frame[frame.train_no != "44444"], name="smaller.csv")
    rep = import_dataset(smaller, db)
    assert rep.status == "success" and _count(db, "train_runs") == len(frame)
    n44 = int((frame.train_no == "44444").sum())
    assert rep.rows_in_db_not_in_csv == n44 and any("--replace" in w for w in rep.warnings)

    rep = import_dataset(smaller, db, replace=True)
    assert rep.status == "success" and rep.rows_replaced == len(frame)
    assert _count(db, "train_runs") == len(frame) - n44
    assert repo.get_train_metadata("44444", db) is None             # orphan metadata removed too


def test_null_handling_and_invalid_values_become_null(tmp_path, frame):
    edited = ["distance_to_next_km", "observation_time", "is_weekend", "station_lat", "next_station_seq"]
    df = frame.copy().astype({c: object for c in edited})
    df.loc[0, "distance_to_next_km"] = "abc"            # unparseable number
    df.loc[1, "observation_time"] = "not a time"        # unparseable timestamp
    df.loc[2, "is_weekend"] = 5                         # outside CHECK range
    df.loc[3, "station_lat"] = 123.0                    # impossible latitude
    df.loc[4, "next_station_seq"] = 2.5                 # INTEGER column with a fraction
    df = df.drop(columns=["station_lng"])               # whole optional column missing
    p = tmp_path / "RailScan_Final_Dataset.csv"
    df.to_csv(p, index=False)
    path = tmp_path / "n.db"
    rep = import_dataset(p, path)
    assert rep.status == "success"
    assert rep.invalid_values_set_null == {"distance_to_next_km": 1, "observation_time": 1, "is_weekend": 1,
                                           "station_lat": 1, "next_station_seq": 1}
    assert rep.missing_optional_columns == ["station_lng"]
    with connect(path) as c:
        assert c.execute("SELECT COUNT(*) FROM train_runs WHERE station_lng IS NOT NULL").fetchone()[0] == 0
        # NaN in the source (first stop has no scheduled arrival) stays NULL, never 0 / ''
        first = frame[frame.current_scheduled_arrival.isna()]
        n_null = c.execute("SELECT COUNT(*) FROM train_runs WHERE scheduled_arrival IS NULL").fetchone()[0]
        assert n_null == len(first)
        assert c.execute("SELECT COUNT(*) FROM train_runs WHERE scheduled_arrival = '' "
                         "OR scheduled_arrival = 'nan'").fetchone()[0] == 0
        # history missing -> NULL with count 0, not filled
        r = c.execute("SELECT historical_avg_delay_min, historical_observation_count FROM train_runs "
                      "WHERE train_no = '33333' ORDER BY seq LIMIT 1").fetchone()
        assert r[0] is None and r[1] == 0
        # derived actual times NULL when the delay is missing; basis recorded when present
        r = c.execute("SELECT actual_arrival, actual_departure, actual_time_basis FROM train_runs "
                      "WHERE train_no = '12301' AND journey_date = '2025-03-01' AND seq = 1").fetchone()
        assert r["actual_arrival"] is None and r["actual_departure"] is not None
        assert r["actual_time_basis"].startswith("derived")


def test_invalid_key_rows_are_rejected_and_counted(tmp_path, frame):
    df = frame.copy().astype({"train_no": object, "station_code": object, "seq": object, "journey_date": object})
    df.loc[0, "train_no"] = "12AB"
    df.loc[1, "journey_date"] = "yesterday"
    df.loc[2, "station_code"] = "  "
    df.loc[3, "seq"] = "1.5"
    p = tmp_path / "RailScan_Final_Dataset.csv"
    df.to_csv(p, index=False)
    rep = import_dataset(p, tmp_path / "k.db")
    assert rep.status == "success" and rep.rows_rejected == 4
    assert set(rep.rejected_reasons) == {"invalid train_no (expected 4-5 digits)", "invalid journey_date",
                                         "missing station_code", "invalid seq (not an integer)"}
    assert _count(tmp_path / "k.db", "train_runs") == len(frame) - 4


def test_missing_required_column_fails_and_writes_nothing(tmp_path, frame, db):
    p = _small_csv(tmp_path, frame.drop(columns=["seq"]), name="bad.csv")
    before = _count(db, "train_runs")
    rep = import_dataset(p, db)
    assert rep.status == "failed" and "seq" in rep.error and rep.rows_imported == 0
    assert _count(db, "train_runs") == before
    with connect(db) as c:
        last = c.execute("SELECT status, notes FROM database_metadata ORDER BY id DESC LIMIT 1").fetchone()
    assert last["status"] == "failed" and "seq" in last["notes"]


def test_identical_duplicates_collapse(tmp_path, frame):
    p = _small_csv(tmp_path, pd.concat([frame, frame.head(3)]))
    rep = import_dataset(p, tmp_path / "d.db")
    assert rep.status == "success" and rep.duplicates_identical == 3
    assert _count(tmp_path / "d.db", "train_runs") == len(frame)


def test_conflicting_duplicates_fail_unless_keep_first(tmp_path, frame):
    dup = frame.head(1).copy()
    dup["current_observed_delay_min"] = 999
    p = _small_csv(tmp_path, pd.concat([frame, dup]))
    rep = import_dataset(p, tmp_path / "c.db")
    assert rep.status == "failed" and "keep-first" in rep.error
    assert not database_ready(tmp_path / "c.db")
    rep = import_dataset(p, tmp_path / "c.db", duplicate_policy="keep-first")
    assert rep.status == "success" and rep.duplicates_conflicting == 1
    with connect(tmp_path / "c.db") as c:
        assert c.execute("SELECT MAX(current_observed_delay_min) FROM train_runs").fetchone()[0] != 999


def test_leading_zero_train_numbers_are_one_key(tmp_path, frame):
    a = frame[frame.train_no == "33333"].copy()
    a["train_no"] = "04601"
    b = a.copy()
    b["train_no"] = "4601"                                   # same train, zero lost
    p = tmp_path / "RailScan_Final_Dataset.csv"
    pd.concat([a, b]).to_csv(p, index=False)
    rep = import_dataset(p, tmp_path / "z.db", duplicate_policy="keep-first")
    assert rep.status == "success" and _count(tmp_path / "z.db", "train_runs") == len(a)
    assert repo.get_train_metadata("4601", tmp_path / "z.db")["train_no"] == "04601"
    assert repo.get_train_metadata("04601", tmp_path / "z.db")["train_no"] == "04601"


def test_split_files_and_missing_dataset_are_refused(tmp_path, frame):
    split = _small_csv(tmp_path, frame, name="train.csv")
    rep = import_dataset(split, tmp_path / "s.db")
    assert rep.status == "failed" and "split" in rep.error
    rep = import_dataset(tmp_path / "absent.csv", tmp_path / "s.db")
    assert rep.status == "failed" and "not found" in rep.error
    assert not database_ready(tmp_path / "s.db")
    with connect(tmp_path / "s.db") as c:            # both failures are on record
        assert [r[0] for r in c.execute("SELECT status FROM database_metadata")] == ["failed", "failed"]


def test_transaction_rolls_back_on_mid_import_error(db, tmp_path, frame, monkeypatch):
    before = _count(db, "train_runs")
    changed = frame.copy()
    changed["current_observed_delay_min"] = 555.0
    p = _small_csv(tmp_path, changed, name="changed.csv")

    def boom(conn):
        raise RuntimeError("simulated failure after the inserts")

    monkeypatch.setattr(imp, "rebuild_train_metadata", boom)
    rep = import_dataset(p, db, replace=True, batch_size=7)    # several batches written before the error
    assert rep.status == "failed" and "simulated" in rep.error
    assert _count(db, "train_runs") == before
    with connect(db) as c:
        assert c.execute("SELECT COUNT(*) FROM train_runs WHERE current_observed_delay_min = 555").fetchone()[0] == 0
        statuses = [r[0] for r in c.execute("SELECT status FROM database_metadata WHERE target_table='train_runs' "
                                            "ORDER BY id")]
    assert statuses == ["success", "failed"]


# ============================================================================ station import
def test_station_import_supplement_precedence_and_validation(tmp_path):
    d = tmp_path / "stations"
    master = _write_stations(d, extra="BAD,,1,2\nXX,BAD COORD,999,77\nNDLS,NEW DELHI DUPLICATE,1,1\n")
    (d / "SOURCE.txt").write_text("source : https://example.org/stations.json\nlicence: CC0\n", encoding="utf-8")
    (d / "station_master_supplement.csv").write_text(
        "station_code,station_name,latitude,longitude,source\nPRYJ,PRAYAGRAJ JN,25.44,81.82,official list\n"
        "CNB,KANPUR OTHER,,,official list\n", encoding="utf-8")
    path = tmp_path / "st.db"
    rep = import_stations(master, db_path=path)
    assert rep.status == "success"
    assert rep.rejected_reasons == {"missing station_name": 1}
    assert rep.invalid_values_set_null == {"latitude": 1}
    assert rep.duplicates_conflicting == 2                       # NDLS twice in master; CNB in supplement
    assert repo.get_station_by_code("NDLS", path)["station_name"] == "NEW DELHI"
    assert repo.get_station_by_code("CNB", path)["station_name"] == "KANPUR CENTRAL"
    assert repo.get_station_by_code("PRYJ", path)["source"] == "official list"
    assert repo.get_station_by_code("NDLS", path)["source"] == "https://example.org/stations.json [licence: CC0]"
    xx = repo.get_station_by_code("XX", path)
    assert xx["latitude"] is None and xx["longitude"] == 77.0


# ============================================================================ repository
def test_station_lookup(db):
    st = repo.get_station_by_code(" ndls ", db)
    assert st["station_code"] == "NDLS" and st["station_name"] == "NEW DELHI" and st["latitude"] == pytest.approx(
        28.6427)
    assert repo.get_station_by_code("ZZZZ", db) is None
    assert repo.get_station_by_code(None, db) is None and repo.get_station_by_code("", db) is None


def test_station_name_search(db):
    res = repo.search_stations("delhi", db_path=db)
    # exact (case-insensitive) > prefix > substring
    assert [(r["station_code"], r["match_type"]) for r in res] == [
        ("DLI", "name_exact"), ("DEE", "name_prefix"), ("NDLS", "name_contains")]
    assert repo.search_stations("cnb", db_path=db)[0]["match_type"] == "code"
    assert repo.search_stations("kan", db_path=db)[0]["station_code"] == "CNB"
    assert len(repo.search_stations("a", limit=2, db_path=db)) == 2
    assert repo.search_stations("NEW DLEHI", db_path=db) == []        # no fuzzy matching at the DB level
    assert repo.search_stations("   ", db_path=db) == []


def test_train_metadata_lookup(db):
    m = repo.get_train_metadata("12301", db)
    assert (m["train_name"], m["train_type"]) == ("FIXTURE RAJDHANI EXPRESS", "Rajdhani")
    amb = repo.get_train_metadata("22222", db)                  # two names in the data -> not forced
    assert amb["train_name"] is None and amb["train_type"] is None
    assert {h["train_name"] for h in amb["name_type_history"]} == {"FIXTURE ALPHA EXPRESS", "FIXTURE BETA EXPRESS"}
    assert repo.get_train_metadata("99999", db) is None
    assert repo.get_train_metadata("not a train", db) is None


def test_train_observations(db, frame):
    obs = repo.get_train_observations("12301", db_path=db)
    assert len(obs) == int((frame.train_no == "12301").sum())
    assert [(o["journey_date"], o["seq"]) for o in obs] == sorted((o["journey_date"], o["seq"]) for o in obs)
    assert not set(TARGET_COLUMNS) & set(obs[0])
    day = repo.get_train_observations("12301", "2025-03-04", db_path=db, include_targets=True)
    assert len(day) == 5 and set(TARGET_COLUMNS) <= set(day[0])
    assert repo.get_train_observations("12301", "2030-01-01", db_path=db) == []
    assert repo.get_train_observations("99999", db_path=db) == []


def test_train_state_lookup(db, frame):
    s = repo.get_train_state("12301", "2025-03-08", "cnb", db)
    src = frame[(frame.train_no == "12301") & (frame.journey_date == "2025-03-08") & (frame.station_code == "CNB")]
    assert s["station_name"] == "KANPUR CENTRAL" and s["seq"] == 2
    assert s["current_observed_delay_min"] == pytest.approx(float(src.current_observed_delay_min.iloc[0]))
    assert s["next_station_code"] == "ALD"
    assert not set(TARGET_COLUMNS) & set(s)                       # no leakage of next-stop outcomes
    exp = (pd.Timestamp(src.current_scheduled_departure.iloc[0])
           + pd.Timedelta(minutes=float(src.current_departure_delay_min.iloc[0])))
    assert s["actual_departure"] == exp.strftime("%Y-%m-%d %H:%M:%S")
    with pytest.raises(repo.AmbiguousRecord) as e:
        repo.get_train_state("44444", "2025-03-05", "CNB", db)
    assert [c["seq"] for c in e.value.candidates] == [1, 3]
    assert repo.get_train_state("44444", "2025-03-05", "CNB", db, seq=3)["next_station_code"] == "GAYA"


def test_next_station_lookup(db):
    n = repo.get_next_station("12301", "2025-03-08", "DHN", db)
    assert n["next_station_code"] == "HWH" and n["next_station_name"] == "HOWRAH JN"
    assert n["next_station_latitude"] == pytest.approx(22.5839)
    assert n["scheduled_travel_to_next_min"] is not None
    assert not {"next_station_arrival_delay_min", "target_next_actual_arrival"} & set(n)


def test_missing_record_handling(db):
    assert repo.get_train_state("12301", "2030-01-01", "CNB", db) is None
    assert repo.get_train_state("99999", "2025-03-08", "CNB", db) is None
    assert repo.get_train_state("12301", "2025-03-08", "HWH", db) is None   # last stop has no row
    assert repo.get_next_station("12301", "2025-03-08", "HWH", db) is None
    assert repo.get_train_state("12301", "not a date", "CNB", db) is None
    assert repo.get_historical_station_reliability("ZZZZ", db_path=db) is None


def test_historical_reliability_respects_cutoff(db, frame):
    cutoff = "2025-03-05 00:00:00"
    r = repo.get_historical_station_reliability("CNB", "12301", db, before=cutoff)
    past = frame[(frame.train_no == "12301") & (frame.next_station_code == "CNB")
                 & (frame.target_next_actual_arrival < pd.Timestamp(cutoff))]
    d = past.next_station_arrival_delay_min
    assert r["leakage_safe_for_prediction"] is True
    assert r["arrivals"]["count"] == len(past)
    assert r["arrivals"]["avg_delay_min"] == pytest.approx(d.mean(), abs=1e-3)
    assert r["arrivals"]["delay_std_min"] == pytest.approx(d.std(ddof=1), abs=1e-3)
    assert r["arrivals"]["on_time_rate_pct"] == pytest.approx((d < 15).mean() * 100, abs=1e-3)
    assert r["arrivals"]["last_journey_date"] < "2025-03-05"
    full = repo.get_historical_station_reliability("CNB", "12301", db)
    assert full["leakage_safe_for_prediction"] is False and full["arrivals"]["count"] > r["arrivals"]["count"]
    with pytest.raises(ValueError):
        repo.get_historical_station_reliability("CNB", db_path=db, before="garbage")


def test_sql_injection_is_inert(db):
    attacks = ["NDLS' OR '1'='1", "'; DROP TABLE stations; --", "NDLS\" OR 1=1 --", "%", "_", "\\"]
    for a in attacks:
        assert repo.get_station_by_code(a, db) is None
        assert repo.search_stations(a, db_path=db) == []           # % and _ are literal, not wildcards
        assert repo.get_train_state("12301", "2025-03-08", a, db) is None
        assert repo.get_historical_station_reliability(a, db_path=db) is None
    assert repo.get_train_observations("12301; DROP TABLE train_runs", db_path=db) == []
    assert repo.get_train_observations("12301", "2025-03-08' OR '1'='1", db_path=db) == []
    assert repo.search_stations("x", limit="1; DROP TABLE stations", db_path=db) == []
    assert _count(db, "stations") == len(FIXTURE_STATIONS) and _count(db, "train_runs") > 0


def test_database_statistics(db, frame):
    st = repo.get_database_statistics(db)
    assert st["tables"]["train_runs"] == len(frame)
    assert st["tables"]["stations"] == len(FIXTURE_STATIONS)
    assert st["train_runs"]["trains"] == frame.train_no.nunique()
    assert st["train_runs"]["dates"] == frame.journey_date.nunique()
    assert st["train_runs"]["station_codes_not_in_stations"] == []
    assert st["foreign_key_violations"] == 0
    assert {r["target_table"]: r["status"] for r in st["last_imports"]} == {"stations": "success",
                                                                            "train_runs": "success"}


# ============================================================================ integration
def _state_keys(frame):
    keys = {(r.train_no, str(pd.Timestamp(r.journey_date).date()), c)
            for r in frame.itertuples() for c in (r.station_code, r.next_station_code)}
    keys |= {("12301", "2030-01-01", "CNB"), ("99999", "2025-03-01", "NDLS"), ("12301", "2025-03-01", "ZZZ"),
             ("abc", "2025-03-01", "NDLS"), ("12301", "nonsense", "NDLS"), ("22222", "2025-03-02", "CNB")}
    return sorted(keys)


def test_train_state_service_sqlite_matches_dataframe_backend(db, frame):
    master = StationMaster.from_records(FIXTURE_STATIONS)
    mem, sql = TrainStateService(frame, master), TrainStateService(db, master)
    assert (mem.backend, sql.backend) == ("dataframe", "sqlite")
    for k in _state_keys(frame):
        assert mem.get_state(*k) == sql.get_state(*k), k
    for tn in ("12301", "22222", "33333", "44444", "99999", "x"):
        for jd in (None, "2025-03-01", "2025-03-05", "2030-01-01"):
            for sc in (None, "CNB"):
                assert mem.lookup_train(tn, jd, sc) == sql.lookup_train(tn, jd, sc), (tn, jd, sc)
        assert mem.data.route_codes(tn) == sql.data.route_codes(tn)
        assert mem.data.available_dates(tn, "CNB") == sql.data.available_dates(tn, "CNB")


def test_pipeline_sqlite_matches_dataframe_backend(db, frame):
    master = StationMaster.from_records(FIXTURE_STATIONS)
    mem, sql = RailwayPipeline(master, frame), RailwayPipeline(master, db)
    cases = [("12301 NEW DELHI", "2025-03-08"), ("12301 KANPUR CENTRAL", None), ("33333 KANPUR CENTRAL", None),
             ("22222 NEW DELHI", "2025-03-02"), ("99999 NEW DELHI", "2025-03-02"), ("HOWRAH JN", None)]
    for text, jd in cases:
        assert mem.process(text, journey_date=jd) == sql.process(text, journey_date=jd), text


def test_default_backend_prefers_populated_database(db, frame, monkeypatch, tmp_path):
    monkeypatch.setenv("RAILSCAN_DB_PATH", str(db))
    svc = TrainStateService(None, StationMaster.from_records(FIXTURE_STATIONS))
    assert svc.backend == "sqlite" and svc.get_state("12301", "2025-03-08", "CNB")["status"] == "ok"
    sm = StationMaster.load()
    assert sm.source == "sqlite" and len(sm) == len(FIXTURE_STATIONS)
    # empty database -> CSV fallback, exactly as before this stage
    empty = tmp_path / "empty.db"
    initialize_database(empty)
    monkeypatch.setenv("RAILSCAN_DB_PATH", str(empty))
    assert TrainStateService(None).backend in ("csv", None)
    assert StationMaster.load().source == "csv"
    with pytest.raises(StationMasterUnavailable):
        StationMaster.from_database(empty)


@pytest.mark.skipif(not REAL_MASTER.is_file(), reason="real station master not present")
def test_real_station_master_from_database_equals_csv(tmp_path):
    path = tmp_path / "real_st.db"
    rep = import_stations(db_path=path)
    assert rep.status == "success" and rep.rows_rejected == 0
    from_db, from_csv = StationMaster.from_database(path), StationMaster.from_csv()
    assert from_db.stations == from_csv.stations
    assert from_db.aliases == from_csv.aliases
    from src.railway.station_matcher import StationMatcher
    a, b = StationMatcher(from_db), StationMatcher(from_csv)
    for q in ("NEW DELHI", "नई दिल्ली", "Mangalore", "Delhi", "KOLHAPUR", "ಮಂಗಳೂರು", "PUNE JN", "zzqx"):
        assert a.match(q) == b.match(q), q


@pytest.mark.skipif(not (REAL_DATASET.is_file() and (REAL_MODELS / "delay_regressor.pkl").is_file()),
                    reason="real dataset or saved delay models not present")
def test_real_data_model_inputs_match_training_path(tmp_path):
    """Saved models must give the SAME predictions from database rows as from the CSV path used in training.

    Uses the real rows of a few trains, copied verbatim (the full dataset is ~930k rows)."""
    path, sample = tmp_path / "real.db", tmp_path / "RailScan_Final_Dataset.csv"
    parts = [c[c.train_no.str.lstrip("0").isin(["12229", "12004", "14164"])]
             for c in pd.read_csv(REAL_DATASET, dtype=str, keep_default_na=False, chunksize=200_000)]
    pd.concat(parts).to_csv(sample, index=False)
    rep = import_dataset(sample, path)
    assert rep.status == "success" and rep.rows_rejected == 0 and rep.rows_imported == rep.rows_read
    from src.database import model_inputs as mi
    try:
        models = mi.load_delay_models(str(REAL_MODELS))
    except Exception as exc:                                    # e.g. incompatible scikit-learn version
        pytest.skip(f"saved models cannot be loaded here: {exc}")
    dc = mi.delay_common()
    csv = pd.read_csv(sample)                                   # exactly how training read it
    with connect(path) as c:
        rows = [dict(r) for r in c.execute(f"SELECT {', '.join(repo.STATE_COLUMNS)} FROM train_runs")]
    Xd = mi.make_model_features(rows, REAL_MODELS)
    Xd.index = [f"{r['train_no']}|{r['journey_date']}|{r['seq']}" for r in rows]
    keys = (csv.train_no.astype(str).str.zfill(5) + "|" + csv.journey_date + "|" + csv.seq.astype(str)).values
    Xd = Xd.loc[keys].reset_index(drop=True)
    Xc = dc.make_features(csv)
    pd.testing.assert_frame_equal(Xc, Xd, check_dtype=False)
    assert list(Xd.columns) == json.loads((REAL_MODELS / "feature_columns.json").read_text())["features"]
    # identical inputs (checked above); a forest averages its trees in parallel, so the last
    # bits of the sum can differ between calls (~1e-14 min) - compare to far below a second
    np.testing.assert_allclose(models["regressor"].predict(Xc), models["regressor"].predict(Xd), rtol=0, atol=1e-9)
    np.testing.assert_allclose(models["classifier"].predict_proba(Xc), models["classifier"].predict_proba(Xd),
                               rtol=0, atol=1e-9)
    assert mi.unknown_categories(Xd, models["regressor"]) == {}
    assert not set(TARGET_COLUMNS) & set(Xd.columns)
