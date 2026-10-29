"""RailScan SQLite schema.

``initialize_database()`` creates every table and index with
``IF NOT EXISTS`` and is safe to run any number of times. It never drops or
alters existing data.

Design notes (decided after inspecting the real data, 2026-10-14)
    * ``train_runs`` uniqueness is ``(train_key, journey_date, seq)``. The
      processed dataset has 0 duplicates on that key (11,697 rows). ``train_key``
      is the train number without leading zeros, so "01023" and "1023" cannot
      be stored twice as different trains.
    * ``train_runs.train_no`` references ``train_metadata``. Station codes are
      NOT foreign keys: the station master is a ~2016 DataMeet snapshot and
      newer stations (e.g. PRYJ) are missing from it; a foreign key would force
      real observations to be dropped. The importer reports unknown codes instead.
    * ``train_type`` is not constant per train in the real data (both trains
      appear as MAIL/EXPRESS on some journey dates and TOD on others).
      ``train_metadata.train_type``
      is therefore only set when a train has exactly one type; the full
      history is in ``name_type_history`` (JSON). Per-journey types stay in
      ``train_runs``.
    * The dataset stores delays, not raw actual timestamps. ``actual_arrival`` /
      ``actual_departure`` are derived as scheduled time + delay, and
      ``actual_time_basis`` says so. They are NULL when either part is missing.
"""
from __future__ import annotations

import sqlite3
from typing import Dict, List, Optional, Tuple

from .connection import PathLike, connect, transaction

SCHEMA_VERSION = 4          # 2: users, user_history, login_attempts; 3: timetable; 4: timetable coordinates

# ----------------------------------------------------------------------------- train_runs
# (database column, SQL type, source column in RailScan_Final_Dataset.csv or None if derived)
TRAIN_RUN_COLUMNS: List[Tuple[str, str, Optional[str]]] = [
    ("journey_date", "TEXT NOT NULL CHECK (length(journey_date) = 10)", "journey_date"),
    ("train_no", "TEXT NOT NULL REFERENCES train_metadata(train_no) ON UPDATE CASCADE", "train_no"),
    ("train_key", "TEXT NOT NULL", None),
    ("train_name", "TEXT", "train_name"),
    ("train_type", "TEXT", "train_type"),
    ("station_code", "TEXT NOT NULL", "station_code"),
    ("seq", "INTEGER NOT NULL", "seq"),
    ("observation_time", "TEXT", "observation_time"),
    ("scheduled_arrival", "TEXT", "current_scheduled_arrival"),
    ("scheduled_departure", "TEXT", "current_scheduled_departure"),
    ("actual_arrival", "TEXT", None),
    ("actual_departure", "TEXT", None),
    ("actual_time_basis", "TEXT", None),
    ("current_arrival_delay_min", "REAL", "current_arrival_delay_min"),
    ("current_departure_delay_min", "REAL", "current_departure_delay_min"),
    ("current_observed_delay_min", "REAL", "current_observed_delay_min"),
    ("scheduled_stop_min", "REAL", "scheduled_stop_min"),
    ("actual_stop_min", "REAL", "actual_stop_min"),
    ("stop_variation_min", "REAL", "stop_variation_min"),
    ("next_station_code", "TEXT", "next_station_code"),
    ("next_station_seq", "INTEGER", "next_station_seq"),
    ("next_scheduled_arrival", "TEXT", "next_scheduled_arrival"),
    ("distance_to_next_km", "REAL", "distance_to_next_km"),
    ("scheduled_travel_to_next_min", "REAL", "scheduled_travel_to_next_min"),
    ("station_lat", "REAL", "station_lat"),
    ("station_lng", "REAL", "station_lng"),
    ("day_of_week", "INTEGER", "day_of_week"),
    ("month", "INTEGER", "month"),
    ("day", "INTEGER", "day"),
    ("hour", "INTEGER", "hour"),
    ("is_weekend", "INTEGER CHECK (is_weekend IN (0, 1))", "is_weekend"),
    ("historical_avg_delay_min", "REAL", "historical_avg_delay_min"),
    ("historical_delay_std_min", "REAL", "historical_delay_std_min"),
    ("historical_observation_count", "INTEGER", "historical_observation_count"),
    ("historical_on_time_rate_pct", "REAL", "historical_on_time_rate_pct"),
    ("historical_avg_actual_stop_min", "REAL", "historical_avg_actual_stop_min"),
    ("historical_stop_std_min", "REAL", "historical_stop_std_min"),
    ("historical_stop_observation_count", "INTEGER", "historical_stop_observation_count"),
    ("historical_stop_reliability_pct", "REAL", "historical_stop_reliability_pct"),
    ("expected_next_station_arrival_baseline", "TEXT", "expected_next_station_arrival_baseline"),
    # ---- targets: outcomes at the NEXT station. Stored as history, never model inputs.
    ("next_station_arrival_delay_min", "REAL", "next_station_arrival_delay_min"),
    ("next_station_delayed_15", "INTEGER CHECK (next_station_delayed_15 IN (0, 1))", "next_station_delayed_15"),
    ("next_delay_risk_label", "TEXT", "next_delay_risk_label"),
    ("target_next_actual_arrival", "TEXT", "target_next_actual_arrival"),
]

TIMETABLE_COLUMNS = ["train_no", "seq", "station_code", "station_name", "train_name", "train_type",
                     "sch_arrival", "arrival_day_offset", "sch_departure", "departure_day_offset",
                     "distance_km", "distance_to_next_km", "journeys_seen", "runs_on", "timetable_window",
                     "journeys_in_window", "latitude", "longitude"]
# columns added after a table was first created: (table, column, type); added by initialize_database
ADDED_COLUMNS = [("timetable", "latitude", "REAL"), ("timetable", "longitude", "REAL")]

TARGET_COLUMNS = ["next_station_arrival_delay_min", "next_station_delayed_15",
                  "next_delay_risk_label", "target_next_actual_arrival"]

# value kinds used by the importer
INTEGER_COLUMNS = {c for c, t, _ in TRAIN_RUN_COLUMNS if t.startswith("INTEGER")}
REAL_COLUMNS = {c for c, t, _ in TRAIN_RUN_COLUMNS if t.startswith("REAL")}
TIMESTAMP_COLUMNS = {"observation_time", "scheduled_arrival", "scheduled_departure", "next_scheduled_arrival",
                     "expected_next_station_arrival_baseline", "target_next_actual_arrival"}

# database column -> dataset column (only columns that come straight from the CSV)
DB_TO_DATASET: Dict[str, str] = {db: src for db, _t, src in TRAIN_RUN_COLUMNS if src}
DATASET_TO_DB: Dict[str, str] = {src: db for db, src in DB_TO_DATASET.items()}

REQUIRED_DATASET_COLUMNS = ["train_no", "journey_date", "station_code", "seq"]
OPTIONAL_DATASET_COLUMNS = [src for src in DB_TO_DATASET.values() if src not in REQUIRED_DATASET_COLUMNS]

REQUIRED_STATION_COLUMNS = ["station_code", "station_name"]
OPTIONAL_STATION_COLUMNS = ["latitude", "longitude", "source"]


def _train_runs_ddl() -> str:
    cols = ",\n    ".join(f"{name} {sqltype}" for name, sqltype, _ in TRAIN_RUN_COLUMNS)
    return (
        "CREATE TABLE IF NOT EXISTS train_runs (\n"
        "    id INTEGER PRIMARY KEY,\n"
        f"    {cols},\n"
        "    UNIQUE (train_key, journey_date, seq)\n"
        ")"
    )


TABLES: Dict[str, str] = {
    "stations": """
CREATE TABLE IF NOT EXISTS stations (
    station_code TEXT PRIMARY KEY,
    station_name TEXT NOT NULL COLLATE NOCASE,
    latitude     REAL CHECK (latitude IS NULL OR (latitude BETWEEN -90 AND 90)),
    longitude    REAL CHECK (longitude IS NULL OR (longitude BETWEEN -180 AND 180)),
    source       TEXT
)""",
    "train_metadata": """
CREATE TABLE IF NOT EXISTS train_metadata (
    train_no           TEXT PRIMARY KEY,
    train_name         TEXT,
    train_type         TEXT,
    name_type_history  TEXT,
    first_journey_date TEXT,
    last_journey_date  TEXT,
    observation_count  INTEGER
)""",
    "database_metadata": """
CREATE TABLE IF NOT EXISTS database_metadata (
    id            INTEGER PRIMARY KEY,
    source_file   TEXT,
    imported_at   TEXT,
    rows_imported INTEGER,
    status        TEXT,
    notes         TEXT,
    target_table  TEXT,
    source_sha256 TEXT
)""",
    "train_runs": _train_runs_ddl(),
    # ---- accounts (schema v2). Passwords are stored only as salted scrypt hashes.
    "users": """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY,
    username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
    display_name  TEXT,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    last_login_at TEXT
)""",
    "user_history": """
CREATE TABLE IF NOT EXISTS user_history (
    id         INTEGER PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    kind       TEXT NOT NULL CHECK (kind IN ('scan', 'forecast', 'typical', 'replay')),
    summary    TEXT NOT NULL,
    details    TEXT
)""",
    # ---- current timetable per train (schema v3), from data/processed/RailScan_Timetable.csv
    "timetable": """
CREATE TABLE IF NOT EXISTS timetable (
    train_no             TEXT NOT NULL,
    seq                  INTEGER NOT NULL,
    station_code         TEXT NOT NULL,
    station_name         TEXT,
    train_name           TEXT,
    train_type           TEXT,
    sch_arrival          TEXT,
    arrival_day_offset   INTEGER,
    sch_departure        TEXT,
    departure_day_offset INTEGER,
    distance_km          REAL,
    distance_to_next_km  REAL,
    journeys_seen        INTEGER,
    runs_on              TEXT,
    timetable_window     TEXT,
    journeys_in_window   INTEGER,
    latitude             REAL,          -- median recorded coordinates of the stop (schema v4)
    longitude            REAL,
    PRIMARY KEY (train_no, seq)
)""",
    "login_attempts": """
CREATE TABLE IF NOT EXISTS login_attempts (
    id           INTEGER PRIMARY KEY,
    username     TEXT NOT NULL COLLATE NOCASE,
    attempted_at TEXT NOT NULL,
    success      INTEGER NOT NULL CHECK (success IN (0, 1))
)""",
}

# creation order respects foreign keys
TABLE_ORDER = ["stations", "train_metadata", "database_metadata", "train_runs",
               "users", "user_history", "login_attempts", "timetable"]

INDEXES: List[str] = [
    "CREATE INDEX IF NOT EXISTS idx_stations_name ON stations(station_name COLLATE NOCASE)",
    "CREATE INDEX IF NOT EXISTS idx_train_runs_train_no ON train_runs(train_no)",
    "CREATE INDEX IF NOT EXISTS idx_train_runs_journey_date ON train_runs(journey_date)",
    "CREATE INDEX IF NOT EXISTS idx_train_runs_station_code ON train_runs(station_code)",
    "CREATE INDEX IF NOT EXISTS idx_train_runs_train_date_seq ON train_runs(train_no, journey_date, seq)",
    "CREATE INDEX IF NOT EXISTS idx_train_runs_key_date ON train_runs(train_key, journey_date)",
    "CREATE INDEX IF NOT EXISTS idx_train_runs_next_station ON train_runs(next_station_code)",
    "CREATE INDEX IF NOT EXISTS idx_database_metadata_table ON database_metadata(target_table, imported_at)",
    "CREATE INDEX IF NOT EXISTS idx_user_history_user ON user_history(user_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_login_attempts_user ON login_attempts(username, attempted_at)",
    "CREATE INDEX IF NOT EXISTS idx_timetable_station ON timetable(station_code)",
]


def initialize_database(db_path: Optional[PathLike] = None) -> Dict[str, object]:
    """Create all tables and indexes if missing. Idempotent; never drops data.

    Returns ``{"db_path", "created_tables", "existing_tables", "schema_problems"}``.
    ``schema_problems`` lists columns that an *older* existing table lacks; the
    function does not alter such tables (rebuild the database instead).
    """
    from .connection import resolve_db_path

    path = resolve_db_path(db_path)
    with connect(path) as conn:
        before = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        with transaction(conn):
            for name in TABLE_ORDER:
                conn.execute(TABLES[name])
            for ddl in INDEXES:
                conn.execute(ddl)
            for table, col, typ in ADDED_COLUMNS:          # upgrade older databases in place
                have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
                if have and col not in have:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
        if conn.execute("PRAGMA user_version").fetchone()[0] < SCHEMA_VERSION:
            conn.execute(f"PRAGMA user_version = {int(SCHEMA_VERSION)}")
        problems = verify_schema(conn)
    return {
        "db_path": str(path),
        "created_tables": [t for t in TABLE_ORDER if t not in before],
        "existing_tables": [t for t in TABLE_ORDER if t in before],
        "schema_problems": problems,
    }


def expected_columns() -> Dict[str, List[str]]:
    return {
        "stations": ["station_code", "station_name", "latitude", "longitude", "source"],
        "train_metadata": ["train_no", "train_name", "train_type", "name_type_history",
                           "first_journey_date", "last_journey_date", "observation_count"],
        "database_metadata": ["id", "source_file", "imported_at", "rows_imported", "status", "notes",
                              "target_table", "source_sha256"],
        "train_runs": ["id"] + [c for c, _t, _s in TRAIN_RUN_COLUMNS],
        "users": ["id", "username", "display_name", "password_hash", "created_at", "last_login_at"],
        "user_history": ["id", "user_id", "created_at", "kind", "summary", "details"],
        "login_attempts": ["id", "username", "attempted_at", "success"],
        "timetable": TIMETABLE_COLUMNS,
    }


def verify_schema(conn: sqlite3.Connection) -> List[str]:
    """List columns missing from existing tables (empty list = schema is current)."""
    problems = []
    for table, cols in expected_columns().items():
        have = {r["name"] for r in conn.execute("SELECT name FROM pragma_table_info(?)", (table,))}
        if not have:
            problems.append(f"table {table} is missing")
            continue
        missing = [c for c in cols if c not in have]
        if missing:
            problems.append(f"table {table} lacks columns {missing}")
    return problems
