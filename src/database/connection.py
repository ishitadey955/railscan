"""SQLite connection handling for RailScan.

Path resolution (first match wins)
    1. an explicit ``db_path`` argument
    2. the ``RAILSCAN_DB_PATH`` environment variable
    3. ``<project root>/data/database/railscan.db``

Every connection
    * returns rows that can be read by column name (``sqlite3.Row``)
    * has foreign-key enforcement switched on (SQLite leaves it OFF by default)
    * uses a configurable busy timeout
    * is closed when the ``connect()`` context manager exits

Only parameterised SQL is used anywhere in ``src.database``.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Union

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = ROOT / "data" / "database" / "railscan.db"
ENV_VAR = "RAILSCAN_DB_PATH"
DEFAULT_TIMEOUT_S = 10.0

PathLike = Union[str, Path]


class DatabaseUnavailable(RuntimeError):
    """The database file is missing, empty or not initialised."""


def resolve_db_path(db_path: Optional[PathLike] = None) -> Path:
    """Return the database path to use (does not create anything)."""
    if db_path:
        return Path(db_path).expanduser().resolve()
    env = os.environ.get(ENV_VAR, "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_DB_PATH


def open_connection(db_path: Optional[PathLike] = None, *, timeout: float = DEFAULT_TIMEOUT_S,
                    create: bool = True, read_only: bool = False) -> sqlite3.Connection:
    """Open a configured connection. The caller must close it (prefer :func:`connect`).

    ``create=False`` raises :class:`DatabaseUnavailable` instead of silently
    creating an empty database file. ``read_only=True`` opens the file in
    SQLite's read-only mode (implies ``create=False``).
    """
    path = resolve_db_path(db_path)
    if read_only or not create:
        if not path.is_file():
            raise DatabaseUnavailable(
                f"Database not found at {path}. Run `python scripts/setup_database.py init` "
                "and then the import commands.")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)

    if read_only:
        conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=timeout)
    else:
        conn = sqlite3.connect(str(path), timeout=timeout)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # PRAGMA values cannot be bound as parameters; int() guarantees a plain number.
    conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")
    return conn


@contextmanager
def connect(db_path: Optional[PathLike] = None, *, timeout: float = DEFAULT_TIMEOUT_S,
            create: bool = True, read_only: bool = False) -> Iterator[sqlite3.Connection]:
    """``with connect() as conn: ...`` - the connection is always closed afterwards.

    Note: unlike ``sqlite3.Connection`` used directly as a context manager,
    this does not commit for you. Use :func:`transaction` for writes.
    """
    conn = open_connection(db_path, timeout=timeout, create=create, read_only=read_only)
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Explicit BEGIN ... COMMIT; any exception rolls the whole block back."""
    conn.execute("BEGIN")
    try:
        yield conn
    except BaseException:
        conn.rollback()
        raise
    else:
        conn.commit()


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)).fetchone()
    return row is not None


def table_count(conn: sqlite3.Connection, name: str) -> int:
    """Row count of a known table. ``name`` is checked against sqlite_master first
    (identifiers cannot be bound as parameters)."""
    if not table_exists(conn, name):
        return 0
    return int(conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0])


def database_ready(db_path: Optional[PathLike] = None, table: str = "train_runs") -> bool:
    """True when the file exists and ``table`` holds at least one row. Never raises."""
    try:
        with connect(db_path, read_only=True) as conn:
            return table_count(conn, table) > 0
    except (DatabaseUnavailable, sqlite3.Error):
        return False
