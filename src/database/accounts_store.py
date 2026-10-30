"""Where accounts and history are stored: the local SQLite file or an external PostgreSQL.

Choice (first match wins)
    1. a ``db_path`` argument that is a ``postgresql://`` URL  -> PostgreSQL
    2. a ``db_path`` argument that is a file path               -> SQLite
    3. environment variable ``RAILSCAN_ACCOUNTS_DB_URL``        -> PostgreSQL
    4. otherwise the RailScan SQLite file (data/database/railscan.db or $RAILSCAN_DB_PATH)

SQL in src/database/users.py is written once with ``?`` placeholders; this
module converts them for PostgreSQL. Usernames are compared with lower() on
both databases, and both enforce one account per lower-cased username.
"""
from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Sequence

from .connection import PathLike, connect, transaction

ENV_URL = "RAILSCAN_ACCOUNTS_DB_URL"
_PG_PREFIXES = ("postgresql://", "postgres://")
_READY: set = set()          # PostgreSQL URLs whose tables were created in this process
# One open PostgreSQL connection per URL, reused: a new connection to a hosted database
# costs several network round trips (TCP, TLS, password exchange) before the first query.
# The lock lets one request use it at a time, so transactions of parallel sessions never mix.
_PG_CONNECTIONS: Dict[str, Any] = {}
_PG_LOCK = threading.Lock()

PG_DDL = [
    """CREATE TABLE IF NOT EXISTS users (
        id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        username      TEXT NOT NULL,
        display_name  TEXT,
        password_hash TEXT NOT NULL,
        created_at    TEXT NOT NULL,
        last_login_at TEXT)""",
    "CREATE UNIQUE INDEX IF NOT EXISTS users_username_lower ON users (lower(username))",
    """CREATE TABLE IF NOT EXISTS user_history (
        id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        user_id    BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        created_at TEXT NOT NULL,
        kind       TEXT NOT NULL CHECK (kind IN ('scan', 'forecast', 'typical', 'replay')),
        summary    TEXT NOT NULL,
        details    TEXT)""",
    "CREATE INDEX IF NOT EXISTS idx_user_history_user ON user_history (user_id, created_at)",
    """CREATE TABLE IF NOT EXISTS login_attempts (
        id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        username     TEXT NOT NULL,
        attempted_at TEXT NOT NULL,
        success      INTEGER NOT NULL CHECK (success IN (0, 1)))""",
    "CREATE INDEX IF NOT EXISTS idx_login_attempts_user ON login_attempts (lower(username), attempted_at)",
    """CREATE TABLE IF NOT EXISTS usage_events (
        id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        subject    TEXT NOT NULL,
        kind       TEXT NOT NULL,
        created_at TEXT NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS idx_usage_events_subject ON usage_events (subject, kind, created_at)",
]


def target(db_path: Optional[PathLike] = None) -> str:
    """'postgres' or 'sqlite' for the given argument / environment."""
    if db_path is not None:
        return "postgres" if str(db_path).startswith(_PG_PREFIXES) else "sqlite"
    return "postgres" if os.environ.get(ENV_URL, "").strip().startswith(_PG_PREFIXES) else "sqlite"


def describe(db_path: Optional[PathLike] = None) -> str:
    """Human-readable location, without credentials."""
    if target(db_path) == "postgres":
        url = str(db_path or os.environ[ENV_URL])
        host = url.split("@")[-1].split("/")[0].split("?")[0]
        return f"PostgreSQL at {host}"
    from .connection import resolve_db_path
    return f"SQLite file {resolve_db_path(db_path)}"


class Store:
    """Minimal common interface: one(), all(), run() and a transaction block."""

    def __init__(self, conn: Any, dialect: str):
        self.conn, self.dialect = conn, dialect

    def _sql(self, sql: str) -> str:
        return sql.replace("?", "%s") if self.dialect == "postgres" else sql

    def run(self, sql: str, params: Sequence[Any] = ()) -> Any:
        return self.conn.execute(self._sql(sql), tuple(params))

    def one(self, sql: str, params: Sequence[Any] = ()) -> Optional[Dict[str, Any]]:
        row = self.run(sql, params).fetchone()
        return dict(row) if row is not None else None

    def all(self, sql: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
        return [dict(r) for r in self.run(sql, params).fetchall()]

    @contextmanager
    def transaction(self) -> Iterator["Store"]:
        if self.dialect == "postgres":
            with self.conn.transaction():
                yield self
        else:
            with transaction(self.conn):
                yield self

    @staticmethod
    def is_unique_violation(exc: BaseException) -> bool:
        if isinstance(exc, sqlite3.IntegrityError):
            return True
        try:
            import psycopg
            return isinstance(exc, psycopg.errors.UniqueViolation)
        except ImportError:
            return False


@contextmanager
def open_store(db_path: Optional[PathLike] = None) -> Iterator[Store]:
    """Open the accounts store, creating its tables if needed."""
    if target(db_path) == "postgres":
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PostgreSQL accounts need psycopg: pip install -r requirements.txt") from exc
        url = str(db_path or os.environ[ENV_URL]).strip()
        with _PG_LOCK:
            conn = _PG_CONNECTIONS.get(url)
            if conn is not None:
                try:
                    conn.execute("SELECT 1")        # the server or pooler may have closed an idle connection
                except psycopg.Error:
                    conn.close()
                    conn = None
            if conn is None:
                # prepare_threshold=None: no server-side prepared statements, which connection
                # poolers in transaction mode (e.g. Supabase port 6543, PgBouncer) do not support
                conn = _PG_CONNECTIONS[url] = psycopg.connect(url, row_factory=dict_row, autocommit=True,
                                                              prepare_threshold=None)
            store = Store(conn, "postgres")
            try:
                if url not in _READY:
                    with store.transaction():
                        for ddl in PG_DDL:
                            conn.execute(ddl)
                    _READY.add(url)
                yield store
            except psycopg.OperationalError:     # connection lost mid-request: reconnect next time
                _PG_CONNECTIONS.pop(url, None)
                conn.close()
                raise
    else:
        from .schema import initialize_database
        initialize_database(db_path)
        with connect(db_path) as conn:
            yield Store(conn, "sqlite")
