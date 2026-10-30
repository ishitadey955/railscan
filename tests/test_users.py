"""Accounts and per-user history, on SQLite AND PostgreSQL (temporary databases only).

PostgreSQL tests use a throw-away local server from the ``pgserver`` package and
are skipped when it is not installed.
"""
import sqlite3

import pytest

from src.database import accounts_store
from src.database import users as u
from src.database.accounts_store import open_store
from src.database.schema import SCHEMA_VERSION, initialize_database


@pytest.fixture(scope="session")
def pg_url(tmp_path_factory):
    pgserver = pytest.importorskip("pgserver")
    srv = pgserver.get_server(str(tmp_path_factory.mktemp("pg")), cleanup_mode="stop")
    yield srv.get_uri()


@pytest.fixture(params=["sqlite", "postgres"])
def db(request, tmp_path):
    if request.param == "sqlite":
        path = tmp_path / "users.db"
        initialize_database(path)
        return path
    url = request.getfixturevalue("pg_url")
    import psycopg
    with psycopg.connect(url, autocommit=True) as c:          # fresh tables for every test
        c.execute("DROP TABLE IF EXISTS user_history, login_attempts, usage_events, users CASCADE")
    accounts_store._READY.discard(url)
    return url


def _scalar(db, sql):
    with open_store(db) as s:
        return list(s.one(sql).values())[0]


def test_sign_up_and_log_in(db):
    created = u.create_user("asha_k", "railway2026", "Asha", db_path=db)
    assert created["username"] == "asha_k" and created["display_name"] == "Asha"
    user = u.authenticate("ASHA_K", "railway2026", db_path=db)          # usernames are case-insensitive
    assert user["id"] == created["id"] and user["last_login_at"]


def test_password_is_never_stored_in_plain_text(db):
    u.create_user("ravi", "secret123x", db_path=db)
    stored = _scalar(db, "SELECT password_hash FROM users")
    assert "secret123x" not in stored and stored.startswith("scrypt$")
    assert u.verify_password("secret123x", stored) and not u.verify_password("secret123y", stored)


def test_same_password_gets_different_hashes():
    assert u.hash_password("samepass1") != u.hash_password("samepass1")


@pytest.mark.parametrize("name", ["ab", "has space", "x" * 31, "drop;table", ""])
def test_invalid_usernames_rejected(db, name):
    with pytest.raises(u.AccountError):
        u.create_user(name, "goodpass1", db_path=db)


@pytest.mark.parametrize("pw", ["short1", "lettersonly", "12345678", "meena123"])
def test_weak_passwords_rejected(db, pw):
    with pytest.raises(u.AccountError):
        u.create_user("meena123", pw, db_path=db)                       # last case: same as username


def test_duplicate_username_case_insensitive(db):
    u.create_user("Priya", "goodpass1", db_path=db)
    with pytest.raises(u.AccountError, match="taken"):
        u.create_user("priya", "otherpass2", db_path=db)


def test_wrong_password_and_unknown_user_give_the_same_message(db):
    u.create_user("kiran", "goodpass1", db_path=db)
    with pytest.raises(u.AccountError) as a:
        u.authenticate("kiran", "wrongpass1", db_path=db)
    with pytest.raises(u.AccountError) as b:
        u.authenticate("nobody", "wrongpass1", db_path=db)
    assert str(a.value) == str(b.value) == u.LOGIN_FAILED


def test_lockout_after_repeated_failures(db):
    u.create_user("sam", "goodpass1", db_path=db)
    for _ in range(u.MAX_FAILED_ATTEMPTS):
        with pytest.raises(u.AccountError):
            u.authenticate("sam", "badpass99", db_path=db)
    with pytest.raises(u.AccountError, match="Too many"):
        u.authenticate("sam", "goodpass1", db_path=db)                   # even the right password waits


def test_successful_login_resets_failure_count(db):
    u.create_user("lee", "goodpass1", db_path=db)
    for _ in range(u.MAX_FAILED_ATTEMPTS - 1):
        with pytest.raises(u.AccountError):
            u.authenticate("lee", "badpass99", db_path=db)
    u.authenticate("lee", "goodpass1", db_path=db)
    for _ in range(u.MAX_FAILED_ATTEMPTS - 1):
        with pytest.raises(u.AccountError, match="Incorrect"):
            u.authenticate("lee", "badpass99", db_path=db)


def test_sql_injection_strings_are_just_text(db):
    with pytest.raises(u.AccountError):
        u.authenticate("' OR '1'='1", "' OR '1'='1", db_path=db)
    uid = u.create_user("safe_user", "goodpass1", db_path=db)["id"]
    u.add_history(uid, "scan", "'); DROP TABLE users; --", {"x": "' OR 1=1"}, db_path=db)
    assert u.list_history(uid, db_path=db)[0]["summary"] == "'); DROP TABLE users; --"
    assert u.get_user(uid, db_path=db) is not None


def test_history_is_private_per_user(db):
    a = u.create_user("user_a", "goodpass1", db_path=db)["id"]
    b = u.create_user("user_b", "goodpass2", db_path=db)["id"]
    e = u.add_history(a, "forecast", "01023 at NIRA", {"delay": 20}, db_path=db)
    u.add_history(b, "scan", "New Delhi board", db_path=db)
    assert [h["summary"] for h in u.list_history(a, db_path=db)] == ["01023 at NIRA"]
    assert u.list_history(a, db_path=db)[0]["details"] == {"delay": 20}
    assert u.delete_history(b, e, db_path=db) == 0                     # b cannot delete a's entry
    assert len(u.list_history(a, db_path=db)) == 1


def test_history_filter_delete_and_bad_kind(db):
    uid = u.create_user("hist", "goodpass1", db_path=db)["id"]
    first = u.add_history(uid, "scan", "one", db_path=db)
    u.add_history(uid, "forecast", "two", db_path=db)
    assert [h["kind"] for h in u.list_history(uid, kind="scan", db_path=db)] == ["scan"]
    assert u.delete_history(uid, first, db_path=db) == 1
    assert u.delete_history(uid, db_path=db) == 1
    assert u.list_history(uid, db_path=db) == []
    with pytest.raises(ValueError):
        u.add_history(uid, "unknown", "x", db_path=db)


def test_change_password(db):
    uid = u.create_user("chg", "goodpass1", db_path=db)["id"]
    with pytest.raises(u.AccountError):
        u.change_password(uid, "wrongold1", "newpass22", db_path=db)
    u.change_password(uid, "goodpass1", "newpass22", db_path=db)
    assert u.authenticate("chg", "newpass22", db_path=db)["id"] == uid


def test_delete_account_needs_password_and_removes_history(db):
    uid = u.create_user("gone", "goodpass1", db_path=db)["id"]
    u.add_history(uid, "scan", "x", db_path=db)
    with pytest.raises(u.AccountError):
        u.delete_user(uid, "wrongpass1", db_path=db)
    u.delete_user(uid, "goodpass1", db_path=db)
    assert u.get_user(uid, db_path=db) is None
    assert _scalar(db, "SELECT COUNT(*) AS n FROM user_history") == 0


def test_existing_database_gains_user_tables(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as c:                                   # a v1 database without user tables
        c.execute("CREATE TABLE stations (station_code TEXT PRIMARY KEY, station_name TEXT NOT NULL, "
                  "latitude REAL, longitude REAL, source TEXT)")
        c.execute("PRAGMA user_version = 1")
    initialize_database(path)
    with sqlite3.connect(path) as c:
        names = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"users", "user_history", "login_attempts"} <= names
        assert c.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_backend_selection(monkeypatch, tmp_path):
    monkeypatch.delenv(accounts_store.ENV_URL, raising=False)
    assert accounts_store.target(None) == "sqlite"
    assert accounts_store.target(tmp_path / "x.db") == "sqlite"
    assert accounts_store.target("postgresql://u:secret@db.example.com/railscan") == "postgres"
    monkeypatch.setenv(accounts_store.ENV_URL, "postgresql://u:secret@db.example.com:5432/railscan?sslmode=require")
    assert accounts_store.target(None) == "postgres"
    assert accounts_store.describe(None) == "PostgreSQL at db.example.com:5432"      # no password shown


# ------------------------------------------------------------------ rate limits and data export
def test_quota_refuses_beyond_the_limit_and_does_not_count_refusals(db):
    for _ in range(3):
        u.take_quota("user:1", "scan", db, limit=3, window_min=10)
    with pytest.raises(u.AccountError, match="Limit reached"):
        u.take_quota("user:1", "scan", db, limit=3, window_min=10)
    assert _scalar(db, "SELECT COUNT(*) FROM usage_events WHERE subject = 'user:1'") == 3


def test_quota_is_per_subject_and_per_kind(db):
    u.take_quota("user:1", "scan", db, limit=1, window_min=10)
    u.take_quota("user:2", "scan", db, limit=1, window_min=10)          # another user
    u.take_quota("user:1", "signup", db, limit=1, window_min=10)        # another kind
    with pytest.raises(u.AccountError):
        u.take_quota("user:1", "scan", db, limit=1, window_min=10)


def test_quota_window_expires(db):
    with open_store(db) as s:
        with s.transaction():
            s.run("INSERT INTO usage_events (subject, kind, created_at) VALUES (?, ?, ?)",
                  ("ip:1.2.3.4", "signup", "2000-01-01T00:00:00+00:00"))
    u.take_quota("ip:1.2.3.4", "signup", db, limit=1, window_min=60)     # the old event no longer counts


def test_export_contains_account_and_history_but_no_password_hash(db):
    user = u.create_user("exporter", "Exporter123", "Ex", db)
    u.add_history(user["id"], "scan", "a scan", {"x": 1}, db)
    data = u.export_user_data(user["id"], db)
    assert data["account"]["username"] == "exporter" and "password_hash" not in data["account"]
    assert [h["summary"] for h in data["history"]] == ["a scan"]


def test_old_login_attempts_are_not_kept(db):
    with open_store(db) as s:
        with s.transaction():
            s.run("INSERT INTO login_attempts (username, attempted_at, success) VALUES (?, ?, ?)",
                  ("someone", "2000-01-01T00:00:00+00:00", 0))
    with pytest.raises(u.AccountError):
        u.authenticate("nobody", "Wrongpass1", db)
    assert _scalar(db, "SELECT COUNT(*) FROM login_attempts WHERE attempted_at < '2001'") == 0
