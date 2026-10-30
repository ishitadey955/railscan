"""A backup restores into an empty database with the same rows, and new sign-ups continue after it."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import backup_accounts as ba  # noqa: E402
from src.database import users as u  # noqa: E402


@pytest.fixture(scope="module")
def pg_server(tmp_path_factory):
    pgserver = pytest.importorskip("pgserver")
    srv = pgserver.get_server(str(tmp_path_factory.mktemp("pg_backup")), cleanup_mode="stop")
    yield srv
    srv.cleanup()


def _fresh_db(srv, name):
    import psycopg
    with psycopg.connect(srv.get_uri(), autocommit=True) as c:
        c.execute(f"DROP DATABASE IF EXISTS {name}")
        c.execute(f"CREATE DATABASE {name}")
    return srv.get_uri(name)


def test_backup_then_restore_into_empty_database(pg_server, tmp_path):
    source, target = _fresh_db(pg_server, "acc_source"), _fresh_db(pg_server, "acc_target")
    a = u.create_user("alice", "Alice12345", "Alice", source)
    b = u.create_user("bob", "Bob1234567", "", source)
    u.add_history(a["id"], "scan", "station board", {"script": "Bengali"}, source)
    u.add_history(b["id"], "forecast", "12229 estimate", {"delay": 5}, source)
    with pytest.raises(u.AccountError):
        u.authenticate("alice", "wrong-pass1", source)

    counts = ba.restore(ba.backup(source, tmp_path), target)
    assert counts["users"] == 2 and counts["user_history"] == 2 and counts["login_attempts"] >= 1

    # the restored accounts work: same password, same history, and new ids continue after them
    assert u.authenticate("alice", "Alice12345", target)["id"] == a["id"]
    assert [h["summary"] for h in u.list_history(b["id"], db_path=target)] == ["12229 estimate"]
    assert u.create_user("carol", "Carol12345", "", target)["id"] > b["id"]


def test_restore_refuses_a_database_with_users(pg_server, tmp_path):
    source = _fresh_db(pg_server, "acc_source2")
    u.create_user("dave", "Dave123456", "", source)
    path = ba.backup(source, tmp_path)
    with pytest.raises(SystemExit, match="already has 1 users"):
        ba.restore(path, source)
