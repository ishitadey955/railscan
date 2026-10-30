"""Logs are JSON lines and never contain passwords, keys, database credentials or e-mail addresses."""
import json
import logging

from src.app_logging import JsonFormatter, redact


def _line(msg, *args, exc=None, **extra):
    rec = logging.makeLogRecord({"name": "railscan.test", "levelname": "ERROR", "levelno": 40,
                                 "msg": msg, "args": args, **extra})
    if exc is not None:
        rec.exc_info = (type(exc), exc, None)
    return json.loads(JsonFormatter().format(rec))


def test_database_url_password_is_removed():
    out = redact("could not connect to postgresql://postgres.abc:S3cr3t!pw@aws-0.pooler.supabase.com:6543/postgres")
    assert "S3cr3t" not in out and "postgresql://postgres.abc:***@aws-0" in out


def test_emails_and_key_values_are_removed():
    out = redact("user ishita@example.com sent password=hunter22 and api_key: abc123")
    assert "example.com" not in out and "hunter22" not in out and "abc123" not in out


def test_secret_named_fields_are_masked_and_ids_kept():
    line = _line("login", user_id=7, password="hunter22", auth_token="t0k", event="login_failed")
    assert line["password"] == "***" and line["auth_token"] == "***"
    assert line["user_id"] == 7 and line["event"] == "login_failed" and line["level"] == "error"


def test_exception_text_is_redacted():
    line = _line("db error", exc=RuntimeError("postgresql://u:pw123@db.example.com/x refused"))
    assert "pw123" not in line["error"] and line["error"].startswith("RuntimeError")
