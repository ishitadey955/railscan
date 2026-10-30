"""Structured logs for RailScan: one JSON object per line, with secrets and personal data removed.

    from src.app_logging import get_logger
    log = get_logger("railscan.app")
    log.error("scan failed", exc_info=True, extra={"event": "scan_failed", "user_id": 7})

Logs go to stderr, which Streamlit Community Cloud shows under Manage app -> Logs. Users see a
short, friendly message; the details (exception type, traceback) go only to the log.

Redaction (applied to the message, the traceback and every extra field):
    * fields named like password / token / secret / authorization / api_key / cookie -> "***"
    * passwords inside database URLs (postgresql://user:***@host)
    * e-mail addresses
Log user ids, never usernames or passwords. Set RAILSCAN_LOG_LEVEL=DEBUG for debug output.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone

SECRET_KEYS = re.compile(r"pass(word)?|token|secret|authori[sz]ation|api[_-]?key|cookie|dsn", re.I)
_URL_PASSWORD = re.compile(r"(\b[a-z][a-z0-9+.-]*://[^:/@\s]+:)[^@\s]+@", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_KEY_VALUE = re.compile(r"\b(password|passwd|pwd|token|secret|api[_-]?key)\s*[=:]\s*\S+", re.I)
_STANDARD = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


def redact(text: str) -> str:
    text = _URL_PASSWORD.sub(r"\1***@", str(text))
    text = _KEY_VALUE.sub(lambda m: f"{m.group(1)}=***", text)
    return _EMAIL.sub("***@***", text)


def _clean(key: str, value):
    if SECRET_KEYS.search(key):
        return "***"
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return redact(value)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {"time": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
               "level": record.levelname.lower(), "logger": record.name, "message": redact(record.getMessage())}
        for key, value in vars(record).items():
            if key not in _STANDARD and not key.startswith("_"):
                out[key] = _clean(key, value)
        if record.exc_info:
            out["error"] = redact(f"{record.exc_info[0].__name__}: {record.exc_info[1]}")
            out["traceback"] = redact(self.formatException(record.exc_info))
        return json.dumps(out, ensure_ascii=False, default=str)


def get_logger(name: str = "railscan") -> logging.Logger:
    """A logger under 'railscan' that writes JSON lines to stderr (configured once per process)."""
    root = logging.getLogger("railscan")
    if not any(getattr(h, "_railscan", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter())
        handler._railscan = True
        root.addHandler(handler)
        root.setLevel(os.environ.get("RAILSCAN_LOG_LEVEL", "INFO").upper())
        root.propagate = False
    return logging.getLogger(name) if name.startswith("railscan") else root.getChild(name)
