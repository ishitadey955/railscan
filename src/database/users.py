"""User accounts and per-user history (SQLite by default, PostgreSQL when configured).

Where the data goes is decided by src/database/accounts_store.py: the local
RailScan SQLite file, or an external PostgreSQL server when
RAILSCAN_ACCOUNTS_DB_URL (or a postgresql:// ``db_path``) is given. The
functions below behave identically on both and are tested on both.

Security choices
    * Passwords are hashed with scrypt (Python standard library), a random
      16-byte salt per user, and compared in constant time. The plain password
      is never stored or logged.
    * Login errors never say whether the username exists, and unknown users
      cost the same single scrypt run as known ones.
    * After MAX_FAILED_ATTEMPTS failed logins within LOCKOUT_WINDOW_MIN minutes
      the username is locked for the rest of that window.
    * History rows belong to one user; every query filters by user id, and
      deleting an account deletes its history (ON DELETE CASCADE).
    * All SQL is parameterised.

History stores what RailScan showed (a short summary plus JSON details),
never the uploaded image itself.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .accounts_store import Store, open_store
from .connection import PathLike

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,30}$")
MIN_PASSWORD_LEN = 8
MAX_PASSWORD_LEN = 128
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_WINDOW_MIN = 15
HISTORY_KINDS = ("scan", "forecast", "typical", "replay")
# scrypt cost parameters (n=2**14, r=8, p=1: ~16 MB, well under OpenSSL's default memory limit)
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2 ** 14, 8, 1
LOGIN_FAILED = "Incorrect username or password."


class AccountError(ValueError):
    """A sign-up / login / account request was refused; the message is safe to show."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ passwords
def hash_password(password: str) -> str:
    """'scrypt$n$r$p$salt$hash' (salt and hash base64)."""
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=32)
    b64 = lambda b: base64.b64encode(b).decode("ascii")
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${b64(salt)}${b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, n, r, p, salt, digest = stored.split("$")
        if algo != "scrypt":
            return False
        expected = base64.b64decode(digest)
        got = hashlib.scrypt(password.encode("utf-8"), salt=base64.b64decode(salt), n=int(n), r=int(r),
                             p=int(p), dklen=len(expected))
        return hmac.compare_digest(got, expected)
    except (ValueError, TypeError):
        return False


_DUMMY: List[str] = []


def _dummy_hash() -> str:
    if not _DUMMY:
        _DUMMY.append(hash_password(secrets.token_urlsafe(16)))
    return _DUMMY[0]


def password_problems(password: str, username: str = "") -> List[str]:
    """Reasons a password is rejected (empty list = acceptable)."""
    problems = []
    if len(password) < MIN_PASSWORD_LEN:
        problems.append(f"Use at least {MIN_PASSWORD_LEN} characters.")
    if len(password) > MAX_PASSWORD_LEN:
        problems.append(f"Use at most {MAX_PASSWORD_LEN} characters.")
    if not (re.search(r"[A-Za-z]", password) and re.search(r"\d", password)):
        problems.append("Include at least one letter and one number.")
    if username and password.casefold() == username.casefold():
        problems.append("The password must not be the same as the username.")
    return problems


# ------------------------------------------------------------------ accounts
def _public(row: Dict[str, Any]) -> Dict[str, Any]:
    return {"id": int(row["id"]), "username": row["username"], "display_name": row["display_name"] or row["username"],
            "created_at": row["created_at"], "last_login_at": row["last_login_at"]}


def _user_by_name(db: Store, username: str) -> Optional[Dict[str, Any]]:
    return db.one("SELECT * FROM users WHERE lower(username) = lower(?)", (username,))


def create_user(username: str, password: str, display_name: str = "",
                db_path: Optional[PathLike] = None) -> Dict[str, Any]:
    """Create an account. Raises AccountError with a user-facing message."""
    username = (username or "").strip()
    if not USERNAME_RE.match(username):
        raise AccountError("Username: 3-30 characters, letters, numbers, '.', '_' or '-' only.")
    problems = password_problems(password or "", username)
    if problems:
        raise AccountError("Password: " + " ".join(problems))
    display_name = (display_name or "").strip()[:60] or None
    with open_store(db_path) as db:
        try:
            with db.transaction():
                row = db.one("INSERT INTO users (username, display_name, password_hash, created_at) "
                             "VALUES (?, ?, ?, ?) RETURNING *",
                             (username, display_name, hash_password(password), _now()))
        except Exception as exc:
            if Store.is_unique_violation(exc):
                raise AccountError("That username is already taken.") from None
            raise
    return _public(row)


def _recent_failures(db: Store, username: str) -> int:
    since = (datetime.now(timezone.utc) - timedelta(minutes=LOCKOUT_WINDOW_MIN)).isoformat(timespec="seconds")
    last_ok = db.one("SELECT MAX(attempted_at) AS t FROM login_attempts "
                     "WHERE lower(username) = lower(?) AND success = 1", (username,))["t"]
    since = max(since, last_ok) if last_ok else since
    return int(db.one("SELECT COUNT(*) AS n FROM login_attempts WHERE lower(username) = lower(?) "
                      "AND success = 0 AND attempted_at > ?", (username, since))["n"])


def authenticate(username: str, password: str, db_path: Optional[PathLike] = None) -> Dict[str, Any]:
    """Return the user on success; raise AccountError otherwise (generic message, lockout)."""
    username = (username or "").strip()
    if not username or not password:
        raise AccountError("Enter your username and password.")
    with open_store(db_path) as db:
        if _recent_failures(db, username) >= MAX_FAILED_ATTEMPTS:
            raise AccountError(f"Too many failed attempts. Try again in {LOCKOUT_WINDOW_MIN} minutes.")
        row = _user_by_name(db, username)
        # one scrypt run either way, so the response time does not reveal which usernames exist
        ok = verify_password(password, row["password_hash"] if row else _dummy_hash()) and row is not None
        with db.transaction():
            db.run("INSERT INTO login_attempts (username, attempted_at, success) VALUES (?, ?, ?)",
                   (username, _now(), int(ok)))
            if ok:
                db.run("UPDATE users SET last_login_at = ? WHERE id = ?", (_now(), row["id"]))
        if not ok:
            raise AccountError(LOGIN_FAILED)
        row = db.one("SELECT * FROM users WHERE id = ?", (row["id"],))
    return _public(row)


def get_user(user_id: int, db_path: Optional[PathLike] = None) -> Optional[Dict[str, Any]]:
    with open_store(db_path) as db:
        row = db.one("SELECT * FROM users WHERE id = ?", (int(user_id),))
    return _public(row) if row else None


def change_password(user_id: int, current: str, new: str, db_path: Optional[PathLike] = None) -> None:
    with open_store(db_path) as db:
        row = db.one("SELECT * FROM users WHERE id = ?", (int(user_id),))
        if row is None or not verify_password(current or "", row["password_hash"]):
            raise AccountError("Current password is incorrect.")
        problems = password_problems(new or "", row["username"])
        if problems:
            raise AccountError("New password: " + " ".join(problems))
        with db.transaction():
            db.run("UPDATE users SET password_hash = ? WHERE id = ?", (hash_password(new), row["id"]))


def delete_user(user_id: int, password: str, db_path: Optional[PathLike] = None) -> None:
    """Delete the account and all of its history. Requires the password."""
    with open_store(db_path) as db:
        row = db.one("SELECT * FROM users WHERE id = ?", (int(user_id),))
        if row is None or not verify_password(password or "", row["password_hash"]):
            raise AccountError("Password is incorrect; the account was not deleted.")
        with db.transaction():
            db.run("DELETE FROM users WHERE id = ?", (row["id"],))      # history cascades
            db.run("DELETE FROM login_attempts WHERE lower(username) = lower(?)", (row["username"],))


# ------------------------------------------------------------------ history
def add_history(user_id: int, kind: str, summary: str, details: Optional[Dict[str, Any]] = None,
                db_path: Optional[PathLike] = None) -> int:
    if kind not in HISTORY_KINDS:
        raise ValueError(f"unknown history kind {kind!r}")
    with open_store(db_path) as db:
        with db.transaction():
            row = db.one("INSERT INTO user_history (user_id, created_at, kind, summary, details) "
                         "VALUES (?, ?, ?, ?, ?) RETURNING id",
                         (int(user_id), _now(), kind, summary[:300],
                          json.dumps(details or {}, ensure_ascii=False, default=str)))
    return int(row["id"])


def list_history(user_id: int, kind: Optional[str] = None, limit: int = 200,
                 db_path: Optional[PathLike] = None) -> List[Dict[str, Any]]:
    sql = "SELECT * FROM user_history WHERE user_id = ?"
    params: List[Any] = [int(user_id)]
    if kind:
        sql += " AND kind = ?"
        params.append(kind)
    sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
    params.append(int(limit))
    with open_store(db_path) as db:
        rows = db.all(sql, params)
    return [{"id": int(r["id"]), "created_at": r["created_at"], "kind": r["kind"], "summary": r["summary"],
             "details": json.loads(r["details"] or "{}")} for r in rows]


def delete_history(user_id: int, entry_id: Optional[int] = None, db_path: Optional[PathLike] = None) -> int:
    """Delete one entry (``entry_id``) or all of the user's history. Returns rows deleted.

    The user id is always part of the WHERE clause, so nobody can delete another user's rows.
    """
    with open_store(db_path) as db:
        with db.transaction():
            if entry_id is None:
                cur = db.run("DELETE FROM user_history WHERE user_id = ?", (int(user_id),))
            else:
                cur = db.run("DELETE FROM user_history WHERE user_id = ? AND id = ?", (int(user_id), int(entry_id)))
    return int(cur.rowcount)
