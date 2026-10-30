#!/usr/bin/env python3
"""
RailScan - back up and restore the accounts database (users, history, security records).

The railway data needs no backup: it is rebuilt from committed files. The accounts are the only
data users create. Supabase's own backups depend on the plan, so this keeps an independent copy.

    python scripts/backup_accounts.py backup  [--url URL] [--out backups/]
    python scripts/backup_accounts.py restore FILE --url TARGET_URL [--force]

URL defaults to $RAILSCAN_ACCOUNTS_DB_URL. A backup is one JSON file with every row of every
accounts table (password hashes included, so keep the file private: backups/ is git-ignored).
``restore`` refuses a target that already has users unless --force; it keeps the original ids
and resets the id counters, so later sign-ups continue after the restored rows.
Test restores into a separate database (e.g. a second Supabase project), never into production.
See docs/runbooks/backups.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.database.accounts_store import open_store  # noqa: E402

TABLES = ["users", "user_history", "login_attempts", "usage_events"]      # parents first


def backup(url: str, out_dir: Path) -> Path:
    data = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tables": {}}
    with open_store(url) as db:
        for t in TABLES:
            data["tables"][t] = db.all(f"SELECT * FROM {t} ORDER BY id")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"railscan_accounts_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")
    counts = {t: len(rows) for t, rows in data["tables"].items()}
    print(f"wrote {path}: {counts}")
    return path


def restore(path: Path, url: str, force: bool = False) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    with open_store(url) as db:
        existing = db.one("SELECT COUNT(*) AS n FROM users")["n"]
        if existing and not force:
            raise SystemExit(f"target already has {existing} users; use --force to add to it")
        postgres = db.dialect == "postgres"
        with db.transaction():
            for t in TABLES:
                rows = data["tables"].get(t, [])
                for r in rows:
                    cols = list(r)
                    db.run(f"INSERT INTO {t} ({', '.join(cols)}) "
                           + ("OVERRIDING SYSTEM VALUE " if postgres else "")
                           + f"VALUES ({', '.join('?' for _ in cols)})", [r[c] for c in cols])
                if postgres and rows:      # continue numbering after the restored ids
                    db.run(f"SELECT setval(pg_get_serial_sequence('{t}', 'id'), (SELECT MAX(id) FROM {t}))")
        counts = {t: db.one(f"SELECT COUNT(*) AS n FROM {t}")["n"] for t in TABLES}
    print(f"restored {path} into the target: {counts}")
    return counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("backup")
    b.add_argument("--url", default=os.environ.get("RAILSCAN_ACCOUNTS_DB_URL"))
    b.add_argument("--out", type=Path, default=ROOT / "backups")
    r = sub.add_parser("restore")
    r.add_argument("file", type=Path)
    r.add_argument("--url", required=True, help="target database (a test database, not production)")
    r.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.command == "backup":
        if not a.url:
            raise SystemExit("no database: pass --url or set RAILSCAN_ACCOUNTS_DB_URL")
        backup(a.url, a.out)
    else:
        restore(a.file, a.url, a.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
