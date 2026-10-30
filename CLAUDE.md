# Project: RailScan

## What this product does
RailScan reads photos of Indian railway signs (script detection with a CNN + OCR), identifies the station
and train numbers, and estimates train delays from recorded journeys. Users are individual travellers with
personal accounts and a private history. Coverage: northern India and West Bengal. No live train status.

## Stack (do not change without asking)
- App: Python 3.11 + Streamlit (one process: UI and logic), entry point `app.py`
- Models: Keras CNN on the PyTorch backend (no TensorFlow at runtime), EasyOCR, scikit-learn RF + XGBoost
- OCR: Google Cloud Vision (`GOOGLE_VISION_API_KEY`, 0007), else Gemini (`GEMINI_API_KEY`, 0008), else EasyOCR
- Railway data: committed Parquet/CSV files; SQLite (`data/database/railscan.db`) where fully built,
  otherwise one train at a time from Parquet (`src/database/on_demand.py`)
- Accounts and history: PostgreSQL (Supabase) via `RAILSCAN_ACCOUNTS_DB_URL`, else the local SQLite file
- Auth: own scrypt-based accounts (`src/database/users.py`); see docs/decisions/0005
- Hosting: Streamlit Community Cloud (redeploys `main` on every push)

## Rules
- Never commit secrets. Settings are environment variables / Streamlit secrets, documented in
  `.env.example` and `.streamlit/secrets.toml.example`.
- Never fabricate data or metrics. Every number in the app or docs comes from a script's real output.
- Run lint and the full test suite before saying a task is done (commands below); show the output.
- Never change a test to make it pass; fix the code, or report the bug.
- Every per-user query filters by user id; history never stores uploaded photos.
- Users see friendly messages; details go to the JSON log (`src/app_logging.py`), redacted.
- Expensive or abusable actions go through `users.take_quota()` (rate limits in the database).
- Schema changes: bump `SCHEMA_VERSION` in `src/database/schema.py`, add the table/column there AND in
  `PG_DDL` (`src/database/accounts_store.py`); changes must be additive (old databases upgrade in place).
- Ask before adding a dependency. Small, reviewable changes.
- Tests use throw-away databases (tmp SQLite, pgserver PostgreSQL), never the real accounts database.
- Port 8501 locally belongs to another app; run RailScan on 8502.

## Commands
- Run: `.venv\Scripts\streamlit run app.py --server.port 8502`
- Lint: `.venv\Scripts\python -m pyflakes app.py src scripts tests`
- Test: `.venv\Scripts\python -m pytest tests -q`   (run from the project root)
- Full railway database (optional, ~5 min): `python scripts/setup_database.py all`
- Account backup: `python scripts/backup_accounts.py backup` (see docs/runbooks/backups.md)

## Architecture and decisions
docs/system-design.md, docs/architecture.md, docs/decisions/ (why things are the way they are),
docs/runbooks/ (deploy, rollback, backups, incidents), docs/security/threat-model.md, docs/monitoring.md.
