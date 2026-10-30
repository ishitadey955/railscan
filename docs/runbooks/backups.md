# Backups (accounts database)

Only the accounts database holds data users create (accounts, history). Railway data and models are rebuilt
from committed files and need no backup.

## Take a backup (weekly, and before risky changes)
```bat
set RAILSCAN_ACCOUNTS_DB_URL=postgresql://...      :: from Streamlit Secrets
.venv\Scripts\python scripts\backup_accounts.py backup
```
Writes `backups/railscan_accounts_<time>.json` (git-ignored; it contains password hashes: keep it private,
e.g. in an encrypted folder; delete old copies you no longer need).

Supabase also keeps its own backups depending on the plan (Dashboard -> Database -> Backups); check what your
plan includes.

## Test a restore (at least once, then every few months)
1. Create a separate, empty Supabase project (never use production as the target).
2. `.venv\Scripts\python scripts\backup_accounts.py restore backups\<file>.json --url <TEST-PROJECT-URL>`
3. Point a local run at the test project and log in with a known account.

The restore keeps the original ids, refuses a target that already has users (unless `--force`), and is
covered by tests/test_backup_accounts.py.

## Real restore after data loss
Restore into a new empty project, check it as above, then change `RAILSCAN_ACCOUNTS_DB_URL` in Streamlit
Secrets to the new project and reboot the app.
