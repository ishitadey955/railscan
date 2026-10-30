# 0006 Accounts and history in Supabase PostgreSQL

**Context.** The host's disk is reset on every restart, so accounts cannot live in the local SQLite file there.

**Decision.** `RAILSCAN_ACCOUNTS_DB_URL` points the accounts store at PostgreSQL (Supabase); unset, accounts
use the local SQLite file (development). One connection per process is reused; prepared statements are off
so the Supabase pooler works.

**Consequences.** Data survives restarts. Supabase's own backups depend on the plan, so
`scripts/backup_accounts.py` keeps an independent copy (runbooks/backups.md). The tables are created by the
app (`PG_DDL`, additive only); there is no separate migration tool.
