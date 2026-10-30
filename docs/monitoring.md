# Monitoring

Kept small: only checks that need a human to act.

## Uptime
- Free uptime monitor (e.g. UptimeRobot) on the app's URL every 5 minutes, e-mail alert after 2 failures.
  Streamlit also answers `/_stcore/health` with `ok` while the server process runs; whether that path is
  reachable from outside on Streamlit Community Cloud is not verified here.
- Supabase: the project's e-mail notifications (paused project, usage limits).

## Logs (Manage app -> Logs)
JSON lines from `src/app_logging.py`, redacted. Useful `event` values:

| event | meaning | act when |
|---|---|---|
| `scan_failed` | a scan raised an error | more than a few in a day |
| `history_save_failed` / `history_read_failed` | accounts database unreachable | any |
| `config_invalid` | bad `RAILSCAN_ACCOUNTS_DB_URL` | any (app refuses to start) |
| `warmup_failed` | scanner models could not load | any |
| `signup`, `login`, `scan` | normal use (user ids only) | usage figures |

## Weekly look (2 minutes)
Sign-ups and scans that week (count `"event": "signup"` / `"scan"` in the logs), any error lines, Supabase
database size, and whether Streamlit showed a throttle banner.

Runbook for anything found: runbooks/incident.md.
