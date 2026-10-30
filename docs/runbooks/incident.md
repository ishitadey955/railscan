# Incidents

## Signs
App does not load, "Oh no" page, every scan fails, log-in fails for everyone, Streamlit "throttled" banner.

## First steps
1. **Manage app -> Logs**: look for `"level": "error"` lines (JSON) and tracebacks; note the `event` field
   (`scan_failed`, `history_save_failed`, `config_invalid`, `warmup_failed`, ...).
2. Was there a push in the last hour? If yes and the app broke after it: roll back (rollback.md).
3. Accounts errors (`history_*`, log-in fails): check Supabase status and that the project is not paused
   (free projects pause after inactivity: Dashboard -> Restore).
4. "Throttled": do not reboot or push repeatedly; wait for the time shown, then reboot once.
5. Memory restarts (app restarts during scans): see docs/HOSTING.md (memory needs).

## Tell users
Post a short note where your users are (e.g. the course group or README): what is broken, since when, and
when you will update next.

## Afterwards (5 minutes)
Write a few lines in `docs/runbooks/incidents-log.md`: date, what users saw, cause, fix, what prevents it
next time.
