# Deploy

Streamlit Community Cloud redeploys `main` on every push. Production = `main`.

## Every change
1. Run lint and tests locally (see CLAUDE.md -> Commands); both must pass.
2. Push. Wait for the **CI** check on GitHub to be green (Actions tab). If it is red, do not deploy: fix first.
3. In Streamlit: **Manage app -> Reboot app** (a live update can leave a half-updated process; the app repairs
   that itself, but a reboot starts clean).
4. Open the app, wait for the "scanner is getting ready" note to disappear (~half a minute), then check:
   log in, open every page, scan one photo, get one delay estimate.
5. Check **Manage app -> Logs** for lines with `"level": "error"`.

## First deploy / new environment
1. Supabase: create a project; Connect -> Session pooler URL.
2. Streamlit: New app -> repo, branch `main`, file `app.py`, Python 3.11.
3. Settings -> Secrets: paste from `.streamlit/secrets.toml.example`, with the real URL.
4. Deploy, then steps 3–5 above.

## Avoid
Many pushes or reboots in a row: each restart loads the models again and the host may throttle the app's
CPU (seen on 30 Oct 2026). Batch changes into one push.
