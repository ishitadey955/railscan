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

## Google OCR (optional)
1. https://console.cloud.google.com -> create a project (e.g. "railscan").
2. Billing -> link a billing account (Google requires one even for the free tier; RailScan caps use at
   1,000 photos a month, inside the free tier).
3. APIs & Services -> Library -> "Cloud Vision API" -> Enable.
4. APIs & Services -> Credentials -> Create credentials -> API key. Edit it: API restrictions ->
   Restrict key -> Cloud Vision API only. Copy the key.
5. Streamlit: Manage app -> Settings -> Secrets: add `GOOGLE_VISION_API_KEY = "<the key>"`, save, reboot.
6. Check: scan a sign; the Text section shows "Engine: Google Cloud Vision". Logs show
   `google_ocr_failed` with the reason if the key or API setup is wrong (the app then uses EasyOCR).
7. Measure it (on your PC, with the key in the environment): `python scripts/evaluate_google_ocr.py`.
