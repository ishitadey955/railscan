"""RailScan - Streamlit interface.

Run:  streamlit run app.py

Pages
    Scan a sign       upload -> CNN script -> OCR -> train number / station / platform
    Journey planner   any date: train, station, current delay -> next-stop estimate, rest of route;
                      without a current delay: typical recorded delays; plus a recorded-journey check
    Data and models   sources, dataset statistics, measured model results
    My history        the signed-in user's saved results; password change; account deletion

Users sign up / log in first (src/database/users.py, SQLite).

The app replays recorded journeys from the historical dataset. It has no live
railway feed; predictions are model estimates, never official running status.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


def _forget_railscan_modules() -> None:
    """Drop RailScan's own modules from the import cache so they load again from disk.

    When a push is pulled into the running server, Streamlit re-imports the changed files
    while other visitors' pages run. That can leave half-initialised RailScan modules
    behind, and every later page load then fails with ImportError / KeyError until the app
    is rebooted. Importing them again from scratch repairs it.
    """
    for name in [n for n in sys.modules if n == "src" or n.startswith("src.")]:
        sys.modules.pop(name, None)
    importlib.invalidate_caches()


for _attempt in (1, 2):
    try:
        from src.app_logging import get_logger
        from src.database import accounts_store
        from src.database import users as accounts
        from src.database.users import AccountError
        from src.delay_prediction import DelayModelUnavailable, DelayPredictor, RISK_RULE
        from src.image.preprocessing import InvalidImageError, letterbox, load_rgb
        from src.image.quality import measure, retake_advice
        from src.railway_assistant import (LIVE_DISCLAIMER, match_options, needs_confirmation, station_services,
                                           train_summary)
        from src.next_station import describe_remaining
        from src.planner import METHOD_LABELS, forecast, model_coverage, typical_delays
        from src.stop_analysis import historical_stop_stats
        from src.position import estimate_from_forecast, estimate_from_typical, map_points
        from src.timetable import (TimetableUnavailable, departure_stations, load_timetable, route, runs_on,
                                   schedule_on)
        from src.railscan_service import (analyze, as_time, journey, journey_dates, load_master,
                                          platform_numbers, station_choices, summarize_ocr_numbers)
        break
    except (ImportError, KeyError):
        if _attempt == 2:
            raise
        _forget_railscan_modules()

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
PAGES = ["Scan a sign", "Journey planner", "Data and models", "My history"]
# OCR language strings -> what the user sees
LANG_NAMES = {"eng+hin+mar": "Devanagari + English", "hin+mar": "Devanagari", "eng+hin": "Hindi + English",
              "eng+mar": "Marathi + English", "eng+kan": "Kannada + English", "eng+tel": "Telugu + English", "eng+ben": "Bengali + English",
              "hin": "Hindi", "mar": "Marathi", "kan": "Kannada", "tel": "Telugu", "ben": "Bengali", "eng": "English"}
NOTICE = ("RailScan covers trains that stop in **northern India** (J&K, Ladakh, Himachal, Punjab, Chandigarh, "
          "Haryana, Delhi, Rajasthan, Uttar Pradesh, Uttarakhand) and **West Bengal**. Estimates come from a model "
          "trained on "
          "**recorded journeys** and the delay you enter. RailScan has **no live train status**. Check official "
          "Indian Railways sources before travelling.")

PRIVACY = ("- **Account:** username, optional display name, a salted scrypt hash of your password (never the "
           "password itself), when the account was created and last used.\n"
           "- **History:** the results RailScan showed you (scans, delay estimates), so you can look them up "
           "again. Uploaded photos are **never** stored.\n"
           "- **Security records:** log-in attempts (kept 30 days, to block password guessing) and usage "
           "counts (kept one day, to limit scans and sign-ups).\n"
           "- **Your control:** download everything with *Download my data*; deleting your account deletes "
           "its history too.")

st.set_page_config(page_title="RailScan", layout="wide")

# Hosted deployments keep the accounts database address in Streamlit secrets
# (Settings -> Secrets: RAILSCAN_ACCOUNTS_DB_URL = "postgresql://..."); locally it may be an environment variable.
try:
    if "RAILSCAN_ACCOUNTS_DB_URL" in st.secrets:
        os.environ.setdefault("RAILSCAN_ACCOUNTS_DB_URL", str(st.secrets["RAILSCAN_ACCOUNTS_DB_URL"]))
except Exception:  # no secrets file: local SQLite accounts
    pass
LOG = get_logger("railscan.app")


def _session_id() -> str:
    """Streamlit's id of this browser session, to follow one visitor's requests through the logs."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx().session_id[:8]
    except Exception:
        return "-"


# Fail loudly at start-up on a malformed setting instead of failing later on a user's click.
_accounts_url = os.environ.get("RAILSCAN_ACCOUNTS_DB_URL", "").strip()
if _accounts_url and not _accounts_url.startswith(("postgresql://", "postgres://")):
    LOG.error("invalid RAILSCAN_ACCOUNTS_DB_URL", extra={"event": "config_invalid"})
    st.error("Configuration error: RAILSCAN_ACCOUNTS_DB_URL must start with postgresql:// "
             "(Streamlit: Manage app -> Settings -> Secrets). Leave it unset to keep accounts in the local "
             "SQLite file.")
    st.stop()
st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1200px;}
  [data-testid="stMetricValue"] {font-size: 1.35rem;}
  [data-testid="stMetricLabel"] p {font-size: 0.85rem; color: #4a5059;}
  .tag {display: inline-block; padding: 0 .45rem; border: 1px solid #c9c5bb; border-radius: 3px;
        font-size: .75rem; color: #4a5059; margin-left: .4rem; vertical-align: middle;}
  h1 {margin-bottom: 0;}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- cached resources
def full_database() -> bool:
    """True where the full railway database has been built (python scripts/setup_database.py all).

    A fresh server never builds it (4-5 minutes of full CPU after every restart). There each
    train is read from the committed Parquet file when it is first needed (~0.4 s, same rows,
    src/database/on_demand.py) and the Data page uses figures exported from the full database,
    so no page ever waits for a build.
    """
    from src.database.app_data import data_ready
    return data_ready()


@st.cache_data(show_spinner=False, max_entries=16)
def train_data(train_no: str) -> pd.DataFrame:
    """One train's observations (the whole table is too large to keep in memory)."""
    if full_database():
        return load_master(train_no=train_no)
    from src.database.on_demand import load_train
    return load_train(train_no)


@st.cache_data(show_spinner=False)
def get_trains() -> pd.DataFrame:
    """train_no, train_name, journeys, rows for every train in the dataset.

    Without the full database: from the committed Parquet copy of the same dataset
    (identical result, under a second).
    """
    from src.database.app_data import train_list_db, train_list_parquet
    return train_list_db() if full_database() else train_list_parquet()


@st.cache_data(show_spinner="Summarising the dataset ...")
def get_overview() -> dict:
    from src.database.app_data import overview_db, overview_files
    if full_database():
        return {**overview_db(), "source": "SQLite"}
    return {**overview_files(), "source": "Parquet"}


@st.cache_resource(show_spinner=False)
def get_predictor():
    try:
        return DelayPredictor.load(), None
    except DelayModelUnavailable as exc:
        return None, str(exc)


@st.cache_data(show_spinner=False)
def get_timetable() -> pd.DataFrame:
    return load_timetable()


@st.cache_resource(show_spinner=False)
def get_coverage() -> set:
    return model_coverage()


def cnn_status():
    """Cheap check (no model import): the models load in the background after login
    (warm_up_scanner), so opening the page stays fast and the server stays responsive."""
    import importlib.util
    from src.image.prediction import LABELS_PATH, MODEL_PATH
    for p in (MODEL_PATH, LABELS_PATH):
        if not p.exists():
            return False, f"{p.name} not found. Run: python scripts/train_cnn.py --arch mobilenetv2 --img-size 224"
    for package in ("keras", "torch"):
        if importlib.util.find_spec(package) is None:
            return False, f"{package} is not installed (pip install -r requirements.txt)"
    return True, None


@st.cache_resource(show_spinner=False)
def warm_up_scanner() -> dict:
    """Load the CNN and the four OCR readers in a background thread, once per server process,
    starting with the first page anyone opens (normally the log-in page).

    After a server restart this downloads the OCR models (~450 MB, once per server) and loads
    them, which takes about as long as logging in or creating an account, so the first scan
    does not wait. A scan started earlier waits for the same load (the loaders are locked).
    Returns a status dict: {"done": bool, "error": str | None, "seconds": float | None}.
    """
    import threading
    import time
    status = {"done": False, "error": None, "seconds": None}

    def run():
        started = time.monotonic()
        try:
            from src.image.prediction import load_model
            from src.ocr import language_candidates
            from src.ocr.ocr_engine import preload_readers
            from src.script_evidence import SCRIPTS
            load_model()
            preload_readers([language_candidates(s_)[0] for s_ in SCRIPTS])
        except Exception as exc:      # the scan itself reports what is missing
            status["error"] = f"{type(exc).__name__}: {exc}"
            LOG.error("scanner warm-up failed", exc_info=True, extra={"event": "warmup_failed"})
        finally:
            _release_free_memory()
            status["seconds"] = round(time.monotonic() - started, 1)
            status["done"] = True

    threading.Thread(target=run, name="railscan-scanner-warm-up", daemon=True).start()
    return status


def _release_free_memory() -> None:
    """Loading the OCR models briefly holds float copies that are then quantised; on Linux
    (Streamlit Cloud) glibc keeps that freed memory unless asked to return it."""
    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):      # not glibc (Windows, macOS): nothing to do
        pass


@st.cache_resource(show_spinner="Loading the scanner models (only right after the server starts)...")
def get_cnn():
    from src.image.prediction import load_model
    return load_model()


@st.cache_resource(show_spinner="Loading OCR engine...")
def get_ocr():
    from src.image.prediction import predict_script
    from src.ocr import OCREngine
    return OCREngine(script_predictor=predict_script)


@st.cache_resource(show_spinner="Loading station list...")
def get_matcher():
    from src.railway import StationMaster, StationMatcher
    # the station CSVs hold the same stations as the database (which is built from them)
    return StationMatcher(StationMaster.load(prefer_database=full_database()))


@st.cache_data(show_spinner=False)
def read_json(path: str):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def tag(text: str) -> str:
    return f'<span class="tag">{text}</span>'


def section(title: str, kind: str = "") -> None:
    st.markdown(f"#### {title}{tag(kind) if kind else ''}", unsafe_allow_html=True)


ACCENT = "#1f4e79"


def bar_chart(df: pd.DataFrame, x: str, y: str, order: list):
    """Bars in a fixed category order (st.bar_chart sorts labels alphabetically)."""
    import altair as alt
    return (alt.Chart(df).mark_bar(color=ACCENT)
            .encode(x=alt.X(f"{x}:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y(f"{y}:Q", title=y), tooltip=[x, y])
            .properties(height=260))


def delay_histogram(delays: pd.Series):
    """10-minute bins from -30 to 150+; the outer bins collect everything beyond them."""
    edges = np.arange(-30, 160, 10)
    idx = np.clip(np.floor((delays.to_numpy(dtype=float) + 30) / 10), 0, len(edges) - 1).astype(int)
    counts = np.bincount(idx, minlength=len(edges))
    labels = [f"{e}+" if i == len(edges) - 1 else (f"<{e + 10}" if i == 0 else str(e)) for i, e in enumerate(edges)]
    df = pd.DataFrame({"delay (min)": labels, "observations": counts})
    return bar_chart(df, "delay (min)", "observations", labels)


def fmt_min(v, signed: bool = True) -> str:
    if v is None or pd.isna(v):
        return "-"
    return f"{v:+.0f} min" if signed else f"{v:.0f} min"


# ---------------------------------------------------------------- navigation
if "page" not in st.session_state:
    st.session_state.page = PAGES[0]


def go_to_lookup(train_no=None, station_code=None):
    st.session_state.page = "Journey planner"
    if train_no:
        st.session_state.plan_train = train_no
    if station_code:
        st.session_state.prefer_station = station_code


# ---------------------------------------------------------------- accounts
def current_user():
    return st.session_state.get("user")


def log_out() -> None:
    for key in list(st.session_state.keys()):
        if key != "page":
            del st.session_state[key]
    st.session_state.page = PAGES[0]


def record(kind: str, summary: str, details: dict) -> None:
    """Save one entry to the signed-in user's history; never breaks the page."""
    user = current_user()
    if not user:
        return
    try:
        accounts.add_history(user["id"], kind, summary, details)
    except Exception:  # database unreachable: keep the result on screen, details go to the log
        LOG.error("history save failed", exc_info=True,
                  extra={"event": "history_save_failed", "user_id": user["id"], "session": _session_id()})
        st.warning("This result could not be saved to your history right now. It is still shown below.")


def page_auth() -> None:
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.title("RailScan")
        st.markdown("CNN-Based Indian Railway Sign Recognition and Delay Prediction")
        st.markdown("Sign in to scan railway signs, plan journeys and keep a history of your results.")
        tab_in, tab_up = st.tabs(["Log in", "Create account"])
        with tab_in:
            with st.form("login_form"):
                username = st.text_input("Username", key="login_username", autocomplete="username")
                password = st.text_input("Password", type="password", key="login_password",
                                         autocomplete="current-password")
                if st.form_submit_button("Log in", type="primary"):
                    try:
                        st.session_state.user = accounts.authenticate(username, password)
                        LOG.info("login", extra={"event": "login", "user_id": st.session_state.user["id"],
                                                 "session": _session_id()})
                        st.session_state.page = PAGES[0]
                        st.rerun()
                    except AccountError as exc:
                        LOG.info("login refused", extra={"event": "login_refused", "session": _session_id()})
                        st.error(str(exc))
        with tab_up:
            with st.form("signup_form"):
                username = st.text_input("Username", key="signup_username", autocomplete="username",
                                         help="3-30 characters: letters, numbers, '.', '_' or '-'.")
                display = st.text_input("Display name (optional)", key="signup_display")
                password = st.text_input("Password", type="password", key="signup_password",
                                         autocomplete="new-password",
                                         help="At least 8 characters, with at least one letter and one number.")
                confirm = st.text_input("Confirm password", type="password", key="signup_confirm",
                                        autocomplete="new-password")
                if st.form_submit_button("Create account", type="primary"):
                    if password != confirm:
                        st.error("The two passwords do not match.")
                    else:
                        try:
                            ip = getattr(st.context, "ip_address", None)
                            if ip:      # sign-up spam: a few new accounts per address and hour
                                accounts.take_quota(f"ip:{ip}", "signup")
                            st.session_state.user = accounts.create_user(username, password, display)
                            LOG.info("signup", extra={"event": "signup", "user_id": st.session_state.user["id"],
                                                      "session": _session_id()})
                            st.session_state.page = PAGES[0]
                            st.rerun()
                        except AccountError as exc:
                            st.error(str(exc))
        st.caption("Your password is stored only as a salted scrypt hash. History keeps the results "
                   "RailScan showed you, never your photos.")
        st.caption(NOTICE)
        with st.expander("What RailScan stores"):
            st.markdown(PRIVACY)


# the scanner models start loading with the first page anyone opens, so they are ready by
# the time that person has logged in or created an account
if cnn_status()[0]:
    warm_up_scanner()

if current_user() is None:
    page_auth()
    st.stop()


with st.sidebar:
    st.markdown("### RailScan")
    st.radio("Page", PAGES, key="page", label_visibility="collapsed")
    st.divider()
    _u = current_user()
    st.caption(f"Signed in as **{_u['display_name']}** ({_u['username']})")
    st.button("Log out", on_click=log_out)
    st.divider()
    st.caption(NOTICE)

st.title("RailScan")
st.markdown("CNN-Based Indian Railway Sign Recognition and Delay Prediction")


# ================================================================= SCAN
def page_scan() -> None:
    st.markdown("Upload a photo of a station board or sign. RailScan identifies the script with a CNN, "
                "reads the text with OCR and looks for a train number, station name or platform number.")
    cnn_ok, cnn_err = cnn_status()
    if not cnn_ok:
        st.warning(f"Script classifier unavailable: {cnn_err}")
    if cnn_ok and not warm_up_scanner()["done"]:
        st.caption("⏳ The scanner is getting ready (only right after the server restarts). You can upload "
                   "a photo now; Scan waits for it automatically.")

    up = st.file_uploader("Sign image (JPG or PNG, up to 10 MB)", type=["jpg", "jpeg", "png"])
    if up is None:
        st.info("No image uploaded. You can also go straight to the **Journey planner**.")
        return
    data = up.getvalue()
    labels = read_json(str(ROOT / "models" / "language_labels.json")) or {}
    size = int(labels.get("image_size", 128))
    try:
        (w, h), shown, cnn_input = photo_preview(data, size)
    except InvalidImageError as exc:
        st.error(f"This file could not be read as an image: {exc}")
        return

    c1, c2 = st.columns([3, 1])
    with c1:
        st.image(shown, caption=f"Original ({w} x {h} px)", width="stretch", output_format="JPEG")
    with c2:
        st.image(cnn_input, caption=f"CNN input ({size} x {size}, aspect ratio kept)", width=180)

    if st.button("Scan image", type="primary", disabled=not cnn_ok):
        user = current_user()
        try:        # a scan is ~20 s of full CPU; a burst could get the whole app throttled
            accounts.take_quota(f"user:{user['id']}", "scan")
        except AccountError as exc:
            st.warning(str(exc))
            return
        get_cnn()
        with st.spinner("Classifying script and reading text..."):
            try:
                st.session_state.scan = run_scan(data)
                st.session_state.scan["file"] = up.name
                st.session_state.scan_name = up.name
                LOG.info("scan", extra={"event": "scan", "user_id": user["id"], "session": _session_id(),
                                        "script": st.session_state.scan["script"]["predicted_script"]})
            except Exception:          # keep the page usable; details go to the log
                st.session_state.pop("scan", None)
                LOG.error("scan failed", exc_info=True,
                          extra={"event": "scan_failed", "user_id": user["id"], "session": _session_id()})
                st.error("The scan failed. Please try again, or try another photo. If it keeps failing, the "
                         "app's log (Manage app -> Logs) has the details.")
        if st.session_state.get("scan") and st.session_state.get("scan_name") == up.name:
            record_scan(up.name, st.session_state.scan)
    scan = st.session_state.get("scan")
    if scan and st.session_state.get("scan_name") == up.name:
        show_scan(scan)


@st.cache_data(show_spinner=False, max_entries=4)
def photo_preview(data: bytes, size: int):
    """Decoded once per photo: (width, height), a copy at most 1280 px for the page (a phone
    photo would otherwise go to the browser as a ~9 MB PNG on every click) and the CNN input."""
    from PIL import Image
    rgb = load_rgb(data)
    shown = Image.fromarray(rgb)
    shown.thumbnail((1280, 1280))
    return (rgb.shape[1], rgb.shape[0]), np.asarray(shown), letterbox(rgb, size)


@st.cache_data(show_spinner=False, max_entries=64)
def services_at(station_code: str) -> pd.DataFrame:
    return station_services(get_timetable(), station_code)


def record_scan(name: str, scan: dict) -> None:
    s_, ocr, station = scan["script"], scan["ocr"], scan["station"] or {}
    where = (f"{station['station_name']} ({station['station_code']})" if station.get("station_code")
             else "no station identified")
    trains = ", ".join(scan["numbers"]["known_trains"]) or "none"
    read_as = LANG_NAMES.get(ocr.get("ocr_language") or "", "") or f"no text (CNN: {s_['predicted_script']})"
    record("scan", f"{name}: {where}; read as {read_as}; trains {trains}",
           {"file": name, "script": s_["predicted_script"], "script_confidence": round(s_["confidence"], 3),
            "script_probabilities": {k: round(v, 3) for k, v in s_["class_probabilities"].items()},
            "ocr_status": ocr["status"], "ocr_text": ocr.get("clean_text"), "ocr_raw_text": ocr.get("raw_text"),
            "ocr_confidence": ocr.get("ocr_confidence"), "read_as": ocr.get("ocr_language"),
            "station": {k: station.get(k) for k in ("status", "station_name", "station_code", "match_score",
                                                    "match_method")},
            "train_numbers": scan["numbers"], "platforms": scan["platforms"]})


def run_scan(data: bytes) -> dict:
    from src.image.prediction import predict_script
    from src.script_evidence import identify_script
    # validated procedure: CNN + an OCR reading with each script, combined (scripts/evaluate_script_fusion.py)
    res = identify_script(data, predict_script, get_ocr())
    c = res["combined"]
    script = {"predicted_script": c["script"], "confidence": c["confidence"],
              "class_probabilities": c["probabilities"], "low_confidence": c["low_confidence"],
              "cnn_probabilities": res["cnn"]["class_probabilities"]}
    ocr = res["ocr"]
    text = ocr.get("clean_text") or ""
    numbers = summarize_ocr_numbers(get_trains(), ocr.get("train_numbers") or [])
    station = get_matcher().match(ocr) if ocr["status"] == "ok" else None
    st.session_state.scan_counter = st.session_state.get("scan_counter", 0) + 1
    return {"script": script, "ocr": ocr, "numbers": numbers, "station": station,
            "platforms": platform_numbers(text), "quality": measure(load_rgb(data)),
            "id": st.session_state.scan_counter}


def show_scan(scan: dict) -> None:
    s, ocr = scan["script"], scan["ocr"]
    section("Script", "CNN + OCR evidence")
    c1, c2 = st.columns([1, 2])
    c1.metric("Detected script", s["predicted_script"])
    c1.caption(f"{s['confidence']:.0%} confidence")
    cnn_p = s.get("cnn_probabilities") or s["class_probabilities"]
    probs = pd.DataFrame({"script": list(s["class_probabilities"]),
                          "combined": list(s["class_probabilities"].values()),
                          "image only (CNN)": [cnn_p.get(k) for k in s["class_probabilities"]]})
    c2.dataframe(probs, hide_index=True, width="stretch",
                 column_config={c_: st.column_config.ProgressColumn(c_, min_value=0.0, max_value=1.0, format="%.2f")
                                for c_ in ("combined", "image only (CNN)")})
    ev = read_json(str(REPORTS / "script_fusion_evaluation.json")) or {}
    at = ev.get("test_at_threshold") or {}
    if s["low_confidence"]:
        st.warning("Uncertain script." + (f" In testing, only {at['accuracy_when_not']:.0%} of such uncertain "
                                          "decisions were right; treat the text below with care." if at else ""))
    elif at:
        st.caption(f"The text was read with each script and combined with the image model. In testing, "
                   f"{at['accuracy_when_confident']:.0%} of confident decisions like this were right.")
    if s["predicted_script"] == "Devanagari":
        st.caption("Devanagari is used by both Hindi and Marathi; OCR tries both.")

    section("Text", "OCR")
    if ocr["status"] != "ok":
        st.error(f"No usable text found. {ocr.get('error') or ''}")
        if ocr.get("raw_text"):
            st.code(ocr["raw_text"], language=None)
    else:
        used = LANG_NAMES.get(ocr["ocr_language"], ocr["ocr_language"])
        c1, c2, c3 = st.columns(3)
        c1.metric("OCR confidence", f"{ocr['ocr_confidence']:.0f} / 100")
        c2.metric("Read as", used)
        c3.metric("Engine", "EasyOCR")
        st.markdown("**Recognised text (cleaned)**")
        st.code(ocr["clean_text"], language=None, wrap_lines=True)
        with st.expander("Raw OCR output and attempts"):
            st.code(ocr["raw_text"] or "(empty)", language=None)
            st.json({k: ocr.get(k) for k in ("preprocessing_method", "warnings", "image_analysis")})

    section("Photo check")
    adv = retake_advice(scan["quality"], ocr, scan["station"])
    m = scan["quality"]
    if adv["retake"]:
        st.warning("Consider taking another photo:\n\n" + "\n".join(f"- {r}" for r in adv["reasons"]))
    else:
        st.success("Text was read and a station was matched.")
    st.caption(f"{m['width']} x {m['height']} px · brightness {m['brightness']:.0f}/255 · contrast "
               f"{m['contrast']:.0f} · sharpness {m['sharpness']:.0f}. For a better photo: fill the frame with "
               "the board, shoot straight on and avoid glare.")

    section("Station", f"matched against {len(get_matcher().master.stations):,} stations")
    chosen = choose_station(scan)

    section("Train number and platform", "read from the sign")
    nums = scan["numbers"]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Train number**")
        if nums["known_trains"]:
            for t in nums["known_trains"]:
                info = train_summary(get_timetable(), t)
                if info:
                    st.write(f"{info['train_no']} {info['train_name']}: {info['origin']} {info['departs']} to "
                             f"{info['destination']} {info['arrives']}, {info['stops']} stops, runs {info['runs_on']}")
        else:
            st.write("No known train number on the sign.")
        if nums["other_numbers"]:
            st.caption("Other numbers read: " + ", ".join(nums["other_numbers"])
                       + ". They are not trains in RailScan's data, so they are not treated as train numbers.")
    with c2:
        st.markdown("**Platform**")
        if scan["platforms"]:
            st.write("Platform printed on the sign: " + ", ".join(scan["platforms"]))
            st.caption("This is the text on the sign. RailScan cannot tell which train will use the platform.")
        else:
            st.write("No platform number on the sign.")

    section("Railway information", "scheduled")
    if not chosen:
        st.info("Confirm a station above to see its information.")
        return
    from src.database.repository import get_station_by_code
    try:
        details = get_station_by_code(chosen["station_code"]) if full_database() else None
    except Exception:
        details = None
    name = (details or {}).get("station_name") or chosen["station_name"]
    st.markdown(f"**{name}** · station code **{chosen['station_code']}**"
                + (f" · {details['latitude']:.4f}, {details['longitude']:.4f}"
                   if details and details.get("latitude") is not None else ""))
    services = services_at(chosen["station_code"])
    if services.empty:
        st.info(f"No train in RailScan's timetable calls at {name}. RailScan covers trains that stop in "
                "northern India or West Bengal, so services elsewhere are not shown.")
    else:
        st.markdown("Scheduled calls of the trains in RailScan's timetable:")
        st.dataframe(services.drop(columns=["train_no"]), hide_index=True, width="stretch")
        first = nums["known_trains"][0] if nums["known_trains"] else services.train_no.iat[0]
        st.button(f"Plan a journey from {name}", on_click=go_to_lookup,
                  kwargs={"train_no": first, "station_code": chosen["station_code"]})
    st.caption(LIVE_DISCLAIMER)


def choose_station(scan: dict):
    """Ranked candidates (plus a search) the user confirms; preselected only when the match is confident."""
    match = scan["station"]
    opts = match_options(match)
    query = st.text_input("Not the right station? Search by name or code", key=f"station_search_{scan['id']}")
    if query.strip():
        for o in match_options(get_matcher().match(query)):
            if o["station_code"] not in {x["station_code"] for x in opts}:
                opts.append(o)
    if not opts:
        st.info("No station could be matched. Search by name above, or take another photo.")
        return None
    auto = not needs_confirmation(match) and match["station_code"] == opts[0]["station_code"]
    label = lambda i: (f"{opts[i]['station_name']} ({opts[i]['station_code']}) - match "
                       f"{opts[i]['match_score']:.0f}/100" if opts[i].get("match_score") is not None
                       else f"{opts[i]['station_name']} ({opts[i]['station_code']})")
    key = f"station_choice_{scan['id']}"
    pick = st.radio("Station", list(range(len(opts))), format_func=label, index=0 if auto else None, key=key,
                    on_change=lambda: record_station_choice(scan, opts, key))
    if pick is None:
        msg = match.get("message") if match else None
        st.warning((f"{msg} " if msg else "") + "Please choose the station.")
        return None
    o = opts[pick]
    if auto and pick == 0:
        st.success(f"Identified from the sign: {o['station_name']} ({o['station_code']}), "
                   f"match score {o['match_score']:.0f}/100.")
    else:
        st.info(f"Selected by you: {o['station_name']} ({o['station_code']}).")
    return o


def record_station_choice(scan: dict, opts: list, key: str) -> None:
    i = st.session_state.get(key)
    if i is not None:
        o = opts[i]
        record("scan", f"{scan.get('file', 'sign')}: station confirmed as {o['station_name']} ({o['station_code']})",
               {"file": scan.get("file"), "confirmed_station": o})


# ================================================================= PLANNER
def now_ist() -> pd.Timestamp:
    return pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)


def clock(ts, journey_date) -> str:
    """'00:22 (+1 day)' relative to the journey date; '-' when unknown."""
    if ts is None or pd.isna(ts):
        return "-"
    ts = pd.Timestamp(ts)
    days = (ts.normalize() - pd.Timestamp(journey_date).normalize()).days
    return ts.strftime("%H:%M") + (f" (+{days} day)" if days == 1 else f" (+{days} days)" if days > 1 else "")


def page_planner() -> None:
    st.markdown("Pick a train, a travel date (today, a future date or a past one) and the station the train is "
                "at or leaving. If you know how late it is, RailScan estimates the delay at the next stop and "
                "along the rest of the route. Otherwise it shows how late this train usually was.")
    with st.spinner("Loading the delay models ..."):
        predictor, predictor_error = get_predictor()
    if predictor_error:
        st.warning(f"Delay model unavailable: {predictor_error}")
    try:
        tt = get_timetable()
    except TimetableUnavailable as exc:
        st.error(str(exc))
        return

    names = tt.groupby("train_no").train_name.first()
    labels = {t: f"{t} - {n}" for t, n in names.items()}
    if st.session_state.get("plan_train") not in labels:
        st.session_state.pop("plan_train", None)
    c1, c2 = st.columns([2, 3])
    train_no = c1.selectbox("Train", list(labels), format_func=labels.get, key="plan_train")
    known = c2.radio("Do you know the train's current delay?", ["Yes, I will enter it", "No, show typical delays"],
                     horizontal=True, key="plan_known",
                     help="The model needs the current delay (for example from the station display or an "
                          "official app). Without it, RailScan can only show past patterns.")
    with_delay = known.startswith("Yes")

    stations = departure_stations(tt, train_no)
    prefer = st.session_state.get("prefer_station")
    idx = next((i for i, c in enumerate(stations) if c["code"] == prefer), 0)
    with st.form("plan_form"):
        f1, f2, f3 = st.columns([1, 2, 1])
        jd = f1.date_input("Journey date (day the train starts)", value=now_ist().date(), key="plan_date",
                           format="DD/MM/YYYY")
        pick = f2.selectbox("Train is at / leaving", stations, index=idx, format_func=lambda c: c["label"],
                            key=f"plan_station_{train_no}")
        delay = (f3.number_input("Current delay (min, negative = early)", -60, 600, 0, step=5, key="plan_delay")
                 if with_delay else None)
        submitted = st.form_submit_button("Show estimate" if with_delay else "Show typical delays", type="primary")

    r0 = route(tt, train_no).iloc[0]
    st.caption(f"Timetable from {r0.journeys_in_window} recorded journeys ({r0.timetable_window}); it is assumed "
               f"to still apply. Runs on: {r0.runs_on}.")

    query = {"train_no": train_no, "train_name": labels[train_no], "journey_date": str(jd),
             "station_code": pick["code"], "station_name": pick["name"],
             "current_delay_min": None if delay is None else float(delay)}
    if submitted:
        st.session_state.plan_query = query
    shown = st.session_state.get("plan_query")
    if not shown or shown["train_no"] != train_no or (shown["current_delay_min"] is None) == with_delay:
        st.info("Choose the date and station, then press the button to see the result.")
    else:
        sjd = pd.Timestamp(shown["journey_date"]).date()
        if runs_on(tt, train_no, sjd) is False:
            st.warning(f"Train {train_no} has not run on a {pd.Timestamp(sjd):%A} in recent records.")
        if with_delay:
            if predictor is None:
                st.info("No estimate without the delay model.")
            else:
                f = forecast(train_data(train_no), tt, predictor, train_no, sjd, shown["station_code"], shown["current_delay_min"],
                             now=now_ist(), covered=get_coverage())
                show_forecast(f, sjd)
                if f["status"] == "ok":
                    show_position(estimate_from_forecast(schedule_on(tt, train_no, sjd), f, now_ist(),
                                                         station_coords()),
                                  "the timetable and the delays estimated above")
                if submitted and f["status"] == "ok":
                    nx = f["next"]
                    record("forecast",
                           f"{train_no} on {sjd:%d %b %Y}: leaving {shown['station_code']} "
                           f"{shown['current_delay_min']:+.0f} min, next stop {nx['station_code']} "
                           f"estimated {nx['predicted_delay_min']:+.0f} min",
                           {**shown, "next_stop": nx["station_code"],
                            "next_stop_estimated_delay_min": round(nx["predicted_delay_min"], 1),
                            "next_stop_expected_arrival": str(nx["expected_arrival"]),
                            "next_stop_likely_range_min": [nx["likely_delay_low_min"], nx["likely_delay_high_min"]],
                            "risk": nx["risk"], "probability_delayed_15": nx["probability_delayed_15"],
                            "route": [{"station": r.station_code, "estimated_delay_min": round(r.predicted_delay_min, 1),
                                       "expected_arrival": str(r.expected_arrival), "basis": r.basis}
                                      for r in f["route"].itertuples()]})
        else:
            t = typical_delays(train_data(train_no), tt, train_no, sjd)
            show_typical(t, sjd)
            sched = schedule_on(tt, train_no, sjd)
            start = sched[sched.station_code == shown["station_code"]]
            if not start.empty and pd.notna(start.sched_dep.iat[0]):
                med = t.loc[t.station_code == shown["station_code"], "median_delay_min"]
                show_position(estimate_from_typical(sched, t, now_ist(), shown["station_code"],
                                                    start.sched_dep.iat[0],
                                                    float(med.iat[0]) if len(med) and pd.notna(med.iat[0]) else 0.0,
                                                    station_coords()),
                              "the timetable and this train's typical (median) recorded delays")
            if submitted:
                record("typical", f"{train_no} on {sjd:%d %b %Y}: typical recorded delays",
                       {**shown, "stops": [{"station": r.station_code, "median_delay_min": r.median_delay_min,
                                            "recorded_arrivals": r.recorded_arrivals} for r in t.itertuples()]})

    with st.expander("Check the model against a recorded journey"):
        page_replay()


# ================================================================= HISTORY
def to_ist(iso: str) -> str:
    ts = pd.Timestamp(iso)
    return ts.tz_convert("Asia/Kolkata").strftime("%d %b %Y, %H:%M") if ts.tzinfo else ts.strftime("%d %b %Y, %H:%M")


def details_table(details: dict) -> pd.DataFrame:
    """Readable field/value rows for a history entry (lists of stops are summarised)."""
    rows = []
    for key, value in details.items():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            value = f"{len(value)} stops (see JSON below)"
        elif isinstance(value, dict):
            value = ", ".join(f"{k}: {v}" for k, v in value.items() if v not in (None, "", []))
        elif isinstance(value, float):
            value = f"{value:g}"
        rows.append({"Field": key.replace("_", " "), "Value": "-" if value in (None, "", []) else str(value)})
    return pd.DataFrame(rows)


def page_history() -> None:
    user = current_user()
    st.markdown("Results RailScan showed you while signed in. Only you can see them.")
    kinds = {"All": None, "Sign scans": "scan", "Delay estimates": "forecast", "Typical delays": "typical"}
    choice = st.selectbox("Show", list(kinds), key="history_filter")
    try:
        items = accounts.list_history(user["id"], kind=kinds[choice])
    except Exception:
        LOG.error("history read failed", exc_info=True,
                  extra={"event": "history_read_failed", "user_id": user["id"], "session": _session_id()})
        st.error("Your history could not be loaded right now. Please reload the page in a moment.")
        return
    if not items:
        st.info("Nothing here yet. Scan a sign or get a delay estimate and it will appear here.")
    else:
        labels = {"scan": "Sign scan", "forecast": "Delay estimate", "typical": "Typical delays", "replay": "Check"}
        st.dataframe(pd.DataFrame({"When (IST)": [to_ist(h["created_at"]) for h in items],
                                   "Type": [labels.get(h["kind"], h["kind"]) for h in items],
                                   "Summary": [h["summary"] for h in items]}),
                     hide_index=True, width="stretch")
        options = {h["id"]: f"{to_ist(h['created_at'])} - {h['summary'][:70]}" for h in items}
        pick = st.selectbox("Details of", list(options), format_func=options.get, key="history_pick")
        entry = next(h for h in items if h["id"] == pick)
        st.dataframe(details_table(entry["details"]), hide_index=True, width="stretch")
        with st.expander("All saved data (JSON)"):
            st.json(entry["details"])
        c1, c2 = st.columns([1, 3])
        if c1.button("Delete this entry", key="history_delete_one"):
            accounts.delete_history(user["id"], pick)
            st.rerun()
        with c2.popover("Clear all history"):
            st.markdown(f"This permanently deletes all **{len(items)}** shown entries and any others in your history.")
            if st.button("Yes, delete all my history", type="primary", key="history_delete_all"):
                n = accounts.delete_history(user["id"])
                st.session_state.history_msg = f"Deleted {n} entries."
                st.rerun()
    if st.session_state.get("history_msg"):
        st.success(st.session_state.pop("history_msg"))

    section("Account")
    st.caption(f"Username: {user['username']} · created {to_ist(user['created_at'])} · "
               f"stored in {accounts_store.describe()}")
    try:
        export = json.dumps(accounts.export_user_data(user["id"]), ensure_ascii=False, indent=1, default=str)
        st.download_button("Download my data (JSON)", export, file_name=f"railscan_{user['username']}.json",
                           mime="application/json",
                           help="Your account details and your whole history, as RailScan stores them.")
    except Exception:
        LOG.error("data export failed", exc_info=True,
                  extra={"event": "export_failed", "user_id": user["id"], "session": _session_id()})
        st.caption("Your data could not be prepared for download right now.")
    with st.expander("What RailScan stores"):
        st.markdown(PRIVACY)
    with st.expander("Change password"):
        with st.form("pw_form", clear_on_submit=True):
            cur = st.text_input("Current password", type="password", autocomplete="current-password")
            new = st.text_input("New password", type="password", autocomplete="new-password")
            again = st.text_input("Confirm new password", type="password", autocomplete="new-password")
            if st.form_submit_button("Change password"):
                if new != again:
                    st.error("The new passwords do not match.")
                else:
                    try:
                        accounts.change_password(user["id"], cur, new)
                        st.success("Password changed.")
                    except AccountError as exc:
                        st.error(str(exc))
    with st.expander("Delete account"):
        st.warning("Deleting your account permanently removes it and all of its history. This cannot be undone.")
        with st.form("delete_form"):
            pw = st.text_input("Password", type="password", autocomplete="current-password")
            sure = st.checkbox("I understand that this cannot be undone")
            if st.form_submit_button("Delete my account"):
                if not sure:
                    st.error("Tick the box to confirm.")
                else:
                    try:
                        accounts.delete_user(user["id"], pw)
                        log_out()
                        st.rerun()
                    except AccountError as exc:
                        st.error(str(exc))


def show_forecast(f: dict, jd) -> None:
    if f["status"] != "ok":
        st.error(f["message"])
        return
    for w in f["warnings"]:
        st.warning(w)
    nx = f["next"]
    section(f"Leaving {f['station_name'] or f['station_code']} ({f['station_code']})", "your input")
    c = st.columns(3)
    c[0].metric("Scheduled departure", clock(f["scheduled_departure"], jd))
    c[1].metric("Current delay", fmt_min(f["current_delay_min"]))
    c[2].metric("Expected departure", clock(f["reference_time"], jd))

    section(f"Next stop: {nx['station_name'] or nx['station_code']} ({nx['station_code']})", "model estimate")
    c = st.columns(4)
    c[0].metric("Scheduled arrival", clock(nx["scheduled_arrival"], jd))
    c[1].metric("Estimated delay", fmt_min(nx["predicted_delay_min"]))
    c[2].metric("Expected arrival", clock(nx["expected_arrival"], jd))
    c[3].metric("Travel time after departure", describe_remaining(nx["minutes_from_departure"]))
    c = st.columns(4)
    if nx["likely_delay_low_min"] is not None:
        c[0].metric("Likely delay range",
                    f"{nx['likely_delay_low_min']:+.0f} to {nx['likely_delay_high_min']:+.0f} min")
    c[1].metric("Delay risk", nx["risk"])
    if nx["probability_delayed_15"] is not None:
        c[2].metric("P(delay of 15 min or more)", f"{nx['probability_delayed_15']:.0%}")
    c[3].metric("Scheduled halt there", fmt_min(nx["scheduled_halt_min"], False))
    if nx["method"] == "no_coverage":
        st.info("The model was not trained on this stretch of the route, so the current delay is carried forward.")
    models = f["models"]
    st.caption(f"Basis: {models.get('regressor', 'regressor')} and {models.get('classifier', 'classifier')} trained "
               "on recorded journeys; inputs are your delay, the timetable on this date, the day of week and month, "
               "and this train's recorded history. Risk: " + RISK_RULE + ".")

    section("Rest of the route", "estimate")
    rd = f["route"]
    show = pd.DataFrame({
        "Station": [f"{n} ({c_})" if isinstance(n, str) and n.strip() else c_ for n, c_ in zip(rd.station_name, rd.station_code)],
        "Scheduled arrival": [clock(t, jd) for t in rd.scheduled_arrival],
        "Estimated delay (min)": rd.predicted_delay_min.round(0),
        "Likely range (min)": [f"{lo:+.0f} to {hi:+.0f}" if lo is not None and not pd.isna(lo) else "-"
                               for lo, hi in zip(rd.likely_delay_low_min, rd.likely_delay_high_min)],
        "Basis": rd.basis,
        "Expected arrival": [clock(t, jd) for t in rd.expected_arrival],
        "After departure": [describe_remaining(m) for m in rd.minutes_from_departure],
        "Halt (min)": [("-" if h is None or pd.isna(h) else f"{h:.0f}") for h in rd.scheduled_halt_min],
    })
    near_now = rd.minutes_from_now.dropna()
    if len(near_now) and (near_now.abs() < 24 * 60).any():
        show.insert(7, "From now", [describe_remaining(m) for m in rd.minutes_from_now])
    st.dataframe(show, hide_index=True, width="stretch",
                 column_config={"Estimated delay (min)": st.column_config.NumberColumn(format="%+.0f")})
    ev = read_json(str(REPORTS / "planner_evaluation.json"))
    if ev:
        b1 = next((b for b in ev["by_stops_ahead"] if b["stops_ahead"] == "1"), None)
        cov = [b["likely_range_test_coverage_pct"] for b in ev["by_stops_ahead"]]
        first = (f"Tested on recorded journeys after the training period: for the next stop the model was off by "
                 f"{b1['test_mae_selected']:.1f} min on average (carrying the delay forward: "
                 f"{b1['test_mae_carry_baseline']:.1f} min). ") if b1 else ""
        later = ev.get("later_stops_method", "carry")
        far = [b for b in ev["by_stops_ahead"] if b["stops_ahead"] != "1"]
        if later == "carry":
            rest = "For later stops no tested method beat carrying the current delay forward, so that is what is shown. "
        else:
            errs = ", ".join(f"{b['stops_ahead']} stops ahead {b['test_mae_selected']:.0f} min "
                             f"(carried forward {b['test_mae_carry_baseline']:.0f})" for b in far)
            rest = (f"For later stops the method chosen on validation is '{METHOD_LABELS.get(later, later)}'; "
                    f"average error on test journeys: {errs}. Where this train has no recorded history for a "
                    "stretch, the current delay is carried forward (see Basis). ")
        st.caption(first + rest + "The likely range comes from past errors at the same distance and contained the "
                   f"actual delay in {min(cov):.0f}-{max(cov):.0f}% of test cases. Estimates, not live running status.")

    sh = historical_stop_stats(train_data(f["train_no"]), f["station_code"], f["train_no"], before=f["reference_time"])
    if sh:
        st.caption(f"Recorded stops of this train at {f['station_code']}: {sh['count']}; average actual stop "
                   f"{sh['avg_actual_stop_min']:.1f} min; within ±{sh['tolerance_min']} min of the scheduled halt "
                   f"{sh['stop_reliability_pct']:.0f}% of the time.")


@st.cache_resource(show_spinner=False)
def station_coords() -> dict:
    """code -> (lat, lon) from the station list, used where the timetable has no coordinates."""
    return {c: (s.latitude, s.longitude) for c, s in get_matcher().master.stations.items()
            if s.latitude is not None and s.longitude is not None}


def show_position(est, basis: str) -> None:
    section("Where the train should be now", "estimate, not live")
    if est is None:
        st.info("Not enough timetable information to place this train.")
        return
    pos = est["position"]
    st.markdown(f"**{now_ist():%d %b %Y, %H:%M}** (now, India time) · {pos['text']}")
    pts = map_points(est["timeline"], pos)
    if len(pts):
        st.map(pts, latitude="lat", longitude="lon", color="color", size="size", height=360)
    st.caption(f"Estimated from {basis}, assuming the train keeps to them. RailScan has **no live location "
               "data**: for the actual position check NTES (enquiry.indianrail.gov.in) or station displays. "
               "Between stations the point is placed on a straight line, so it is approximate. "
               "Grey = stations on the route, red = estimated train position.")


def show_typical(t: pd.DataFrame, jd) -> None:
    section("Typical delays for this train", "recorded history")
    st.markdown("Without the current delay RailScan cannot make a prediction for your date. This is how late the "
                "train actually was at each stop in the recorded journeys.")

    def fmt(v, f_):
        return "-" if v is None or pd.isna(v) else f_(v)

    show = pd.DataFrame({
        "Station": [f"{n} ({c_})" if isinstance(n, str) and n.strip() else c_ for n, c_ in zip(t.station_name, t.station_code)],
        "Scheduled arrival": [clock(x, jd) for x in t.scheduled_arrival],
        "Scheduled departure": [clock(x, jd) for x in t.scheduled_departure],
        "Median delay (min)": [fmt(v, lambda x: f"{x:+.0f}") for v in t.median_delay_min],
        f"Median on {pd.Timestamp(jd):%A}s": [fmt(v, lambda x: f"{x:+.0f}") for v in t.median_delay_same_weekday_min],
        "Late 15+ min": [fmt(v, lambda x: f"{x:.0f}%") for v in t.share_delayed_15_pct],
        "Recorded arrivals": t.recorded_arrivals,
    })
    st.dataframe(show, hide_index=True, width="stretch")
    st.caption("From the recorded journeys in RailScan's dataset (Data and models page). '-' means no recorded "
               "arrival at that stop; weekday medians need at least 5 journeys. Past patterns, not a forecast.")


# ================================================================= RECORDED-JOURNEY CHECK
def page_replay() -> None:
    st.markdown("Pick a journey that was actually recorded and a stop on it. RailScan predicts the next stop as "
                "if it were that moment, and shows what really happened.")
    trains = get_trains()
    predictor, _ = get_predictor()
    labels = {r.train_no: f"{r.train_no} - {r.train_name} ({r.journeys} recorded journeys)"
              for r in trains.itertuples()}
    c1, c2, c3 = st.columns([2, 1, 2])
    train_no = c1.selectbox("Train", list(labels), format_func=labels.get, key="replay_train")
    data = train_data(train_no)
    dates = journey_dates(data, train_no)
    jd = c2.selectbox("Recorded journey date", dates, format_func=lambda d: d.strftime("%d %b %Y"),
                      key="replay_date")
    j = journey(data, train_no, jd)
    choices = station_choices(j)
    pick = c3.selectbox("Observed at station", choices, format_func=lambda c: c["label"], key="replay_station")

    mode = st.radio("Current delay", ["Recorded delay", "Enter a delay (what-if)"], horizontal=True,
                    key="delay_mode",
                    help="What-if keeps the timetable and history of the selected stop and replaces only "
                         "the current delay.")
    override = None
    if mode != "Recorded delay":
        override = st.number_input("Current delay in minutes (negative = early)", -60, 600, 20, step=5,
                                   key="replay_delay")

    res = analyze(data, predictor, train_no, jd, pick["code"], seq=pick["seq"], delay_override=override)
    if res["status"] != "ok":
        st.error(res["message"])
        return
    show_analysis(res, j)


def show_analysis(res: dict, j: pd.DataFrame) -> None:
    for w in res["warnings"]:
        st.warning(w)
    sch, rec, nxt = res["scheduled"], res["recorded"], res["next_station"]
    st.markdown(f"**{res['train_no']} {res['train_name']}** · {res['train_type']} · "
                f"journey of {res['journey_date']:%d %b %Y}")

    section(f"At {res['station']['name'] or res['station']['code']} ({res['station']['code']})",
            "scenario" if res["scenario"] else "recorded")
    c = st.columns(4)
    c[0].metric("Scheduled arrival", as_time(sch["arrival"])[-5:])
    c[0].caption(as_time(sch["arrival"])[:-7])
    c[1].metric("Scheduled departure", as_time(sch["departure"])[-5:])
    c[1].caption(as_time(sch["departure"])[:-7])
    c[2].metric("Scheduled halt", fmt_min(sch["halt_min"], False))
    c[2].caption(f"{sch['stop_category']} stop" if sch["stop_category"] else "")
    if res["scenario"]:
        c[3].metric("Current delay (entered)", fmt_min(res["current_delay_used"]))
    else:
        c[3].metric("Recorded delay", fmt_min(rec["current_delay_min"]))
        c[3].caption(f"on arrival: {fmt_min(rec['arrival_delay_min'])}")
        st.caption(f"Recorded actual stop: {fmt_min(rec['actual_stop_min'], False)} "
                   f"(arrived {as_time(rec['actual_arrival'])[-5:]}, departed {as_time(rec['actual_departure'])[-5:]}; "
                   "derived as scheduled time + recorded delay). Halt category uses the scheduled halt: "
                   "Short < 5 min, Moderate 5-10 min, Long > 10 min.")

    pred = res["prediction"]
    section(f"Next station: {nxt['name'] or nxt['code']} ({nxt['code']})", "model estimate")
    if pred is None:
        st.info("No prediction available.")
    else:
        c = st.columns(4)
        c[0].metric("Scheduled arrival", as_time(nxt["scheduled_arrival"])[-5:])
        c[1].metric("Predicted delay", fmt_min(pred["predicted_next_station_delay_min"]))
        c[2].metric("Expected arrival", as_time(pred["expected_arrival"])[-5:])
        c[3].metric("Time remaining", pred["remaining_text"])
        c = st.columns(4)
        c[0].metric("Delay risk", pred["risk_label"])
        c[1].metric("P(delay of 15 min or more)", f"{pred['probability_delayed_15']:.0%}")
        c[2].metric("Distance", f"{nxt['distance_km']:.1f} km" if nxt["distance_km"] else "-")
        c[3].metric("Scheduled run", fmt_min(nxt["scheduled_travel_min"], False))
        models = pred["models"]
        st.caption(f"Basis: {models.get('regressor', 'regressor')} for the delay, "
                   f"{models.get('classifier', 'classifier')} for the probability; inputs are the current delay, "
                   f"timetable and this train's earlier journeys only. Time remaining is measured from "
                   f"{as_time(res['reference_time'])} ({'entered delay applied to scheduled departure' if res['scenario'] else 'when the observation was recorded'}). "
                   f"Risk: {RISK_RULE}.")

        rows = [{"Source": "Model prediction", "Next-station delay": fmt_min(pred["predicted_next_station_delay_min"]),
                 "Arrival": as_time(pred["expected_arrival"])},
                {"Source": "Baseline: current delay carried forward",
                 "Next-station delay": fmt_min(res["baseline"]["next_arrival_delay_min"]),
                 "Arrival": as_time(res["baseline"]["expected_arrival"])}]
        if not res["scenario"]:
            ro = res["recorded_outcome"]
            rows.append({"Source": "Recorded outcome (what actually happened)",
                         "Next-station delay": fmt_min(ro["next_arrival_delay_min"]),
                         "Arrival": as_time(ro["next_actual_arrival"])})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    section("Upcoming stations", "model estimate")
    up = res["upcoming"]
    if up.empty:
        st.info("No upcoming stations with recorded timetable data on this journey.")
    else:
        show = up.copy()
        for col in ("Scheduled arrival", "Expected arrival"):
            show[col] = show[col].map(lambda t: as_time(t)[-5:])
        # values the data does not contain are shown as "not recorded", never as "None"
        show["Scheduled halt (min)"] = show["Scheduled halt (min)"].map(
            lambda v: "not recorded" if v is None or pd.isna(v) else f"{v:.0f}")
        show["Stop reliability (%)"] = show["Stop reliability (%)"].map(
            lambda v: "no history" if v is None or pd.isna(v) else f"{v:.0f}")
        st.dataframe(show, hide_index=True, width="stretch",
                     column_config={"Predicted delay (min)": st.column_config.NumberColumn(format="%+.0f")})
        st.caption("The first row is the model's next-station prediction. Later rows feed each estimate back "
                   "in as the current delay (iterative estimate), so uncertainty grows along the route. The list "
                   "stops where the dataset has no record of the following stop.")

    section("Stop history at this station", "recorded")
    sh = res["stop_history"]
    h = res["history"]
    if sh is None:
        st.info("No earlier observed stops of this train at this station in the dataset, so no stop "
                "statistics are shown.")
    else:
        c = st.columns(5)
        c[0].metric("Earlier stops", sh["count"])
        c[1].metric("Average actual stop", f"{sh['avg_actual_stop_min']:.1f} min")
        c[2].metric("Median actual stop", f"{sh['median_actual_stop_min']:.1f} min")
        c[3].metric("Variability (std)", f"{sh['std_actual_stop_min']:.1f} min" if sh["std_actual_stop_min"] else "-")
        c[4].metric("Stop reliability", f"{sh['stop_reliability_pct']:.0f}%")
        st.caption(f"Recorded stops from {sh['first_date']} to {sh['last_date']} before this observation. "
                   f"Stop reliability = share of stops within ±{sh['tolerance_min']} min of the scheduled halt. "
                   "These describe past patterns, not a guarantee.")
    if h["observation_count"]:
        st.caption(f"Arrival at {nxt['code']} on {h['observation_count']} earlier journeys: average delay "
                   f"{h['avg_delay_min']:.0f} min, on time (under 15 min late) {h['on_time_rate_pct']:.0f}% of the time.")

    with st.expander("All recorded stops on this journey"):
        cols = {"seq": "Seq", "station_name": "Station", "station_code": "Code",
                "current_scheduled_arrival": "Sch. arrival", "current_scheduled_departure": "Sch. departure",
                "scheduled_stop_min": "Sch. halt (min)", "current_arrival_delay_min": "Arrival delay",
                "current_departure_delay_min": "Departure delay", "actual_stop_min": "Actual stop (min)"}
        t = j[list(cols)].rename(columns=cols)
        for c_ in ("Sch. arrival", "Sch. departure"):
            t[c_] = t[c_].dt.strftime("%H:%M")
        st.dataframe(t, hide_index=True, width="stretch")
        st.caption("Gaps in the sequence are stops the source did not report or that failed validation.")


# ================================================================= DATA
def page_data() -> None:
    info = get_overview()
    ov = info["overview"]
    section("Railway dataset", "recorded")
    src = info["source"]
    from src.database.connection import resolve_db_path
    if src == "SQLite":
        st.caption(f"Read from the SQLite database ({resolve_db_path()}).")
    else:
        st.caption("Figures exported from the full database (data/processed/dataset_overview.json); each train's "
                   "rows are read from the committed Parquet file when needed.")
        if info.get("stale"):
            st.warning("The dataset file changed after these figures were exported. Run "
                       "`python scripts/setup_database.py all` and `python scripts/setup_database.py export-overview`.")
    c = st.columns(4)
    c[0].metric("Observations", f"{ov['rows']:,}")
    c[1].metric("Journeys", f"{ov['journeys']:,}")
    c[2].metric("Trains", f"{ov['trains']:,}")
    c[3].metric("Stations", f"{ov['stations']:,}")
    st.markdown(
        "Source: [yadavsaurabh/indian-railways-runs](https://huggingface.co/datasets/yadavsaurabh/indian-railways-runs) "
        "(GPL-3.0), stop-level recorded arrivals and departures, "
        f"**{ov['date_min']:%d %b %Y} to {ov['date_max']:%d %b %Y}**. One row is one observed stop of a train "
        "on one journey, paired with what happened at the next stop. It holds every recorded journey in this "
        "period of the trains that stop in northern India or West Bengal, at stations in those states; see "
        "`data/processed/dataset_sources.md`.")
    st.caption("Coverage: northern India (J&K, Ladakh, Himachal Pradesh, Punjab, Chandigarh, Haryana, Delhi, "
               "Rajasthan, Uttar Pradesh, Uttarakhand) and West Bengal only, and only the recent weeks above. Seasonal effects such as "
               "winter fog are not in the data, so estimates may be too optimistic in those periods.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Next-station arrival delay (minutes)**")
        st.altair_chart(delay_histogram(info["delays"]), width="stretch")
        st.caption("10-minute bins (negative = early). The first bar holds every delay under -20 min, "
                   "the last every delay of 150 min or more.")
    with c2:
        st.markdown("**Delay category at the next station**")
        cat = (pd.Series(info["risk"]).reindex(["Low", "Medium", "High"], fill_value=0)
               .rename_axis("category").reset_index(name="observations"))
        st.altair_chart(bar_chart(cat, "category", "observations", ["Low", "Medium", "High"]), width="stretch")
        st.caption("Low < 15 min, Medium 15-59 min, High 60 min or more.")
    with st.expander("Missing values"):
        ms = info["missing"]
        st.dataframe(ms, hide_index=True, width="stretch") if not ms.empty else st.write("None")

    section("Delay models", "measured on held-out data")
    meta = read_json(str(ROOT / "models" / "delay" / "metadata.json"))
    reg_csv, clf_csv = REPORTS / "results" / "regression_results.csv", REPORTS / "results" / "classification_results.csv"
    if not meta or not reg_csv.exists() or not clf_csv.exists():
        st.info("Delay models not trained yet.")
    else:
        rr, cr = pd.read_csv(reg_csv), pd.read_csv(clf_csv)
        rr, cr = rr[rr.split == "test"], cr[(cr.split == "test") & (cr.task == "binary_delayed_15")]
        reg = rr[rr.role.str.startswith("selected")].iloc[0]
        base_r = rr[rr.role == "baseline"].iloc[0]
        clf = cr[cr.role.str.startswith("selected")].iloc[0]
        base_c = cr[cr.role == "baseline"].iloc[0]
        st.markdown(f"Chronological split: trained on journeys up to {meta['dataset_date_ranges']['train']['max']}, "
                    f"tested on {meta['dataset_date_ranges']['test']['min']} to "
                    f"{meta['dataset_date_ranges']['test']['max']} ({meta['test_rows']} rows).")
        st.dataframe(pd.DataFrame([
            {"Task": "Next-station delay (min)", "Model": meta["model_type"]["regressor"],
             "MAE": reg["MAE"], "RMSE": reg["RMSE"], "Baseline MAE": base_r["MAE"], "Baseline RMSE": base_r["RMSE"]},
        ]), hide_index=True, width="stretch", column_config={
            c_: st.column_config.NumberColumn(format="%.2f") for c_ in ("MAE", "RMSE", "Baseline MAE", "Baseline RMSE")})
        st.dataframe(pd.DataFrame([
            {"Task": "Delayed 15 min or more", "Model": meta["model_type"]["classifier"],
             "Precision": clf["precision"], "Recall": clf["recall"], "F1": clf["f1"], "ROC-AUC": clf["roc_auc"],
             "Baseline F1": base_c["f1"]},
        ]), hide_index=True, width="stretch", column_config={
            c_: st.column_config.NumberColumn(format="%.3f") for c_ in ("Precision", "Recall", "F1", "ROC-AUC", "Baseline F1")})
        st.caption("Baseline = carry the current delay forward unchanged. Delays are strongly persistent between "
                   "neighbouring stations, which is why both the baseline and the model score highly.")
        figs = [p for p in (REPORTS / "results" / "confusion_matrix.png", REPORTS / "results" / "feature_importance.png")
                if p.exists()]
        if figs:
            with st.expander("Confusion matrix and feature importance"):
                for p in figs:
                    st.image(str(p))

    section("Script classifier (CNN)", "measured on held-out images")
    cnn = read_json(str(REPORTS / "cnn_evaluation.json"))
    if not cnn:
        st.info("CNN not trained yet. Run `python scripts/train_cnn.py`.")
    else:
        te = cnn["metrics"]["test"]
        arch = {"mobilenetv2": "MobileNetV2 (transfer learning)", "small": "Small CNN"}.get(
            cnn.get("architecture", ""), "CNN")
        st.markdown(f"{arch}, {cnn['input_shape'][0]} px input, trained on "
                    f"{cnn['data']['train']} signboard images from INDICSTR12 "
                    "([jai23023/indicstr12-signboards](https://huggingface.co/datasets/jai23023/indicstr12-signboards), "
                    f"CC BY 4.0); tested on {te['n']} held-out images. Test accuracy: **{te['accuracy']:.1%}** "
                    f"(chance level {1 / len(cnn['classes']):.0%}).")
        st.dataframe(pd.DataFrame([{"Script": k, "Precision": v["precision"], "Recall": v["recall"],
                                    "F1": v["f1-score"], "Images": int(v["support"])}
                                   for k, v in te["per_class"].items()]),
                     hide_index=True, width="stretch",
                     column_config={c_: st.column_config.NumberColumn(format="%.3f") for c_ in ("Precision", "Recall", "F1")})
        fus = read_json(str(REPORTS / "script_fusion_evaluation.json"))
        if fus:
            at = fus["test_at_threshold"]
            st.markdown(f"**Used in the app: image model + OCR evidence.** The sign is read with each script and the "
                        f"readings are combined with the image model. Test accuracy "
                        f"**{fus['cnn_plus_ocr_evidence']['test']:.1%}** (image model alone "
                        f"{fus['cnn_only']['test']:.1%}). {at['share_confident']:.0%} of test images were confident, "
                        f"and {at['accuracy_when_confident']:.0%} of those were right. Settings were chosen on "
                        f"validation images only.")
        four = read_json(str(REPORTS / "cnn_evaluation_4class.json"))
        if four:
            st.caption(f"Experiment: the same network trained on four languages (Hindi, Kannada, Marathi, Telugu) "
                       f"reached {four['metrics']['test']['accuracy']:.1%} test accuracy. Hindi and Marathi share "
                       "the Devanagari script, so the app classifies scripts and lets OCR try both languages.")
        figs = [p for p in (REPORTS / "figures" / "cnn_confusion_matrix.png", REPORTS / "figures" / "cnn_training_curves.png")
                if p.exists()]
        if figs:
            with st.expander("Confusion matrix and training curves"):
                for p in figs:
                    st.image(str(p))
        st.caption("The images are general Indian signboards (shops, streets), not only railway boards.")

    section("OCR", "measured on held-out images")
    ocr = read_json(str(REPORTS / "ocr_evaluation.json"))
    if not ocr:
        st.info("OCR not evaluated yet. Run `python scripts/evaluate_ocr.py`.")
    else:
        o = ocr["overall"]
        st.markdown(f"EasyOCR on {ocr['images']} test images with the true language given. "
                    f"No usable text on **{o['ocr_failed']}** images; on average **{o['word_recall']:.0%}** of the "
                    "ground-truth words were found.")
        st.dataframe(pd.DataFrame(ocr["by_language"]).rename(columns={
            "language": "Language", "images": "Images", "mean_cer": "Mean CER", "median_cer": "Median CER",
            "word_recall": "Word recall", "ocr_failed": "No text"}), hide_index=True, width="stretch",
            column_config={c_: st.column_config.NumberColumn(format="%.2f")
                           for c_ in ("Mean CER", "Median CER", "Word recall")})
        st.caption("CER = character edit distance / reference length (lower is better; above 1 when the OCR also "
                   "reads English or numbers missing from the reference). Scene photos are hard; clear station "
                   "boards read much better, but that has not been measured on a labelled railway set.")


{"Scan a sign": page_scan, "Journey planner": page_planner, "Data and models": page_data,
 "My history": page_history}[st.session_state.page]()
