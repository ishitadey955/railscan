"""Journey planner: delay estimates for ANY date (past, today or future).

Two modes, kept clearly apart:

forecast (the user knows the train's current delay)
    Builds the model input from the timetable placed on the chosen date, the
    date's calendar features, the delay the user entered, and history computed
    from recorded journeys before the prediction moment (the same past-only
    definitions as scripts/build_dataset.py). The model predicts the next stop;
    later stops are an iterative estimate (each prediction fed back in).

typical (no current delay known, e.g. planning a trip next month)
    No prediction is possible without a current delay, so this reports how late
    the train actually was at each stop in the recorded data. That is history,
    not a forecast for the chosen date.

Assumes the timetable in data/processed/RailScan_Timetable.csv still applies.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .delay_prediction import DelayPredictor, risk_label
from .next_station import expected_arrival, minutes_until
from .stop_analysis import STOP_TOLERANCE_MIN, stop_category
from .timetable import runs_on, schedule_on

DELAY_THRESHOLD = 15
ROOT = Path(__file__).resolve().parents[1]
TRAIN_SPLIT = ROOT / "data" / "processed" / "train.csv"
COVERAGE = ROOT / "models" / "delay" / "coverage.csv"     # written by scripts/train_delay_models.py


def model_coverage(path: Optional[Path] = None) -> set:
    """(train, station, next station) segments that occur in the model's TRAINING rows.

    Outside these segments the model has never seen the route, so the planner
    carries the delay forward instead of extrapolating. Read from the small
    coverage file saved with the models, else from the training split.
    """
    path = Path(path) if path else (COVERAGE if COVERAGE.exists() else TRAIN_SPLIT)
    if not path.exists():
        return set()
    tr = pd.read_csv(path, dtype=str, usecols=["train_no", "station_code", "next_station_code"])
    return {(_key(t), a, b) for t, a, b in zip(tr.train_no, tr.station_code, tr.next_station_code)}


def _key(train_no) -> str:
    return str(train_no).lstrip("0")


def history_features(df: pd.DataFrame, train_no: str, station_code: str, next_code: Optional[str],
                     cutoff: pd.Timestamp) -> Dict[str, Any]:
    """Past-only history, same definitions as the training data.

    Delay history: this train's arrivals at ``next_code`` known before ``cutoff``.
    Stop history: this train's completed stops at ``station_code`` before ``cutoff``.
    """
    t = df[df.train_no.astype(str).str.lstrip("0") == _key(train_no)]
    out: Dict[str, Any] = {}
    arr = t[(t.next_station_code == next_code) & (pd.to_datetime(t.target_next_actual_arrival) < cutoff)]
    d = arr.next_station_arrival_delay_min.astype(float)
    out["historical_observation_count"] = int(len(d))
    out["historical_avg_delay_min"] = float(d.mean()) if len(d) else np.nan
    out["historical_delay_std_min"] = float(d.std(ddof=1)) if len(d) > 1 else np.nan
    out["historical_on_time_rate_pct"] = float((d < DELAY_THRESHOLD).mean() * 100) if len(d) else np.nan
    stops = t[(t.station_code == station_code) & (pd.to_datetime(t.observation_time) < cutoff)]
    a = stops.actual_stop_min.dropna().astype(float)
    out["historical_avg_actual_stop_min"] = float(a.mean()) if len(a) else np.nan
    out["historical_stop_std_min"] = float(a.std(ddof=1)) if len(a) > 1 else np.nan
    both = stops[stops.actual_stop_min.notna() & stops.scheduled_stop_min.notna()]
    out["historical_stop_observation_count"] = int(len(both))
    out["historical_stop_reliability_pct"] = (
        float(((both.actual_stop_min - both.scheduled_stop_min).abs() <= STOP_TOLERANCE_MIN).mean() * 100)
        if len(both) else np.nan)
    return out


def feature_row(df: pd.DataFrame, stop: pd.Series, train_no: str, train_type: str, journey_date,
                delay_min: float, cutoff: pd.Timestamp,
                departure_delay_min: Optional[float] = None) -> Dict[str, Any]:
    """One model input row for a train standing at ``stop`` with ``delay_min`` of delay.

    ``delay_min`` is the arrival delay; ``departure_delay_min`` defaults to it.
    Calendar features come from the JOURNEY date (the day the train started), as in training.
    """
    dep_delay = delay_min if departure_delay_min is None else departure_delay_min
    jd = pd.Timestamp(journey_date)
    # the source (and so the training data) records the origin's arrival as its departure, halt 0
    origin = stop.sched_arr is None or pd.isna(stop.sched_arr)
    sched_arr = stop.sched_dep if origin else stop.sched_arr
    halt = 0.0 if origin else stop.halt_min
    row = {
        "train_no": train_no, "train_type": train_type, "station_code": stop.station_code,
        "next_station_code": stop.next_station_code, "seq": int(stop.seq),
        "current_scheduled_arrival": sched_arr, "current_scheduled_departure": stop.sched_dep,
        "scheduled_stop_min": halt, "scheduled_travel_to_next_min": stop.travel_to_next_min,
        "distance_to_next_km": stop.distance_to_next_km,
        "day_of_week": jd.day_of_week, "month": jd.month, "is_weekend": int(jd.day_of_week >= 5),
        "current_arrival_delay_min": delay_min, "current_departure_delay_min": dep_delay,
    }
    row.update(history_features(df, train_no, stop.station_code, stop.next_station_code, cutoff))
    return row


# Ways to estimate the delay at a stop further ahead than the next one.
METHODS = ("iterative", "carry", "model_then_carry", "history_change")
METHOD_LABELS = {
    "model": "model (next stop)",
    "iterative": "model applied stop by stop",
    "carry": "current delay carried forward",
    "model_then_carry": "next-stop estimate carried forward",
    "history_change": "current delay + typical recorded change",
    "no_coverage": "no model coverage: delay carried forward",
}
BANDS = ((1, 1), (2, 3), (4, 6), (7, 10), (11, 99))
EVALUATION = ROOT / "reports" / "planner_evaluation.json"


def band_of(stops_ahead: int) -> str:
    for lo, hi in BANDS:
        if lo <= stops_ahead <= hi:
            return f"{lo}" if lo == hi else f"{lo}-{hi}"
    return f"{BANDS[-1][0]}-{BANDS[-1][1]}"


def likely_ranges(path: Path = EVALUATION) -> Dict[str, List[float]]:
    """Per distance band: 10th/90th percentile of (actual - estimate) on the validation period."""
    try:
        import json
        return dict(json.loads(Path(path).read_text())["likely_range_by_band"])
    except (OSError, KeyError, ValueError):
        return {}


def selected_methods(path: Path = EVALUATION) -> Dict[str, str]:
    """Method per distance band chosen on the VALIDATION period by scripts/evaluate_planner.py.

    Without that report, stops beyond the next one carry the current delay forward
    (the simplest estimate, never worse than an unvalidated method).
    """
    try:
        import json
        return dict(json.loads(Path(path).read_text())["selected_method_by_band"])
    except (OSError, KeyError, ValueError):
        return {}


def arrival_delays(df: pd.DataFrame, train_no: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Recorded arrival delay per (journey, station) for one train, outcomes known before ``cutoff``."""
    t = df[df.train_no.astype(str).str.lstrip("0") == _key(train_no)]
    a = t[pd.to_datetime(t.observation_time) < cutoff][["journey_date", "station_code", "current_arrival_delay_min"]]
    a = a.rename(columns={"current_arrival_delay_min": "arr"})
    b = t[pd.to_datetime(t.target_next_actual_arrival) < cutoff][
        ["journey_date", "next_station_code", "next_station_arrival_delay_min"]].rename(
        columns={"next_station_code": "station_code", "next_station_arrival_delay_min": "arr"})
    return pd.concat([b, a]).dropna().drop_duplicates(["journey_date", "station_code"])


def typical_change(df: pd.DataFrame, train_no: str, start: str, target: str, cutoff: pd.Timestamp,
                   min_journeys: int = 5) -> Optional[float]:
    """Median recorded change: arrival delay at ``target`` minus departure delay at ``start``.

    Same journey, same train, only journeys whose outcome was known before ``cutoff``.
    None when fewer than ``min_journeys`` journeys recorded both stops.
    """
    t = df[(df.train_no.astype(str).str.lstrip("0") == _key(train_no)) & (df.station_code == start)
           & (pd.to_datetime(df.observation_time) < cutoff)]
    dep = t[["journey_date", "current_departure_delay_min", "current_arrival_delay_min"]].copy()
    dep["dep"] = dep.current_departure_delay_min.fillna(dep.current_arrival_delay_min)
    arr = arrival_delays(df, train_no, cutoff)
    m = dep.merge(arr[arr.station_code == target], on="journey_date")
    return float((m.arr - m.dep).median()) if len(m) >= min_journeys else None


def halt_overrun(df: pd.DataFrame, train_no: str, station_code: str, cutoff: pd.Timestamp) -> float:
    """Average recorded (actual - scheduled) stop at a station before ``cutoff``; 0 without history."""
    t = df[(df.train_no.astype(str).str.lstrip("0") == _key(train_no)) & (df.station_code == station_code)
           & (pd.to_datetime(df.observation_time) < cutoff)]
    v = (t.actual_stop_min - t.scheduled_stop_min).dropna()
    return float(v.mean()) if len(v) else 0.0


def forecast(df: pd.DataFrame, tt: pd.DataFrame, predictor: DelayPredictor, train_no: str, journey_date,
             station_code: str, current_delay: float, now: Optional[pd.Timestamp] = None,
             covered: Optional[set] = None, method: str = "selected") -> Dict[str, Any]:
    """The model's estimate for the next stop, and an estimate for every later stop.

    ``method`` for stops after the next one: one of METHODS, or "selected" (the
    method per distance band chosen on the validation period, see
    scripts/evaluate_planner.py). ``covered``: segments seen in training
    (``model_coverage()``); outside them the delay is carried forward, labelled.
    """
    covered = model_coverage() if covered is None else covered
    if method != "selected" and method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose 'selected' or one of {METHODS}")
    sched = schedule_on(tt, train_no, journey_date)
    if sched.empty:
        return {"status": "unknown_train", "message": f"Train {train_no} is not in the timetable."}
    here = sched[sched.station_code == station_code]
    if here.empty:
        return {"status": "unknown_station",
                "message": f"Train {train_no} does not call at {station_code} in the current timetable."}
    i0 = int(here.index[0])
    start = sched.loc[i0]
    if pd.isna(start.sched_dep) or pd.isna(start.next_station_code):
        return {"status": "no_next_station", "message": f"{station_code} is the last stop of train {train_no}."}

    # the prediction is made when the train leaves this stop
    reference = start.sched_dep + pd.Timedelta(minutes=float(current_delay))
    train_type = str(sched.train_type.iat[0])
    warnings: List[str] = []
    ran = runs_on(tt, train_no, journey_date)
    if ran is False:
        warnings.append(f"Train {train_no} has not run on a {pd.Timestamp(journey_date):%A} in recent records.")

    # at the starting stop the user's delay is both the arrival and departure delay;
    # at later stops a train that arrives early still leaves on time (recorded data:
    # 0.4% of departures are early), and usually overruns its halt a little
    selection = selected_methods() if method == "selected" else {}
    ranges = likely_ranges() if method == "selected" else {}
    rows, delay, dep_delay = [], float(current_delay), float(current_delay)
    for k in range(i0, len(sched) - 1):
        stop = sched.loc[k]
        ahead = k - i0 + 1
        in_model = (_key(train_no), stop.station_code, stop.next_station_code) in covered
        # candidate 1: the model, applied stop by stop (for the next stop this IS the model's prediction)
        prob, unseen = None, []
        if in_model:
            x = feature_row(df, stop, train_no, train_type, journey_date, delay, cutoff=reference,
                            departure_delay_min=dep_delay)
            p = predictor.predict(x)[0]
            d_iter = p["predicted_next_station_delay_min"]
            prob = p["probability_delayed_15"] if ahead == 1 else None
            unseen = sorted(p["unknown_categories"].get("station_code", [])
                            + p["unknown_categories"].get("next_station_code", []))
        else:
            d_iter = dep_delay
        if ahead == 1:
            chosen, d = ("model" if in_model else "no_coverage"), d_iter
        else:
            name = selection.get(band_of(ahead), "carry") if method == "selected" else method
            candidates = {"iterative": d_iter, "carry": float(current_delay),
                          "model_then_carry": rows[0]["predicted_delay_min"]}
            if name == "history_change":
                change = typical_change(df, train_no, station_code, stop.next_station_code, reference)
                if change is not None:
                    candidates["history_change"] = float(current_delay) + change
            chosen = name if name in candidates else "carry"
            d = candidates[chosen]
        exp = expected_arrival(stop.next_sched_arr, d)
        rng = ranges.get(band_of(ahead)) if chosen != "no_coverage" else None
        rows.append({
            "station_code": stop.next_station_code, "station_name": stop.next_station_name,
            "stops_ahead": ahead, "scheduled_arrival": stop.next_sched_arr,
            "predicted_delay_min": d, "risk": risk_label(d), "probability_delayed_15": prob,
            "expected_arrival": exp,
            "minutes_from_departure": minutes_until(exp, reference),
            "minutes_from_now": minutes_until(exp, now) if now is not None else None,
            "scheduled_halt_min": sched.loc[k + 1].halt_min,
            "stop_category": stop_category(sched.loc[k + 1].halt_min),
            "likely_delay_low_min": d + rng[0] if rng else None,
            "likely_delay_high_min": d + rng[1] if rng else None,
            "unseen_inputs": unseen, "method": chosen, "basis": METHOD_LABELS[chosen],
        })
        # the stop-by-stop model state: leave the next stop after its usual recorded halt overrun
        overrun = halt_overrun(df, train_no, stop.next_station_code, reference)
        delay, dep_delay = d_iter, max(d_iter + overrun, 0.0)
    first = rows[0]
    unseen = sorted({c for r in rows for c in r["unseen_inputs"]})
    if unseen:
        warnings.append("Stations not in the model's training data (their identity is ignored by the model): "
                        + ", ".join(unseen) + ".")
    return {"status": "ok", "mode": "forecast", "train_no": train_no, "journey_date": pd.Timestamp(journey_date),
            "station_code": station_code, "station_name": start.station_name,
            "scheduled_departure": start.sched_dep, "current_delay_min": float(current_delay),
            "reference_time": reference, "next": first, "route": pd.DataFrame(rows),
            "models": predictor.metadata.get("model_type", {}), "warnings": warnings}


def typical_delays(df: pd.DataFrame, tt: pd.DataFrame, train_no: str, journey_date) -> pd.DataFrame:
    """Recorded arrival delays per stop (history, not a forecast), with the chosen date's times."""
    sched = schedule_on(tt, train_no, journey_date)
    t = df[df.train_no.astype(str).str.lstrip("0") == _key(train_no)]
    # each recorded arrival once: as the current stop's arrival delay, or as the previous row's next-stop delay
    a = t[["journey_date", "station_code", "current_arrival_delay_min"]].rename(
        columns={"current_arrival_delay_min": "delay"})
    b = t[["journey_date", "next_station_code", "next_station_arrival_delay_min"]].rename(
        columns={"next_station_code": "station_code", "next_station_arrival_delay_min": "delay"})
    arr = (pd.concat([b, a]).dropna().drop_duplicates(["journey_date", "station_code"]))
    dow = pd.Timestamp(journey_date).day_of_week
    arr["same_weekday"] = pd.to_datetime(arr.journey_date).dt.day_of_week == dow
    out = []
    for s in sched.itertuples():
        d = arr[arr.station_code == s.station_code]
        same = d[d.same_weekday]
        out.append({"station_code": s.station_code, "station_name": s.station_name,
                    "scheduled_arrival": s.sched_arr, "scheduled_departure": s.sched_dep,
                    "scheduled_halt_min": s.halt_min, "recorded_arrivals": int(len(d)),
                    "median_delay_min": float(d.delay.median()) if len(d) else None,
                    "share_delayed_15_pct": float((d.delay >= DELAY_THRESHOLD).mean() * 100) if len(d) else None,
                    "median_delay_same_weekday_min": float(same.delay.median()) if len(same) >= 5 else None,
                    "same_weekday_arrivals": int(len(same))})
    return pd.DataFrame(out)
