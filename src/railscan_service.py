"""Application logic for the Streamlit UI (no Streamlit imports here).

Every value returned is tagged with what it is:
    recorded   - taken from the historical dataset (a real observation)
    scheduled  - timetable value from the same records
    predicted  - output of the trained delay model
    estimate   - iterative use of the model beyond the next station
    scenario   - the user typed the current delay ("what if")
There is no live data source: RailScan replays recorded journeys.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .delay_prediction import DelayPredictor, risk_label
from .next_station import describe_remaining, expected_arrival, minutes_until
from .stop_analysis import historical_stop_stats, stop_category

ROOT = Path(__file__).resolve().parents[1]
MASTER_CSV = ROOT / "data" / "processed" / "RailScan_Master_Dataset.csv"
DATETIME_COLS = ["observation_time", "current_scheduled_arrival", "current_scheduled_departure",
                 "next_scheduled_arrival", "expected_next_station_arrival_baseline",
                 "target_next_actual_arrival", "actual_arrival", "actual_departure"]
MAX_ITERATIVE_STEPS = 12


class DataUnavailable(RuntimeError):
    pass


def load_master(path: Path = MASTER_CSV, source: str = "auto", train_no: Optional[str] = None) -> pd.DataFrame:
    """The observations (all, or one train's). ``source``: "auto" (SQLite if it holds the data,
    else CSV), "sqlite", "csv". ``df.attrs["source"]`` says which one was used.
    """
    if source in ("auto", "sqlite"):
        from src.database.app_data import data_ready, load_master_db
        if data_ready():
            df = load_master_db(train_no=train_no)
            df.attrs["source"] = "SQLite"
            return df
        if source == "sqlite":
            raise DataUnavailable("The SQLite database has no railway data. Run: python scripts/setup_database.py all")
    if not Path(path).exists():
        raise DataUnavailable(f"Dataset not found: {path}. Run: python scripts/prepare_dataset.py")
    if train_no is None:
        df = pd.read_csv(path, dtype={"train_no": str})
    else:                                   # the file is large: read in chunks, keep one train
        want = str(train_no).zfill(5)
        df = pd.concat([c[c.train_no.str.zfill(5) == want]
                        for c in pd.read_csv(path, dtype={"train_no": str}, chunksize=200_000)],
                       ignore_index=True)
    df["train_no"] = df["train_no"].str.zfill(5)
    for c in DATETIME_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    df["journey_date"] = pd.to_datetime(df["journey_date"]).dt.date
    df.attrs["source"] = "CSV"
    return df


def list_trains(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("train_no").agg(train_name=("train_name", lambda s: s.mode().iat[0]),
                                   journeys=("journey_date", "nunique"), rows=("seq", "size"))
    return g.reset_index()


def journey_dates(df: pd.DataFrame, train_no: str) -> List:
    return sorted(df.loc[df.train_no == train_no, "journey_date"].unique(), reverse=True)


def journey(df: pd.DataFrame, train_no: str, journey_date) -> pd.DataFrame:
    return df[(df.train_no == train_no) & (df.journey_date == journey_date)].sort_values("seq")


def _station_label(name, code) -> str:
    """'NEW DELHI (NDLS)', or just the code when no name is known (missing names are NaN)."""
    return f"{name} ({code})" if isinstance(name, str) and name.strip() else str(code)


def _num(v) -> Optional[float]:
    return None if v is None or pd.isna(v) else float(v)


def _scenario_row(row: pd.Series, current_delay: float) -> pd.Series:
    """Copy of a row with the current delay replaced by a user value (history unchanged)."""
    r = row.copy()
    r["current_arrival_delay_min"] = current_delay
    r["current_departure_delay_min"] = current_delay
    r["current_observed_delay_min"] = current_delay
    return r


def analyze(df: pd.DataFrame, predictor: Optional[DelayPredictor], train_no: str, journey_date,
            station_code: str, seq: Optional[int] = None,
            delay_override: Optional[float] = None) -> Dict[str, Any]:
    """Everything the delay-analysis view shows for one observed stop."""
    j = journey(df, train_no, journey_date)
    rows = j[j.station_code == station_code]
    if seq is not None:
        rows = rows[rows.seq == seq]
    if rows.empty:
        return {"status": "not_found",
                "message": f"No recorded observation of train {train_no} at {station_code} on {journey_date}."}
    row = rows.iloc[0]
    scenario = delay_override is not None
    model_row = _scenario_row(row, float(delay_override)) if scenario else row

    # reference time = when the prediction is made
    if scenario:
        reference = row.current_scheduled_departure + pd.Timedelta(minutes=float(delay_override))
    else:
        reference = row.observation_time

    out: Dict[str, Any] = {
        "status": "ok", "scenario": scenario, "reference_time": reference,
        "train_no": train_no, "train_name": row.train_name, "train_type": row.train_type,
        "journey_date": journey_date,
        "station": {"code": row.station_code, "name": row.get("station_name"), "seq": int(row.seq)},
        "scheduled": {"arrival": row.current_scheduled_arrival, "departure": row.current_scheduled_departure,
                      "halt_min": _num(row.scheduled_stop_min),
                      "stop_category": stop_category(_num(row.scheduled_stop_min))},
        "recorded": {"arrival_delay_min": _num(row.current_arrival_delay_min),
                     "departure_delay_min": _num(row.current_departure_delay_min),
                     "current_delay_min": _num(row.current_observed_delay_min),
                     "actual_arrival": row.get("actual_arrival"), "actual_departure": row.get("actual_departure"),
                     "actual_stop_min": _num(row.actual_stop_min)},
        "current_delay_used": float(delay_override) if scenario else _num(row.current_observed_delay_min),
        "next_station": {"code": row.next_station_code, "name": row.get("next_station_name"),
                         "scheduled_arrival": row.next_scheduled_arrival,
                         "distance_km": _num(row.distance_to_next_km),
                         "scheduled_travel_min": _num(row.scheduled_travel_to_next_min)},
        "history": {"observation_count": int(row.historical_observation_count),
                    "avg_delay_min": _num(row.historical_avg_delay_min),
                    "delay_std_min": _num(row.historical_delay_std_min),
                    "on_time_rate_pct": _num(row.historical_on_time_rate_pct)},
        "stop_history": historical_stop_stats(df, row.station_code, train_no, before=row.observation_time),
        "prediction": None, "upcoming": pd.DataFrame(), "warnings": [],
    }

    # recorded outcome (only meaningful for a replay, not for a scenario)
    if not scenario:
        out["recorded_outcome"] = {"next_arrival_delay_min": _num(row.next_station_arrival_delay_min),
                                   "next_actual_arrival": row.target_next_actual_arrival,
                                   "risk_label": row.next_delay_risk_label}
    # persistence baseline: carry the current delay forward
    cur = out["current_delay_used"]
    out["baseline"] = {"next_arrival_delay_min": cur,
                       "expected_arrival": expected_arrival(row.next_scheduled_arrival, cur)}

    if predictor is None:
        out["warnings"].append("Delay model not loaded; showing schedule and recorded values only.")
        return out

    pred = predictor.predict(model_row)[0]
    if pred["unknown_categories"]:
        out["warnings"].append("The model has not seen: " + ", ".join(
            f"{k}={v}" for k, v in pred["unknown_categories"].items()) + "; those inputs are ignored.")
    exp = expected_arrival(row.next_scheduled_arrival, pred["predicted_next_station_delay_min"])
    mins = minutes_until(exp, reference)
    out["prediction"] = {**pred, "expected_arrival": exp, "minutes_until": mins,
                         "remaining_text": describe_remaining(mins)}
    out["upcoming"] = upcoming_table(df, j, row, model_row, predictor, reference)
    return out


def upcoming_table(df: pd.DataFrame, j: pd.DataFrame, row: pd.Series, model_row: pd.Series,
                   predictor: DelayPredictor, reference) -> pd.DataFrame:
    """Next station from the model, later stations as a labelled iterative estimate.

    For station k+1 the predicted delay at k is fed back as the current delay
    (assuming the scheduled halt is kept). The chain stops where the dataset has
    no record of the following stop, rather than guessing a timetable.
    """
    records = []
    current, step = model_row, 0
    while step < MAX_ITERATIVE_STEPS and pd.notna(current.get("next_station_code")):
        p = predictor.predict(current)[0]
        d = p["predicted_next_station_delay_min"]
        exp = expected_arrival(current.next_scheduled_arrival, d)
        nxt = j[(j.station_code == current.next_station_code) & (j.seq == current.next_station_seq)]
        halt = _num(nxt.iloc[0].scheduled_stop_min) if not nxt.empty else None
        stats = historical_stop_stats(df, current.next_station_code, row.train_no, before=row.observation_time)
        records.append({
            "Station": _station_label(current.get("next_station_name"), current.next_station_code),
            "Scheduled arrival": current.next_scheduled_arrival,
            "Predicted delay (min)": round(d, 1),
            "Risk": risk_label(d),
            "Expected arrival": exp,
            "Time remaining": describe_remaining(minutes_until(exp, reference)),
            "Scheduled halt (min)": halt,
            "Stop reliability (%)": round(stats["stop_reliability_pct"], 1) if stats else None,
            "Basis": "model: next station" if step == 0 else "iterative estimate",
        })
        if nxt.empty:
            break
        current = _scenario_row(nxt.iloc[0], d)
        step += 1
    return pd.DataFrame(records)


def dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    stations = set(df.station_code) | set(df.next_station_code.dropna())
    return {"rows": len(df), "columns": df.shape[1], "trains": df.train_no.nunique(),
            "journeys": df.groupby(["train_no", "journey_date"]).ngroups, "stations": len(stations),
            "date_min": min(df.journey_date), "date_max": max(df.journey_date)}


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    m = (df.isna().mean() * 100).round(2)
    m = m[m > 0].sort_values(ascending=False)
    return pd.DataFrame({"column": m.index, "missing %": m.values})


def station_choices(j: pd.DataFrame) -> List[Dict[str, Any]]:
    return [{"code": r.station_code, "name": r.station_name, "seq": int(r.seq),
             "label": f"{int(r.seq):>2}. {_station_label(r.station_name, r.station_code)}"}
            for r in j.itertuples()]


def summarize_ocr_numbers(df: pd.DataFrame, numbers: List[str]) -> Dict[str, List[str]]:
    """Split numbers read by OCR into known trains and other numbers (never assume)."""
    known = set(df.train_no) | set(df.train_no.str.lstrip("0"))
    found = [n for n in numbers if n in known or n.lstrip("0") in known]
    return {"known_trains": [n.zfill(5) for n in found], "other_numbers": [n for n in numbers if n not in found]}


def platform_numbers(text: str) -> List[str]:
    """'PF 3', 'Platform No. 2', 'प्लेटफॉर्म 4' -> ['3', '2', '4'] (only when the word is present)."""
    import re
    pat = re.compile(r"(?:platform|plat\.?|pf|प्लेटफॉर्म|प्लॅटफॉर्म|ప్లాట్‌ఫాం|ಪ್ಲಾಟ್‌ಫಾರ್ಮ್)\s*(?:no\.?|number|#)?\s*[:.-]?\s*(\d{1,2})\b",
                     re.IGNORECASE)
    return list(dict.fromkeys(pat.findall(text or "")))


def as_time(ts) -> str:
    if ts is None or (isinstance(ts, float) and np.isnan(ts)) or pd.isna(ts):
        return "-"
    return pd.Timestamp(ts).strftime("%d %b %Y, %H:%M")
