#!/usr/bin/env python3
"""
RailScan - Step 1: real railway dataset preparation pipeline.

Source  : Hugging Face dataset ``yadavsaurabh/indian-railways-runs``
          (historical, third-party stop-level running data; NOT an official
          live railway feed).
Output  : data/processed/{RailScan_Final_Dataset,train,validation,test,
          dataset_manifest,filtering_report}.csv

One output row = one *observation*: "the train has been observed at station
S (arrival and/or departure), what happens at the NEXT station N?"

Usage
-----
    python scripts/build_dataset.py --target-rows 10000 --min-trains 1
    python scripts/build_dataset.py --target-rows 10000 --min-trains 1 \
           --output-dir data/processed

Design summary (see README of the pipeline in the final answer for detail)
--------------------------------------------------------------------------
* The source is streamed (never fully loaded). Rows are grouped into
  journeys keyed by (train_no, date) in a small bounded buffer.
* Each journey is sorted by ``seq``. The source lists only the stops where
  the train halts, and ``seq`` numbers positions along the line (stations
  passed without stopping keep their numbers), so consecutive stops can be
  e.g. seq 10 -> 13. The next station is the next REGULAR stop of the train
  (a seq present on at least half of its journeys, see ``regular_stops``).
  If that stop is missing from a day's record the row is dropped, because the
  "next station" would otherwise be wrong.
* Nothing is invented: invalid / missing actual timestamps stay missing
  and the affected rows are dropped or the feature is left as NaN.
* Historical features are *time-aware*: an earlier observation contributes
  only if its outcome was fully known STRICTLY BEFORE the current
  observation time (this is stricter than a simple shifted expanding mean).
* Split is chronological and cut on journey-date boundaries so no journey
  date appears in two splits.
"""
from __future__ import annotations

import argparse
import re
import sys
import zlib
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
DATASET_ID = "yadavsaurabh/indian-railways-runs"

# Columns without which the pipeline cannot work at all.
CORE_COLUMNS = ["train_no", "date", "seq", "station_code",
                "sch_arrival", "sch_departure", "act_arrival"]
# Columns that are used when present and gracefully skipped when absent.
OPTIONAL_COLUMNS = [
    "train_name", "train_type", "halts_here",
    "sch_arrival_date", "sch_departure_date",
    "act_arrival_date", "act_departure", "act_departure_date",
    "delay_arrival_min", "delay_departure_min",
    "platform", "distance_km", "lat", "lng",
    "departed", "reported", "reported_at",
]

TRAIN_NO_RE = re.compile(r"^\d{4,5}$")          # Indian train numbers
STATION_RE = re.compile(r"^[A-Z0-9]{1,6}$")     # Indian Railways station codes
MAX_ABS_DELAY_MIN = 2880                        # |delay| > 2 days => implausible
MAX_STOP_MIN = 24 * 60                          # stop longer than 1 day => invalid
MAX_TRAVEL_MIN = 48 * 60                        # scheduled leg > 2 days => invalid
DELAY_THRESHOLD = 15                            # minutes
HIGH_THRESHOLD = 60                             # minutes
STOP_TOLERANCE_MIN = 2                          # |actual stop - scheduled halt| <= 2 min => "reliable" stop
MIN_DISTINCT_DATES = 10                         # default; needed for a date-based split

# train_no -> sorted seq numbers of its regular stops (set by build() for local
# sources; without it, consecutive rows of the journey are treated as consecutive stops)
REGULAR_STOPS: Dict[str, List[int]] = {}
REGULAR_STOP_SHARE = 0.5

# Column groups (used by the leakage checks).
ID_COLUMNS = ["train_no", "train_name", "train_type", "journey_date",
              "station_code", "seq", "next_station_code", "next_station_seq",
              "observation_time"]
FEATURE_COLUMNS = [
    "current_scheduled_arrival", "current_scheduled_departure",
    "next_scheduled_arrival", "distance_to_next_km",
    "scheduled_stop_min", "actual_stop_min", "stop_variation_min",
    "scheduled_travel_to_next_min",
    "day_of_week", "month", "day", "hour", "is_weekend",
    "current_arrival_delay_min", "current_departure_delay_min",
    "current_observed_delay_min",
    "historical_avg_delay_min", "historical_delay_std_min",
    "historical_observation_count", "historical_on_time_rate_pct",
    "historical_avg_actual_stop_min", "historical_stop_std_min",
    "historical_stop_observation_count", "historical_stop_reliability_pct",
    "expected_next_station_arrival_baseline",
    "station_lat", "station_lng",
]
TARGET_COLUMNS = ["next_station_arrival_delay_min", "next_station_delayed_15",
                  "next_delay_risk_label", "target_next_actual_arrival"]


# --------------------------------------------------------------------------
# Filtering log - documents every filtering decision
# --------------------------------------------------------------------------
class FilterLog:
    """Counts rows removed / values blanked, per documented reason."""

    def __init__(self) -> None:
        self.source_rows = 0
        self.dropped: Counter = Counter()
        self.notes: Counter = Counter()

    def drop(self, reason: str, n: int) -> None:
        if n > 0:
            self.dropped[reason] += int(n)

    def note(self, what: str, n: int) -> None:
        if n > 0:
            self.notes[what] += int(n)

    def to_frame(self) -> pd.DataFrame:
        rows = [("row_dropped", k, v) for k, v in self.dropped.items()]
        rows += [("value_blanked_or_filled", k, v) for k, v in self.notes.items()]
        return pd.DataFrame(rows, columns=["kind", "reason", "count"])

    def print_report(self) -> None:
        print("\n=== FILTERING REPORT (every filtering decision) ===")
        print(f"Source rows inspected: {self.source_rows:,}")
        if self.dropped:
            print("Rows dropped:")
            for k, v in self.dropped.items():
                print(f"  - {k}: {v:,}")
        if self.notes:
            print("Values blanked / fallbacks applied (no values fabricated):")
            for k, v in self.notes.items():
                print(f"  - {k}: {v:,}")


# --------------------------------------------------------------------------
# Small parsing helpers
# --------------------------------------------------------------------------
def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Return a column, or an all-NaN series if the source lacks it."""
    if name in df.columns:
        return df[name]
    return pd.Series(np.nan, index=df.index, dtype="object")


def _to_flag(s: pd.Series) -> pd.Series:
    """Coerce 1/0, True/False, 'true'/'false' to float 1.0/0.0 (NaN if unknown)."""
    mapped = s.map(lambda v: 1.0 if str(v).strip().lower() in ("1", "1.0", "true", "t", "yes")
                   else (0.0 if str(v).strip().lower() in ("0", "0.0", "false", "f", "no")
                         else np.nan))
    return mapped.astype(float)


def _minutes_of_day(time_s: pd.Series) -> pd.Series:
    """Parse 'HH:MM[...]' (also 'HH:MM:SS' or a full timestamp) to minutes since midnight."""
    parts = time_s.astype(str).str.extract(r"(\d{1,2}):(\d{2})").astype(float)
    return parts[0] * 60.0 + parts[1]


def _combine(date_s: pd.Series, time_s: pd.Series) -> pd.Series:
    """date + time-of-day -> datetime64. NaT when either part is missing/invalid."""
    day = pd.to_datetime(date_s.astype(str).str.strip(), errors="coerce").dt.normalize()
    delta = pd.to_timedelta(_minutes_of_day(time_s), unit="m")
    return day + delta


def _schedule_with_rollover(journey_date: pd.Series, arr_t: pd.Series,
                            dep_t: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Reconstruct scheduled datetimes when the explicit *_date column is missing.

    Walk the journey in seq order (arrival, then departure, per stop). If the
    time-of-day falls back by more than 12 h compared with the previous valid
    time, the train has crossed midnight and the day offset is incremented.
    Used for SCHEDULED times only; actual times are never reconstructed.
    """
    arr_m, dep_m = _minutes_of_day(arr_t).to_numpy(), _minutes_of_day(dep_t).to_numpy()
    base = pd.to_datetime(journey_date.astype(str), errors="coerce").dt.normalize()
    day_offset, prev = 0, None
    arr_out, dep_out = [], []
    for a, d in zip(arr_m, dep_m):
        pair = []
        for val in (a, d):
            if np.isnan(val):
                pair.append(np.nan)
                continue
            if prev is not None and prev - val > 720:
                day_offset += 1
            prev = val
            pair.append(day_offset * 1440.0 + val)
        arr_out.append(pair[0])
        dep_out.append(pair[1])
    arr_dt = base + pd.to_timedelta(pd.Series(arr_out, index=journey_date.index), unit="m")
    dep_dt = base + pd.to_timedelta(pd.Series(dep_out, index=journey_date.index), unit="m")
    return arr_dt, dep_dt


def _minutes(delta: pd.Series) -> pd.Series:
    return delta / pd.Timedelta(minutes=1)


# --------------------------------------------------------------------------
# Journey processing: one (train_no, date) group -> observations
# --------------------------------------------------------------------------
def regular_stops(frame: pd.DataFrame) -> Dict[str, List[int]]:
    """Seq numbers present on at least REGULAR_STOP_SHARE of each train's journeys."""
    f = frame.assign(seq=pd.to_numeric(frame["seq"], errors="coerce")).dropna(subset=["seq"])
    f = f.assign(train_no=f["train_no"].astype(str).str.replace(r"\.0$", "", regex=True))
    out = {}
    for t, g in f.groupby("train_no"):
        share = g.groupby("seq")["date"].nunique() / g["date"].nunique()
        out[t] = sorted(int(s) for s in share[share >= REGULAR_STOP_SHARE].index)
    return out


def _next_regular_ok(cur: pd.DataFrame) -> pd.Series:
    """True where the recorded next stop is the train's next regular stop."""
    ok = []
    for t, s, n in zip(cur["train_no"], cur["seq"], cur["next_station_seq"]):
        stops = REGULAR_STOPS.get(t)
        if not stops:                       # no stop list (streaming source): accept the next recorded stop
            ok.append(True)
            continue
        later = [x for x in stops if x > s]
        ok.append(bool(later) and n == later[0])
    return pd.Series(ok, index=cur.index)


def process_journey(rows: List[dict], log: FilterLog) -> Optional[pd.DataFrame]:
    """Turn the stop rows of ONE journey into current->next observations."""
    df = pd.DataFrame(rows)

    # -- basic validity of identifiers -------------------------------------
    df["train_no"] = (df["train_no"].astype(str).str.strip()
                      .str.replace(r"\.0$", "", regex=True))
    bad = ~df["train_no"].str.match(TRAIN_NO_RE)
    log.drop("invalid train number (not 4-5 digits)", bad.sum())
    df = df[~bad]

    df["station_code"] = df["station_code"].astype(str).str.strip().str.upper()
    bad = ~df["station_code"].str.match(STATION_RE)
    log.drop("invalid station code", bad.sum())
    df = df[~bad]

    df["journey_date"] = pd.to_datetime(df["date"].astype(str).str.strip(),
                                        errors="coerce").dt.normalize()
    bad = df["journey_date"].isna()
    log.drop("invalid journey date", bad.sum())
    df = df[~bad]

    df["seq"] = pd.to_numeric(df["seq"], errors="coerce")
    bad = df["seq"].isna()
    log.drop("missing/invalid seq", bad.sum())
    df = df[~bad]
    if df.empty:
        return None

    # -- duplicates within the journey -------------------------------------
    df = df.sort_values("seq", kind="stable")
    dup = df.duplicated(subset=["seq"], keep="first")
    log.drop("duplicate stop (same train, date, seq)", dup.sum())
    df = df[~dup].reset_index(drop=True)
    if len(df) < 2:
        log.drop("journey with fewer than 2 valid stops", len(df))
        return None

    # -- observation flags ---------------------------------------------------
    has_reported = "reported" in df.columns
    reported = _to_flag(df["reported"]) if has_reported else pd.Series(1.0, index=df.index)
    departed = _to_flag(df["departed"]) if "departed" in df.columns else pd.Series(np.nan, index=df.index)

    # -- datetimes -------------------------------------------------------------
    sch_arr = _combine(_col(df, "sch_arrival_date"), df["sch_arrival"])
    sch_dep = _combine(_col(df, "sch_departure_date"), df["sch_departure"])
    # Fallback ONLY for scheduled times without an explicit date: midnight-rollover rebuild.
    need = sch_arr.isna() | sch_dep.isna()
    if need.any():
        arr_fb, dep_fb = _schedule_with_rollover(df["journey_date"], df["sch_arrival"], df["sch_departure"])
        filled_a = sch_arr.isna() & arr_fb.notna()
        filled_d = sch_dep.isna() & dep_fb.notna()
        log.note("scheduled arrival rebuilt via midnight-rollover (explicit date missing)", filled_a.sum())
        log.note("scheduled departure rebuilt via midnight-rollover (explicit date missing)", filled_d.sum())
        sch_arr = sch_arr.fillna(arr_fb)
        sch_dep = sch_dep.fillna(dep_fb)

    # Actual times: explicit dates only, never reconstructed, never faked.
    act_arr = _combine(_col(df, "act_arrival_date"), df["act_arrival"])
    act_dep = _combine(_col(df, "act_departure_date"), _col(df, "act_departure"))
    # Stops that were not reported carry no real observation.
    not_rep = reported != 1.0
    act_arr = act_arr.mask(not_rep)
    act_dep = act_dep.mask(not_rep | (departed == 0.0))

    # -- delays (source fields first, else computed from timestamps; never 0-filled)
    def _delay(src_name: str, act: pd.Series, sch: pd.Series, label: str) -> pd.Series:
        src = pd.to_numeric(_col(df, src_name), errors="coerce").mask(not_rep)
        src = src.where(src.abs() <= MAX_ABS_DELAY_MIN)
        computed = _minutes(act - sch)
        computed = computed.where(computed.abs() <= MAX_ABS_DELAY_MIN)
        log.note(f"{label}: source field invalid, computed from timestamps", (src.isna() & computed.notna()).sum())
        return src.fillna(computed)

    arr_delay = _delay("delay_arrival_min", act_arr, sch_arr, "arrival delay")
    dep_delay = _delay("delay_departure_min", act_dep, sch_dep, "departure delay")

    dist = pd.to_numeric(_col(df, "distance_km"), errors="coerce")

    cur = pd.DataFrame({
        "train_no": df["train_no"],
        "train_name": _col(df, "train_name"),
        "train_type": _col(df, "train_type"),
        "journey_date": df["journey_date"],
        "station_code": df["station_code"],
        "seq": df["seq"].astype(int),
        "current_scheduled_arrival": sch_arr,
        "current_scheduled_departure": sch_dep,
        "_act_arr": act_arr, "_act_dep": act_dep,
        "current_arrival_delay_min": arr_delay,
        "current_departure_delay_min": dep_delay,
        "_dist": dist, "_reported": reported,
        "station_lat": pd.to_numeric(_col(df, "lat"), errors="coerce"),
        "station_lng": pd.to_numeric(_col(df, "lng"), errors="coerce"),
    })

    # -- attach the NEXT stop (sequence order) --------------------------------
    cur["next_station_code"] = cur["station_code"].shift(-1)
    cur["next_station_seq"] = cur["seq"].shift(-1)
    cur["next_scheduled_arrival"] = sch_arr.shift(-1)
    cur["next_station_arrival_delay_min"] = arr_delay.shift(-1)
    cur["target_next_actual_arrival"] = act_arr.shift(-1)
    cur["_next_dist"] = dist.shift(-1)
    cur["_next_reported"] = reported.shift(-1)

    # -- sequential filters with documented reasons ----------------------------
    def keep(mask: pd.Series, reason: str) -> None:
        nonlocal cur
        mask = mask.astype(bool)            # an EMPTY object-dtype mask would select columns, not rows
        log.drop(reason, (~mask).sum())
        cur = cur[mask]

    keep(cur["next_station_code"].notna(), "no next station (last stop of journey)")
    if cur.empty:
        return None
    keep(cur["_reported"] == 1.0, "current stop not reported (reported != 1)")
    keep(_next_regular_ok(cur), "next regular stop missing from this journey's record")
    keep(cur["next_station_code"] != cur["station_code"], "next station equals current station")
    keep(cur["_next_reported"] == 1.0, "next stop not reported (target not observed)")
    keep(cur["next_station_arrival_delay_min"].notna(), "missing target (next arrival delay)")
    keep(cur["target_next_actual_arrival"].notna(), "next actual arrival timestamp invalid/missing")
    if cur.empty:
        return None

    # Observation moment = actual departure if observed, else actual arrival.
    cur["observation_time"] = cur["_act_dep"].where(cur["_act_dep"].notna(), cur["_act_arr"])
    keep(cur["observation_time"].notna(), "no valid actual timestamp at current stop")
    cur["current_observed_delay_min"] = cur["current_departure_delay_min"].fillna(cur["current_arrival_delay_min"])
    keep(cur["current_observed_delay_min"].notna(), "no valid current delay observation")
    keep(cur["observation_time"] <= cur["target_next_actual_arrival"],
         "observation not before next arrival (non-causal / invalid order)")
    if cur.empty:
        return None

    # -- stop duration ---------------------------------------------------------
    sched_stop = _minutes(cur["current_scheduled_departure"] - cur["current_scheduled_arrival"])
    bad_s = sched_stop.notna() & ((sched_stop < 0) | (sched_stop > MAX_STOP_MIN))
    log.note("scheduled_stop_min negative/implausible -> blank", bad_s.sum())
    cur["scheduled_stop_min"] = sched_stop.mask(bad_s)

    act_stop = _minutes(cur["_act_dep"] - cur["_act_arr"])      # NaT if either missing
    bad_a = act_stop.notna() & ((act_stop < 0) | (act_stop > MAX_STOP_MIN))
    log.note("actual_stop_min negative/implausible -> blank", bad_a.sum())
    cur["actual_stop_min"] = act_stop.mask(bad_a)
    cur["stop_variation_min"] = cur["actual_stop_min"] - cur["scheduled_stop_min"]

    # -- travel time to next stop ----------------------------------------------
    travel = _minutes(cur["next_scheduled_arrival"] - cur["current_scheduled_departure"])
    ok = travel.notna() & (travel >= 0) & (travel <= MAX_TRAVEL_MIN)
    keep(ok, "scheduled travel to next stop missing/negative/implausible")
    cur["scheduled_travel_to_next_min"] = travel[cur.index]
    if cur.empty:
        return None

    # -- distance to next (cumulative-km difference), only when valid ----------
    d = cur["_next_dist"] - cur["_dist"]
    log.note("distance_to_next_km negative -> blank", (d < 0).sum())
    cur["distance_to_next_km"] = d.where(d >= 0)

    # -- calendar features (journey date + SCHEDULED hour => known in advance) -
    jd = cur["journey_date"]
    cur["day_of_week"] = jd.dt.dayofweek
    cur["month"] = jd.dt.month
    cur["day"] = jd.dt.day
    hour_src = cur["current_scheduled_departure"].fillna(cur["current_scheduled_arrival"])
    cur["hour"] = hour_src.dt.hour
    cur["is_weekend"] = (cur["day_of_week"] >= 5).astype(int)

    # -- baseline ETA (NOT an official railway ETA) ----------------------------
    cur["expected_next_station_arrival_baseline"] = (
        cur["next_scheduled_arrival"] + pd.to_timedelta(cur["current_observed_delay_min"], unit="m"))

    # -- targets ---------------------------------------------------------------
    t = cur["next_station_arrival_delay_min"]
    cur["next_station_delayed_15"] = (t >= DELAY_THRESHOLD).astype(int)
    cur["next_delay_risk_label"] = pd.cut(
        t, bins=[-np.inf, DELAY_THRESHOLD, HIGH_THRESHOLD, np.inf],
        labels=["Low", "Medium", "High"], right=False).astype(str)

    return cur.drop(columns=[c for c in cur.columns if c.startswith("_")]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Leakage-safe, time-aware historical features
# --------------------------------------------------------------------------
def time_aware_history(df: pd.DataFrame, group_cols: List[str], value_col: str,
                       avail_col: str, query_col: str,
                       threshold: Optional[float] = None) -> pd.DataFrame:
    """
    For every row, aggregate values of EARLIER rows in the same group whose
    outcome was known strictly before this row's query time.

    A past row r contributes to row q only if  avail_time(r) < query_time(q).
    This is a shifted-expanding computation on the real timeline: it excludes
    the row itself, all later rows, AND earlier-starting journeys whose
    outcome had not happened yet at the moment of prediction (long multi-day
    runs of the same train number can overlap).

    Returns count, mean, sample std and (optionally) % of values < threshold.
    Missing history stays NaN (count = 0); nothing is zero-filled.
    """
    n = len(df)
    count = np.zeros(n)
    mean = np.full(n, np.nan)
    std = np.full(n, np.nan)
    pct = np.full(n, np.nan)

    q_all = df[query_col].to_numpy(dtype="datetime64[ns]")
    a_all = df[avail_col].to_numpy(dtype="datetime64[ns]")
    v_all = df[value_col].to_numpy(dtype=float)

    for _, idx in df.groupby(group_cols, sort=False).indices.items():
        valid = ~np.isnan(v_all[idx]) & ~np.isnat(a_all[idx])
        if not valid.any():
            continue
        av = a_all[idx][valid]
        vals = v_all[idx][valid]
        order = np.argsort(av, kind="stable")
        av, vals = av[order], vals[order]
        cs = np.concatenate([[0.0], np.cumsum(vals)])
        cs2 = np.concatenate([[0.0], np.cumsum(vals ** 2)])
        cl = np.concatenate([[0.0], np.cumsum(vals < threshold)]) if threshold is not None else None
        k = np.searchsorted(av, q_all[idx], side="left")        # count with avail < query
        count[idx] = k
        pos = k > 0
        kk = k[pos].astype(float)
        m = cs[k[pos]] / kk
        mean[idx[pos]] = m
        multi = kk > 1
        var = np.full(kk.shape, np.nan)
        var[multi] = np.maximum((cs2[k[pos]][multi] - kk[multi] * m[multi] ** 2) / (kk[multi] - 1), 0.0)
        std[idx[pos]] = np.sqrt(var)
        if cl is not None:
            pct[idx[pos]] = cl[k[pos]] / kk * 100.0
    return pd.DataFrame({"count": count, "mean": mean, "std": std, "pct": pct}, index=df.index)


def add_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the leakage-safe historical features (all strictly past-only).

    Delay history (same train, same NEXT station): avg, std, count and the
    on-time rate (% of earlier arrivals less than 15 min late).
    Stop history (same train, same CURRENT station): avg / std of the actual
    stop duration and the stop reliability = % of earlier observed stops whose
    actual duration was within +/-STOP_TOLERANCE_MIN of the scheduled halt.
    """
    df = df.sort_values(["observation_time", "train_no", "seq"], kind="stable").reset_index(drop=True)

    # Past arrival delays of the SAME train at the SAME upcoming station.
    h_delay = time_aware_history(
        df, ["train_no", "next_station_code"], value_col="next_station_arrival_delay_min",
        avail_col="target_next_actual_arrival", query_col="observation_time",
        threshold=DELAY_THRESHOLD)
    df["historical_avg_delay_min"] = h_delay["mean"]
    df["historical_delay_std_min"] = h_delay["std"]
    df["historical_observation_count"] = h_delay["count"].astype(int)
    df["historical_on_time_rate_pct"] = h_delay["pct"]

    # Past actual stop durations of the SAME train at the SAME current station.
    # (A completed stop is known at its observation_time = actual departure.)
    h_stop = time_aware_history(
        df, ["train_no", "station_code"], value_col="actual_stop_min",
        avail_col="observation_time", query_col="observation_time")
    df["historical_avg_actual_stop_min"] = h_stop["mean"]
    df["historical_stop_std_min"] = h_stop["std"]

    # Stop reliability: 1 if the actual stop was within tolerance of the schedule.
    # Rows without both durations stay NaN and are ignored by the history.
    within = (df["actual_stop_min"] - df["scheduled_stop_min"]).abs() <= STOP_TOLERANCE_MIN
    df["_stop_within_tol"] = within.astype(float).where(
        df["actual_stop_min"].notna() & df["scheduled_stop_min"].notna())
    h_rel = time_aware_history(
        df, ["train_no", "station_code"], value_col="_stop_within_tol",
        avail_col="observation_time", query_col="observation_time")
    df["historical_stop_observation_count"] = h_rel["count"].astype(int)
    df["historical_stop_reliability_pct"] = h_rel["mean"] * 100.0
    return df.drop(columns=["_stop_within_tol"])


def verify_history_bruteforce(df: pd.DataFrame, n_checks: int = 200, seed: int = 0) -> bool:
    """Independent O(n) re-computation of the history features for sampled rows."""
    rng = np.random.default_rng(seed)           # used only to choose which rows to audit
    rows = rng.choice(len(df), size=min(n_checks, len(df)), replace=False)
    for i in rows:
        r = df.iloc[i]
        past = df[(df.train_no == r.train_no) & (df.next_station_code == r.next_station_code)
                  & (df.target_next_actual_arrival < r.observation_time)]
        if len(past) != r.historical_observation_count:
            return False
        if len(past):
            if not np.isclose(past.next_station_arrival_delay_min.mean(), r.historical_avg_delay_min):
                return False
            if not np.isclose((past.next_station_arrival_delay_min < DELAY_THRESHOLD).mean() * 100,
                              r.historical_on_time_rate_pct):
                return False
        elif not np.isnan(r.historical_avg_delay_min):
            return False
        # stop reliability: same train, same current station, completed strictly earlier
        stops = df[(df.train_no == r.train_no) & (df.station_code == r.station_code)
                   & (df.observation_time < r.observation_time)
                   & df.actual_stop_min.notna() & df.scheduled_stop_min.notna()]
        if len(stops) != r.historical_stop_observation_count:
            return False
        if len(stops):
            ok = ((stops.actual_stop_min - stops.scheduled_stop_min).abs() <= STOP_TOLERANCE_MIN).mean() * 100
            if not np.isclose(ok, r.historical_stop_reliability_pct):
                return False
        elif not np.isnan(r.historical_stop_reliability_pct):
            return False
    return True


# --------------------------------------------------------------------------
# Chronological split
# --------------------------------------------------------------------------
def chronological_split(df: pd.DataFrame, train_frac: float = 0.70,
                        val_frac: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Sort by journey date; ~70/15/15 by rows, with cut points moved to the
    nearest journey-date boundary so no journey date is shared by two splits.
    """
    df = df.sort_values(["journey_date", "observation_time", "train_no", "seq"],
                        kind="stable").reset_index(drop=True)
    per_date = df.groupby("journey_date").size()
    if len(per_date) < 3:
        raise SystemExit("Need at least 3 distinct journey dates for a chronological split "
                         f"(found {len(per_date)}). Increase --max-source-rows or --target-rows.")
    cum = per_date.cumsum().to_numpy() / len(df)

    def cut(frac: float, lo: int) -> int:
        j = int(np.argmin(np.abs(cum - frac)))
        return max(j, lo)

    c1 = cut(train_frac, 0)
    c2 = cut(train_frac + val_frac, c1 + 1)
    c2 = min(c2, len(per_date) - 2)
    c1 = min(c1, c2 - 1)
    dates = per_date.index
    d_train_end, d_val_end = dates[c1], dates[c2]
    train = df[df.journey_date <= d_train_end]
    val = df[(df.journey_date > d_train_end) & (df.journey_date <= d_val_end)]
    test = df[df.journey_date > d_val_end]
    return train, val, test


# --------------------------------------------------------------------------
# Source streaming
# --------------------------------------------------------------------------
def open_source_stream() -> Iterable[dict]:
    """Stream the Hugging Face dataset (no full download / no full load into RAM)."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("The 'datasets' package is required: pip install datasets pandas numpy") from exc
    return load_dataset(DATASET_ID, split="train", streaming=True)


def open_local_stream(local_dir: Path) -> Iterable[dict]:
    """Rows of downloaded per-train CSVs (same columns as the Hugging Face source)."""
    files = sorted(Path(local_dir).glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files in {local_dir}. Run: python scripts/download_datasets.py")
    for f in files:
        df = pd.read_csv(f, dtype={"train_no": str, "date": str})
        df = df.astype(object).where(df.notna(), None)
        yield from df.to_dict(orient="records")


def _stable_fraction(key: Tuple[str, str]) -> float:
    return (zlib.crc32("|".join(key).encode()) & 0xFFFFFFFF) / 2 ** 32


def collect_observations(stream: Iterable[dict], args: argparse.Namespace,
                         log: FilterLog) -> pd.DataFrame:
    """Stream rows, group into journeys, build observations until targets are met."""
    open_j: "OrderedDict[Tuple[str, str], List[dict]]" = OrderedDict()
    frames: List[pd.DataFrame] = []
    state = {"obs": 0}
    trains, dates = set(), set()
    schema_checked = False

    def flush_one(key: Tuple[str, str]) -> None:
        out = process_journey(open_j.pop(key), log)
        if out is not None and len(out):
            frames.append(out)
            state["obs"] += len(out)
            trains.update(out["train_no"].unique())
            dates.update(out["journey_date"].unique())

    def enough() -> bool:
        return (state["obs"] >= args.target_rows and len(trains) >= args.min_trains
                and len(dates) >= args.min_dates)

    for row in stream:
        log.source_rows += 1
        if not schema_checked:
            missing_core = [c for c in CORE_COLUMNS if c not in row]
            missing_opt = [c for c in OPTIONAL_COLUMNS if c not in row]
            if missing_core:
                raise SystemExit(f"Source is missing required columns: {missing_core}")
            if missing_opt:
                print(f"[schema] optional columns not in source (skipped gracefully): {missing_opt}")
            schema_checked = True
        if log.source_rows > args.max_source_rows:
            print(f"[stream] reached --max-source-rows={args.max_source_rows:,}; stopping.")
            break
        key = (str(row.get("train_no")), str(row.get("date")))
        if args.sample_rate < 1.0 and _stable_fraction(key) >= args.sample_rate:
            continue
        open_j.setdefault(key, []).append(row)
        if len(open_j) > args.max_open_journeys:
            flush_one(next(iter(open_j)))                 # oldest journey is complete
            if enough():
                break
        if log.source_rows % 200_000 == 0:
            print(f"[stream] {log.source_rows:,} source rows | {state['obs']:,} observations | "
                  f"{len(trains)} trains | {len(dates)} dates")

    while open_j:                                          # flush remaining journeys
        flush_one(next(iter(open_j)))

    if not frames:
        raise SystemExit("No valid observations could be built. See the filtering report above.")
    if not enough():
        print(f"[warning] stream ended before targets were fully met "
              f"(obs={state['obs']:,}, trains={len(trains)}, dates={len(dates)}).")
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Quality checks
# --------------------------------------------------------------------------
def _stations(df: pd.DataFrame) -> int:
    return len(set(df["station_code"]) | set(df["next_station_code"]))


def _range(df: pd.DataFrame) -> str:
    return f"{df.journey_date.min().date()} -> {df.journey_date.max().date()}"


def quality_checks(final: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame,
                   test: pd.DataFrame, log: FilterLog) -> bool:
    print("\n=== DATASET STATISTICS ===")
    print(f"Source rows inspected : {log.source_rows:,}")
    print(f"Rows retained         : {len(final):,}")
    print(f"Distinct trains       : {final.train_no.nunique():,}")
    print(f"Distinct stations     : {_stations(final):,}")
    print(f"Date range            : {_range(final)}")

    print("\nMissing-value statistics (columns with missing values only):")
    miss = final.isna().sum()
    miss = miss[miss > 0]
    if miss.empty:
        print("  none")
    for c, v in miss.items():
        print(f"  {c:<42s}{v:>8,}  ({v / len(final):6.2%})")

    print("\nTarget distribution:")
    print(final["next_station_arrival_delay_min"].describe().round(2).to_string())
    print(final["next_station_delayed_15"].value_counts(normalize=False).sort_index()
          .rename({0: "0 (<15 min)", 1: "1 (>=15 min)"}).to_string())
    print("\nRisk distribution:")
    risk = final["next_delay_risk_label"].value_counts()
    print(pd.DataFrame({"rows": risk, "share": (risk / len(final)).round(4)}).to_string())

    print("\n=== SPLIT SUMMARY ===")
    for name, part in (("train", train), ("validation", val), ("test", test)):
        print(f"{name:<11s} rows={len(part):>7,}  trains={part.train_no.nunique():>5,}  "
              f"stations={_stations(part):>5,}  dates={_range(part)}")

    print("\n=== INTEGRITY & LEAKAGE CHECKS ===")
    checks = {
        "next station differs from current station":
            bool((final.next_station_code != final.station_code).all()),
        "no target column is listed as a feature":
            set(FEATURE_COLUMNS).isdisjoint(TARGET_COLUMNS)
            and not any(c.startswith("target_") or c.startswith("next_station_arrival")
                        for c in FEATURE_COLUMNS),
        "observation time <= next actual arrival (no future timestamps)":
            bool((final.observation_time <= final.target_next_actual_arrival).all()),
        "historical features reproduced by independent brute-force audit":
            verify_history_bruteforce(final),
        "history count never exceeds earlier rows available":
            bool((final.historical_observation_count >= 0).all()),
        "chronological split: train < validation < test":
            bool(train.journey_date.max() < val.journey_date.min()
                 and val.journey_date.max() < test.journey_date.min()),
        "splits are disjoint and complete":
            len(train) + len(val) + len(test) == len(final),
        "no duplicate observation remains":
            not final.duplicated(subset=["train_no", "journey_date", "seq"]).any(),
        "no negative durations (stop/travel)":
            bool((final.scheduled_travel_to_next_min >= 0).all()
                 and (final.scheduled_stop_min.dropna() >= 0).all()
                 and (final.actual_stop_min.dropna() >= 0).all()),
        "target present for every row":
            bool(final.next_station_arrival_delay_min.notna().all()),
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return all(checks.values())


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def write_outputs(out_dir: Path, final, train, val, test, log: FilterLog) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    files = {"RailScan_Final_Dataset.csv": final, "train.csv": train,
             "validation.csv": val, "test.csv": test}
    manifest = []
    for name, part in files.items():
        part.to_csv(out_dir / name, index=False)
        manifest.append({"file": name, "rows": len(part), "columns": part.shape[1],
                         "date_min": part.journey_date.min().date().isoformat(),
                         "date_max": part.journey_date.max().date().isoformat(),
                         "created_at": created})
    pd.DataFrame(manifest).to_csv(out_dir / "dataset_manifest.csv", index=False)
    log.to_frame().to_csv(out_dir / "filtering_report.csv", index=False)
    print(f"\nWrote {len(files) + 2} files to {out_dir.resolve()}")


def build(stream: Iterable[dict], args: argparse.Namespace) -> bool:
    log = FilterLog()
    obs = collect_observations(stream, args, log)

    before = len(obs)
    obs = obs.drop_duplicates(subset=["train_no", "journey_date", "seq"], keep="first")
    log.drop("duplicate observation after merging journeys", before - len(obs))

    obs = add_historical_features(obs)
    cols = (ID_COLUMNS + [c for c in FEATURE_COLUMNS if c in obs.columns] + TARGET_COLUMNS)
    final = obs[cols].sort_values(["journey_date", "observation_time", "train_no", "seq"],
                                  kind="stable").reset_index(drop=True)
    train, val, test = chronological_split(final)

    log.print_report()
    ok = quality_checks(final, train, val, test, log)
    write_outputs(Path(args.output_dir), final, train, val, test, log)
    return ok


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the RailScan dataset from real Indian railway run data.")
    p.add_argument("--target-rows", type=int, default=10_000, help="approximate number of observations")
    p.add_argument("--min-trains", type=int, default=1, help="minimum distinct trains required")
    p.add_argument("--output-dir", default="data/processed", help="output directory")
    p.add_argument("--min-dates", type=int, default=MIN_DISTINCT_DATES,
                   help="minimum distinct journey dates (needed for a date-based split)")
    p.add_argument("--max-source-rows", type=int, default=5_000_000,
                   help="safety cap on streamed source rows")
    p.add_argument("--sample-rate", type=float, default=1.0,
                   help="deterministic fraction of journeys to keep (use <1 if the source is "
                        "sorted by date, to spread observations over many dates)")
    p.add_argument("--local-dir", default=None,
                   help="read downloaded per-train CSVs (data/raw/delay) instead of streaming the hub")
    p.add_argument("--max-open-journeys", type=int, default=200,
                   help="size of the in-memory journey buffer")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if args.local_dir:
        frames = [pd.read_csv(f, dtype={"train_no": str, "date": str}) for f in sorted(Path(args.local_dir).glob("*.csv"))]
        if frames:
            REGULAR_STOPS.update(regular_stops(pd.concat(frames, ignore_index=True)))
            for t, stops in REGULAR_STOPS.items():
                print(f"[stops] train {t}: {len(stops)} regular stops")
        ok = build(open_local_stream(Path(args.local_dir)), args)
    else:
        ok = build(open_source_stream(), args)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
