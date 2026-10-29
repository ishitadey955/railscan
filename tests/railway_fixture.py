"""SYNTHETIC test fixture in the schema of data/processed/RailScan_Final_Dataset.csv.

!!! NOT REAL RAILWAY DATA !!!  Delays, times and train names are generated
so the code paths of src/railway can be tested (history, missing values,
ambiguity, last stop ...). Train names carry the word FIXTURE. Never use
this for evaluation or report results. The real file is produced by
scripts/build_dataset.py from the Hugging Face dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FIXTURE_BANNER = "SYNTHETIC FIXTURE - not real railway data (test/demo of the code path only)"
DELAY_THRESHOLD = 15


def _journey(train_no, name, ttype, date, stops, delay_fn):
    """stops: [(station_code, sched_arr_hhmm|None, sched_dep_hhmm|None, km)]"""
    base = pd.Timestamp(date)
    rows = []
    arr_d, dep_d = [], []
    for i, (code, sa, sd, km) in enumerate(stops):
        a = None if sa is None else delay_fn(date, i)
        dd = None if sd is None else (a if a is not None else delay_fn(date, i)) + 2
        arr_d.append(a)
        dep_d.append(dd)
    # scheduled timestamps with midnight roll-over: a clock time earlier than the previous one is the next day
    sched_ts, prev, day = [], None, 0
    for _, sa, sd, _km in stops:
        pair = []
        for hhmm in (sa, sd):
            if hhmm is None:
                pair.append(None)
                continue
            mins = int(hhmm[:2]) * 60 + int(hhmm[2:])
            if prev is not None and mins < prev:
                day += 1
            prev = mins
            pair.append(base + pd.Timedelta(days=day, minutes=mins))
        sched_ts.append(tuple(pair))
    for i in range(len(stops) - 1):
        code, sa, sd, km = stops[i]
        ncode, nsa, nsd, nkm = stops[i + 1]
        ts = lambda hhmm: None if hhmm is None else base + pd.Timedelta(hours=int(hhmm[:2]), minutes=int(hhmm[2:]))
        sched_arr, sched_dep, next_arr = (sched_ts[i][0], sched_ts[i][1], sched_ts[i + 1][0])
        cur_delay = dep_d[i] if dep_d[i] is not None else arr_d[i]
        obs = (sched_dep if dep_d[i] is not None else sched_arr) + pd.Timedelta(minutes=cur_delay)
        nd = arr_d[i + 1]
        rows.append(dict(
            train_no=train_no, train_name=name, train_type=ttype, journey_date=base, station_code=code, seq=i + 1,
            next_station_code=ncode, next_station_seq=i + 2, observation_time=obs,
            current_scheduled_arrival=sched_arr, current_scheduled_departure=sched_dep,
            next_scheduled_arrival=next_arr, distance_to_next_km=nkm - km,
            scheduled_travel_to_next_min=(next_arr - sched_dep).total_seconds() / 60 if next_arr is not None and sched_dep is not None else np.nan,
            station_lat=np.nan, station_lng=np.nan,
            scheduled_stop_min=(sched_dep - sched_arr).total_seconds() / 60 if sched_arr is not None and sched_dep is not None else np.nan,
            actual_stop_min=np.nan, stop_variation_min=np.nan,
            day_of_week=base.day_of_week, month=base.month, day=base.day, hour=(sched_dep or sched_arr).hour,
            is_weekend=int(base.day_of_week >= 5),
            current_arrival_delay_min=np.nan if arr_d[i] is None else arr_d[i],
            current_departure_delay_min=np.nan if dep_d[i] is None else dep_d[i],
            current_observed_delay_min=cur_delay,
            next_station_arrival_delay_min=nd,
            target_next_actual_arrival=next_arr + pd.Timedelta(minutes=nd),
        ))
    return rows


def _add_history(df: pd.DataFrame) -> pd.DataFrame:
    """Time-aware history: only EARLIER journeys of the same train (and next station) count."""
    df = df.sort_values(["train_no", "journey_date", "seq"]).reset_index(drop=True)
    cols = ["historical_avg_delay_min", "historical_delay_std_min", "historical_observation_count",
            "historical_avg_actual_stop_min", "historical_on_time_rate_pct", "historical_stop_std_min",
            "historical_stop_observation_count", "historical_stop_reliability_pct"]
    for c in cols:
        df[c] = np.nan
    df["historical_observation_count"] = 0
    df["historical_stop_observation_count"] = 0
    for idx, r in df.iterrows():
        stops = df[(df.train_no == r.train_no) & (df.station_code == r.station_code)
                   & (df.observation_time < r.observation_time)
                   & df.actual_stop_min.notna() & df.scheduled_stop_min.notna()]
        df.at[idx, "historical_stop_observation_count"] = len(stops)
        if len(stops):
            df.at[idx, "historical_avg_actual_stop_min"] = stops.actual_stop_min.mean()
            df.at[idx, "historical_stop_std_min"] = stops.actual_stop_min.std(ddof=1)
            df.at[idx, "historical_stop_reliability_pct"] = (
                (stops.actual_stop_min - stops.scheduled_stop_min).abs() <= 2).mean() * 100
        past = df[(df.train_no == r.train_no) & (df.next_station_code == r.next_station_code)
                  & (df.target_next_actual_arrival < r.observation_time)]
        n = len(past)
        df.at[idx, "historical_observation_count"] = n
        if n:
            d = past.next_station_arrival_delay_min
            df.at[idx, "historical_avg_delay_min"] = d.mean()
            df.at[idx, "historical_delay_std_min"] = d.std(ddof=0)
            df.at[idx, "historical_on_time_rate_pct"] = (d < DELAY_THRESHOLD).mean() * 100
    df["expected_next_station_arrival_baseline"] = df.next_scheduled_arrival + pd.to_timedelta(
        df.current_observed_delay_min, unit="m")
    df["next_station_delayed_15"] = (df.next_station_arrival_delay_min >= DELAY_THRESHOLD).astype(int)
    df["next_delay_risk_label"] = np.where(df.next_station_arrival_delay_min >= 60, "High",
                                           np.where(df.next_station_arrival_delay_min >= 15, "Medium", "Low"))
    return df


def build_fixture() -> pd.DataFrame:
    rows = []
    dates = [f"2025-03-0{d}" for d in range(1, 9)]

    # 12301: NDLS -> CNB -> ALD -> GAYA -> DHN -> HWH, 8 days  (history grows day by day;
    # HWH is the last stop, so it never has a state row of its own)
    route_a = [("NDLS", None, "1630", 0), ("CNB", "2150", "2155", 440), ("ALD", "0020", "0025", 630),
               ("GAYA", "0400", "0403", 990), ("DHN", "0545", "0548", 1100), ("HWH", "0955", None, 1450)]
    for k, d in enumerate(dates):
        rows += _journey("12301", "FIXTURE RAJDHANI EXPRESS", "Rajdhani", d, route_a,
                         lambda dt, i, k=k: [-2, 3, 6, 12, 18, 25][i] + (k * 7) % 31 - 9)

    # 22222: same number, two different names on different dates -> ambiguity
    route_b = [("NDLS", None, "0700", 0), ("CNB", "1300", "1305", 440), ("ALD", "1530", None, 630)]
    for k, d in enumerate(dates[:3]):
        rows += _journey("22222", "FIXTURE ALPHA EXPRESS", "Superfast", d, route_b, lambda dt, i, k=k: 5 + i * 4 + k)
    for k, d in enumerate(dates[3:5]):
        rows += _journey("22222", "FIXTURE BETA EXPRESS", "Express", d, route_b, lambda dt, i, k=k: 10 + i * 3 + k)

    # 33333: one single journey -> insufficient history
    route_c = [("CNB", None, "0910", 0), ("ALD", "1130", "1135", 190), ("GAYA", "1600", None, 540)]
    rows += _journey("33333", "FIXTURE PASSENGER", "Passenger", "2025-03-04", route_c, lambda dt, i: 20 + 5 * i)

    # 44444: calls at CNB twice on the same day -> ambiguous stop
    route_d = [("CNB", None, "0600", 0), ("ALD", "0830", "0835", 190), ("CNB", "1100", "1105", 380),
               ("GAYA", "1500", None, 700)]
    rows += _journey("44444", "FIXTURE LOOP SPECIAL", "Special", "2025-03-05", route_d, lambda dt, i: 3 * i)

    return _add_history(pd.DataFrame(rows))
