#!/usr/bin/env python3
"""
RailScan - build the current timetable of each train from the raw running records.

Input : data/raw/delay_north/<train_no>.csv  (scripts/download_northern.py; falls back to data/raw/delay)
Output: data/processed/RailScan_Timetable.csv

The journey planner needs a schedule for ANY date, not only recorded ones.
The source repeats the published schedule on every recorded journey, so the
current timetable is taken from the most recent --window-days of journeys:
for every stop, the most common scheduled arrival / departure (and day offset).
Stops without a sequence number cannot be placed on the route and are left
out (reported). Nothing is invented: a time missing in the source stays empty.

Usage:  python scripts/build_timetable.py [--window-days 90]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def station_names() -> dict:
    names = {}
    for f in ("station_master.csv", "station_master_supplement.csv"):
        p = ROOT / "data" / "stations" / f
        if p.exists():
            s = pd.read_csv(p, dtype=str)
            names.update(dict(zip(s.station_code, s.station_name)))
    return names


def day_offset(date_col: pd.Series, journey: pd.Series) -> pd.Series:
    d = pd.to_datetime(date_col, format="%d %b %Y", errors="coerce")
    return (d - pd.to_datetime(journey)).dt.days


def mode(s: pd.Series):
    s = s.dropna()
    return s.mode().iat[0] if not s.empty else None


def build_train(raw: pd.DataFrame, window_days: int, names: dict) -> pd.DataFrame | None:
    raw = raw.copy()
    raw["seq"] = pd.to_numeric(raw.seq, errors="coerce")
    raw["date"] = pd.to_datetime(raw.date)
    last = raw.date.max()
    recent = raw[raw.date > last - pd.Timedelta(days=window_days)]
    dropped = sorted(recent.loc[recent.seq.isna(), "station_code"].unique())
    recent = recent[recent.seq.notna() & (recent.halts_here == 1)]
    # Some train numbers (mostly specials) changed route inside the window. The current
    # timetable is the route of the LATEST journey: keep journeys whose halts are mostly on it.
    if not recent.empty:
        latest_codes = set(recent.loc[recent.date == recent.date.max(), "station_code"])
        share = recent.groupby("date").station_code.agg(lambda c: c.isin(latest_codes).mean())
        other = share[share < 0.8].index
        if len(other):
            print(f"{str(raw.train_no.iat[0]).zfill(5)}: {len(other)} journey(s) on a different route left out")
        recent = recent[~recent.date.isin(other)]
    recent = recent.assign(arr_off=day_offset(recent.sch_arrival_date, recent.date),
                           dep_off=day_offset(recent.sch_departure_date, recent.date))
    rows = []
    for (code, seq), g in recent.groupby(["station_code", "seq"]):
        rows.append({"station_code": code, "seq": int(seq),
                     "sch_arrival": mode(g.sch_arrival), "arrival_day_offset": mode(g.arr_off),
                     "sch_departure": mode(g.sch_departure), "departure_day_offset": mode(g.dep_off),
                     "distance_km": round(float(g.distance_km.median()), 3) if g.distance_km.notna().any() else None,
                     "journeys_seen": int(g.date.nunique()), "_last_seen": g.date.max(),
                     # where the stop is: median of the coordinates in the running records
                     "latitude": round(float(pd.to_numeric(g.lat, errors="coerce").median()), 6)
                     if pd.to_numeric(g.lat, errors="coerce").notna().any() else None,
                     "longitude": round(float(pd.to_numeric(g.lng, errors="coerce").median()), 6)
                     if pd.to_numeric(g.lng, errors="coerce").notna().any() else None})
    # a station renamed inside the window appears under two codes at the same position:
    # keep the code of the most recent journey
    rows = (pd.DataFrame(rows).sort_values(["seq", "_last_seen", "journeys_seen"])
            .drop_duplicates("seq", keep="last").drop(columns="_last_seen").to_dict("records")) if rows else []
    if len(rows) < 2:                       # nothing to plan with
        print(f"{str(raw.train_no.iat[0]).zfill(5)}: fewer than 2 scheduled stops with a sequence number, left out")
        return None
    tt = pd.DataFrame(rows).sort_values("seq").reset_index(drop=True)
    # distance to the next stop: median of each journey's own difference, as in the training data
    seg = recent.sort_values(["date", "seq"])
    seg = seg.assign(next_code=seg.groupby("date").station_code.shift(-1),
                     leg=seg.groupby("date").distance_km.shift(-1) - seg.distance_km)
    nxt = tt.station_code.shift(-1)
    tt["distance_to_next_km"] = [
        round(float(seg[(seg.station_code == a) & (seg.next_code == b)].leg.median()), 3)
        if b is not None and not pd.isna(b) and seg[(seg.station_code == a) & (seg.next_code == b)].leg.notna().any()
        else None
        for a, b in zip(tt.station_code, nxt)]
    # the source repeats the origin's departure as its arrival and the destination's
    # arrival as its departure; blank them so the first/last stop has no fake halt
    tt.loc[0, ["sch_arrival", "arrival_day_offset"]] = [None, None]
    tt.loc[len(tt) - 1, ["sch_departure", "departure_day_offset"]] = [None, None]
    train_no = str(raw.train_no.iat[0]).zfill(5)
    runs = sorted(recent.date.dt.day_of_week.unique())
    latest = recent[recent.date == recent.date.max()].iloc[0]
    tt.insert(0, "train_no", train_no)
    tt.insert(1, "train_name", latest.train_name)
    tt.insert(2, "train_type", latest.train_type)
    tt.insert(5, "station_name", tt.station_code.map(names))
    tt["runs_on"] = ",".join(DAY_NAMES[d] for d in runs)
    tt["timetable_window"] = f"{recent.date.min().date()} to {last.date()}"
    tt["journeys_in_window"] = int(recent.date.nunique())
    print(f"{train_no} {latest.train_name}: {len(tt)} stops, {tt.journeys_in_window.iat[0]} journeys in window, "
          f"runs on {tt.runs_on.iat[0]}; stops without a sequence number left out: {dropped or 'none'}")
    return tt


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Build RailScan_Timetable.csv from the raw running records.")
    ap.add_argument("--window-days", type=int, default=90)
    ap.add_argument("--raw-dir", default=None, help="default: data/raw/delay_north, else data/raw/delay")
    args = ap.parse_args(argv)
    north = ROOT / "data" / "raw" / "delay_north"
    raw_dir = Path(args.raw_dir) if args.raw_dir else (north if any(north.glob("*.csv")) else ROOT / "data" / "raw" / "delay")
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"No raw files in {raw_dir}. Run: python scripts/download_northern.py")
    names = station_names()
    parts = []
    for f in files:
        raw = pd.read_csv(f, dtype={"train_no": str})
        if not raw.empty:
            parts.append(build_train(raw, args.window_days, names))
    tt = pd.concat([t for t in parts if t is not None], ignore_index=True)
    out = ROOT / "data" / "processed" / "RailScan_Timetable.csv"
    tt.to_csv(out, index=False)
    print(f"Wrote {out} ({len(tt)} rows)")


if __name__ == "__main__":
    sys.exit(main())
