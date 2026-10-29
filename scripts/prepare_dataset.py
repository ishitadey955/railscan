#!/usr/bin/env python3
"""
RailScan - reduce the processed dataset to a fixed number of REAL records and
write the master CSV, splits, summary and source notes.

Input   : data/interim/RailScan_Full_Dataset.csv  (11,697 real observations
          written by scripts/build_dataset.py from the Hugging Face dataset
          yadavsaurabh/indian-railways-runs)
Outputs : data/processed/RailScan_Final_Dataset.csv   (model-ready, N rows)
          data/processed/RailScan_Master_Dataset.csv  (same rows + names,
                                                       derived times, stop category)
          data/processed/{train,validation,test}.csv
          data/processed/dataset_manifest.csv, filtering_report.csv
          data/processed/dataset_summary.json

How rows are selected (no row is invented or modified)
-------------------------------------------------------
* The unit of selection is a whole journey (train_no, journey_date), so a
  journey is either fully in or fully out.
* Journeys are sorted by date and visited in a low-discrepancy (golden-ratio)
  order, so any prefix of that order is spread evenly over the whole
  2023-11 .. 2026-08 period and over both trains.
* Journeys are added while they fit into the target; afterwards smaller
  journeys fill the remaining gap so the result has exactly --target-rows
  rows whenever that is possible.

After selection, every historical_* feature is RECOMPUTED inside the reduced
sample with the builder's own time-aware, past-only code, because the old
values referred to journeys that are no longer in the dataset.

Usage:  python scripts/prepare_dataset.py --target-rows 1500
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_dataset as bd  # noqa: E402

sys.path.insert(0, str(ROOT))
from src.stop_analysis import stop_category  # noqa: E402  (one definition for the app and the dataset)

GOLDEN = (np.sqrt(5) - 1) / 2
# Stop categories use the SCHEDULED halt (documented rule).
OLD_HISTORY = ["historical_avg_delay_min", "historical_delay_std_min", "historical_observation_count",
               "historical_avg_actual_stop_min", "historical_stop_reliability_pct"]
DATETIME_COLS = ["observation_time", "current_scheduled_arrival", "current_scheduled_departure",
                 "next_scheduled_arrival", "expected_next_station_arrival_baseline",
                 "target_next_actual_arrival"]


# ------------------------------------------------------------------ selection
def select_journeys(df: pd.DataFrame, target: int) -> pd.DataFrame:
    """Pick whole journeys, evenly spread over time, totalling `target` rows."""
    sizes = (df.groupby(["journey_date", "train_no"]).size()
             .reset_index(name="n").sort_values(["journey_date", "train_no"]).reset_index(drop=True))
    if sizes.n.sum() <= target:
        return df.copy()
    # golden-ratio order: position i gets priority frac(i * 0.618...)
    order = np.argsort(np.mod(np.arange(len(sizes)) * GOLDEN, 1.0), kind="stable")
    chosen, total = [], 0
    for i in order:                                  # first pass: fill in even order
        n = int(sizes.n.iat[i])
        if total + n <= target:
            chosen.append(i)
            total += n
        if total == target:
            break
    if total < target:                               # second pass: close the gap exactly
        rest = [i for i in order if i not in set(chosen)]
        for i in rest:
            n = int(sizes.n.iat[i])
            if total + n <= target:
                chosen.append(i)
                total += n
            if total == target:
                break
    keys = sizes.iloc[sorted(chosen)][["journey_date", "train_no"]]
    return df.merge(keys, on=["journey_date", "train_no"], how="inner")


# ------------------------------------------------------------------ master CSV
def load_station_names() -> dict:
    names = {}
    for f in ("station_master.csv", "station_master_supplement.csv"):
        p = ROOT / "data" / "stations" / f
        if p.exists():
            s = pd.read_csv(p, dtype=str)
            names.update(dict(zip(s.station_code, s.station_name)))
    return names


def build_master(final: pd.DataFrame) -> pd.DataFrame:
    """Final dataset + readable names and documented derived columns (spec naming)."""
    names = load_station_names()
    m = final.copy()
    m.insert(m.columns.get_loc("station_code") + 1, "station_name", m.station_code.map(names))
    m.insert(m.columns.get_loc("next_station_code") + 1, "next_station_name",
             m.next_station_code.map(names))
    m["observation_date"] = m.observation_time.dt.date.astype(str)
    m["scheduled_halt_min"] = m.scheduled_stop_min
    m["stop_category"] = m.scheduled_halt_min.map(stop_category)   # Short < 5, Moderate 5-10, Long > 10
    # actual times are derived: scheduled time + recorded delay (source stores both,
    # the processed dataset keeps the delay); NaN when the delay is missing.
    m["actual_arrival"] = m.current_scheduled_arrival + pd.to_timedelta(m.current_arrival_delay_min, unit="m")
    m["actual_departure"] = m.current_scheduled_departure + pd.to_timedelta(m.current_departure_delay_min, unit="m")
    m["arrival_delay_min"] = m.current_arrival_delay_min
    m["departure_delay_min"] = m.current_departure_delay_min
    m["current_delay_min"] = m.current_observed_delay_min
    return m


# ------------------------------------------------------------------ checks
def raw_cross_check(final: pd.DataFrame) -> dict:
    """Compare delays/schedule against the raw per-train CSVs, if they were downloaded."""
    raw_dir = ROOT / "data" / "raw" / "delay"
    frames = [pd.read_csv(p, dtype={"train_no": str}) for p in sorted(raw_dir.glob("*.csv"))]
    if not frames:
        return {"status": "skipped (data/raw/delay/*.csv not found)"}
    raw = pd.concat(frames, ignore_index=True)
    raw["seq"] = pd.to_numeric(raw.seq, errors="coerce")
    raw = raw.drop_duplicates(["train_no", "date", "seq"], keep="first")
    f = final.assign(train_no=final.train_no.astype(str).str.zfill(5),
                     date=final.journey_date.dt.strftime("%Y-%m-%d"))
    j = f.merge(raw, on=["train_no", "date", "seq", "station_code"], how="left", indicator=True)
    matched = j["_merge"] == "both"
    sch = pd.to_datetime(j.current_scheduled_arrival).dt.strftime("%H:%M")
    sch_ok = (sch == j.sch_arrival) | j.sch_arrival.isna()
    # where the source delay field was valid the builder kept it; elsewhere it recomputed
    # from timestamps, so compare only rows that have a source value
    has = j.delay_arrival_min.notna() & j.current_arrival_delay_min.notna()
    delay_ok = (j.current_arrival_delay_min[has] - j.delay_arrival_min[has]).abs() <= 1
    return {"status": "checked",
            "rows_matched_to_raw": int(matched.sum()), "rows_total": int(len(j)),
            "scheduled_arrival_matches_raw_pct": round(float(sch_ok[matched].mean() * 100), 2),
            "arrival_delay_within_1min_of_raw_pct": round(float(delay_ok.mean() * 100), 2),
            "rows_compared_for_delay": int(has.sum())}


def consistency_checks(m: pd.DataFrame) -> dict:
    """Recompute derived values from their definitions and count disagreements."""
    def mins(a, b):
        return (a - b).dt.total_seconds() / 60
    halt = mins(m.current_scheduled_departure, m.current_scheduled_arrival)
    travel = mins(m.next_scheduled_arrival, m.current_scheduled_departure)
    target = mins(m.target_next_actual_arrival, m.next_scheduled_arrival)
    stop = mins(m.actual_departure, m.actual_arrival)
    ok = lambda a, b: int(((a - b).abs() > 0.01).sum())
    both = m.actual_stop_min.notna() & stop.notna()
    return {
        "scheduled_halt_mismatches": ok(halt.dropna(), m.scheduled_stop_min[halt.notna()]),
        "scheduled_travel_mismatches": ok(travel, m.scheduled_travel_to_next_min),
        "next_arrival_delay_mismatches": ok(target, m.next_station_arrival_delay_min),
        "actual_stop_mismatches": ok(stop[both], m.actual_stop_min[both]),
        "delayed_15_label_mismatches": int(((m.next_station_arrival_delay_min >= 15).astype(int)
                                            != m.next_station_delayed_15).sum()),
        "negative_scheduled_halts": int((m.scheduled_stop_min < 0).sum()),
        "negative_scheduled_travel": int((m.scheduled_travel_to_next_min < 0).sum()),
    }


def summarize(final, master, train, val, test, selection, checks, raw_check) -> dict:
    stations = sorted(set(final.station_code) | set(final.next_station_code))
    miss = (master.isna().mean() * 100).round(2)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"dataset": bd.DATASET_ID, "license": "GPL-3.0",
                   "url": f"https://huggingface.co/datasets/{bd.DATASET_ID}"},
        "selection": selection,
        "rows": len(final), "columns_final": final.shape[1], "columns_master": master.shape[1],
        "journeys": int(final.groupby(["train_no", "journey_date"]).ngroups),
        "unique_trains": int(final.train_no.nunique()),
        "trains": sorted(final.train_no.astype(str).str.zfill(5).unique().tolist()),
        "unique_stations": len(stations), "station_codes": stations,
        "date_range": {"min": str(final.journey_date.min().date()), "max": str(final.journey_date.max().date())},
        "timezone": "Asia/Kolkata (IST, UTC+05:30); timestamps are naive local times",
        "splits": {n: {"rows": len(d), "min": str(d.journey_date.min().date()),
                       "max": str(d.journey_date.max().date())}
                   for n, d in (("train", train), ("validation", val), ("test", test))},
        "duplicate_keys_train_date_seq": int(final.duplicated(["train_no", "journey_date", "seq"]).sum()),
        "missing_pct_master": {k: float(v) for k, v in miss.items() if v > 0},
        "target_next_station_delayed_15": {str(k): int(v) for k, v in
                                           final.next_station_delayed_15.value_counts().sort_index().items()},
        "target_risk_label": {k: int(v) for k, v in final.next_delay_risk_label.value_counts().items()},
        "next_station_arrival_delay_min": {k: round(float(v), 2) for k, v in
                                           final.next_station_arrival_delay_min.describe().items()},
        "stop_category_scheduled": {k: int(v) for k, v in master.stop_category.value_counts().items()},
        "definitions": {
            "next_station_delayed_15": "1 if next_station_arrival_delay_min >= 15 else 0",
            "next_delay_risk_label": "Low < 15 min, Medium 15-59 min, High >= 60 min (next-station arrival delay)",
            "stop_category": "based on SCHEDULED halt: Short < 5, Moderate 5-10 inclusive, Long > 10 min",
            "historical_on_time_rate_pct": "% of EARLIER arrivals of this train at the next station < 15 min late",
            "historical_stop_reliability_pct": (f"% of EARLIER observed stops of this train at this station whose "
                                                f"actual stop was within +/-{bd.STOP_TOLERANCE_MIN} min of the "
                                                "scheduled halt"),
            "history_rule": "a past row counts only if its outcome was known strictly before observation_time",
            "actual_arrival/actual_departure": "derived as scheduled time + recorded delay",
        },
        "validation": {"derived_value_consistency": checks, "raw_source_cross_check": raw_check},
    }


# ------------------------------------------------------------------ main
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--input", default=str(ROOT / "data" / "interim" / "RailScan_Full_Dataset.csv"))
    ap.add_argument("--target-rows", type=int, default=1500)
    ap.add_argument("--output-dir", default=str(ROOT / "data" / "processed"))
    args = ap.parse_args(argv)
    src, out = Path(args.input), Path(args.output_dir)
    if not src.exists():
        raise SystemExit(f"Input not found: {src}\nRun scripts/build_dataset.py --output-dir data/interim "
                         "and rename RailScan_Final_Dataset.csv to RailScan_Full_Dataset.csv.")

    full = pd.read_csv(src, dtype={"train_no": str})
    for c in DATETIME_COLS:
        full[c] = pd.to_datetime(full[c], errors="coerce")
    full["journey_date"] = pd.to_datetime(full["journey_date"])
    print(f"Input: {len(full):,} rows, {full.groupby(['train_no', 'journey_date']).ngroups:,} journeys")

    reduced = select_journeys(full, args.target_rows)
    reduced = reduced.drop(columns=[c for c in OLD_HISTORY if c in reduced.columns])
    reduced = bd.add_historical_features(reduced)
    cols = bd.ID_COLUMNS + [c for c in bd.FEATURE_COLUMNS if c in reduced.columns] + bd.TARGET_COLUMNS
    final = reduced[cols].sort_values(["journey_date", "observation_time", "train_no", "seq"],
                                      kind="stable").reset_index(drop=True)
    train, val, test = bd.chronological_split(final)

    selection = {"method": "whole journeys, golden-ratio even spacing over date-sorted journeys, "
                           "gap filled exactly; history features recomputed inside the sample",
                 "input_file": src.name, "input_rows": len(full),
                 "input_journeys": int(full.groupby(["train_no", "journey_date"]).ngroups),
                 "target_rows": args.target_rows, "selected_rows": len(final)}
    print(f"Selected: {len(final):,} rows from "
          f"{final.groupby(['train_no', 'journey_date']).ngroups} journeys")

    master = build_master(final)
    checks = consistency_checks(master)
    raw_check = raw_cross_check(final)
    log = bd.FilterLog()
    log.source_rows = len(full)
    ok = bd.quality_checks(final, train, val, test, log)
    print("\nDerived-value consistency:", json.dumps(checks))
    print("Raw-source cross-check:", json.dumps(raw_check))
    ok = ok and all(v == 0 for v in checks.values())

    out.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    files = {"RailScan_Final_Dataset.csv": final, "train.csv": train,
             "validation.csv": val, "test.csv": test}
    manifest = []
    for name, part in files.items():
        part.to_csv(out / name, index=False)
        manifest.append({"file": name, "rows": len(part), "columns": part.shape[1],
                         "date_min": part.journey_date.min().date().isoformat(),
                         "date_max": part.journey_date.max().date().isoformat(), "created_at": created})
    master.to_csv(out / "RailScan_Master_Dataset.csv", index=False)
    manifest.append({"file": "RailScan_Master_Dataset.csv", "rows": len(master), "columns": master.shape[1],
                     "date_min": manifest[0]["date_min"], "date_max": manifest[0]["date_max"],
                     "created_at": created})
    pd.DataFrame(manifest).to_csv(out / "dataset_manifest.csv", index=False)

    # keep the builder's filtering log and append the reduction step
    flog = ROOT / "data" / "interim" / "filtering_report_full.csv"
    rep = pd.read_csv(flog) if flog.exists() else pd.DataFrame(columns=["kind", "reason", "count"])
    rep = pd.concat([rep, pd.DataFrame([{
        "kind": "row_dropped",
        "reason": f"size reduction to {args.target_rows} rows (whole journeys, even time spacing)",
        "count": len(full) - len(final)}])], ignore_index=True)
    rep.to_csv(out / "filtering_report.csv", index=False)

    summary = summarize(final, master, train, val, test, selection, checks, raw_check)
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {len(files) + 4} files to {out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
