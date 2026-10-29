#!/usr/bin/env python3
"""
RailScan - build the northern-India delay dataset.

Input : data/raw/delay_north/<train_no>.csv   (scripts/download_northern.py)
        data/interim/northern/station_states.csv
Output: data/processed/ - the same files as scripts/prepare_dataset.py
        (RailScan_Final_Dataset, RailScan_Master_Dataset, train/validation/test,
        dataset_manifest, filtering_report, dataset_summary.json) and
        RailScan_Final_Dataset.parquet (the same rows, compact, for GitHub)

Every journey goes through the same code as before (build_dataset.process_journey:
next REGULAR stop, no invented values, documented drops). Then only observations
whose CURRENT station is in northern India are kept: a passenger in the north
asks about the next stop, which may lie just across the border.

If more than --max-rows observations remain, the OLDEST journey dates are dropped
(whole dates) until the dataset fits. History features are computed afterwards,
inside the kept rows, with the builder's past-only code.

Usage:  python scripts/build_northern_dataset.py --max-rows 950000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_dataset as bd  # noqa: E402
import prepare_dataset as pdp  # noqa: E402
import select_northern_trains as sel  # noqa: E402

ROOT = sel.ROOT
sys.path.insert(0, str(ROOT))
RAW = ROOT / "data" / "raw" / "delay_north"


def northern_stations() -> set:
    st = pd.read_csv(sel.OUT / "station_states.csv", dtype=str)
    return set(st.loc[st.state.isin(sel.COVERED_STATES), "station_code"])


def process_train(path: str, north: set) -> tuple[pd.DataFrame | None, dict, dict]:
    """All observations of one train file whose current station is northern."""
    raw = pd.read_csv(path, dtype=str)
    log = bd.FilterLog()
    log.source_rows = len(raw)
    if raw.empty:
        return None, {}, {}
    bd.REGULAR_STOPS.clear()
    bd.REGULAR_STOPS.update(bd.regular_stops(raw))
    raw = raw.astype(object).where(raw.notna(), None)
    frames = []
    for _, g in raw.groupby("date", sort=True):
        out = bd.process_journey(g.to_dict("records"), log)
        if out is not None and len(out):
            frames.append(out)
    if not frames:
        return None, dict(log.dropped), dict(log.notes)
    obs = pd.concat(frames, ignore_index=True)
    keep = obs.station_code.isin(north)
    log.drop("current station outside the covered states", int((~keep).sum()))
    log.dropped["_source_rows"] = log.source_rows
    return obs[keep], dict(log.dropped), dict(log.notes)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=950_000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--output-dir", default=str(ROOT / "data" / "processed"))
    args = ap.parse_args(argv)
    north = northern_stations()
    files = sorted(str(p) for p in RAW.glob("*.csv"))
    print(f"{len(files)} train files, {len(north)} northern stations", flush=True)

    t0, frames, dropped, notes = time.time(), [], Counter(), Counter()
    with ProcessPoolExecutor(args.workers) as pool:
        for i, (obs, d, n) in enumerate(pool.map(process_train, files, [north] * len(files), chunksize=8), 1):
            if obs is not None and len(obs):
                frames.append(obs)
            dropped.update(d)
            notes.update(n)
            if i % 200 == 0:
                print(f"  {i}/{len(files)} trains, {sum(map(len, frames)):,} observations "
                      f"({time.time() - t0:.0f}s)", flush=True)
    source_rows = dropped.pop("_source_rows", 0)
    obs = pd.concat(frames, ignore_index=True)
    obs = obs.drop_duplicates(subset=["train_no", "journey_date", "seq"], keep="first")
    print(f"observations at northern stations: {len(obs):,} from {obs.train_no.nunique()} trains", flush=True)

    # a few source rows contradict themselves: the recorded delay field disagrees with the
    # recorded timestamps. Neither value can be trusted, so the row is dropped (not corrected).
    mins = lambda a, b: (a - b).dt.total_seconds() / 60
    target_gap = mins(obs.target_next_actual_arrival, obs.next_scheduled_arrival) - obs.next_station_arrival_delay_min
    stop_gap = (mins(obs.current_scheduled_departure + pd.to_timedelta(obs.current_departure_delay_min, unit="m"),
                     obs.current_scheduled_arrival + pd.to_timedelta(obs.current_arrival_delay_min, unit="m"))
                - obs.actual_stop_min)
    contradict = (target_gap.abs() > 0.01) | (stop_gap.notna() & (stop_gap.abs() > 0.01))
    dropped["source delay field contradicts its own timestamps"] += int(contradict.sum())
    obs = obs[~contradict]
    print(f"dropped {int(contradict.sum())} self-contradicting source rows", flush=True)

    # fit the row budget by dropping the oldest whole journey dates
    trimmed = 0
    if len(obs) > args.max_rows:
        per_date = obs.groupby("journey_date").size().sort_index(ascending=False).cumsum()
        keep_dates = per_date[per_date <= args.max_rows].index
        before = len(obs)
        obs = obs[obs.journey_date.isin(keep_dates)]
        trimmed = before - len(obs)
        print(f"row budget {args.max_rows:,}: dropped the oldest dates ({trimmed:,} rows); "
              f"kept {obs.journey_date.min().date()} .. {obs.journey_date.max().date()}", flush=True)

    obs = bd.add_historical_features(obs)
    cols = bd.ID_COLUMNS + [c for c in bd.FEATURE_COLUMNS if c in obs.columns] + bd.TARGET_COLUMNS
    final = obs[cols].sort_values(["journey_date", "observation_time", "train_no", "seq"],
                                  kind="stable").reset_index(drop=True)
    train, val, test = bd.chronological_split(final)

    log = bd.FilterLog()
    log.source_rows = source_rows
    log.dropped.update(dropped)
    log.notes.update(notes)
    if trimmed:
        log.drop(f"row budget {args.max_rows}: oldest journey dates dropped", trimmed)
    log.print_report()
    master = pdp.build_master(final)
    checks = pdp.consistency_checks(master)
    ok = bd.quality_checks(final, train, val, test, log)
    print("\nDerived-value consistency:", json.dumps(checks))
    ok = ok and all(v == 0 for v in checks.values())

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    files_out = {"RailScan_Final_Dataset.csv": final, "train.csv": train, "validation.csv": val,
                 "test.csv": test, "RailScan_Master_Dataset.csv": master}
    manifest = []
    for name, part in files_out.items():
        part.to_csv(out / name, index=False)
        manifest.append({"file": name, "rows": len(part), "columns": part.shape[1],
                         "date_min": part.journey_date.min().date().isoformat(),
                         "date_max": part.journey_date.max().date().isoformat(), "created_at": created})
    pd.DataFrame(manifest).to_csv(out / "dataset_manifest.csv", index=False)
    # compact copy for GitHub / fresh servers (the CSVs exceed GitHub's 100 MB file limit)
    from src.database.import_data import export_parquet
    export_parquet(out / "RailScan_Final_Dataset.csv", out / "RailScan_Final_Dataset.parquet")
    log.to_frame().to_csv(out / "filtering_report.csv", index=False)

    dl = json.loads((RAW / "_download.json").read_text()) if (RAW / "_download.json").exists() else {}
    selection = {"method": "trains with a scheduled halt in a covered state (northern India, West Bengal); their "
                           "journeys in the download window; observations whose current station is in a covered "
                           "state; oldest dates "
                           "dropped to fit the row budget; history features computed inside the kept rows",
                 "covered_states": sel.COVERED_STATES, "download": dl, "max_rows": args.max_rows,
                 "source_rows": int(source_rows), "selected_rows": len(final)}
    summary = pdp.summarize(final, master, train, val, test, selection, checks,
                            {"status": "skipped (rows come straight from the downloaded source files)"})
    summary["station_codes"] = len(summary["station_codes"])        # thousands: keep the summary readable
    summary["trains"] = len(summary["trains"])
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {len(files_out) + 3} files to {out}  ({time.time() - t0:.0f}s)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
