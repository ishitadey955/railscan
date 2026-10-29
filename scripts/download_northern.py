#!/usr/bin/env python3
"""
RailScan - download the recent journeys of the northern-India trains.

Input : data/interim/northern/northern_trains.csv  (scripts/select_northern_trains.py)
Output: data/raw/delay_north/<train_no>.csv  - every journey of the train that started
        within the last --days days of the dataset (whole journeys, all 24 source columns)

Only the end of each source file is downloaded (HTTP range request): the files are
sorted oldest first, so the end holds the recent journeys. The range is sized from
the bytes one journey takes; if the downloaded part does not reach back to the cut-off
date, a larger range is requested.

Trains whose latest journey is more than --active-days before the dataset's last date
are skipped (no longer running).

Usage:  python scripts/download_northern.py --days 120
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_northern_trains as sel  # noqa: E402

ROOT = sel.ROOT
OUT = ROOT / "data" / "raw" / "delay_north"


def fetch_window(row: dict, cutoff: str) -> tuple[str, int, str]:
    """Download the journeys of one train that started on/after `cutoff`."""
    size = int(row["file_size"]) or 0
    want = int(row["bytes_per_journey"] * row["days"] * 1.3) + 20_000
    while True:
        want = min(want, size) if size else want
        body = sel.fetch(f"{sel.RAW}/{row['path']}", headers={"Range": f"bytes=-{want}"}, timeout=120)
        text = body.decode("utf-8", errors="replace")
        whole = size and want >= size
        if not whole:
            text = text.split("\n", 1)[1] if "\n" in text else ""
        else:
            text = text.split("\n", 1)[1]                      # header line of the full file
        df = pd.read_csv(io.StringIO(text), names=sel.COLUMNS, header=None, dtype=str, on_bad_lines="skip")
        df = df[df.date.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)]
        if whole or df.empty or df.date.min() < cutoff:
            break
        want *= 2                                           # did not reach back far enough
    # the range may start mid-journey, but that journey is older than the cut-off (loop above)
    df = df[df.date >= cutoff]
    path = OUT / f"{row['train_no']}.csv"
    df.to_csv(path, index=False)
    return row["train_no"], len(df), df.date.min() if len(df) else ""


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=120, help="journey window, counted back from the dataset's last date")
    ap.add_argument("--active-days", type=int, default=30,
                    help="skip trains whose latest journey is older than this (days before the last date)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    trains = pd.read_csv(sel.OUT / "northern_trains.csv", dtype={"train_no": str})
    last = pd.to_datetime(pd.read_csv(sel.OUT / "all_trains_summary.csv", usecols=["last_date"]).last_date).max()
    cutoff = (last - pd.Timedelta(days=args.days - 1)).strftime("%Y-%m-%d")
    active = pd.to_datetime(trains.last_date) >= last - pd.Timedelta(days=args.active_days)
    print(f"dataset last date {last.date()}; window {cutoff} .. {last.date()} ({args.days} days)")
    print(f"northern trains: {len(trains)}; still running (journey in the last {args.active_days} days): "
          f"{int(active.sum())}")
    todo = trains[active].assign(days=args.days)
    done = {p.stem for p in OUT.glob("*.csv")}
    todo = todo[~todo.train_no.isin(done)]
    print(f"to download: {len(todo)} (already present: {len(done)})", flush=True)

    t0, rows = time.time(), 0
    failed = []
    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(fetch_window, r, cutoff): r["train_no"] for r in todo.to_dict("records")}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                _, n, _ = f.result()
                rows += n
            except Exception as exc:
                failed.append(futs[f])
                print(f"  ! {futs[f]}: {exc}", flush=True)
            if i % 200 == 0:
                print(f"  {i}/{len(todo)} trains, {rows:,} source rows ({time.time() - t0:.0f}s)", flush=True)
    meta = {"window": {"from": cutoff, "to": str(last.date()), "days": args.days},
            "active_days": args.active_days, "northern_trains": int(len(trains)),
            "downloaded": len(list(OUT.glob("*.csv"))), "failed": failed,
            "source": f"https://huggingface.co/datasets/{sel.REPO}", "license": "GPL-3.0"}
    (OUT / "_download.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    if failed:
        print("Some downloads failed; run the script again to retry them.")


if __name__ == "__main__":
    main()
