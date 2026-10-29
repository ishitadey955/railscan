#!/usr/bin/env python3
"""
RailScan - how far ahead can the journey planner's estimates be trusted?

Replays recorded journeys AFTER the model's training period, using the planner
exactly as the app does:
  start = the first recorded stop of the journey with a recorded departure delay
  input = that recorded delay, the journey date, the current timetable
then compares the planner's estimate at every later recorded stop with the
recorded arrival delay, by number of stops ahead, against the baseline of
carrying the starting delay forward unchanged.

Evaluation data: data/interim/RailScan_Full_Dataset.csv (all 35k corrected
observations), journeys dated after the last training date. History features
come from the 1,500-row dataset, as in the app.

Output: reports/planner_evaluation.json
Usage:  python scripts/evaluate_planner.py [--max-journeys 300]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.delay_prediction import DelayPredictor  # noqa: E402
from src.planner import BANDS, METHODS, band_of, forecast, model_coverage  # noqa: E402
from src.railscan_service import load_master  # noqa: E402
from src.timetable import load_timetable  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-journeys", type=int, default=300)
    args = ap.parse_args(argv)

    full = pd.read_csv(ROOT / "data" / "interim" / "RailScan_Full_Dataset.csv", dtype={"train_no": str})
    full["train_no"] = full.train_no.str.zfill(5)
    train_end = pd.read_csv(ROOT / "data" / "processed" / "train.csv").journey_date.max()
    full = full[full.journey_date > train_end]
    df, tt, p, cov = load_master(), load_timetable(), DelayPredictor.load(), model_coverage()

    val_end = pd.read_csv(ROOT / "data" / "processed" / "validation.csv").journey_date.max()
    journeys = full.groupby(["train_no", "journey_date"]).size().index.tolist()
    step = max(1, len(journeys) // args.max_journeys)
    journeys = journeys[::step][:args.max_journeys]
    rows = []
    for t, jd in journeys:
        j = full[(full.train_no == t) & (full.journey_date == jd)].sort_values("seq")
        start = j[j.current_departure_delay_min.notna()]
        if start.empty:
            continue
        s0 = start.iloc[0]
        cur = float(s0.current_departure_delay_min)
        actual = dict(zip(j.next_station_code, j.next_station_arrival_delay_min))
        out = {}
        for m in METHODS:
            f = forecast(df, tt, p, t, jd, s0.station_code, cur, covered=cov, method=m)
            if f["status"] == "ok":
                out[m] = f["route"]
        if len(out) < len(METHODS):
            continue
        for k, code in enumerate(out[METHODS[0]].station_code, 1):
            if code in actual:
                row = {"period": "validation" if jd <= val_end else "test", "stops_ahead": k,
                       "band": band_of(k), "actual": actual[code], "baseline": cur}
                for m in METHODS:
                    row[m] = out[m].predicted_delay_min.iat[k - 1]
                rows.append(row)
    res = pd.DataFrame(rows)
    mae = lambda part, m: round(float((part[m] - part.actual).abs().mean()), 2)
    bias = lambda part, m: round(float((part[m] - part.actual).mean()), 2)

    # one method for every stop after the next, chosen on pooled VALIDATION error (per-band
    # choices over a few hundred comparisons proved noisy); reported per band on TEST
    va_all, te_all = res[res.period == "validation"], res[res.period == "test"]
    later_va = va_all[va_all.stops_ahead >= 2]
    later_method = min(METHODS, key=lambda m: mae(later_va, m))
    selected, table = {}, []
    for lo, hi in BANDS:
        band = f"{lo}" if lo == hi else f"{lo}-{hi}"
        va, te = va_all[va_all.band == band], te_all[te_all.band == band]
        if va.empty or te.empty:
            continue
        m = "iterative" if band == "1" else later_method          # band 1: the model's own prediction
        selected[band] = "model" if band == "1" else later_method
        err = va.actual - va[m]                                     # likely range from VALIDATION errors
        lo_q, hi_q = float(err.quantile(0.10)), float(err.quantile(0.90))
        inside = ((te.actual - te[m]) >= lo_q) & ((te.actual - te[m]) <= hi_q)
        table.append({"stops_ahead": band, "validation_comparisons": int(len(va)), "test_comparisons": int(len(te)),
                      "validation_mae": {**{k: mae(va, k) for k in METHODS}, "baseline": mae(va, "baseline")},
                      "selected": selected[band],
                      "test_mae_selected": mae(te, m), "test_bias_selected": bias(te, m),
                      "test_mae_carry_baseline": mae(te, "baseline"),
                      "test_mae_all": {**{k: mae(te, k) for k in METHODS}, "baseline": mae(te, "baseline")},
                      "likely_range_min": [round(lo_q, 1), round(hi_q, 1)],
                      "likely_range_test_coverage_pct": round(float(inside.mean() * 100), 1)})
    report = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "journeys": int(len(journeys)), "model_trained_until": str(train_end),
              "validation_period": f"after {train_end} to {val_end}", "test_period": f"after {val_end}",
              "comparisons": int(len(res)), "selected_method_by_band": selected,
              "later_stops_method": later_method,
              "likely_range_by_band": {r["stops_ahead"]: r["likely_range_min"] for r in table},
              "by_stops_ahead": table,
              "note": ("carry = starting delay carried forward (baseline); model_then_carry = model for the next "
                       "stop, then that estimate carried forward; iterative = model applied stop by stop; "
                       "history_change = starting delay + median recorded change between the two stops for this "
                       "train (past journeys only). Methods chosen on validation, reported on test. "
                       "bias = mean(estimate - actual); negative means too optimistic.")}
    (ROOT / "reports" / "planner_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("comparisons", "validation_period", "test_period", "selected_method_by_band")}, indent=2))
    for r in table:
        print(f"{r['stops_ahead']:>6}  {r['selected']:<10} test MAE {r['test_mae_selected']:>6} "
              f"(bias {r['test_bias_selected']:+.1f}) vs carry baseline {r['test_mae_carry_baseline']:>6}  "
              f"likely range {r['likely_range_min']} min, covers {r['likely_range_test_coverage_pct']}% of test  "
              f"[n={r['test_comparisons']}]")


if __name__ == "__main__":
    main()
