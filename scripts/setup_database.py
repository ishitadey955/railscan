#!/usr/bin/env python
"""RailScan database setup (SQLite, data/database/railscan.db).

Run from the project root:

    python scripts/setup_database.py init              # create tables + indexes (safe to repeat)
    python scripts/setup_database.py import-stations   # data/stations/station_master.csv (+ supplement)
    python scripts/setup_database.py import-data       # data/processed/RailScan_Final_Dataset.csv
    python scripts/setup_database.py stats             # real row counts
    python scripts/setup_database.py all               # init + import-stations + import-data + stats

After regenerating the processed dataset:

    python scripts/setup_database.py import-data --replace

Options: --db PATH (or env RAILSCAN_DB_PATH), --csv PATH, --master PATH,
--supplement PATH, --replace, --duplicates {fail,keep-first}, --json.
Exit code is 1 when a step fails; a failed import writes nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.database import (get_database_statistics, import_dataset, import_stations,  # noqa: E402
                          initialize_database, resolve_db_path)
from src.database.connection import DatabaseUnavailable  # noqa: E402


def _print_report(rep, as_json: bool) -> bool:
    d = rep.as_dict()
    if as_json:
        print(json.dumps(d, indent=2, ensure_ascii=False, default=str))
        return rep.status == "success"
    ok = rep.status == "success"
    print(f"\n[{'OK' if ok else 'FAILED'}] import into {rep.target_table}")
    print(f"  source            : {rep.source_file}")
    if not ok:
        print(f"  error             : {rep.error}")
        print("  nothing was written to the table (transaction rolled back); failure recorded in database_metadata")
        return False
    print(f"  mode              : {rep.mode}")
    print(f"  rows read         : {rep.rows_read}")
    print(f"  rows imported     : {rep.rows_imported}  (inserted {rep.rows_inserted}, updated {rep.rows_updated})")
    if rep.rows_replaced:
        print(f"  rows replaced     : {rep.rows_replaced} (previous contents removed by --replace)")
    print(f"  rows rejected     : {rep.rows_rejected} {rep.rejected_reasons or ''}")
    print(f"  duplicates        : {rep.duplicates_identical} identical collapsed, "
          f"{rep.duplicates_conflicting} conflicting")
    print(f"  invalid -> NULL   : {rep.invalid_values_set_null or 'none'}")
    print(f"  missing optional  : {rep.missing_optional_columns or 'none'}")
    if rep.unknown_station_codes:
        print(f"  unknown stations  : {rep.unknown_station_codes}")
    print(f"  table rows now    : {rep.table_rows_after}")
    for w in rep.warnings:
        print(f"  warning           : {w}")
    return True


def cmd_init(a) -> bool:
    res = initialize_database(a.db)
    print(f"Database: {res['db_path']}")
    print(f"  created tables : {res['created_tables'] or 'none (already existed)'}")
    print(f"  existing tables: {res['existing_tables'] or 'none'}")
    if res["schema_problems"]:
        print("  SCHEMA PROBLEMS (existing tables are older than this code):")
        for p in res["schema_problems"]:
            print(f"    - {p}")
        print("  Fix: move the old file away and run init + imports again.")
        return False
    print("  schema OK")
    return True


def cmd_import_data(a) -> bool:
    rep = import_dataset(a.csv, a.db, replace=a.replace, duplicate_policy=a.duplicates)
    return _print_report(rep, a.json)


def cmd_import_stations(a) -> bool:
    rep = import_stations(a.master, a.supplement, a.db, replace=a.replace)
    return _print_report(rep, a.json)


def cmd_import_timetable(a) -> bool:
    from src.database.import_data import import_timetable
    rep = import_timetable(getattr(a, "timetable", None), a.db)
    return _print_report(rep, a.json)


def cmd_export_overview(a) -> bool:
    """Data page figures for servers without the full database (data/processed/dataset_overview.json)."""
    from src.database.app_data import export_overview
    print(f"wrote {export_overview(a.db)}")
    return True


def cmd_stats(a) -> bool:
    try:
        st = get_database_statistics(a.db)
    except DatabaseUnavailable as exc:
        print(str(exc))
        return False
    if a.json:
        print(json.dumps(st, indent=2, ensure_ascii=False, default=str))
        return True
    print(f"Database: {resolve_db_path(a.db)}  (schema v{st['schema_version']})")
    print("Row counts:")
    for t, n in st["tables"].items():
        print(f"  {t:<18} {n:>8}")
    if "train_runs" in st:
        tr = st["train_runs"]
        print(f"train_runs: {tr['trains']} train(s), {tr['journeys']} journeys, {tr['dates']} journey dates "
              f"({tr['d0']} .. {tr['d1']}), {tr['stations']} station codes")
        print(f"  rows with earlier history : {tr['rows_with_history']}")
        print(f"  NULL counts               : {tr['null_counts']}")
        if "station_codes_not_in_stations" in tr:
            print(f"  codes not in stations     : {tr['station_codes_not_in_stations'] or 'none'}")
    if "stations" in st:
        s = st["stations"]
        print(f"stations: {s['total']} total, {s['with_coordinates']} with coordinates")
    for key, label in (("last_successful_imports", "Last successful import per table (current contents):"),
                       ("last_imports", "Last import attempt per table:")):
        if st.get(key):
            print(label)
            for r in st[key]:
                print(f"  {r['target_table']:<12} {r['status']:<8} {r['rows_imported']:>7} rows  {r['imported_at']}  "
                      f"{r['source_file']}")
    print(f"Foreign-key violations: {st['foreign_key_violations']}")
    if not st["tables"]["train_runs"]:
        print("train_runs is EMPTY - run: python scripts/setup_database.py import-data")
    if not st["tables"]["stations"]:
        print("stations is EMPTY - run: python scripts/setup_database.py import-stations")
    return True


def main(argv=None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", help="database path (default data/database/railscan.db or $RAILSCAN_DB_PATH)")
    common.add_argument("--json", action="store_true", help="machine-readable output")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("init", parents=[common], help="create tables and indexes")
    p = sub.add_parser("import-data", parents=[common], help="import RailScan_Final_Dataset.csv")
    p.add_argument("--csv", help="dataset path (default data/processed/RailScan_Final_Dataset.csv)")
    p.add_argument("--replace", action="store_true", help="replace train_runs instead of upserting")
    p.add_argument("--duplicates", choices=["fail", "keep-first"], default="fail")
    p = sub.add_parser("import-stations", parents=[common], help="import the station master")
    p.add_argument("--master", help="default data/stations/station_master.csv")
    p.add_argument("--supplement", help="default station_master_supplement.csv next to the master")
    p.add_argument("--replace", action="store_true", help="replace the stations table instead of upserting")
    p = sub.add_parser("import-timetable", parents=[common], help="replace the timetable table")
    p.add_argument("--timetable", help="default data/processed/RailScan_Timetable.csv")
    sub.add_parser("stats", parents=[common], help="print real row counts")
    sub.add_parser("export-overview", parents=[common],
                   help="write the Data page figures to data/processed/dataset_overview.json (commit it)")
    p = sub.add_parser("all", parents=[common],
                       help="init + import-stations + import-data + import-timetable + export-overview + stats")
    p.add_argument("--csv"); p.add_argument("--master"); p.add_argument("--supplement")
    p.add_argument("--timetable")
    p.add_argument("--replace", action="store_true")
    p.add_argument("--duplicates", choices=["fail", "keep-first"], default="fail")
    a = ap.parse_args(argv)

    if a.command == "all":
        steps = [cmd_init, cmd_import_stations, cmd_import_data, cmd_import_timetable, cmd_export_overview,
                 cmd_stats]
    else:
        steps = [{"init": cmd_init, "import-data": cmd_import_data, "import-stations": cmd_import_stations,
                  "import-timetable": cmd_import_timetable, "export-overview": cmd_export_overview,
                  "stats": cmd_stats}[a.command]]
    for step in steps:
        if not step(a):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
