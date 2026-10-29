#!/usr/bin/env python3
"""Build data/stations/station_master.csv from a REAL public source.

Source: DataMeet "Indian Railways" stations.json (GeoJSON, ~9,000 stations,
licence CC0) - https://github.com/datameet/railways

Only what the source contains is written: station_code, station_name,
latitude, longitude. Stations without coordinates keep empty
latitude/longitude. Placeholder rows (codes such as "XX-BECE") are dropped.
Nothing is invented. Note the source is a 2016 snapshot, so newer or renamed
stations (e.g. CSMT, PRYJ) may be missing; add them with
data/stations/station_master_supplement.csv (see data/stations/README.md).

Usage (project root):
    python scripts/build_station_master.py                    # download + build
    python scripts/build_station_master.py --input stations.json   # use a local copy
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sys
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/datameet/railways/master/stations.json"
ROOT = Path(__file__).resolve().parents[1]
CODE_RE = re.compile(r"^[A-Z0-9]{1,6}$")  # same rule as scripts/build_dataset.py


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="local stations.json instead of downloading")
    ap.add_argument("--output", default=str(ROOT / "data" / "stations" / "station_master.csv"))
    args = ap.parse_args(argv)

    if args.input:
        data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    else:
        print(f"downloading {URL}")
        with urllib.request.urlopen(URL, timeout=60) as r:
            data = json.loads(r.read().decode("utf-8"))

    rows, dropped, seen = [], 0, set()
    for feat in data["features"]:
        p = feat.get("properties") or {}
        code = (p.get("code") or "").strip().upper()
        name = (p.get("name") or "").strip()
        if not CODE_RE.match(code) or not name or code in seen:
            dropped += 1
            continue
        seen.add(code)
        geom = feat.get("geometry")
        lon = lat = ""
        if geom and geom.get("type") == "Point" and len(geom.get("coordinates", [])) >= 2:
            lon, lat = (round(float(v), 6) for v in geom["coordinates"][:2])
        rows.append((code, name, lat, lon))
    rows.sort()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["station_code", "station_name", "latitude", "longitude"])
        w.writerows(rows)
    no_geo = sum(1 for r in rows if r[2] == "")
    print(f"wrote {out}: {len(rows)} stations ({no_geo} without coordinates); dropped {dropped} placeholder/invalid rows")
    (out.parent / "SOURCE.txt").write_text(
        f"station_master.csv\nsource : {URL}\nlicence: CC0 (DataMeet)\nbuilt  : {dt.date.today().isoformat()}\n"
        f"rows   : {len(rows)}\nscript : scripts/build_station_master.py\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
