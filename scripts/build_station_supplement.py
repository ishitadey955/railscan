#!/usr/bin/env python3
"""
RailScan - names for station codes that are missing from the DataMeet station master.

The DataMeet list is a ~2016 snapshot. The northern-India running records use newer or
renamed codes (PRYJ Prayagraj Jn, VGLJ Jhansi Jn, DDU Pt. Deen Dayal Upadhyaya Jn, ...),
which then have no name in the app and cannot be matched from a sign photo.

For each such code this script searches English Wikipedia and accepts a name ONLY when the
station article's infobox states exactly that code. The name is the infobox's own name
field when present (it follows renames), otherwise the article title. Codes without such
an article are left out (the app shows the bare code). Coordinates come from the running
records themselves (data/interim/northern/station_states.csv).

Output: data/stations/station_master_supplement.csv
        (station_code, station_name, latitude, longitude, source)
Licence of the names: Wikipedia text, CC BY-SA 4.0 (attribution in the `source` column).

Usage:  python scripts/build_station_supplement.py
Then:   python scripts/setup_database.py import-stations
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
API = "https://en.wikipedia.org/w/api.php"
OUT = ROOT / "data" / "stations" / "station_master_supplement.csv"
NOT_FOUND = ROOT / "data" / "stations" / "station_names_not_found.txt"   # codes already searched without a verified article
CODE_RE = re.compile(r"^\s*\|\s*(?:code|station_code|stationcode)\s*=\s*(.+)$", re.I | re.M)
NAME_RE = re.compile(r"^\s*\|\s*name\s*=\s*(.+)$", re.I | re.M)


def api(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json", "formatversion": 2})
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "RailScan-research/1.0 (station names)"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except Exception:
            time.sleep(2 ** attempt)
    return {}


def plain(wikitext: str) -> str:
    """'[[Ayodhya Cantt|Ayodhya Cantonment]]<ref>..</ref>' -> 'Ayodhya Cantonment'."""
    t = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>|<!--.*?-->", "", wikitext)
    t = re.sub(r"\[\[(?:File|Image):[^\]]*\]\]", "", t, flags=re.I)        # logos: [[File:x.svg|50px]]
    t = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"\{\{[^}]*\}\}|'''?|<[^>]+>", " ", t)
    t = re.sub(r"\b\d+\s*px\b", "", t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip()


def clean_name(name: str, code: str) -> str:
    """'Thiruvananthapuram North railway station (TVCN)' -> 'Thiruvananthapuram North'."""
    n = re.sub(rf"\(\s*{re.escape(code)}\s*\)", "", name, flags=re.I)
    n = re.sub(r"\s+railway station\b", "", n, flags=re.I)
    return re.sub(r"\s+", " ", n).strip(" ,.-")


def lookup(code: str) -> dict | None:
    hits = (api({"action": "query", "list": "search", "srsearch": f'"{code}" railway station India',
                 "srlimit": 5}).get("query") or {}).get("search", [])
    for hit in hits:
        title = hit["title"]
        if not re.search(r"railway station|junction|terminal", title, re.I):
            continue
        text = (api({"action": "parse", "page": title, "prop": "wikitext", "section": 0})
                .get("parse") or {}).get("wikitext", "")
        codes = {plain(c).split()[0].strip("'\"").upper() for c in CODE_RE.findall(text) if plain(c)}
        if code not in codes:
            continue
        names = [clean_name(plain(n), code) for n in NAME_RE.findall(text)]
        name = next((n for n in names if re.search(r"[A-Za-z]{3}", n)), clean_name(title, code))
        return {"station_code": code, "station_name": name.upper(), "title": title}
    return None


def main() -> None:
    master = pd.read_csv(ROOT / "data" / "stations" / "station_master.csv", dtype=str)
    known = set(master.station_code)
    used = set()
    tt = ROOT / "data" / "processed" / "RailScan_Timetable.csv"
    if tt.exists():
        used |= set(pd.read_csv(tt, dtype=str, usecols=["station_code"]).station_code)
    ds = ROOT / "data" / "processed" / "RailScan_Final_Dataset.csv"
    if ds.exists():
        d = pd.read_csv(ds, dtype=str, usecols=["station_code", "next_station_code"])
        used |= set(d.station_code) | set(d.next_station_code.dropna())
    missing = sorted(used - known)
    print(f"{len(missing)} station codes used by RailScan are not in the station master", flush=True)
    # incremental: keep names found earlier, skip codes already searched without success
    previous = pd.read_csv(OUT, dtype=str) if OUT.exists() and OUT.stat().st_size > 60 else pd.DataFrame()
    done = set(previous.station_code) if len(previous) else set()
    tried = set(NOT_FOUND.read_text().split()) if NOT_FOUND.exists() else set()
    todo = [c for c in missing if c not in done and c not in tried]
    print(f"  {len(done)} named earlier, {len(tried)} searched earlier without result, {len(todo)} to look up",
          flush=True)

    states = ROOT / "data" / "interim" / "northern" / "station_states.csv"
    coords = (pd.read_csv(states, dtype=str).set_index("station_code")[["lat", "lng"]]
              if states.exists() else pd.DataFrame(columns=["lat", "lng"]))
    rows = []
    for i, code in enumerate(todo, 1):
        r = lookup(code)
        if r:
            lat, lng = (coords.loc[code, "lat"], coords.loc[code, "lng"]) if code in coords.index else (None, None)
            rows.append({"station_code": code, "station_name": r["station_name"], "latitude": lat,
                         "longitude": lng,
                         "source": f"Wikipedia '{r['title']}' (infobox code {code}), CC BY-SA 4.0; "
                                   "coordinates: indian-railways-runs records"})
        if i % 25 == 0:
            print(f"  {i}/{len(todo)} looked up, {len(rows)} named", flush=True)
    cols = ["station_code", "station_name", "latitude", "longitude", "source"]
    out = pd.concat([previous[cols] if len(previous) else pd.DataFrame(columns=cols),
                     pd.DataFrame(rows, columns=cols)], ignore_index=True)
    out = out[out.station_code.isin(missing)].sort_values("station_code")
    out.to_csv(OUT, index=False)
    not_found = sorted(set(missing) - set(out.station_code))
    NOT_FOUND.write_text("\n".join(not_found) + "\n")
    print(f"named {len(out)} of {len(missing)}; wrote {OUT}")
    print("not found (shown as code only): " + ", ".join(not_found))


if __name__ == "__main__":
    sys.exit(main())
