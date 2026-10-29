#!/usr/bin/env python3
"""
RailScan - find the trains that serve northern India in the Indian Railways runs dataset.

The dataset (yadavsaurabh/indian-railways-runs, GPL-3.0) has one CSV per train,
12,935 files and about 35 GB in total, oldest journeys first. Downloading all of
it is not needed: this script reads only the END of each file (an HTTP range
request), which holds the train's most recent journeys, and records

  * the date of its latest journey (is the train still running?)
  * its route on that journey: station codes, halts, coordinates
  * how many bytes one recent journey takes (to size the later download)

Each station is placed in a state: the DataMeet station list's own state where it
has one, otherwise the state polygon (geoBoundaries IND ADM1, from DataMeet maps,
CC BY 2.5 IN) that contains its coordinates.

Covered states (COVERED_STATES): West Bengal and northern India, i.e. the Northern Zonal Council (Jammu and Kashmir, Ladakh,
Himachal Pradesh, Punjab, Chandigarh, Haryana, Delhi, Rajasthan) plus Uttar
Pradesh and Uttarakhand.

Outputs (data/interim/northern/):
  file_list.json        every train file with its size
  train_routes.jsonl    one line per train (resumable cache of the tail reads)
  station_states.csv    station_code -> state, and how it was decided
  northern_trains.csv   trains with at least one scheduled halt in a covered state

Usage:
  python scripts/select_northern_trains.py            (~30-60 min, ~0.6 GB of range reads)
  python scripts/select_northern_trains.py --workers 8
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "interim" / "northern"
REPO = "yadavsaurabh/indian-railways-runs"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main"
RAW = f"https://huggingface.co/datasets/{REPO}/resolve/main"
FOLDERS = ["trains", "trains_1"]
STATIONS_URL = "https://raw.githubusercontent.com/datameet/railways/master/stations.json"
STATES_API = "https://www.geoboundaries.org/api/current/gbOpen/IND/ADM1/"
TAIL_BYTES = 32_000

NORTHERN_STATES = ["Jammu and Kashmir", "Ladakh", "Himachal Pradesh", "Punjab", "Chandigarh",
                   "Haryana", "Delhi", "Rajasthan", "Uttar Pradesh", "Uttarakhand"]
# States RailScan covers: northern India plus West Bengal (added 2026-10-30 at the user's request).
COVERED_STATES = NORTHERN_STATES + ["West Bengal"]
# DataMeet / geoBoundaries spellings -> one name
STATE_ALIASES = {"Delhi NCT": "Delhi", "Orissa": "Odisha"}
COLUMNS = ["date", "train_no", "train_name", "train_type", "seq", "station_code", "halts_here",
           "sch_arrival", "sch_arrival_date", "sch_departure", "sch_departure_date", "act_arrival",
           "act_arrival_date", "act_departure", "act_departure_date", "delay_arrival_min",
           "delay_departure_min", "platform", "distance_km", "lat", "lng", "departed", "reported",
           "reported_at"]


def fetch(url: str, headers: dict | None = None, timeout: int = 60, tries: int = 6) -> bytes:
    """GET with retries. Hugging Face allows 3,000 anonymous requests per 5 minutes; on
    429 the wait the server asks for (Retry-After) is honoured and not counted as a failure."""
    attempt = waits = 0
    while attempt < tries:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "railscan-research", **(headers or {})})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 416):
                raise
            if exc.code == 429 and waits < 30:
                waits += 1
                time.sleep(int(exc.headers.get("Retry-After") or 30) + 1)
                continue
            attempt += 1
            time.sleep(min(60, 2 ** attempt))
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            attempt += 1
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"failed after {tries} tries: {url}")


def strip_accents(s: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def canonical_state(name: str | None) -> str | None:
    if not name:
        return None
    name = strip_accents(name).strip()
    return STATE_ALIASES.get(name, name)


# ------------------------------------------------------------------ file list
def list_files() -> list[dict]:
    path = OUT / "file_list.json"
    if path.exists():
        return json.loads(path.read_text())
    files = []
    for folder in FOLDERS:
        url = f"{API}/{folder}"
        while url:
            req = urllib.request.Request(url, headers={"User-Agent": "railscan-research"})
            for attempt in range(6):
                try:
                    with urllib.request.urlopen(req, timeout=120) as r:
                        page, link = json.loads(r.read()), r.headers.get("Link", "")
                    break
                except (urllib.error.URLError, TimeoutError, OSError):
                    time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"cannot list {url}")
            files += [{"path": f["path"], "size": f.get("size", 0)} for f in page if f.get("type") == "file"
                      and f["path"].endswith(".csv")]
            url = None
            for part in link.split(","):
                if 'rel="next"' in part:
                    url = part.split(";")[0].strip().strip("<>")
        print(f"  {folder}: {sum(f['path'].startswith(folder + '/') for f in files)} files", flush=True)
    path.write_text(json.dumps(files))
    return files


# ------------------------------------------------------------------ station states
_POLYS: list = []


def locate(lon, lat):
    """State whose boundary contains the point (geoBoundaries IND ADM1), or None."""
    if not _POLYS:
        from matplotlib.path import Path as Poly
        meta = json.loads(fetch(STATES_API))
        for f in json.loads(fetch(meta["simplifiedGeometryGeoJSON"], timeout=300))["features"]:
            g = f["geometry"]          # outer rings only; a station is never inside a state's hole
            rings = [g["coordinates"][0]] if g["type"] == "Polygon" else [p[0] for p in g["coordinates"]]
            _POLYS.extend((canonical_state(f["properties"]["shapeName"]), Poly(np.asarray(r)[:, :2]))
                          for r in rings)
    if lon is None or lat is None or pd.isna(lon) or pd.isna(lat):
        return None
    for name, poly in _POLYS:
        if poly.contains_point((float(lon), float(lat))):
            return name
    return None


def station_states() -> pd.DataFrame:
    path = OUT / "station_states.csv"
    if path.exists():
        return pd.read_csv(path, dtype=str)
    stations = json.loads(fetch(STATIONS_URL, timeout=300))["features"]
    rows = []
    for s in stations:
        p, geom = s["properties"], s.get("geometry")
        code = (p.get("code") or "").strip().upper()
        if not code:
            continue
        lon, lat = (geom["coordinates"][:2] if geom and geom.get("coordinates") else (None, None))
        listed = canonical_state(p.get("state"))
        located = locate(lon, lat) if lon is not None else None
        state, how = (listed, "datameet state") if listed else (located, "coordinates in state boundary")             if located else (None, "unknown")
        rows.append({"station_code": code, "state": state, "how": how, "zone": p.get("zone") or "",
                     "lat": lat, "lng": lon})
    df = pd.DataFrame(rows).drop_duplicates("station_code")
    df.to_csv(path, index=False)
    return df


def locate_unlisted(routes: list[dict], states: pd.DataFrame, workers: int) -> pd.DataFrame:
    """States of halt stations without one, from the coordinates in the running records.

    Many current codes are missing from the station list (renamed stations such as
    PRYJ Prayagraj, VGLJ Jhansi, AYC Ayodhya Cantt). For each such station one train
    halting there is read (greedy cover: few requests) and the station's lat/lng in
    the records is placed inside a state boundary.
    """
    known = set(states.dropna(subset=["state"]).station_code)
    need = {s for r in routes if r.get("ok") for s in r["halts"] if s not in known}
    if not need:
        return states
    cover, left = [], set(need)
    for r in sorted((r for r in routes if r.get("ok")), key=lambda r: -len(set(r["halts"]) & need)):
        hit = set(r["halts"]) & left
        if hit:
            cover.append(r)
            left -= hit
        if not left:
            break
    print(f"  {len(need)} halt stations without a state; reading {len(cover)} trains for their coordinates",
          flush=True)

    def coords(r):
        try:
            body = fetch(f"{RAW}/{r['path']}", headers={"Range": f"bytes=-{max(r['bytes_per_journey'] * 2, 8000)}"})
        except urllib.error.HTTPError as exc:          # 416: the file is smaller than the range
            if exc.code != 416:
                raise
            body = fetch(f"{RAW}/{r['path']}")
        text = body.decode("utf-8", errors="replace").split("\n", 1)[-1]
        df = pd.read_csv(io.StringIO(text), names=COLUMNS, header=None, dtype=str, on_bad_lines="skip")
        return df[["station_code", "lat", "lng"]]

    found = []
    with ThreadPoolExecutor(workers) as pool:
        for f in as_completed([pool.submit(coords, r) for r in cover]):
            try:
                found.append(f.result())
            except Exception as exc:
                print(f"  ! {exc}", flush=True)
    pts = pd.concat(found).dropna()
    pts = pts[pts.station_code.isin(need)]
    pts = pts.assign(lat=pd.to_numeric(pts.lat, errors="coerce"), lng=pd.to_numeric(pts.lng, errors="coerce"))         .dropna().groupby("station_code")[["lat", "lng"]].median().reset_index()
    rows = [{"station_code": r.station_code, "state": locate(r.lng, r.lat),
             "how": "running-record coordinates in state boundary", "zone": "", "lat": r.lat, "lng": r.lng}
            for r in pts.itertuples()]
    add = pd.DataFrame(rows, columns=["station_code", "state", "how", "zone", "lat", "lng"])
    add.loc[add.state.isna(), "how"] = "unknown"
    out = pd.concat([states[~states.station_code.isin(add.station_code)], add], ignore_index=True)
    out.to_csv(OUT / "station_states.csv", index=False)
    print(f"  placed {int(add.state.notna().sum())} of {len(need)} stations", flush=True)
    return out


# ------------------------------------------------------------------ tail reads
def read_tail(file: dict, tail: int = TAIL_BYTES) -> dict:
    """Route and recency of one train from the end of its file (re-read 4x larger if one journey did not fit)."""
    url = f"{RAW}/{file['path']}"
    size = int(file.get("size") or 0)
    want = min(tail, size) if size else tail
    body = fetch(url, headers={"Range": f"bytes=-{want}"})
    text = body.decode("utf-8", errors="replace")
    if want < size:                       # drop the partial first line
        text = text.split("\n", 1)[1] if "\n" in text else ""
    df = pd.read_csv(io.StringIO(text), names=COLUMNS, header=None, dtype=str, on_bad_lines="skip")
    df = df[df.date.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)]
    train = Path(file["path"]).stem
    if df.empty:
        return {"train_no": train, "path": file["path"], "size": size, "ok": False}
    last = df.date.max()
    trip = df[df.date == last]
    complete = want >= size or (df.date.nunique() > 1)       # the last journey is wholly inside the tail
    if not complete and tail < 1_000_000:
        return read_tail(file, tail * 4)
    journeys = df.date.nunique() - (0 if want >= size else 1)
    bytes_per_journey = (len(body) / max(journeys, 1)) if complete else len(body)
    halts = trip[trip.halts_here == "1"]
    return {"train_no": train, "path": file["path"], "size": size, "ok": True,
            "train_name": trip.train_name.iloc[0], "train_type": trip.train_type.iloc[0],
            "last_date": last, "route_complete": bool(complete),
            "stations": trip.station_code.tolist(), "halts": halts.station_code.tolist(),
            "bytes_per_journey": int(bytes_per_journey),
            "journeys_in_tail": int(journeys), "first_date_in_tail": df.date.min()}


def scan_routes(files: list[dict], workers: int) -> list[dict]:
    cache = OUT / "train_routes.jsonl"
    done = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["path"]] = r
    todo = [f for f in files if f["path"] not in done]
    print(f"  routes: {len(done)} cached, {len(todo)} to read", flush=True)
    t0 = time.time()
    with open(cache, "a", encoding="utf-8") as out, ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(read_tail, f): f for f in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                r = fut.result()
            except Exception as exc:                      # recorded, retried on the next run
                print(f"  ! {futures[fut]['path']}: {exc}", flush=True)
                continue
            done[r["path"]] = r
            out.write(json.dumps(r) + "\n")
            if i % 250 == 0:
                out.flush()
                rate = i / (time.time() - t0)
                print(f"  {i}/{len(todo)}  ({rate:.1f} files/s, ~{(len(todo) - i) / rate / 60:.0f} min left)",
                      flush=True)
    return list(done.values())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    states = station_states()
    print(f"stations: {len(states)}  " + str(states.how.value_counts().to_dict()), flush=True)
    files = list_files()
    print(f"train files: {len(files)}  ({sum(f['size'] for f in files) / 1e9:.1f} GB)", flush=True)
    routes = scan_routes(files, args.workers)
    states = locate_unlisted(routes, states, args.workers)

    state_of = dict(zip(states.station_code, states.state))
    north = set(COVERED_STATES)
    rows = []
    for r in routes:
        if not r.get("ok"):
            continue
        halts_n = [s for s in r["halts"] if state_of.get(s) in north]
        unknown = [s for s in r["halts"] if s not in state_of or pd.isna(state_of.get(s))]
        rows.append({"train_no": r["train_no"], "path": r["path"], "train_name": r["train_name"],
                     "train_type": r["train_type"], "last_date": r["last_date"],
                     "halts": len(r["halts"]), "northern_halts": len(halts_n),
                     "halts_unknown_state": len(unknown), "route_complete": r["route_complete"],
                     "bytes_per_journey": r["bytes_per_journey"], "file_size": r["size"]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "all_trains_summary.csv", index=False)
    sel = df[df.northern_halts > 0].sort_values("train_no")
    sel.to_csv(OUT / "northern_trains.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"trains read: {len(df)}; with a northern halt: {len(sel)}; latest journey overall: {df.last_date.max()}")


if __name__ == "__main__":
    sys.exit(main())
