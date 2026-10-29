"""Where should the train be now? An ESTIMATE from the timetable and the expected delays.

RailScan has no live feed. This module places the train on its route for a given moment
from what the planner already computes:

    expected arrival at a stop   = scheduled arrival + estimated delay at that stop
    expected departure           = expected arrival + scheduled halt
                                   (at the start station: scheduled departure + the current delay)

Between two stops the position is interpolated by time along the straight line between
the stations' recorded coordinates (rails curve, so the map point is approximate). Nothing
here is a measured position; the app labels it "estimate, not live".
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def timeline(tt_route: pd.DataFrame, start_code: str, departure: pd.Timestamp,
             stops: pd.DataFrame, delay_col: str,
             fallback_coords: Optional[Dict[str, tuple]] = None) -> pd.DataFrame:
    """Expected arrival/departure at the start station and every later stop.

    ``tt_route``: ``schedule_on(...)`` rows of the train (coordinates, halts, scheduled times).
    ``stops``: later stops with ``station_code``, ``scheduled_arrival`` and ``delay_col`` (minutes).
    ``fallback_coords``: code -> (lat, lon) used where the timetable has no coordinates.
    """
    tt_route = tt_route.copy()
    for c in ("latitude", "longitude"):
        if c not in tt_route.columns:
            tt_route[c] = np.nan
    for code, (lat, lon) in (fallback_coords or {}).items():
        gap = (tt_route.station_code == code) & tt_route.latitude.isna()
        tt_route.loc[gap, ["latitude", "longitude"]] = [lat, lon]
    coords = tt_route.set_index("station_code")[["latitude", "longitude", "station_name"]]
    coords = coords[~coords.index.duplicated()]
    halts = tt_route.set_index("station_code")["halt_min"]
    halts = halts[~halts.index.duplicated()]
    rows = [{"station_code": start_code, "arrival": pd.NaT, "departure": pd.Timestamp(departure)}]
    for r in stops.itertuples():
        d = getattr(r, delay_col)
        arr = pd.Timestamp(r.scheduled_arrival) + pd.to_timedelta(d if pd.notna(d) else 0.0, unit="m")
        halt = halts.get(r.station_code)
        dep = arr + pd.to_timedelta(float(halt), unit="m") if halt is not None and pd.notna(halt) else pd.NaT
        rows.append({"station_code": r.station_code, "arrival": arr, "departure": dep})
    out = pd.DataFrame(rows)
    out["station_name"] = out.station_code.map(coords["station_name"])
    out["latitude"] = pd.to_numeric(out.station_code.map(coords["latitude"]), errors="coerce")
    out["longitude"] = pd.to_numeric(out.station_code.map(coords["longitude"]), errors="coerce")
    return out


def _name(row) -> str:
    n = row.station_name
    return f"{n} ({row.station_code})" if isinstance(n, str) and n.strip() else str(row.station_code)


def locate(tl: pd.DataFrame, now: pd.Timestamp) -> Dict[str, Any]:
    """Estimated state at ``now``: 'not_started' | 'at_station' | 'between' | 'arrived'.

    Returns the state, a sentence, and (when coordinates are known) lat/lng of the estimate.
    """
    now = pd.Timestamp(now)
    rows = list(tl.itertuples())
    first, last = rows[0], rows[-1]
    out: Dict[str, Any] = {"state": None, "text": "", "latitude": None, "longitude": None,
                           "from": None, "to": None, "progress": None}

    def at(r):
        return {"latitude": r.latitude if pd.notna(r.latitude) else None,
                "longitude": r.longitude if pd.notna(r.longitude) else None}

    if now < first.departure:
        mins = (first.departure - now).total_seconds() / 60
        out.update(state="not_started", **at(first),
                   text=f"Not yet departed from {_name(first)}; expected departure in {_fmt(mins)}.")
        return out
    final_arr = last.arrival if pd.notna(last.arrival) else last.departure
    if pd.notna(final_arr) and now >= final_arr:
        out.update(state="arrived", **at(last), text=f"Expected to have arrived at {_name(last)}.")
        return out
    for prev, nxt in zip(rows, rows[1:]):
        dep = prev.departure if pd.notna(prev.departure) else prev.arrival
        if pd.notna(prev.arrival) and prev.arrival <= now <= (dep if pd.notna(dep) else prev.arrival):
            out.update(state="at_station", **at(prev), text=f"Expected to be at {_name(prev)} (halt).")
            return out
        if pd.notna(dep) and pd.notna(nxt.arrival) and dep <= now < nxt.arrival:
            span = (nxt.arrival - dep).total_seconds()
            frac = float(np.clip((now - dep).total_seconds() / span, 0, 1)) if span > 0 else 0.0
            out.update(state="between", progress=frac, **{"from": prev.station_code, "to": nxt.station_code},
                       text=(f"Expected between {_name(prev)} and {_name(nxt)}, about {frac:.0%} of the way; "
                             f"arrival at {nxt.station_code} expected in "
                             f"{_fmt((nxt.arrival - now).total_seconds() / 60)}."))
            if all(pd.notna(v) for v in (prev.latitude, prev.longitude, nxt.latitude, nxt.longitude)):
                out["latitude"] = float(prev.latitude + frac * (nxt.latitude - prev.latitude))
                out["longitude"] = float(prev.longitude + frac * (nxt.longitude - prev.longitude))
            return out
    out.update(state=None, text="The estimate cannot place the train (incomplete timetable times).")
    return out


def _fmt(minutes: float) -> str:
    m = int(round(minutes))
    return f"{m} min" if m < 60 else f"{m // 60} h {m % 60:02d} min"


def map_points(tl: pd.DataFrame, pos: Dict[str, Any]) -> pd.DataFrame:
    """Stations (grey) and the estimated train position (red) for st.map."""
    pts: List[Dict[str, Any]] = [{"lat": r.latitude, "lon": r.longitude, "color": "#8a8f98", "size": 120,
                                  "label": _name(r)}
                                 for r in tl.itertuples() if pd.notna(r.latitude) and pd.notna(r.longitude)]
    if pos.get("latitude") is not None and pos.get("longitude") is not None:
        pts.append({"lat": pos["latitude"], "lon": pos["longitude"], "color": "#d1342f", "size": 600,
                    "label": "estimated train position"})
    return pd.DataFrame(pts, columns=["lat", "lon", "color", "size", "label"])


def estimate_from_forecast(tt_route: pd.DataFrame, f: Dict[str, Any], now: pd.Timestamp,
                           fallback_coords: Optional[Dict[str, tuple]] = None) -> Optional[Dict[str, Any]]:
    """Position estimate for a planner forecast (``planner.forecast`` result with status 'ok')."""
    if f.get("status") != "ok":
        return None
    tl = timeline(tt_route, f["station_code"], f["reference_time"], f["route"], "predicted_delay_min",
                  fallback_coords)
    return {"timeline": tl, "position": locate(tl, now)}


def estimate_from_typical(tt_route: pd.DataFrame, typical: pd.DataFrame, now: pd.Timestamp,
                          start_code: str, start_departure: pd.Timestamp, start_delay: Optional[float],
                          fallback_coords: Optional[Dict[str, tuple]] = None) -> Optional[Dict[str, Any]]:
    """Position estimate from the recorded median delays (no current delay known)."""
    later = typical[typical.scheduled_arrival.notna()].copy()
    if later.empty:
        return None
    later = later[later.scheduled_arrival > pd.Timestamp(start_departure)]
    dep = pd.Timestamp(start_departure) + pd.to_timedelta(start_delay or 0.0, unit="m")
    tl = timeline(tt_route, start_code, dep, later, "median_delay_min", fallback_coords)
    return {"timeline": tl, "position": locate(tl, now)}
