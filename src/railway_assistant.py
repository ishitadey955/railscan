"""Turn a validated scan result into railway information from RailScan's own data.

Only information that exists in the database / timetable is returned:
station details from the station master, and the scheduled calls of the
trains RailScan knows (the current timetable). Nothing here says where a train
is now, which platform it will use, or whether it is running.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from .timetable import route

LIVE_DISCLAIMER = ("Scheduled information from recorded timetables. RailScan cannot see live train positions, "
                   "platform assignments or cancellations; check the station displays or official sources.")


def match_options(match: Optional[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    """Ranked station candidates from a StationMatcher result (best first, no duplicates)."""
    if not match:
        return []
    seen, out = set(), []
    for c in match.get("candidates") or []:
        if c.get("station_code") and c["station_code"] not in seen:
            seen.add(c["station_code"])
            out.append({"station_code": c["station_code"], "station_name": c.get("station_name"),
                        "match_score": c.get("match_score"), "match_method": c.get("match_method")})
    return out[:limit]


def needs_confirmation(match: Optional[Dict[str, Any]]) -> bool:
    """True unless the matcher accepted one station with high confidence."""
    return not (match and match.get("status") == "matched" and match.get("accepted"))


def _clock(hhmm, offset) -> str:
    if hhmm is None or pd.isna(hhmm):
        return "-"
    days = 0 if offset is None or pd.isna(offset) else int(offset)
    return str(hhmm) + (f" (+{days} day)" if days == 1 else f" (+{days} days)" if days > 1 else "")


def _label(row) -> str:
    """'NEW DELHI (NDLS)', or just the code when the station list has no name for it."""
    name = row.station_name
    return f"{name} ({row.station_code})" if isinstance(name, str) and name.strip() else str(row.station_code)


def station_services(tt: pd.DataFrame, station_code: str) -> pd.DataFrame:
    """Scheduled calls at a station by every train in the timetable (empty if none)."""
    rows = []
    for train_no in sorted(tt.train_no.unique()):
        r = route(tt, train_no)
        hit = r[r.station_code == station_code]
        if hit.empty:
            continue
        s, i = hit.iloc[0], int(hit.index[0])
        origin, dest = r.iloc[0], r.iloc[-1]
        nxt = r.iloc[i + 1] if i + 1 < len(r) else None
        rows.append({
            "Train": f"{train_no} {s.train_name}",
            "From": f"{_label(origin)} {origin.sch_departure}",
            "To": _label(dest),
            "Arrives": _clock(s.sch_arrival, s.arrival_day_offset),
            "Departs": _clock(s.sch_departure, s.departure_day_offset),
            "Next stop": _label(nxt) if nxt is not None else "terminates",
            "Runs on": s.runs_on,
            "train_no": train_no,
        })
    return pd.DataFrame(rows)


def train_summary(tt: pd.DataFrame, train_no: str) -> Optional[Dict[str, Any]]:
    r = route(tt, train_no)
    if r.empty:
        return None
    o, d = r.iloc[0], r.iloc[-1]
    return {"train_no": str(train_no).zfill(5), "train_name": o.train_name, "stops": int(len(r)),
            "origin": _label(o), "departs": o.sch_departure,
            "destination": _label(d),
            "arrives": _clock(d.sch_arrival, d.arrival_day_offset), "runs_on": o.runs_on}
