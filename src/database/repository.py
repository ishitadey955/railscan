"""Read-only database access for RailScan.

Every function takes an optional ``db_path`` (or an open ``conn``) and returns
plain dicts / lists of dicts. Missing records give ``None`` or ``[]``.

Field names follow the processed dataset where possible. Time values are
ISO strings ``YYYY-MM-DD HH:MM:SS``; dates ``YYYY-MM-DD``.

No fuzzy matching happens here: ``search_stations`` only does exact-code,
prefix and substring matches. Fuzzy matching stays in
``src.railway.station_matcher``, which reads station records from here.

Leakage
    * Target columns (outcomes at the next station) are never returned by
      ``get_train_state`` and are excluded from ``get_train_observations``
      unless ``include_targets=True``.
    * ``get_historical_station_reliability`` aggregates real outcomes. Pass
      ``before=<observation time>`` to count only outcomes that had already
      happened (strictly earlier); without it the result covers the whole
      table and is flagged ``leakage_safe_for_prediction: False``. For model
      features use the precomputed ``historical_*`` columns of ``train_runs``.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from .connection import PathLike, connect, table_count, table_exists
from .schema import TARGET_COLUMNS, TRAIN_RUN_COLUMNS

DELAYED_THRESHOLD_MIN = 15          # same rule as the dataset builder (delayed = >= 15 min)
STOP_TOLERANCE_MIN = 2              # same rule as the dataset builder (reliable stop = within +/-2 min)
MAX_SEARCH_LIMIT = 100

STATE_COLUMNS = [c for c, _t, _s in TRAIN_RUN_COLUMNS if c not in TARGET_COLUMNS and c != "train_key"]


class AmbiguousRecord(LookupError):
    """More than one record matches and nothing in the arguments decides between them."""

    def __init__(self, message: str, candidates: List[Dict[str, Any]]):
        super().__init__(message)
        self.candidates = candidates


# ------------------------------------------------------------------------------ helpers
@contextmanager
def _conn(db_path: Optional[PathLike], conn: Optional[sqlite3.Connection]) -> Iterator[sqlite3.Connection]:
    if conn is not None:
        yield conn
    else:
        with connect(db_path, read_only=True) as c:
            yield c


def normalize_station_code(code: Any) -> Optional[str]:
    s = str(code).strip().upper() if code is not None else ""
    return s or None


def train_key(train_no: Any) -> Optional[str]:
    """Database lookup key: normalised train number without leading zeros ("01023" -> "1023")."""
    from src.railway.railway_data import normalize_train_no  # local import keeps the package light
    tn = normalize_train_no(train_no)
    if tn is None:
        return None
    return tn.lstrip("0") or "0"


def normalize_date(value: Any) -> Optional[str]:
    from src.railway.railway_data import parse_date
    ts = parse_date(value)
    return ts.date().isoformat() if ts is not None else None


def _row(r: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    return dict(r) if r is not None else None


def _escape_like(s: str) -> str:
    return re.sub(r"([\\%_])", r"\\\1", s)


# ------------------------------------------------------------------------------ stations
def get_station_by_code(station_code: Any, db_path: Optional[PathLike] = None, *,
                        conn: Optional[sqlite3.Connection] = None) -> Optional[Dict[str, Any]]:
    code = normalize_station_code(station_code)
    if code is None:
        return None
    with _conn(db_path, conn) as c:
        return _row(c.execute("SELECT station_code, station_name, latitude, longitude, source FROM stations "
                              "WHERE station_code = ?", (code,)).fetchone())


def get_all_stations(db_path: Optional[PathLike] = None, *,
                     conn: Optional[sqlite3.Connection] = None) -> List[Dict[str, Any]]:
    """Every station (used once by the station matcher to build its indexes)."""
    with _conn(db_path, conn) as c:
        return [dict(r) for r in c.execute("SELECT station_code, station_name, latitude, longitude, source "
                                           "FROM stations ORDER BY station_code")]


def search_stations(query: Any, limit: int = 10, db_path: Optional[PathLike] = None, *,
                    conn: Optional[sqlite3.Connection] = None) -> List[Dict[str, Any]]:
    """Exact code, then name prefix, then name substring (case-insensitive). No fuzzy matching.

    Each result has ``match_type``: ``code`` | ``name_exact`` | ``name_prefix`` | ``name_contains``.
    """
    q = str(query).strip() if query is not None else ""
    if not q:
        return []
    try:
        limit = max(1, min(int(limit), MAX_SEARCH_LIMIT))
    except (TypeError, ValueError):
        limit = 10
    like = _escape_like(q)
    sql = """
        SELECT station_code, station_name, latitude, longitude, source,
               CASE WHEN station_code = :code THEN 'code'
                    WHEN station_name = :q THEN 'name_exact'
                    WHEN station_name LIKE :prefix ESCAPE '\\' THEN 'name_prefix'
                    ELSE 'name_contains' END AS match_type,
               CASE WHEN station_code = :code THEN 0
                    WHEN station_name = :q THEN 1
                    WHEN station_name LIKE :prefix ESCAPE '\\' THEN 2
                    ELSE 3 END AS rank
        FROM stations
        WHERE station_code = :code
           OR station_name LIKE :contains ESCAPE '\\'
        ORDER BY rank, length(station_name), station_name
        LIMIT :limit"""
    params = {"code": q.upper(), "q": q, "prefix": like + "%", "contains": "%" + like + "%", "limit": limit}
    with _conn(db_path, conn) as c:
        out = []
        for r in c.execute(sql, params):
            d = dict(r)
            d.pop("rank", None)
            out.append(d)
        return out


# ------------------------------------------------------------------------------ trains
def get_train_metadata(train_no: Any, db_path: Optional[PathLike] = None, *,
                       conn: Optional[sqlite3.Connection] = None) -> Optional[Dict[str, Any]]:
    """``train_no, train_name, train_type`` (+ history). ``train_type`` is None when the
    train ran under several types; see ``name_type_history`` for which type applied when."""
    key = train_key(train_no)
    if key is None:
        return None
    with _conn(db_path, conn) as c:
        # "01023" and "1023" are the same train; compare without leading zeros
        r = c.execute("SELECT * FROM train_metadata WHERE ltrim(train_no, '0') = ? LIMIT 1",
                      (key.lstrip("0"),)).fetchone()
    if r is None:
        return None
    d = dict(r)
    d["name_type_history"] = json.loads(d["name_type_history"]) if d.get("name_type_history") else []
    return d


def get_train_observations(train_no: Any, journey_date: Any = None, db_path: Optional[PathLike] = None, *,
                           include_targets: bool = False, limit: Optional[int] = None,
                           conn: Optional[sqlite3.Connection] = None) -> List[Dict[str, Any]]:
    """Rows of ``train_runs`` for a train (optionally one journey date), ordered by date and seq."""
    key = train_key(train_no)
    if key is None:
        return []
    cols = STATE_COLUMNS + (TARGET_COLUMNS if include_targets else [])
    sql = f"SELECT {', '.join(cols)} FROM train_runs WHERE train_key = ?"
    params: List[Any] = [key]
    if journey_date is not None:
        jd = normalize_date(journey_date)
        if jd is None:
            return []
        sql += " AND journey_date = ?"
        params.append(jd)
    sql += " ORDER BY journey_date, seq"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(max(0, int(limit)))
    with _conn(db_path, conn) as c:
        return [dict(r) for r in c.execute(sql, params)]


def get_train_state(train_no: Any, journey_date: Any, station_code: Any, db_path: Optional[PathLike] = None, *,
                    seq: Optional[int] = None, conn: Optional[sqlite3.Connection] = None
                    ) -> Optional[Dict[str, Any]]:
    """The state row of a train at a station on a journey date (no target columns).

    ``None`` if there is no such row. Raises :class:`AmbiguousRecord` when the
    train calls at the station more than once that day and ``seq`` is not given.
    For status messages (unknown train, last stop, insufficient history ...)
    use ``src.railway.train_state.TrainStateService``, which builds on this data.
    """
    key, jd, code = train_key(train_no), normalize_date(journey_date), normalize_station_code(station_code)
    if None in (key, jd, code):
        return None
    sql = (f"SELECT {', '.join(STATE_COLUMNS)} FROM train_runs WHERE train_key = ? AND journey_date = ? "
           "AND station_code = ?")
    params: List[Any] = [key, jd, code]
    if seq is not None:
        sql += " AND seq = ?"
        params.append(int(seq))
    with _conn(db_path, conn) as c:
        rows = [dict(r) for r in c.execute(sql + " ORDER BY seq", params)]
        if not rows:
            return None
        if len(rows) > 1:
            raise AmbiguousRecord(f"train {train_no} calls at {code} {len(rows)} times on {jd}; pass seq",
                                  [{"seq": r["seq"]} for r in rows])
        st = get_station_by_code(code, conn=c)
    state = rows[0]
    state["station_name"] = st["station_name"] if st else None
    return state


def get_next_station(train_no: Any, journey_date: Any, station_code: Any, db_path: Optional[PathLike] = None, *,
                     seq: Optional[int] = None, conn: Optional[sqlite3.Connection] = None
                     ) -> Optional[Dict[str, Any]]:
    """Next stop after ``station_code`` (schedule information only; no next-stop outcomes)."""
    with _conn(db_path, conn) as c:
        state = get_train_state(train_no, journey_date, station_code, seq=seq, conn=c)
        if state is None or state.get("next_station_code") is None:
            return None
        nxt = get_station_by_code(state["next_station_code"], conn=c)
    return {
        "train_no": state["train_no"], "journey_date": state["journey_date"],
        "station_code": state["station_code"], "seq": state["seq"],
        "next_station_code": state["next_station_code"],
        "next_station_name": nxt["station_name"] if nxt else None,
        "next_station_seq": state["next_station_seq"],
        "next_scheduled_arrival": state["next_scheduled_arrival"],
        "distance_to_next_km": state["distance_to_next_km"],
        "scheduled_travel_to_next_min": state["scheduled_travel_to_next_min"],
        "next_station_latitude": nxt["latitude"] if nxt else None,
        "next_station_longitude": nxt["longitude"] if nxt else None,
    }


def _stats(n: int, s: Optional[float], s2: Optional[float], on_time: Optional[int]) -> Dict[str, Any]:
    if not n:
        return {"count": 0, "avg_delay_min": None, "delay_std_min": None, "on_time_rate_pct": None}
    mean = s / n
    var = (s2 - n * mean * mean) / (n - 1) if n > 1 else None
    return {"count": n, "avg_delay_min": round(mean, 3),
            "delay_std_min": round(math.sqrt(max(var, 0.0)), 3) if var is not None else None,
            "on_time_rate_pct": round(100.0 * on_time / n, 3)}


def get_historical_station_reliability(station_code: Any, train_no: Any = None, db_path: Optional[PathLike] = None,
                                       *, before: Any = None, conn: Optional[sqlite3.Connection] = None
                                       ) -> Optional[Dict[str, Any]]:
    """Observed punctuality at a station.

    ``arrivals``  : delays of trains ARRIVING at the station (rows whose next stop is it);
                    an arrival counts only if it happened strictly before ``before``.
    ``departures``: current-stop delays observed AT the station; counted only if
                    ``observation_time`` is strictly before ``before``.
    ``on_time_rate_pct`` = share of observations below 15 min late (std is the sample std, ddof=1).
    Returns ``None`` when the station has no observations at all.
    """
    code = normalize_station_code(station_code)
    if code is None:
        return None
    cutoff = None
    if before is not None:
        import pandas as pd
        ts = pd.to_datetime(str(before), errors="coerce")
        if pd.isna(ts):
            raise ValueError(f"invalid cutoff time: {before!r}")
        cutoff = ts.strftime("%Y-%m-%d %H:%M:%S")
    key = None
    if train_no is not None:
        key = train_key(train_no)
        if key is None:
            return None

    def agg(c: sqlite3.Connection, value_col: str, station_col: str, time_col: str) -> Dict[str, Any]:
        sql = (f"SELECT COUNT({value_col}) n, SUM({value_col}) s, SUM({value_col} * {value_col}) s2, "
               f"SUM(CASE WHEN {value_col} < ? THEN 1 ELSE 0 END) ok, MIN(journey_date) d0, MAX(journey_date) d1 "
               f"FROM train_runs WHERE {station_col} = ? AND {value_col} IS NOT NULL")
        params: List[Any] = [DELAYED_THRESHOLD_MIN, code]
        if key is not None:
            sql += " AND train_key = ?"
            params.append(key)
        if cutoff is not None:
            sql += f" AND {time_col} IS NOT NULL AND {time_col} < ?"
            params.append(cutoff)
        r = c.execute(sql, params).fetchone()
        out = _stats(int(r["n"] or 0), r["s"], r["s2"], r["ok"])
        out.update(first_journey_date=r["d0"], last_journey_date=r["d1"])
        return out

    with _conn(db_path, conn) as c:
        arrivals = agg(c, "next_station_arrival_delay_min", "next_station_code", "target_next_actual_arrival")
        departures = agg(c, "current_observed_delay_min", "station_code", "observation_time")
        st = get_station_by_code(code, conn=c)
    if arrivals["count"] == 0 and departures["count"] == 0:
        return None
    return {
        "station_code": code, "station_name": st["station_name"] if st else None,
        "train_no": train_no if train_no is None else key, "cutoff": cutoff,
        "leakage_safe_for_prediction": cutoff is not None,
        "delayed_threshold_min": DELAYED_THRESHOLD_MIN,
        "arrivals": arrivals, "departures": departures,
        "note": ("Aggregates of observations inside the processed dataset sample only. "
                 + ("Only outcomes strictly before the cutoff are counted."
                    if cutoff else "No cutoff: includes every observation - do not use as a model feature.")),
    }


def get_historical_stop_statistics(station_code: Any, train_no: Any = None, db_path: Optional[PathLike] = None,
                                   *, before: Any = None, conn: Optional[sqlite3.Connection] = None
                                   ) -> Optional[Dict[str, Any]]:
    """Recorded ACTUAL stop durations at a station (not the timetable halt).

    Only rows where both the actual and the scheduled stop are known are used.
    ``stop_reliability_pct`` = share of those stops within +/-STOP_TOLERANCE_MIN of
    the scheduled halt. With ``before``, only stops completed strictly earlier count.
    Returns ``None`` when no valid observed stop exists.
    """
    code = normalize_station_code(station_code)
    if code is None:
        return None
    sql = ("SELECT actual_stop_min a, scheduled_stop_min s FROM train_runs WHERE station_code = ? "
           "AND actual_stop_min IS NOT NULL AND scheduled_stop_min IS NOT NULL")
    params: List[Any] = [code]
    if train_no is not None:
        key = train_key(train_no)
        if key is None:
            return None
        sql += " AND train_key = ?"
        params.append(key)
    if before is not None:
        import pandas as pd
        ts = pd.to_datetime(str(before), errors="coerce")
        if pd.isna(ts):
            raise ValueError(f"invalid cutoff time: {before!r}")
        sql += " AND observation_time < ?"
        params.append(ts.strftime("%Y-%m-%d %H:%M:%S"))
    with _conn(db_path, conn) as c:
        rows = c.execute(sql, params).fetchall()
    if not rows:
        return None
    actual = sorted(float(r["a"]) for r in rows)
    n = len(actual)
    mean = sum(actual) / n
    mid = n // 2
    median = actual[mid] if n % 2 else (actual[mid - 1] + actual[mid]) / 2
    std = math.sqrt(sum((x - mean) ** 2 for x in actual) / (n - 1)) if n > 1 else None
    within = sum(abs(float(r["a"]) - float(r["s"])) <= STOP_TOLERANCE_MIN for r in rows)
    return {"station_code": code, "count": n, "avg_actual_stop_min": round(mean, 3),
            "median_actual_stop_min": round(median, 3),
            "std_actual_stop_min": round(std, 3) if std is not None else None,
            "stop_reliability_pct": round(100.0 * within / n, 3),
            "tolerance_min": STOP_TOLERANCE_MIN, "leakage_safe_for_prediction": before is not None}


# ------------------------------------------------------------------------------ statistics
def get_database_statistics(db_path: Optional[PathLike] = None, *,
                            conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
    with _conn(db_path, conn) as c:
        tables = {t: table_count(c, t) for t in ("stations", "train_metadata", "train_runs", "database_metadata")}
        stats: Dict[str, Any] = {"tables": tables}
        if tables["train_runs"]:
            r = c.execute("SELECT COUNT(DISTINCT train_key) trains, COUNT(DISTINCT journey_date) dates, "
                          "MIN(journey_date) d0, MAX(journey_date) d1, COUNT(DISTINCT station_code) stations, "
                          "COUNT(DISTINCT train_key || '|' || journey_date) journeys FROM train_runs").fetchone()
            stats["train_runs"] = dict(r)
            stats["train_runs"]["null_counts"] = {
                col: int(c.execute(f"SELECT COUNT(*) FROM train_runs WHERE {col} IS NULL").fetchone()[0])
                for col in ("actual_departure", "current_departure_delay_min", "actual_stop_min",
                            "historical_avg_delay_min", "historical_delay_std_min", "next_station_code")}
            stats["train_runs"]["rows_with_history"] = int(c.execute(
                "SELECT COUNT(*) FROM train_runs WHERE historical_observation_count > 0").fetchone()[0])
            if tables["stations"]:
                stats["train_runs"]["station_codes_not_in_stations"] = [r[0] for r in c.execute(
                    "SELECT DISTINCT code FROM (SELECT station_code code FROM train_runs UNION "
                    "SELECT next_station_code FROM train_runs WHERE next_station_code IS NOT NULL) "
                    "WHERE code NOT IN (SELECT station_code FROM stations) ORDER BY code")]
        if tables["stations"]:
            stats["stations"] = dict(c.execute(
                "SELECT COUNT(*) total, SUM(latitude IS NOT NULL AND longitude IS NOT NULL) with_coordinates, "
                "COUNT(DISTINCT source) sources FROM stations").fetchone())
        if table_exists(c, "database_metadata"):
            q = ("SELECT d.id, d.target_table, d.source_file, d.imported_at, d.rows_imported, d.status "
                 "FROM database_metadata d JOIN (SELECT target_table, MAX(id) id FROM database_metadata {w} "
                 "GROUP BY target_table) last ON last.id = d.id ORDER BY d.target_table")
            stats["last_imports"] = [dict(r) for r in c.execute(q.format(w=""))]
            stats["last_successful_imports"] = [dict(r) for r in c.execute(q.format(w="WHERE status = 'success'"))]
        stats["schema_version"] = c.execute("PRAGMA user_version").fetchone()[0]
        stats["foreign_key_violations"] = len(c.execute("PRAGMA foreign_key_check").fetchall())
    return stats
