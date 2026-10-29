"""Train identity lookup and current train state from the processed dataset.

``lookup_train``  : train number -> train_name / train_type
``get_state``     : (train_no, journey_date, station_code) -> current delay,
                    scheduled / actual times, next station, historical stats

Only values present in ``RailScan_Final_Dataset.csv`` are returned; anything
missing is ``None`` and listed in ``missing_fields``. Target columns
(``next_station_arrival_delay_min`` ...) are deliberately never returned: they
describe the future and must not leak into the ETA stage.

Every call returns a dict with ``status`` and ``message`` instead of raising.

Data source: with no ``data`` argument the service reads the SQLite database
(``data/database/railscan.db``, table ``train_runs``) when it has been
populated, and falls back to the processed CSV otherwise. Results are the same
either way (tests/test_database.py checks this row by row); ``self.backend``
says which one is in use.

Status values of ``get_state``
    ok                        full state, enough history
    insufficient_history      state returned; historical statistics rest on too few
                              earlier journeys (see ``historical.sufficient``)
    no_next_station           the station is the last observed stop of the journey
                              (or its row has no next stop): no state row / next-stop data
    unknown_train             train number not in the dataset ("Train information unavailable")
    no_record_for_date        train known, but no records on that journey date
    unknown_station_for_train station not in the records of that train / date
    ambiguous_train           several train names/types share the number; context did not decide
    ambiguous_stop            the train calls at the station more than once that day
    invalid_input             missing or unparseable train_no / journey_date / station_code
    data_unavailable          processed dataset missing or unreadable
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .railway_data import (
    RailwayData, RailwayDataUnavailable, clean_value, normalize_train_no, parse_date,
)
from .station_master import StationMaster
from .text_normalizer import display_name

TRAIN_UNAVAILABLE_MESSAGE = "Train information unavailable"


@dataclass
class TrainStateConfig:
    min_history_observations: int = 5   # fewer earlier journeys => "insufficient_history"
    max_listed_dates: int = 10


def _name(sm: Optional[StationMaster], code: Optional[str]) -> Optional[str]:
    st = sm.get(code) if (sm and code) else None
    return display_name(st.name) if st else None


class TrainStateService:
    """Look up trains and their state in a :class:`RailwayData` set."""

    def __init__(self, data: Union[RailwayData, pd.DataFrame, str, Path, None] = None,
                 station_master: Optional[StationMaster] = None,
                 config: Optional[TrainStateConfig] = None):
        self.cfg = config or TrainStateConfig()
        self.station_master = station_master
        self.load_error: Optional[str] = None
        self.data = None            # RailwayData (CSV / DataFrame) or SQLiteRailwayData
        self.backend: Optional[str] = None
        try:
            self.data, self.backend = load_railway_data(data)
        except RailwayDataUnavailable as exc:
            self.load_error = str(exc)

    # ------------------------------------------------------------------ helpers
    def _unavailable(self) -> Dict[str, Any]:
        return {"status": "data_unavailable", "message": self.load_error or "Railway dataset not loaded."}

    @staticmethod
    def _station_mask(rows: pd.DataFrame, code: str) -> pd.Series:
        m = rows["station_code"] == code
        if "next_station_code" in rows.columns:
            m = m | (rows["next_station_code"] == code)
        return m

    # ------------------------------------------------------------------ train identity
    def lookup_train(self, train_no: Any, journey_date: Any = None,
                     station_code: Optional[str] = None) -> Dict[str, Any]:
        """Return ``train_no, train_name, train_type`` or "Train information unavailable".

        If one train number carries several names/types in the data, the journey
        date and then the station are used to narrow them down; if that does
        not leave exactly one, ``status`` is ``ambiguous_train`` and all
        candidates are listed (none is forced).
        """
        out: Dict[str, Any] = {"status": "unknown_train", "message": TRAIN_UNAVAILABLE_MESSAGE,
                               "train_no": None, "train_name": None, "train_type": None,
                               "method": None, "candidates": [], "warnings": []}
        if self.data is None:
            out.update(self._unavailable())
            return out
        tn = normalize_train_no(train_no)
        if tn is None:
            out.update(status="invalid_input", message="Invalid train number (expected 4-5 digits).")
            return out
        if not self.data.has_train(tn):
            out["train_no"] = tn
            return out

        # name/type groups, narrowed by journey date and then station when one number has several
        cands = self.data.name_type_counts(tn)
        method = "unique_train_number"
        if len(cands) > 1:
            used_date = None
            jd = parse_date(journey_date)
            if jd is not None:
                c = self.data.name_type_counts(tn, journey_date=jd)
                if not c.empty:
                    cands, used_date, method = c, jd, "journey_date"
                else:
                    out["warnings"].append("no records on the given journey date; date not used to pick the train")
            if len(cands) > 1 and station_code:
                sc = station_code.strip().upper()
                c = self.data.name_type_counts(tn, journey_date=used_date, station_code=sc)
                if not c.empty:
                    cands, method = c, (method + "+station" if method == "journey_date" else "station")
            if len(cands) > 1:
                out.update(status="ambiguous_train", train_no=tn,
                           message=f"Train {tn} appears under {len(cands)} different names/types; not selecting one.",
                           candidates=[{k: clean_value(v) for k, v in r.items()} for r in cands.to_dict("records")])
                return out
        row = cands.iloc[0]
        out.update(status="ok", message="Train found.", train_no=tn, method=method,
                   train_name=clean_value(row.get("train_name")), train_type=clean_value(row.get("train_type")))
        if out["train_name"] is None:
            out["warnings"].append("train_name is not present in the dataset for this train")
        return out

    # ------------------------------------------------------------------ state
    def get_state(self, train_no: Any, journey_date: Any, station_code: Optional[str],
                  seq: Optional[int] = None) -> Dict[str, Any]:
        """State of ``train_no`` at ``station_code`` on ``journey_date`` (see module doc)."""
        res = _blank_state()
        if self.data is None:
            res.update(self._unavailable())
            return res

        tn = normalize_train_no(train_no)
        jd = parse_date(journey_date)
        sc = (station_code or "").strip().upper() or None
        bad = [n for n, v in (("train_no", tn), ("journey_date", jd), ("station_code", sc)) if v is None]
        if bad:
            res.update(status="invalid_input", message=f"Missing or invalid input: {', '.join(bad)}.")
            return res
        res.update(train_no=tn, journey_date=jd.date().isoformat(), station_code=sc,
                   station_name=_name(self.station_master, sc))

        ident = self.lookup_train(tn, jd, sc)
        res["warnings"].extend(ident.get("warnings", []))
        if ident["status"] == "unknown_train":
            res.update(status="unknown_train", message=TRAIN_UNAVAILABLE_MESSAGE)
            return res
        if ident["status"] == "ambiguous_train":
            res.update(status="ambiguous_train", message=ident["message"], candidates=ident["candidates"])
            return res
        res.update(train_name=ident["train_name"], train_type=ident["train_type"])

        day = self.data.train_day_rows(tn, jd)
        if day.empty:
            dates = [pd.Timestamp(d).date().isoformat() for d in self.data.available_dates(tn)]
            res.update(status="no_record_for_date",
                       message=f"No railway record for train {tn} on {jd.date().isoformat()}.",
                       available_dates=dates[-self.cfg.max_listed_dates:])
            return res

        here = day[day["station_code"] == sc]
        if seq is not None:
            here = here[here["seq"] == seq]
        if here.empty:
            seen_as_next = ("next_station_code" in day.columns) and bool((day["next_station_code"] == sc).any())
            if seen_as_next:
                prev = day[day["next_station_code"] == sc].sort_values("seq").iloc[-1]
                res.update(status="no_next_station",
                           message=(f"{sc} is the last observed stop of train {tn} on {jd.date().isoformat()}: "
                                    "the processed dataset has no state row or next station for it."),
                           previous_station_code=clean_value(prev["station_code"]),
                           scheduled_arrival=clean_value(prev.get("next_scheduled_arrival")))
                return res
            res.update(status="unknown_station_for_train",
                       message=f"Train {tn} has no record at {sc} on {jd.date().isoformat()}.",
                       stations_in_record=sorted(set(day["station_code"].dropna())))
            return res
        if len(here) > 1:
            res.update(status="ambiguous_stop",
                       message=f"Train {tn} calls at {sc} {len(here)} times that day; pass seq to choose.",
                       candidates=[{"seq": clean_value(s)} for s in here["seq"]])
            return res

        self._fill_state(res, here.iloc[0])
        return res

    # ------------------------------------------------------------------ fill
    def _fill_state(self, res: Dict[str, Any], r: pd.Series) -> None:
        g = lambda col: clean_value(r.get(col)) if col in r.index else None
        missing: List[str] = []

        sched_arr, sched_dep = r.get("current_scheduled_arrival"), r.get("current_scheduled_departure")
        arr_delay, dep_delay = g("current_arrival_delay_min"), g("current_departure_delay_min")

        def derive(sched, delay):
            if sched is None or pd.isna(sched) or delay is None:
                return None
            return clean_value(sched + pd.Timedelta(minutes=float(delay)))

        res.update(
            seq=g("seq"),
            current_delay_min=g("current_observed_delay_min"),
            arrival_delay_min=arr_delay,
            departure_delay_min=dep_delay,
            scheduled_arrival=clean_value(sched_arr),
            scheduled_departure=clean_value(sched_dep),
            actual_arrival=derive(sched_arr, arr_delay),
            actual_departure=derive(sched_dep, dep_delay),
            observation_time=g("observation_time"),
            next_station_code=g("next_station_code"),
            next_scheduled_arrival=g("next_scheduled_arrival"),
            distance_to_next_km=g("distance_to_next_km"),
            scheduled_travel_to_next_min=g("scheduled_travel_to_next_min"),
        )
        res["next_station_name"] = _name(self.station_master, res["next_station_code"])
        res["field_sources"] = {
            "actual_arrival": "derived: scheduled arrival + arrival delay (dataset stores delays, not raw actual times)",
            "actual_departure": "derived: scheduled departure + departure delay",
            "current_delay_min": "dataset current_observed_delay_min (departure delay, else arrival delay)",
        }

        count = g("historical_observation_count")
        count = int(count) if count is not None else 0
        hist = {
            "observation_count": count,
            # delay at the NEXT station on this train's earlier journeys (dataset definition)
            "avg_delay_min": g("historical_avg_delay_min"),
            "delay_std_min": g("historical_delay_std_min"),
            "on_time_rate_pct": g("historical_on_time_rate_pct"),
            "avg_actual_stop_min": g("historical_avg_actual_stop_min"),
            "stop_std_min": g("historical_stop_std_min"),
            "stop_observation_count": g("historical_stop_observation_count"),
            "stop_reliability_pct": g("historical_stop_reliability_pct"),
            "sufficient": count >= self.cfg.min_history_observations,
            "note": ("avg_delay_min / delay_std_min / on_time_rate_pct describe this train's arrival at its next "
                     "station on EARLIER journeys inside the processed dataset; on_time_rate_pct = share of those "
                     "journeys less than 15 min late. stop_reliability_pct = share of EARLIER observed stops at "
                     "this station within +/-2 min of the scheduled halt."),
        }
        res["historical"] = hist

        for key in ("current_delay_min", "scheduled_arrival", "scheduled_departure", "actual_arrival",
                    "actual_departure", "next_station_code", "next_scheduled_arrival"):
            if res[key] is None:
                missing.append(key)
        for key in ("avg_delay_min", "on_time_rate_pct"):
            if hist[key] is None:
                missing.append("historical." + key)
        res["missing_fields"] = missing

        if res["next_station_code"] is None:
            res.update(status="no_next_station", message="The dataset row has no next station.")
        elif not hist["sufficient"]:
            res.update(status="insufficient_history",
                       message=(f"State found, but only {count} earlier journey(s) of this train are in the "
                                f"dataset (minimum {self.cfg.min_history_observations}); historical statistics "
                                "are unreliable." if count else
                                "State found, but no earlier journey of this train is in the dataset: "
                                "historical statistics are unavailable."))
        else:
            res.update(status="ok", message="Train state found.")


def load_railway_data(data: Any = None):
    """Pick the railway data backend. Returns ``(data, backend_name)``.

    * ``RailwayData`` / object with the same lookup methods  -> used as is
    * ``pandas.DataFrame``                                   -> in-memory (``"dataframe"``)
    * path ending in ``.db`` / ``.sqlite`` / ``.sqlite3``    -> SQLite (``"sqlite"``)
    * other path                                             -> that CSV (``"csv"``)
    * ``None``  -> SQLite ``data/database/railscan.db`` (or ``RAILSCAN_DB_PATH``) when its
                   ``train_runs`` table has rows, otherwise the processed CSV as before.
    Raises :class:`RailwayDataUnavailable` when the chosen source cannot be used.
    """
    if isinstance(data, RailwayData):
        return data, "dataframe"
    if hasattr(data, "train_day_rows") and hasattr(data, "name_type_counts"):
        return data, getattr(data, "backend", "custom")
    if isinstance(data, pd.DataFrame):
        return RailwayData(data), "dataframe"
    try:
        from src.database.connection import database_ready
        from src.database.railway_source import SQLiteRailwayData, is_sqlite_path
    except Exception:  # database package missing -> CSV only, as before
        SQLiteRailwayData = None  # type: ignore
    if SQLiteRailwayData is not None:
        if is_sqlite_path(data):
            return SQLiteRailwayData(data), "sqlite"
        if data is None and database_ready():
            return SQLiteRailwayData(), "sqlite"
    return RailwayData.from_csv(data), "csv"


def _blank_state() -> Dict[str, Any]:
    keys = ["train_no", "train_name", "train_type", "journey_date", "station_code", "station_name", "seq",
            "current_delay_min", "arrival_delay_min", "departure_delay_min", "scheduled_arrival",
            "scheduled_departure", "actual_arrival", "actual_departure", "observation_time",
            "next_station_code", "next_station_name", "next_scheduled_arrival", "distance_to_next_km",
            "scheduled_travel_to_next_min"]
    d: Dict[str, Any] = {k: None for k in keys}
    d.update(status="invalid_input", message="", historical=None, missing_fields=[], field_sources={},
             candidates=[], warnings=[])
    return d


def get_train_state(train_no: Any, journey_date: Any, station_code: str,
                    data: Union[RailwayData, pd.DataFrame, str, Path, None] = None,
                    station_master: Optional[StationMaster] = None, **kw) -> Dict[str, Any]:
    """One-shot helper: ``get_train_state("12301", "2025-03-04", "NDLS")``."""
    return TrainStateService(data, station_master).get_state(train_no, journey_date, station_code, **kw)
