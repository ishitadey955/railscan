"""Loader for the processed RailScan railway dataset.

Reads ``data/processed/RailScan_Final_Dataset.csv`` (written by
``scripts/build_dataset.py`` from the real Hugging Face dataset
``yadavsaurabh/indian-railways-runs``). Nothing is generated or filled here:
missing values stay missing (``None``).

Each row of that file is one *stop* of one *journey*: the state of the train
at ``station_code`` on ``journey_date`` plus the next stop. The dataset only
holds stops whose next stop was also observed, so the last observed stop of a
journey has no row of its own.
"""
from __future__ import annotations

import math
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, List, Optional, Union

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "data" / "processed" / "RailScan_Final_Dataset.csv"

REQUIRED_COLUMNS = ["train_no", "journey_date", "station_code", "seq"]
TIME_COLUMNS = ["current_scheduled_arrival", "current_scheduled_departure",
                "next_scheduled_arrival", "observation_time"]
ID_COLUMNS = ["train_no", "station_code", "next_station_code"]

DATASET_HELP = (
    "Processed railway dataset not found at {path}. Build it with "
    "`python scripts/build_dataset.py --target-rows 10000 --min-trains 1 --output-dir data/processed` "
    "(needs access to huggingface.co) or pass the CSV path explicitly."
)


class RailwayDataUnavailable(RuntimeError):
    """The processed dataset is missing or unusable."""


def normalize_train_no(value: Any) -> Optional[str]:
    """'12301', 12301, '12301.0', ' 12301 ' -> '12301'; native digits are
    converted. Returns None unless the result is 4-5 digits."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    s = str(value).strip()
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".")[0]
    s = "".join(str(int(c)) if (not c.isascii() and c.isdecimal()) else c for c in s)
    return s if re.fullmatch(r"\d{4,5}", s) else None


def parse_date(value: Any) -> Optional[pd.Timestamp]:
    """str / date / datetime / Timestamp -> normalised (midnight) Timestamp, or None."""
    if value is None or value == "":
        return None
    try:
        if isinstance(value, (date, datetime, pd.Timestamp)):
            ts = pd.Timestamp(value)
        else:
            ts = pd.to_datetime(str(value).strip(), errors="raise")
    except (ValueError, TypeError, OverflowError):
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def clean_value(v: Any) -> Any:
    """NaN / NaT / numpy types -> JSON-friendly Python values (missing -> None)."""
    if v is None:
        return None
    if isinstance(v, pd.Timestamp):
        return None if pd.isna(v) else v.isoformat(sep=" ")
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float):
        return round(v, 3)
    return v


class RailwayData:
    """In-memory view of the processed dataset with train / date / station lookups."""

    def __init__(self, frame: pd.DataFrame):
        missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
        if missing:
            raise RailwayDataUnavailable(f"dataset is missing required columns: {missing}")
        df = frame.copy()
        for c in ID_COLUMNS:
            if c in df.columns:
                df[c] = df[c].astype("string").str.strip()
        df["station_code"] = df["station_code"].str.upper()
        if "next_station_code" in df.columns:
            df["next_station_code"] = df["next_station_code"].str.upper()
        df["journey_date"] = pd.to_datetime(df["journey_date"], errors="coerce").dt.normalize()
        for c in TIME_COLUMNS:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
        df["seq"] = pd.to_numeric(df["seq"], errors="coerce")
        df["_tn"] = df["train_no"].str.lstrip("0")        # "04601" == "4601"
        df = df[df["train_no"].notna() & df["journey_date"].notna() & df["station_code"].notna()]
        self.df = df.reset_index(drop=True)

    # ---- construction ---------------------------------------------------
    @classmethod
    def from_csv(cls, path: Optional[Union[str, Path]] = None) -> "RailwayData":
        p = Path(path) if path else DEFAULT_DATASET
        if not p.is_file():
            raise RailwayDataUnavailable(DATASET_HELP.format(path=p))
        try:
            frame = pd.read_csv(p, dtype={"train_no": "string", "station_code": "string",
                                          "next_station_code": "string"})
        except Exception as exc:
            raise RailwayDataUnavailable(f"could not read {p}: {exc}") from exc
        if frame.empty:
            raise RailwayDataUnavailable(f"{p} contains no rows")
        return cls(frame)

    def __len__(self) -> int:
        return len(self.df)

    @property
    def columns(self) -> List[str]:
        return list(self.df.columns)

    # ---- lookups ----------------------------------------------------------
    def train_rows(self, train_no: Any) -> pd.DataFrame:
        tn = normalize_train_no(train_no)
        if tn is None:
            return self.df.iloc[0:0]
        return self.df[self.df["_tn"] == tn.lstrip("0")]

    def route_codes(self, train_no: Any) -> List[str]:
        """Every station code seen for the train (as current or next stop)."""
        r = self.train_rows(train_no)
        codes = set(r["station_code"].dropna())
        if "next_station_code" in r.columns:
            codes |= set(r["next_station_code"].dropna())
        return sorted(codes)

    # Narrow lookups. TrainStateService uses these so that the SQLite backend
    # (src.database.railway_source.SQLiteRailwayData) can answer them with an
    # indexed query instead of loading a train's whole history.
    def train_day_rows(self, train_no: Any, journey_date: Any) -> pd.DataFrame:
        r = self.train_rows(train_no)
        jd = parse_date(journey_date)
        return r.iloc[0:0] if jd is None else r[r["journey_date"] == jd]

    def has_train(self, train_no: Any) -> bool:
        return not self.train_rows(train_no).empty

    def name_type_counts(self, train_no: Any, journey_date: Any = None,
                         station_code: Optional[str] = None) -> pd.DataFrame:
        """``train_name, train_type, records`` groups, optionally narrowed by date and station."""
        r = self.train_rows(train_no)
        if journey_date is not None:
            jd = parse_date(journey_date)
            r = r.iloc[0:0] if jd is None else r[r["journey_date"] == jd]
        if station_code:
            sc = station_code.strip().upper()
            mask = r["station_code"] == sc
            if "next_station_code" in r.columns:
                mask = mask | (r["next_station_code"] == sc)
            r = r[mask]
        return name_type_groups(r)

    def available_dates(self, train_no: Any, station_code: Optional[str] = None) -> List[pd.Timestamp]:
        r = self.train_rows(train_no)
        if station_code:
            sc = station_code.strip().upper()
            mask = r["station_code"] == sc
            if "next_station_code" in r.columns:
                mask |= r["next_station_code"] == sc
            r = r[mask]
        return sorted(r["journey_date"].dropna().unique())


def name_type_groups(rows: pd.DataFrame) -> pd.DataFrame:
    """Distinct ``(train_name, train_type)`` pairs with their record counts.

    Same grouping the train lookup has always used (missing values kept as a
    group; sorted ascending, missing last). Empty input gives an empty frame.
    """
    if rows.empty:
        return pd.DataFrame({"train_name": [], "train_type": [], "records": []})
    cols = [c for c in ("train_name", "train_type") if c in rows.columns]
    if not cols:
        return pd.DataFrame({"train_name": [None], "train_type": [None], "records": [len(rows)]})
    g = rows.groupby(cols, dropna=False).size().reset_index(name="records")
    for c in ("train_name", "train_type"):
        if c not in g.columns:
            g[c] = None
    return g
