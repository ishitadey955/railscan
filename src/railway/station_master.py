"""Station master loader.

The matcher needs a *real* list of Indian Railways stations. This module only
loads and validates it; it never fabricates entries.

Files (default folder ``data/stations/``)
    station_master.csv             station_code, station_name, latitude, longitude
    station_aliases.csv            alias, station_code, language, source        (optional)
    station_master_supplement.csv  same columns as the master (+ source)        (optional)

``station_master.csv`` is produced from DataMeet's CC0 station list by
``scripts/build_station_master.py``. If you have a more complete official
source (NTES / data.gov.in / Indian Railways station list), provide it in the
same four-column format.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / "data" / "stations"

MASTER_COLUMNS = ["station_code", "station_name", "latitude", "longitude"]

REQUIRED_SOURCE_HELP = (
    "Station master not found. Required: a CSV with columns "
    "station_code, station_name, latitude, longitude built from a REAL source. "
    "Run `python scripts/build_station_master.py` (downloads DataMeet's CC0 "
    "Indian Railways station list) or place an official list (NTES / "
    "data.gov.in / Indian Railways) at {path}."
)


class StationMasterUnavailable(RuntimeError):
    """The station master file is missing or unusable."""


@dataclass(frozen=True)
class Station:
    code: str
    name: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None


@dataclass(frozen=True)
class Alias:
    alias: str
    code: str
    language: str = ""
    source: str = ""


def _float(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


@dataclass
class StationMaster:
    stations: Dict[str, Station] = field(default_factory=dict)
    aliases: List[Alias] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    source: str = "records"          # "csv" | "sqlite" | "records"

    # ---- construction ---------------------------------------------------
    @classmethod
    def from_records(cls, stations: Iterable[Sequence], aliases: Iterable[Sequence] = ()) -> "StationMaster":
        """Build from tuples ``(code, name[, lat, lon])`` / ``(alias, code[, language, source])``."""
        sm = cls()
        for rec in stations:
            sm._add_station(*rec)
        for rec in aliases:
            sm._add_alias(*rec)
        return sm

    @classmethod
    def from_csv(cls, master_path: Optional[Path] = None, alias_path: Optional[Path] = None,
                 supplement_path: Optional[Path] = None, directory: Optional[Path] = None) -> "StationMaster":
        d = Path(directory) if directory else DEFAULT_DIR
        master_path = Path(master_path) if master_path else d / "station_master.csv"
        alias_path = Path(alias_path) if alias_path else d / "station_aliases.csv"
        supplement_path = Path(supplement_path) if supplement_path else d / "station_master_supplement.csv"
        if not master_path.is_file():
            raise StationMasterUnavailable(REQUIRED_SOURCE_HELP.format(path=master_path))

        sm = cls(source="csv")
        for path in (master_path, supplement_path):
            if not path.is_file():
                continue
            with path.open(newline="", encoding="utf-8-sig") as fh:
                reader = csv.DictReader(fh)
                missing = [c for c in MASTER_COLUMNS if c not in (reader.fieldnames or [])]
                if missing:
                    raise StationMasterUnavailable(f"{path} is missing required columns: {missing}")
                for row in reader:
                    sm._add_station(row["station_code"], row["station_name"], row["latitude"], row["longitude"])
        if not sm.stations:
            raise StationMasterUnavailable(f"{master_path} contains no stations")

        if alias_path.is_file():
            with alias_path.open(newline="", encoding="utf-8-sig") as fh:
                for row in csv.DictReader(fh):
                    sm._add_alias(row.get("alias"), row.get("station_code"), row.get("language", ""), row.get("source", ""))
        return sm

    @classmethod
    def from_database(cls, db_path: Optional[Path] = None, alias_path: Optional[Path] = None) -> "StationMaster":
        """Stations from the SQLite ``stations`` table; aliases still from ``station_aliases.csv``.

        Raises :class:`StationMasterUnavailable` if the database or its station table is missing/empty.
        """
        try:
            from src.database.repository import get_all_stations
            rows = get_all_stations(db_path)
        except Exception as exc:  # missing file, missing table, sqlite error, package absent
            raise StationMasterUnavailable(f"station table unavailable: {exc}") from exc
        if not rows:
            raise StationMasterUnavailable("stations table is empty (run `python scripts/setup_database.py "
                                           "import-stations`)")
        sm = cls()
        for r in rows:
            sm._add_station(r["station_code"], r["station_name"], r["latitude"], r["longitude"])
        alias_path = Path(alias_path) if alias_path else DEFAULT_DIR / "station_aliases.csv"
        if alias_path.is_file():
            with alias_path.open(newline="", encoding="utf-8-sig") as fh:
                for row in csv.DictReader(fh):
                    sm._add_alias(row.get("alias"), row.get("station_code"), row.get("language", ""),
                                  row.get("source", ""))
        sm.source = "sqlite"
        return sm

    @classmethod
    def load(cls, db_path: Optional[Path] = None, directory: Optional[Path] = None,
             prefer_database: bool = True) -> "StationMaster":
        """SQLite station table when populated, otherwise the CSV files (``from_csv``)."""
        if prefer_database:
            try:
                return cls.from_database(db_path, alias_path=(Path(directory) / "station_aliases.csv")
                                         if directory else None)
            except StationMasterUnavailable:
                pass
        sm = cls.from_csv(directory=directory)
        sm.source = "csv"
        return sm

    def _add_station(self, code, name, lat=None, lon=None) -> None:
        code = (code or "").strip().upper()
        name = (name or "").strip()
        if not code or not name:
            return
        if code in self.stations:
            self.warnings.append(f"duplicate station code {code}: kept first entry")
            return
        self.stations[code] = Station(code, name, _float(lat), _float(lon))

    def _add_alias(self, alias, code, language="", source="") -> None:
        alias = (alias or "").strip()
        code = (code or "").strip().upper()
        if not alias or not code:
            return
        if code not in self.stations:
            self.warnings.append(f"alias {alias!r} skipped: station code {code} is not in the master")
            return
        self.aliases.append(Alias(alias, code, language or "", source or ""))

    # ---- access ---------------------------------------------------------
    def get(self, code: Optional[str]) -> Optional[Station]:
        return self.stations.get((code or "").strip().upper())

    def __contains__(self, code: str) -> bool:
        return (code or "").strip().upper() in self.stations

    def __len__(self) -> int:
        return len(self.stations)
