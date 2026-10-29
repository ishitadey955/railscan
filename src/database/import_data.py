"""Import RailScan CSV sources into SQLite.

    import_dataset()   data/processed/RailScan_Final_Dataset.csv  -> train_runs + train_metadata
    import_stations()  data/stations/station_master.csv (+ supplement) -> stations

Rules
    * Required columns are validated; missing optional columns are reported.
    * Empty / NaN cells stay SQL NULL. Unparseable numbers, dates or
      out-of-range values become NULL and are counted per column - never
      replaced by an invented value.
    * Rows whose key (train_no, journey_date, station_code, seq) is missing
      or invalid are rejected and counted; they cannot be stored honestly.
    * Duplicate keys inside the CSV are handled explicitly: identical copies
      are collapsed; conflicting copies stop the import unless
      ``duplicate_policy="keep-first"``.
    * Everything runs in ONE transaction with batched ``executemany``. Any
      error rolls the whole import back; the failure is then recorded in
      ``database_metadata`` in a separate transaction.
    * Default mode is a non-destructive upsert on the natural key. Rows that
      are in the database but no longer in the CSV are reported, not deleted.
      ``replace=True`` swaps the table contents for the CSV (the number of
      replaced rows is reported and recorded).

``train.csv`` / ``validation.csv`` / ``test.csv`` are deliberately NOT
imported: they are splits of RailScan_Final_Dataset.csv, so importing them
would duplicate every observation.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.railway.railway_data import normalize_train_no

from .connection import ROOT, PathLike, connect, resolve_db_path, table_count, transaction
from .schema import (
    INTEGER_COLUMNS, OPTIONAL_DATASET_COLUMNS, OPTIONAL_STATION_COLUMNS, REAL_COLUMNS,
    REQUIRED_DATASET_COLUMNS, REQUIRED_STATION_COLUMNS, TIMESTAMP_COLUMNS, TRAIN_RUN_COLUMNS,
    initialize_database,
)

DEFAULT_DATASET = ROOT / "data" / "processed" / "RailScan_Final_Dataset.csv"
DEFAULT_STATION_DIR = ROOT / "data" / "stations"
DEFAULT_TIMETABLE = ROOT / "data" / "processed" / "RailScan_Timetable.csv"
SPLIT_FILES = ("train.csv", "validation.csv", "test.csv")
TS_FORMAT = "%Y-%m-%d %H:%M:%S"
DERIVED_BASIS = "derived: scheduled time + observed delay (dataset stores delays, not raw actual timestamps)"


class ImportFailed(RuntimeError):
    """The import was stopped; nothing was written to the target tables."""


@dataclass
class ImportReport:
    source_file: str
    target_table: str
    status: str = "pending"                     # success | failed
    mode: str = "upsert"
    rows_read: int = 0
    rows_imported: int = 0                      # rows written (inserted + updated)
    rows_inserted: int = 0
    rows_updated: int = 0
    rows_replaced: int = 0                      # rows removed by replace=True
    rows_rejected: int = 0
    rejected_reasons: Dict[str, int] = field(default_factory=dict)
    duplicates_identical: int = 0
    duplicates_conflicting: int = 0
    invalid_values_set_null: Dict[str, int] = field(default_factory=dict)
    missing_optional_columns: List[str] = field(default_factory=list)
    unknown_station_codes: List[str] = field(default_factory=list)
    rows_in_db_not_in_csv: int = 0
    table_rows_after: int = 0
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    metadata_id: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================ helpers
def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def find_dataset_file(path: Optional[PathLike] = None) -> Optional[Path]:
    """Explicit path > default location > the first RailScan_Final_Dataset.csv found under the project."""
    if path:
        p = Path(path)
        return p if p.is_file() else None
    if DEFAULT_DATASET.is_file():
        return DEFAULT_DATASET
    for p in sorted(ROOT.rglob("RailScan_Final_Dataset.csv")):
        if ".git" not in p.parts and "site-packages" not in p.parts:
            return p
    return None


def find_station_files(master: Optional[PathLike] = None, supplement: Optional[PathLike] = None
                       ) -> Tuple[Optional[Path], Optional[Path]]:
    if master:
        mp = Path(master)
        mp = mp if mp.is_file() else None
    else:
        mp = DEFAULT_STATION_DIR / "station_master.csv"
        if not mp.is_file():
            mp = next((p for p in sorted(ROOT.rglob("station_master.csv")) if ".git" not in p.parts), None)
    if supplement:
        sp = Path(supplement)
    elif mp is not None:
        sp = mp.parent / "station_master_supplement.csv"
    else:
        sp = None
    return mp, (sp if sp is not None and sp.is_file() else None)


def _py(v: Any) -> Any:
    """pandas / numpy scalar -> plain Python value that sqlite3 can bind; missing -> None."""
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, np.generic):
        v = v.item()
        if isinstance(v, float) and math.isnan(v):
            return None
    return v


def _blank(s: pd.Series) -> pd.Series:
    """True where the raw cell is empty / NaN / whitespace."""
    return s.isna() | (s.astype("string").str.strip() == "")


def _clean_text(s: pd.Series, upper: bool = False) -> pd.Series:
    out = s.astype("string").str.strip()
    if upper:
        out = out.str.upper()
    return out.mask(out == "", pd.NA)


def _numeric(raw: pd.Series, col: str, report: ImportReport, integer: bool = False,
             lo: Optional[float] = None, hi: Optional[float] = None) -> pd.Series:
    vals = pd.to_numeric(raw.astype("string").str.strip(), errors="coerce").astype("float64")
    vals = vals.where(np.isfinite(vals))
    bad = vals.isna() & ~_blank(raw)
    if integer:
        frac = vals.notna() & (vals != np.round(vals))
        bad |= frac
        vals = vals.mask(frac)
    if lo is not None or hi is not None:
        oor = vals.notna() & (((vals < lo) if lo is not None else False) | ((vals > hi) if hi is not None else False))
        bad |= oor
        vals = vals.mask(oor)
    if int(bad.sum()):
        report.invalid_values_set_null[col] = report.invalid_values_set_null.get(col, 0) + int(bad.sum())
    return vals.round().astype("Int64") if integer else vals


def _timestamps(raw: pd.Series, col: str, report: ImportReport) -> pd.Series:
    ts = pd.to_datetime(raw.astype("string").str.strip(), errors="coerce", format="ISO8601")
    bad = ts.isna() & ~_blank(raw)
    if int(bad.sum()):
        report.invalid_values_set_null[col] = report.invalid_values_set_null.get(col, 0) + int(bad.sum())
    return ts


def _reject(report: ImportReport, reason: str, n: int) -> None:
    if n:
        report.rejected_reasons[reason] = report.rejected_reasons.get(reason, 0) + int(n)
        report.rows_rejected += int(n)


def _batches(rows: Sequence[tuple], size: int) -> Iterable[Sequence[tuple]]:
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def _record_metadata(conn: sqlite3.Connection, report: ImportReport, sha: Optional[str]) -> int:
    notes = {k: v for k, v in report.as_dict().items()
             if k not in ("source_file", "target_table", "status", "metadata_id") and v not in (None, [], {}, 0)}
    cur = conn.execute(
        "INSERT INTO database_metadata (source_file, imported_at, rows_imported, status, notes, target_table, "
        "source_sha256) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (report.source_file, _utc_now(), report.rows_imported, report.status,
         json.dumps(notes, ensure_ascii=False, default=str), report.target_table, sha))
    return int(cur.lastrowid)


def _record_failure(db_path: Path, report: ImportReport, sha: Optional[str]) -> None:
    """Separate transaction, after the import's rollback. Never raises."""
    try:
        initialize_database(db_path)
        with connect(db_path) as conn, transaction(conn):
            report.metadata_id = _record_metadata(conn, report, sha)
    except Exception as exc:  # pragma: no cover - only if the DB itself is broken
        report.warnings.append(f"could not record the failure in database_metadata: {exc}")


# ============================================================================ dataset
def prepare_train_runs(frame: pd.DataFrame, report: ImportReport) -> pd.DataFrame:
    """Validate and normalise the processed dataset into ``train_runs`` columns."""
    missing_req = [c for c in REQUIRED_DATASET_COLUMNS if c not in frame.columns]
    if missing_req:
        raise ImportFailed(f"dataset is missing required columns: {missing_req}")
    report.missing_optional_columns = [c for c in OPTIONAL_DATASET_COLUMNS if c not in frame.columns]
    src = frame.copy()
    for c in report.missing_optional_columns:
        src[c] = pd.NA                      # absent column -> NULL, never a made-up value

    out = pd.DataFrame(index=src.index)
    # ---- keys
    out["train_no"] = src["train_no"].map(normalize_train_no).astype("string")
    out["train_key"] = out["train_no"].str.lstrip("0")
    out.loc[out["train_no"].notna() & (out["train_key"] == ""), "train_key"] = "0"
    jd = pd.to_datetime(src["journey_date"].astype("string").str.strip(), errors="coerce", format="ISO8601")
    out["journey_date"] = jd.dt.strftime("%Y-%m-%d").astype("string")
    out["station_code"] = _clean_text(src["station_code"], upper=True)
    seq_report = ImportReport("", "")
    out["seq"] = _numeric(src["seq"], "seq", seq_report, integer=True)

    reasons = {
        "invalid train_no (expected 4-5 digits)": out["train_no"].isna(),
        "invalid journey_date": out["journey_date"].isna(),
        "missing station_code": out["station_code"].isna(),
        "invalid seq (not an integer)": out["seq"].isna(),
    }
    bad = pd.Series(False, index=out.index)
    for reason, mask in reasons.items():
        _reject(report, reason, int((mask & ~bad).sum()))
        bad |= mask

    # ---- other columns
    for db_col, sqltype, src_col in TRAIN_RUN_COLUMNS:
        if src_col is None or db_col in out.columns:
            continue
        raw = src[src_col]
        if db_col in TIMESTAMP_COLUMNS:
            out[db_col] = _timestamps(raw, db_col, report)
        elif db_col in INTEGER_COLUMNS:
            lo_hi = (0, 1) if db_col in ("is_weekend", "next_station_delayed_15") else (None, None)
            out[db_col] = _numeric(raw, db_col, report, integer=True, lo=lo_hi[0], hi=lo_hi[1])
        elif db_col in REAL_COLUMNS:
            lo, hi = {"station_lat": (-90, 90), "station_lng": (-180, 180)}.get(db_col, (None, None))
            out[db_col] = _numeric(raw, db_col, report, lo=lo, hi=hi)
        else:
            out[db_col] = _clean_text(raw, upper=(db_col == "next_station_code"))

    # ---- derived actual times (scheduled + delay); NULL when either part is missing
    arr = out["scheduled_arrival"] + pd.to_timedelta(out["current_arrival_delay_min"], unit="m")
    dep = out["scheduled_departure"] + pd.to_timedelta(out["current_departure_delay_min"], unit="m")
    out["actual_arrival"] = arr
    out["actual_departure"] = dep
    out["actual_time_basis"] = pd.Series(DERIVED_BASIS, index=out.index, dtype="string").where(
        arr.notna() | dep.notna())
    for c in TIMESTAMP_COLUMNS | {"actual_arrival", "actual_departure"}:
        out[c] = out[c].dt.strftime(TS_FORMAT).astype("string")

    out = out[~bad]
    ordered = [c for c, _t, _s in TRAIN_RUN_COLUMNS]
    return out[ordered].reset_index(drop=True)


def _dedupe(df: pd.DataFrame, key: List[str], report: ImportReport, policy: str) -> pd.DataFrame:
    dup_any = df.duplicated(key, keep=False)
    if not dup_any.any():
        return df
    identical = df.duplicated(keep="first")            # every column equal to an earlier row
    report.duplicates_identical = int(identical.sum())
    df = df[~identical]
    conflicting = df.duplicated(key, keep="first")
    report.duplicates_conflicting = int(conflicting.sum())
    if report.duplicates_conflicting:
        sample = df[df.duplicated(key, keep=False)][key].drop_duplicates().head(5).to_dict("records")
        if policy != "keep-first":
            raise ImportFailed(
                f"{report.duplicates_conflicting} rows share a key {key} with different values "
                f"(e.g. {sample}). Fix the CSV or re-run with --duplicates keep-first.")
        report.warnings.append(f"kept the first of {report.duplicates_conflicting} conflicting duplicate(s), "
                               f"e.g. {sample}")
        df = df[~conflicting]
    return df


def rebuild_train_metadata(conn: sqlite3.Connection) -> int:
    """Recompute train_metadata from train_runs (inside the caller's transaction)."""
    rows = conn.execute(
        "SELECT train_no, train_name, train_type, COUNT(*) AS n, MIN(journey_date) AS first_d, "
        "MAX(journey_date) AS last_d FROM train_runs GROUP BY train_no, train_name, train_type "
        "ORDER BY train_no, first_d").fetchall()
    per: Dict[str, List[sqlite3.Row]] = {}
    for r in rows:
        per.setdefault(r["train_no"], []).append(r)
    for tno, groups in per.items():
        names = {g["train_name"] for g in groups if g["train_name"] is not None}
        types = {g["train_type"] for g in groups if g["train_type"] is not None}
        history = [{"train_name": g["train_name"], "train_type": g["train_type"], "first_journey_date": g["first_d"],
                    "last_journey_date": g["last_d"], "rows": g["n"]} for g in groups]
        conn.execute(
            "INSERT INTO train_metadata (train_no, train_name, train_type, name_type_history, first_journey_date, "
            "last_journey_date, observation_count) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(train_no) DO UPDATE SET train_name = excluded.train_name, train_type = excluded.train_type, "
            "name_type_history = excluded.name_type_history, first_journey_date = excluded.first_journey_date, "
            "last_journey_date = excluded.last_journey_date, observation_count = excluded.observation_count",
            (tno, next(iter(names)) if len(names) == 1 else None, next(iter(types)) if len(types) == 1 else None,
             json.dumps(history, ensure_ascii=False), min(g["first_d"] for g in groups),
             max(g["last_d"] for g in groups), sum(g["n"] for g in groups)))
    conn.execute("DELETE FROM train_metadata WHERE train_no NOT IN (SELECT DISTINCT train_no FROM train_runs)")
    return len(per)


def import_dataset(csv_path: Optional[PathLike] = None, db_path: Optional[PathLike] = None, *,
                   replace: bool = False, duplicate_policy: str = "fail", batch_size: int = 1000) -> ImportReport:
    """Import RailScan_Final_Dataset.csv into ``train_runs`` / ``train_metadata``. Never raises."""
    db = resolve_db_path(db_path)
    path = find_dataset_file(csv_path)
    report = ImportReport(source_file=str(csv_path or DEFAULT_DATASET), target_table="train_runs",
                          mode="replace" if replace else "upsert")
    if path is None:
        report.status = "failed"
        report.error = (f"Processed dataset not found ({report.source_file}). Build it with "
                        "`python scripts/build_dataset.py ...` or pass --csv PATH. "
                        "train.csv / validation.csv / test.csv are not used: they are splits of the same data.")
        _record_failure(db, report, None)
        return report
    if path.name in SPLIT_FILES:
        report.status = "failed"
        report.error = (f"{path.name} is a split of RailScan_Final_Dataset.csv; importing it as independent "
                        "observations would duplicate records. Import RailScan_Final_Dataset.csv instead.")
        _record_failure(db, report, None)
        return report

    report.source_file = str(path)
    sha = None
    try:
        sha = file_sha256(path)
        frame = pd.read_csv(path, dtype=str, keep_default_na=True, encoding="utf-8-sig")
        report.rows_read = len(frame)
        if frame.empty:
            raise ImportFailed(f"{path} contains no rows")
        df = prepare_train_runs(frame, report)
        df = _dedupe(df, ["train_key", "journey_date", "seq"], report, duplicate_policy)
        if df.empty:
            raise ImportFailed("no valid rows left after validation")

        cols = list(df.columns)
        rows = [tuple(_py(v) for v in rec) for rec in df.astype(object).itertuples(index=False, name=None)]
        placeholders = ", ".join("?" for _ in cols)
        updates = ", ".join(f"{c} = excluded.{c}" for c in cols if c not in ("train_key", "journey_date", "seq"))
        sql = (f"INSERT INTO train_runs ({', '.join(cols)}) VALUES ({placeholders}) "
               f"ON CONFLICT(train_key, journey_date, seq) DO UPDATE SET {updates}")

        initialize_database(db)
        with connect(db) as conn:
            with transaction(conn):
                if replace:
                    report.rows_replaced = table_count(conn, "train_runs")
                    conn.execute("DELETE FROM train_runs")
                before = table_count(conn, "train_runs")
                conn.executemany("INSERT OR IGNORE INTO train_metadata (train_no) VALUES (?)",
                                 sorted({(r[cols.index("train_no")],) for r in rows}))
                for chunk in _batches(rows, batch_size):
                    conn.executemany(sql, chunk)
                after = table_count(conn, "train_runs")
                rebuild_train_metadata(conn)
                report.rows_imported = len(rows)
                report.rows_inserted = after - before
                report.rows_updated = len(rows) - report.rows_inserted
                report.table_rows_after = after
                report.rows_in_db_not_in_csv = after - len(rows)
                if report.rows_in_db_not_in_csv:
                    report.warnings.append(
                        f"{report.rows_in_db_not_in_csv} row(s) in train_runs are not in this CSV (kept). "
                        "Use --replace after regenerating the dataset.")
                if table_count(conn, "stations"):
                    codes = set(df["station_code"].dropna()) | set(df["next_station_code"].dropna())
                    known = set()
                    code_list = sorted(codes)
                    for i in range(0, len(code_list), 500):
                        part = code_list[i:i + 500]
                        q = f"SELECT station_code FROM stations WHERE station_code IN ({', '.join('?' * len(part))})"
                        known |= {r[0] for r in conn.execute(q, part)}
                    report.unknown_station_codes = sorted(codes - known)
                    if report.unknown_station_codes:
                        report.warnings.append(f"{len(report.unknown_station_codes)} station code(s) are not in "
                                               "the stations table (rows kept)")
                else:
                    report.warnings.append("stations table is empty: station codes were not cross-checked "
                                           "(run import-stations)")
                report.status = "success"
                report.metadata_id = _record_metadata(conn, report, sha)
    except Exception as exc:
        report.status = "failed"
        report.error = f"{type(exc).__name__}: {exc}"
        report.rows_imported = report.rows_inserted = report.rows_updated = report.rows_replaced = 0
        _record_failure(db, report, sha)
    return report


# ============================================================================ stations
def _read_source_label(master_path: Path) -> Optional[str]:
    """'<source url> [licence: <licence>]' from SOURCE.txt next to the master, if present."""
    f = master_path.parent / "SOURCE.txt"
    if not f.is_file():
        return None
    fields = {}
    for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
        key, sep, val = line.partition(":")
        if sep and val.strip():
            fields.setdefault(key.strip().lower(), val.strip())
    src = fields.get("source")
    if not src:
        return None
    lic = fields.get("licence") or fields.get("license")
    return f"{src} [licence: {lic}]" if lic else src


def prepare_stations(frame: pd.DataFrame, default_source: str, report: ImportReport) -> pd.DataFrame:
    missing = [c for c in REQUIRED_STATION_COLUMNS if c not in frame.columns]
    if missing:
        raise ImportFailed(f"station file is missing required columns: {missing}")
    report.missing_optional_columns = sorted(set(report.missing_optional_columns) |
                                             {c for c in OPTIONAL_STATION_COLUMNS if c not in frame.columns})
    out = pd.DataFrame(index=frame.index)
    out["station_code"] = _clean_text(frame["station_code"], upper=True)
    out["station_name"] = _clean_text(frame["station_name"])
    for c, (lo, hi) in (("latitude", (-90, 90)), ("longitude", (-180, 180))):
        out[c] = _numeric(frame[c], c, report, lo=lo, hi=hi) if c in frame.columns else np.nan
    src = _clean_text(frame["source"]) if "source" in frame.columns else pd.Series(pd.NA, index=frame.index,
                                                                                 dtype="string")
    out["source"] = src.fillna(default_source)
    bad_code, bad_name = out["station_code"].isna(), out["station_name"].isna()
    _reject(report, "missing station_code", int(bad_code.sum()))
    _reject(report, "missing station_name", int((bad_name & ~bad_code).sum()))
    return out[~(bad_code | bad_name)].reset_index(drop=True)


def import_stations(master_path: Optional[PathLike] = None, supplement_path: Optional[PathLike] = None,
                    db_path: Optional[PathLike] = None, *, replace: bool = False,
                    batch_size: int = 1000) -> ImportReport:
    """Import the station master (+ optional supplement) into ``stations``. Never raises.

    Same precedence as ``StationMaster.from_csv``: within the files the FIRST
    entry of a code wins (master before supplement); later ones are reported.
    """
    db = resolve_db_path(db_path)
    mp, sp = find_station_files(master_path, supplement_path)
    report = ImportReport(source_file=str(master_path or DEFAULT_STATION_DIR / "station_master.csv"),
                          target_table="stations", mode="replace" if replace else "upsert")
    if mp is None:
        report.status = "failed"
        report.error = (f"Station master not found ({report.source_file}). Run "
                        "`python scripts/build_station_master.py` or pass --master PATH.")
        _record_failure(db, report, None)
        return report
    report.source_file = str(mp) + (f" + {sp}" if sp else "")
    sha = None
    try:
        sha = file_sha256(mp)
        parts = []
        mframe = pd.read_csv(mp, dtype=str, encoding="utf-8-sig")
        parts.append(prepare_stations(mframe, _read_source_label(mp) or mp.name, report))
        report.rows_read = len(mframe)
        if sp is not None:
            sframe = pd.read_csv(sp, dtype=str, encoding="utf-8-sig")
            report.rows_read += len(sframe)
            if len(sframe):
                parts.append(prepare_stations(sframe, sp.name, report))
        df = pd.concat(parts, ignore_index=True)
        if df.empty:
            raise ImportFailed("no valid station rows")
        identical = df.duplicated(keep="first")
        report.duplicates_identical = int(identical.sum())
        df = df[~identical]
        conflicting = df.duplicated("station_code", keep="first")
        report.duplicates_conflicting = int(conflicting.sum())
        if report.duplicates_conflicting:
            report.warnings.append(f"{report.duplicates_conflicting} duplicate station code(s) with different "
                                   f"values: kept the first entry (master before supplement), e.g. "
                                   f"{df[conflicting]['station_code'].head(5).tolist()}")
            df = df[~conflicting]

        cols = ["station_code", "station_name", "latitude", "longitude", "source"]
        rows = [tuple(_py(v) for v in rec) for rec in df[cols].astype(object).itertuples(index=False, name=None)]
        sql = ("INSERT INTO stations (station_code, station_name, latitude, longitude, source) "
               "VALUES (?, ?, ?, ?, ?) ON CONFLICT(station_code) DO UPDATE SET "
               "station_name = excluded.station_name, latitude = excluded.latitude, "
               "longitude = excluded.longitude, source = excluded.source")
        initialize_database(db)
        with connect(db) as conn:
            with transaction(conn):
                if replace:
                    report.rows_replaced = table_count(conn, "stations")
                    conn.execute("DELETE FROM stations")
                before = table_count(conn, "stations")
                for chunk in _batches(rows, batch_size):
                    conn.executemany(sql, chunk)
                after = table_count(conn, "stations")
                report.rows_imported = len(rows)
                report.rows_inserted = after - before
                report.rows_updated = len(rows) - report.rows_inserted
                report.table_rows_after = after
                report.rows_in_db_not_in_csv = after - len(rows)
                if report.rows_in_db_not_in_csv:
                    report.warnings.append(f"{report.rows_in_db_not_in_csv} station(s) in the database are not "
                                           "in these files (kept). Use --replace to mirror the files exactly.")
                report.status = "success"
                report.metadata_id = _record_metadata(conn, report, sha)
    except Exception as exc:
        report.status = "failed"
        report.error = f"{type(exc).__name__}: {exc}"
        report.rows_imported = report.rows_inserted = report.rows_updated = report.rows_replaced = 0
        _record_failure(db, report, sha)
    return report


# ============================================================================ timetable
def import_timetable(csv_path: Optional[PathLike] = None, db_path: Optional[PathLike] = None) -> ImportReport:
    """Replace the ``timetable`` table with RailScan_Timetable.csv (a snapshot). Never raises."""
    from .schema import TIMETABLE_COLUMNS
    db = resolve_db_path(db_path)
    path = Path(csv_path) if csv_path else DEFAULT_TIMETABLE
    report = ImportReport(source_file=str(path), target_table="timetable", mode="replace")
    if not path.is_file():
        report.status = "failed"
        report.error = f"Timetable not found ({path}). Run `python scripts/build_timetable.py`."
        _record_failure(db, report, None)
        return report
    sha = None
    try:
        sha = file_sha256(path)
        df = pd.read_csv(path, dtype={"train_no": str, "sch_arrival": str, "sch_departure": str})
        report.rows_read = len(df)
        missing = [c for c in TIMETABLE_COLUMNS if c not in df.columns]
        if missing:
            raise ImportFailed(f"timetable columns missing: {missing}")
        df["train_no"] = df["train_no"].str.zfill(5)
        if df.duplicated(["train_no", "seq"]).any():
            raise ImportFailed("duplicate (train_no, seq) in the timetable")
        rows = [tuple(_py(v) for v in rec) for rec in df[TIMETABLE_COLUMNS].astype(object).itertuples(index=False, name=None)]
        sql = (f"INSERT INTO timetable ({', '.join(TIMETABLE_COLUMNS)}) "
               f"VALUES ({', '.join('?' for _ in TIMETABLE_COLUMNS)})")
        initialize_database(db)
        with connect(db) as conn:
            with transaction(conn):
                report.rows_replaced = table_count(conn, "timetable")
                conn.execute("DELETE FROM timetable")
                conn.executemany(sql, rows)
                report.rows_imported = report.rows_inserted = len(rows)
                report.table_rows_after = table_count(conn, "timetable")
                report.status = "success"
                report.metadata_id = _record_metadata(conn, report, sha)
    except Exception as exc:
        report.status = "failed"
        report.error = f"{type(exc).__name__}: {exc}"
        report.rows_imported = report.rows_inserted = 0
        _record_failure(db, report, sha)
    return report
