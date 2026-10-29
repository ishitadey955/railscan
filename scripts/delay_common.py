"""
RailScan - delay module: shared helpers.

Used by scripts/train_delay_models.py and scripts/evaluate_models.py (and,
later, by the ETA engine, which should call ``make_features`` and load the
saved pipelines so training and serving share exactly the same code).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- targets
REG_TARGET = "next_station_arrival_delay_min"
CLF_TARGET = "next_station_delayed_15"
RISK_TARGET = "next_delay_risk_label"
RISK_LABELS = ["Low", "Medium", "High"]

# --------------------------------------------------------------- features
# Everything below is known at the moment the train is observed at the
# current station (schedule data, current observed delays, past history).
NUMERIC_FEATURES = [
    "seq",
    "current_arrival_delay_min", "current_departure_delay_min",
    "scheduled_arrival_hour", "scheduled_arrival_minute",
    "scheduled_departure_hour", "scheduled_departure_minute",
    "scheduled_stop_min", "scheduled_travel_to_next_min", "distance_to_next_km",
    "day_of_week", "month", "is_weekend",
    "historical_avg_delay_min", "historical_delay_std_min",
    "historical_observation_count", "historical_on_time_rate_pct",
    "historical_avg_actual_stop_min", "historical_stop_std_min",
    "historical_stop_observation_count", "historical_stop_reliability_pct",
]
# Identifier-like columns: one-hot encoded (no target encoding => no leakage).
# next_station_code is an addition to the suggested list: the identity of the
# upcoming station is schedule information known in advance.
CATEGORICAL_FEATURES = ["train_no", "train_type", "station_code", "next_station_code"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Names/patterns that must never appear among the features.
FORBIDDEN_FEATURE_PATTERNS = [
    r"^next_station_arrival_delay_min$", r"^next_station_delayed_15$",
    r"^next_delay_risk_label$", r"^target_", r"next.*actual", r"actual.*arrival",
    r"actual.*departure", r"future", r"^next_.*delay", r"expected_next_station",
]
# 'actual_stop_min' of the CURRENT stop is allowed only through its
# historical average; the current-stop value itself is not in the list above.


# ------------------------------------------------------------------ data
def load_splits(data_dir: Path) -> Dict[str, pd.DataFrame]:
    splits = {}
    for name in ("train", "validation", "test"):
        path = Path(data_dir) / f"{name}.csv"
        if not path.exists():
            raise SystemExit(f"Missing input file: {path}. Run scripts/build_dataset.py first.")
        df = pd.read_csv(path)
        for c in ("journey_date", "observation_time", "target_next_actual_arrival",
                  "current_scheduled_arrival", "current_scheduled_departure"):
            df[c] = pd.to_datetime(df[c], errors="coerce")
        splits[name] = df
    return splits


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the model input from a dataset frame (also reusable at serving time)."""
    sa = pd.to_datetime(df["current_scheduled_arrival"], errors="coerce")
    sd = pd.to_datetime(df["current_scheduled_departure"], errors="coerce")
    X = df.copy()
    # A departure not yet recorded: the latest known delay is the arrival delay
    # (same rule as current_observed_delay_min). Median imputation would replace
    # e.g. a 294-min-late arrival with a ~30-min delay.
    X["current_departure_delay_min"] = X["current_departure_delay_min"].fillna(X["current_arrival_delay_min"])
    X["scheduled_arrival_hour"] = sa.dt.hour
    X["scheduled_arrival_minute"] = sa.dt.minute
    X["scheduled_departure_hour"] = sd.dt.hour
    X["scheduled_departure_minute"] = sd.dt.minute
    X["train_no"] = X["train_no"].astype(str)
    for c in ("train_type", "station_code", "next_station_code"):
        X[c] = X[c].astype(str)
    return X[FEATURE_COLUMNS]


def build_preprocessor() -> ColumnTransformer:
    """Median imputation (+ missing indicators) and scaling for numbers; one-hot for categories."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler())])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def source_feature_name(transformed_name: str) -> str:
    """Map a transformed column name (e.g. 'cat__station_code_JSP') back to its raw feature."""
    n = transformed_name.split("__", 1)[1]
    if n.startswith("missingindicator_"):
        return n[len("missingindicator_"):]
    for col in sorted(CATEGORICAL_FEATURES, key=len, reverse=True):
        if n.startswith(col + "_"):
            return col
    return n


# --------------------------------------------------------- leakage audit
def leakage_audit(features: List[str], splits: Dict[str, pd.DataFrame]) -> None:
    """Print the feature list and stop with a clear error if anything leaks."""
    print("\n=== LEAKAGE AUDIT ===")
    print(f"Final feature list ({len(features)}):")
    for f in features:
        print(f"  - {f}")

    problems = []
    for f in features:
        for pat in FORBIDDEN_FEATURE_PATTERNS:
            if re.search(pat, f):
                problems.append(f"feature '{f}' matches forbidden pattern '{pat}'")
    for t in (REG_TARGET, CLF_TARGET, RISK_TARGET):
        if t in features:
            problems.append(f"target '{t}' is used as a feature")

    for name, df in splits.items():
        late = (df["observation_time"] > df["target_next_actual_arrival"]).sum()
        if late:
            problems.append(f"{name}: {late} rows observed AFTER the next actual arrival")
    tr, va, te = splits["train"], splits["validation"], splits["test"]
    if not (tr.journey_date.max() < va.journey_date.min() and va.journey_date.max() < te.journey_date.min()):
        problems.append("splits are not strictly chronological")

    checks = [
        "no target column used as a feature",
        "no next-station actual arrival/departure in features",
        "no 'future' delay / stop-duration features",
        "observation_time <= next actual arrival in every split",
        "train < validation < test in time",
    ]
    if problems:
        raise SystemExit("\nLEAKAGE DETECTED - stopping.\n  " + "\n  ".join(problems))
    for c in checks:
        print(f"  [PASS] {c}")


# --------------------------------------------------------------- metrics
def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    return {"MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": float(r2_score(y_true, y_pred))}


def classification_metrics(y_true, y_pred, proba=None) -> Dict[str, float]:
    """Precision/recall/F1 are for the positive class (delayed >= 15 min)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {"accuracy": float(accuracy_score(y_true, y_pred)),
           "precision": float(precision_score(y_true, y_pred, zero_division=0)),
           "recall": float(recall_score(y_true, y_pred, zero_division=0)),
           "f1": float(f1_score(y_true, y_pred, zero_division=0)),
           "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
           "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
           "roc_auc": float("nan"),
           "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    if proba is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, proba))
    return out


def delay_to_risk_label(delay_min) -> pd.Series:
    """Low < 15, Medium 15-59, High >= 60 (same rule as the dataset builder)."""
    s = pd.Series(np.asarray(delay_min, dtype=float))
    return pd.cut(s, bins=[-np.inf, 15, 60, np.inf], labels=RISK_LABELS, right=False).astype(str)
