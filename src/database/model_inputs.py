"""Turn ``train_runs`` rows into the exact input of the saved delay models.

The models in ``models/delay/`` were trained by ``scripts/train_delay_models.py``
on frames read with ``pd.read_csv`` (no dtypes) and passed through
``delay_common.make_features``. This module rebuilds that same frame from
database rows and then calls the SAME ``make_features`` - it does not
re-implement feature engineering.

One detail matters: ``pd.read_csv`` parsed train numbers as integers, so the
saved one-hot encoder learned ``"1023"``, not ``"01023"``. The encoder ignores
unknown categories silently, so passing the zero-padded form would change
predictions without any error. ``to_training_frame`` reproduces the integer
form, and ``unknown_categories`` reports any value the model has never seen.

Target columns never enter the features: ``make_features`` selects only
``FEATURE_COLUMNS`` and ``train_runs`` rows from ``repository.get_train_state``
do not even contain them.
"""
from __future__ import annotations

import importlib
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from .connection import ROOT
from .schema import DB_TO_DATASET, TARGET_COLUMNS

DELAY_MODEL_DIR = ROOT / "models" / "delay"
DATASET_TARGETS = set(TARGET_COLUMNS)


def delay_common():
    """Import ``scripts/delay_common.py`` (the training pipeline's own feature code)."""
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module("delay_common")


def _training_train_no(v: Any) -> Any:
    """'01023' -> '1023' (what pd.read_csv + astype(str) produced at training time)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = str(v).strip()
    return str(int(s)) if s.isdigit() else s


def to_training_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Database rows (dicts) -> frame with dataset column names and training-time dtypes.

    Target columns are dropped even if present in ``rows``.
    """
    rows = list(rows)
    frame = pd.DataFrame({src: [r.get(db) for r in rows] for db, src in DB_TO_DATASET.items()
                          if src not in DATASET_TARGETS})
    dc = delay_common()
    for c in dc.NUMERIC_FEATURES:
        if c in frame.columns:
            frame[c] = pd.to_numeric(frame[c], errors="coerce").astype("float64")
    frame["train_no"] = frame["train_no"].map(_training_train_no)
    for c in ("train_type", "station_code", "next_station_code"):
        frame[c] = frame[c].astype(object).where(frame[c].notna(), np.nan)   # None -> NaN, as read_csv gives
    return frame


def make_model_features(rows: Sequence[Mapping[str, Any]], model_dir: Path = DELAY_MODEL_DIR) -> pd.DataFrame:
    """Rows -> X in the exact column order stored in ``models/delay/feature_columns.json``."""
    dc = delay_common()
    X = dc.make_features(to_training_frame(rows))
    saved = feature_spec(str(model_dir))["features"]
    if list(X.columns) != saved:
        raise RuntimeError(f"feature order differs from {model_dir}/feature_columns.json: "
                           f"{list(X.columns)} != {saved}")
    leaked = [c for c in X.columns if c in DATASET_TARGETS]
    if leaked:
        raise RuntimeError(f"target column(s) in model input: {leaked}")
    return X


@lru_cache(maxsize=4)
def feature_spec(model_dir: str) -> Dict[str, List[str]]:
    with open(Path(model_dir) / "feature_columns.json", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=2)
def load_delay_models(model_dir: str = str(DELAY_MODEL_DIR)) -> Dict[str, Any]:
    """Load the saved pipelines (no retraining). Cached per directory."""
    import joblib
    d = Path(model_dir)
    missing = [f for f in ("delay_regressor.pkl", "delay_classifier.pkl", "feature_columns.json")
               if not (d / f).is_file()]
    if missing:
        raise FileNotFoundError(f"missing model artifacts in {d}: {missing}. Run scripts/train_delay_models.py.")
    return {"regressor": joblib.load(d / "delay_regressor.pkl"),
            "classifier": joblib.load(d / "delay_classifier.pkl"),
            "features": feature_spec(str(d))}


def unknown_categories(X: pd.DataFrame, pipeline: Any) -> Dict[str, List[str]]:
    """Categorical values the saved one-hot encoder never saw (they are ignored by the model)."""
    try:
        ct = pipeline.named_steps["prep"]
        enc = ct.named_transformers_["cat"]
        cols = next(c for name, _t, c in ct.transformers_ if name == "cat")
    except Exception:
        return {}
    out: Dict[str, List[str]] = {}
    for col, cats in zip(cols, enc.categories_):
        unseen = sorted({str(v) for v in X[col] if str(v) not in set(map(str, cats))})
        if unseen:
            out[col] = unseen
    return out


def predict_delay_for_rows(rows: Sequence[Mapping[str, Any]], model_dir: str = str(DELAY_MODEL_DIR)
                           ) -> List[Dict[str, Any]]:
    """Saved-model predictions for database rows (used to verify parity with the CSV path).

    This is not the ETA engine; it only proves the database feeds the models
    exactly what they were trained on.
    """
    models = load_delay_models(model_dir)
    X = make_model_features(rows, Path(model_dir))
    reg = models["regressor"].predict(X)
    proba = models["classifier"].predict_proba(X)[:, 1]
    pred = models["classifier"].predict(X)
    unseen = unknown_categories(X, models["regressor"])
    return [{"predicted_next_delay_min": float(reg[i]), "delayed_15_probability": float(proba[i]),
             "delayed_15_prediction": int(pred[i]), "unknown_categories": unseen} for i in range(len(X))]
