"""Serve the trained delay models (separate from the CNN).

Loads the two scikit-learn pipelines written by scripts/train_delay_models.py
and builds their input with the training code's own ``make_features``, so the
app cannot drift from the training-time feature engineering.

    predictor = DelayPredictor.load()
    out = predictor.predict(row)   # row: one dataset row (dict / Series / 1-row DataFrame)

Only columns available at the current station enter the model (see
``scripts/delay_common.py``: FEATURE_COLUMNS and the leakage audit).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "delay"
_SCRIPTS = str(ROOT / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
import delay_common as dc  # noqa: E402

RISK_RULE = "Low < 15 min, Medium 15-59 min, High >= 60 min (predicted next-station arrival delay)"


class DelayModelUnavailable(RuntimeError):
    """Model files are missing or do not match the feature specification."""


def risk_label(delay_min: Optional[float]) -> Optional[str]:
    if delay_min is None or pd.isna(delay_min):
        return None
    return str(dc.delay_to_risk_label([delay_min]).iloc[0])


@dataclass
class DelayPredictor:
    regressor: Any
    classifier: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, model_dir: Union[str, Path] = MODEL_DIR) -> "DelayPredictor":
        model_dir = Path(model_dir)
        files = {n: model_dir / f for n, f in (("regressor", "delay_regressor.pkl"),
                                                 ("classifier", "delay_classifier.pkl"),
                                                 ("features", "feature_columns.json"))}
        missing = [str(p) for p in files.values() if not p.exists()]
        if missing:
            raise DelayModelUnavailable("Delay model files missing: " + ", ".join(missing)
                                        + ". Run: python scripts/train_delay_models.py")
        spec = json.loads(files["features"].read_text())
        if spec.get("features") != dc.FEATURE_COLUMNS:
            raise DelayModelUnavailable("Saved feature list differs from scripts/delay_common.py; retrain the "
                                        "models with python scripts/train_delay_models.py")
        meta_path = model_dir / "metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return cls(joblib.load(files["regressor"]), joblib.load(files["classifier"]), meta)

    @staticmethod
    def features(rows: Union[Mapping[str, Any], pd.Series, pd.DataFrame]) -> pd.DataFrame:
        df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame([dict(rows)])
        missing = [c for c in dc.FEATURE_COLUMNS if c not in df.columns and c not in (
            "scheduled_arrival_hour", "scheduled_arrival_minute",
            "scheduled_departure_hour", "scheduled_departure_minute")]
        for c in missing:            # an absent optional column becomes NaN -> imputed, never invented
            df[c] = np.nan
        # the encoder learned train numbers as read from CSV without dtypes ("1023")
        df["train_no"] = df["train_no"].astype(str).str.lstrip("0")
        return dc.make_features(df)

    def unknown_categories(self, X: pd.DataFrame) -> Dict[str, List[str]]:
        enc = self.regressor.named_steps["prep"].named_transformers_["cat"]
        out = {}
        for col, cats in zip(dc.CATEGORICAL_FEATURES, enc.categories_):
            unseen = sorted(set(X[col].astype(str)) - set(map(str, cats)))
            if unseen:
                out[col] = unseen
        return out

    def predict(self, rows: Union[Mapping[str, Any], pd.Series, pd.DataFrame]) -> List[Dict[str, Any]]:
        X = self.features(rows)
        delay = self.regressor.predict(X)
        proba = self.classifier.predict_proba(X)[:, 1]
        unknown = self.unknown_categories(X)
        return [{
            "predicted_next_station_delay_min": float(d),
            "probability_delayed_15": float(p),
            "predicted_delayed_15": int(p >= 0.5),
            "risk_label": risk_label(float(d)),
            "unknown_categories": unknown,
            "models": self.metadata.get("model_type", {}),
        } for d, p in zip(delay, proba)]
