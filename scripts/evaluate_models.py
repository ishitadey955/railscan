#!/usr/bin/env python3
"""
RailScan - one-time evaluation of the SELECTED delay models.

Loads the artifacts written by train_delay_models.py, evaluates the selected
regressor and classifier on validation and on the held-out TEST set, adds a
simple persistence baseline for context, and writes:

  reports/results/regression_results.csv
  reports/results/classification_results.csv
  reports/results/confusion_matrix.png
  reports/results/risk_confusion_matrix.png
  models/delay/metadata.json   (test metrics are added)

Candidate models are listed with their VALIDATION metrics (read from
metadata.json, computed at training time). Only the selected models are
scored on the test set; nothing is tuned on it.

Run:  python scripts/evaluate_models.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import delay_common as dc  # noqa: E402

BASELINE = "Persistence baseline (next delay = current delay)"


def plot_confusion(cm: np.ndarray, labels: list, title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title(f"{title}\ncounts")
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    ConfusionMatrixDisplay(norm, display_labels=labels).plot(ax=axes[1], cmap="Blues",
                                                             colorbar=False, values_format=".2f")
    axes[1].set_title(f"{title}\nrow-normalised (recall per class)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(dc.ROOT / "data" / "processed"))
    ap.add_argument("--models-dir", default=str(dc.ROOT / "models" / "delay"))
    ap.add_argument("--results-dir", default=str(dc.ROOT / "reports" / "results"))
    args = ap.parse_args()
    models_dir, results_dir = Path(args.models_dir), Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for f in ("delay_regressor.pkl", "delay_classifier.pkl", "feature_columns.json", "metadata.json"):
        if not (models_dir / f).exists():
            raise SystemExit(f"Missing {models_dir / f}. Run scripts/train_delay_models.py first.")
    reg = joblib.load(models_dir / "delay_regressor.pkl")
    clf = joblib.load(models_dir / "delay_classifier.pkl")
    meta = json.loads((models_dir / "metadata.json").read_text())
    feats = json.loads((models_dir / "feature_columns.json").read_text())["features"]

    splits = dc.load_splits(Path(args.data_dir))
    dc.leakage_audit(feats, splits)
    reg_name, clf_name = meta["model_type"]["regressor"], meta["model_type"]["classifier"]

    # ------------------------------------------------------ regression
    reg_rows = []
    for name, m in meta["metrics"]["validation"]["regression"].items():
        reg_rows.append({"model": name, "role": "selected" if name == reg_name else "candidate",
                         "split": "validation", **m, "n_rows": meta["validation_rows"]})
    test_reg = {}
    for split in ("validation", "test"):
        df = splits[split]
        X = dc.make_features(df)
        m = dc.regression_metrics(df[dc.REG_TARGET], reg.predict(X))
        if split == "test":
            test_reg = m
        reg_rows.append({"model": reg_name, "role": "selected (recomputed)", "split": split, **m,
                         "n_rows": len(df)})
        base = dc.regression_metrics(df[dc.REG_TARGET], df["current_observed_delay_min"])
        reg_rows.append({"model": BASELINE, "role": "baseline", "split": split, **base,
                         "n_rows": len(df)})
    reg_df = pd.DataFrame(reg_rows)
    reg_df.to_csv(results_dir / "regression_results.csv", index=False)

    # -------------------------------------------------- classification
    clf_rows = []
    for name, m in meta["metrics"]["validation"]["classification"].items():
        clf_rows.append({"model": name, "role": "selected" if name == clf_name else "candidate",
                         "split": "validation", "task": "binary_delayed_15",
                         "average": "positive_class", **m, "n_rows": meta["validation_rows"]})
    test_clf, cm_test = {}, None
    for split in ("validation", "test"):
        df = splits[split]
        X, y = dc.make_features(df), df[dc.CLF_TARGET]
        pred, proba = clf.predict(X), clf.predict_proba(X)[:, 1]
        m = dc.classification_metrics(y, pred, proba)
        if split == "test":
            test_clf, cm_test = m, confusion_matrix(y, pred, labels=[0, 1])
        clf_rows.append({"model": clf_name, "role": "selected (recomputed)", "split": split,
                         "task": "binary_delayed_15", "average": "positive_class", **m, "n_rows": len(df)})
        bpred = (df["current_observed_delay_min"] >= 15).astype(int)
        clf_rows.append({"model": BASELINE, "role": "baseline", "split": split,
                         "task": "binary_delayed_15", "average": "positive_class",
                         **dc.classification_metrics(y, bpred), "n_rows": len(df)})

    # Three-level risk label (Low / Medium / High) derived from the selected regressor.
    risk_cm = None
    for split in ("validation", "test"):
        df = splits[split]
        pred_risk = dc.delay_to_risk_label(reg.predict(dc.make_features(df))).values
        true_risk = df[dc.RISK_TARGET].values
        row = {"model": f"{reg_name} -> risk label thresholds", "role": "derived", "split": split,
               "task": "risk_3class", "average": "macro",
               "accuracy": accuracy_score(true_risk, pred_risk),
               "precision": precision_score(true_risk, pred_risk, average="macro", zero_division=0),
               "recall": recall_score(true_risk, pred_risk, average="macro", zero_division=0),
               "f1": f1_score(true_risk, pred_risk, average="macro", zero_division=0),
               "n_rows": len(df)}
        clf_rows.append(row)
        if split == "test":
            risk_cm = confusion_matrix(true_risk, pred_risk, labels=dc.RISK_LABELS)
    clf_df = pd.DataFrame(clf_rows)
    clf_df.to_csv(results_dir / "classification_results.csv", index=False)

    plot_confusion(cm_test, ["On time (<15)", "Delayed (>=15)"],
                   f"{clf_name} - test set", results_dir / "confusion_matrix.png")
    plot_confusion(risk_cm, dc.RISK_LABELS, "Risk label from regressor - test set",
                   results_dir / "risk_confusion_matrix.png")

    # ------------------------------------------------------ metadata
    meta["metrics"]["test"] = {"regression": {reg_name: test_reg}, "classification": {clf_name: test_clf}}
    (models_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    # ---------------------------------------------------- final print
    base_t = reg_df[(reg_df.role == "baseline") & (reg_df.split == "test")].iloc[0]
    cbase_t = clf_df[(clf_df.role == "baseline") & (clf_df.split == "test")
                     & (clf_df.task == "binary_delayed_15")].iloc[0]
    print("\n=== FINAL RESULTS (held-out TEST set) ===")
    print(f"Best regression model : {reg_name}")
    print(f"  MAE  = {test_reg['MAE']:.3f} min")
    print(f"  RMSE = {test_reg['RMSE']:.3f} min")
    print(f"  R2   = {test_reg['R2']:.3f}")
    print(f"  (persistence baseline: MAE={base_t.MAE:.3f}, RMSE={base_t.RMSE:.3f}, R2={base_t.R2:.3f})")
    print(f"Best classification model : {clf_name}")
    print(f"  Accuracy  = {test_clf['accuracy']:.3f}")
    print(f"  Precision = {test_clf['precision']:.3f}")
    print(f"  Recall    = {test_clf['recall']:.3f}")
    print(f"  F1        = {test_clf['f1']:.3f}")
    print(f"  (persistence baseline: acc={cbase_t.accuracy:.3f}, P={cbase_t.precision:.3f}, "
          f"R={cbase_t.recall:.3f}, F1={cbase_t.f1:.3f})")
    print(f"  Confusion matrix [[TN FP],[FN TP]] = {cm_test.tolist()}")
    r3 = clf_df[(clf_df.task == "risk_3class") & (clf_df.split == "test")].iloc[0]
    print(f"Risk label (Low/Medium/High) from regressor: accuracy={r3.accuracy:.3f}, macro-F1={r3.f1:.3f}")
    print(f"\nSaved results to {results_dir} and updated {models_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
