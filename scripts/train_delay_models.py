#!/usr/bin/env python3
"""
RailScan - delay prediction training.

MODEL A: next-station delay regression  (next_station_arrival_delay_min)
MODEL B: next-station delay-risk classification (next_station_delayed_15)

Models are fitted on train.csv ONLY and compared on validation.csv. The test
set is never touched here (scripts/evaluate_models.py does the one-time test
evaluation of the selected models).

Run:  python scripts/train_delay_models.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import delay_common as dc  # noqa: E402

TREE_MODELS = ("Random Forest", "XGBoost")
# Sized for the northern-India dataset (hundreds of thousands of rows): every tree sees a
# 10% bootstrap sample (~65k rows), which keeps training time and the saved model size practical.
RF_SIZE = {"n_estimators": 100, "min_samples_leaf": 10, "max_samples": 0.10}
IMPORTANCE_ROWS = 20_000          # validation rows used for permutation importance (seeded sample)


def regression_candidates(seed: int) -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(**RF_SIZE, n_jobs=-1, random_state=seed),
        "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                                subsample=0.8, colsample_bytree=0.8, random_state=seed,
                                n_jobs=-1),
    }


def classification_candidates(seed: int, scale_pos_weight: float) -> dict:
    # Imbalance handling: class weights (LR, RF) / scale_pos_weight (XGB),
    # all computed from the TRAIN labels only.
    return {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(**RF_SIZE, class_weight="balanced", n_jobs=-1,
                                                random_state=seed),
        "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                                 subsample=0.8, colsample_bytree=0.8,
                                 scale_pos_weight=scale_pos_weight, eval_metric="logloss",
                                 random_state=seed, n_jobs=-1),
    }


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([("prep", dc.build_preprocessor()), ("model", estimator)])


def impurity_importance(pipe: Pipeline) -> pd.Series:
    """Built-in tree importance, summed back onto the raw feature names."""
    names = pipe.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(pipe.named_steps["model"].feature_importances_, index=names)
    return imp.groupby(dc.source_feature_name).sum()


def importance_table(pipe, X_val, y_val, scoring, task, model_name, seed) -> pd.DataFrame:
    """Permutation importance on VALIDATION (raw columns) + built-in importance."""
    if len(X_val) > IMPORTANCE_ROWS:
        idx = X_val.sample(IMPORTANCE_ROWS, random_state=seed).index
        X_val, y_val = X_val.loc[idx], y_val.loc[idx]
    perm = permutation_importance(pipe, X_val, y_val, scoring=scoring, n_repeats=10,
                                  random_state=seed, n_jobs=1)
    out = pd.DataFrame({"feature": X_val.columns,
                        "permutation_importance": perm.importances_mean,
                        "permutation_std": perm.importances_std})
    out["builtin_importance"] = out["feature"].map(impurity_importance(pipe)).fillna(0.0)
    out.insert(0, "model", model_name)
    out.insert(0, "task", task)
    return out.sort_values("permutation_importance", ascending=False).reset_index(drop=True)


def plot_importance(tables: list, path: Path) -> None:
    fig, axes = plt.subplots(1, len(tables), figsize=(7 * len(tables), 6))
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, tables):
        top = t.head(12).iloc[::-1]
        ax.barh(top["feature"], top["permutation_importance"], xerr=top["permutation_std"],
                color="#2a6f97")
        ax.set_title(f"{t['task'].iloc[0]}: {t['model'].iloc[0]}\n(permutation importance, validation)")
        ax.set_xlabel("increase in error when feature is shuffled")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(dc.ROOT / "data" / "processed"))
    ap.add_argument("--models-dir", default=str(dc.ROOT / "models" / "delay"))
    ap.add_argument("--results-dir", default=str(dc.ROOT / "reports" / "results"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    models_dir, results_dir = Path(args.models_dir), Path(args.results_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    splits = dc.load_splits(Path(args.data_dir))
    dc.leakage_audit(dc.FEATURE_COLUMNS, splits)          # stops on any leakage

    tr, va = splits["train"], splits["validation"]
    X_tr, X_va = dc.make_features(tr), dc.make_features(va)
    print(f"\nRows  train={len(tr):,}  validation={len(va):,}  test={len(splits['test']):,} "
          "(test is not used in training)")

    # ------------------------------------------------------- MODEL A
    print("\n=== MODEL A: next-station delay regression (selection on VALIDATION) ===")
    reg_fitted, reg_val = {}, {}
    for name, est in regression_candidates(args.seed).items():
        pipe = make_pipeline(est).fit(X_tr, tr[dc.REG_TARGET])
        reg_fitted[name] = pipe
        reg_val[name] = dc.regression_metrics(va[dc.REG_TARGET], pipe.predict(X_va))
        m = reg_val[name]
        print(f"  {name:<20s} MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  R2={m['R2']:.3f}")
    best_reg = min(reg_val, key=lambda k: reg_val[k]["MAE"])
    print(f"  -> selected regressor (lowest validation MAE): {best_reg}")

    # ------------------------------------------------------- MODEL B
    y_tr, y_va = tr[dc.CLF_TARGET], va[dc.CLF_TARGET]
    pos_rate = y_tr.mean()
    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    print("\n=== MODEL B: delay-risk classification (selection on VALIDATION) ===")
    print(f"  class balance  train: {pos_rate:.1%} delayed | validation: {y_va.mean():.1%} | "
          f"test: {splits['test'][dc.CLF_TARGET].mean():.1%}")
    print(f"  imbalance handling: class_weight='balanced' (LR, RF), scale_pos_weight={spw:.3f} (XGB); "
          "weights computed from train labels only; decision threshold 0.5")
    clf_fitted, clf_val = {}, {}
    for name, est in classification_candidates(args.seed, spw).items():
        pipe = make_pipeline(est).fit(X_tr, y_tr)
        clf_fitted[name] = pipe
        clf_val[name] = dc.classification_metrics(y_va, pipe.predict(X_va),
                                                  pipe.predict_proba(X_va)[:, 1])
        m = clf_val[name]
        print(f"  {name:<20s} acc={m['accuracy']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"F1={m['f1']:.3f}  macroF1={m['macro_f1']:.3f}  AUC={m['roc_auc']:.3f}")
    # Macro-F1 is used for selection: with ~80% positives the positive-class F1
    # can look good even for a model that predicts 'delayed' almost always.
    best_clf = max(clf_val, key=lambda k: clf_val[k]["macro_f1"])
    print(f"  -> selected classifier (highest validation macro-F1): {best_clf}")

    # ---------------------------------------------- feature importance
    best_tree_reg = min(TREE_MODELS, key=lambda k: reg_val[k]["MAE"])
    best_tree_clf = max(TREE_MODELS, key=lambda k: clf_val[k]["macro_f1"])
    t_reg = importance_table(reg_fitted[best_tree_reg], X_va, va[dc.REG_TARGET],
                             "neg_mean_absolute_error", "regression", best_tree_reg, args.seed)
    t_clf = importance_table(clf_fitted[best_tree_clf], X_va, y_va, "f1_macro",
                             "classification", best_tree_clf, args.seed)
    pd.concat([t_reg, t_clf], ignore_index=True).to_csv(results_dir / "feature_importance.csv", index=False)
    plot_importance([t_reg, t_clf], results_dir / "feature_importance.png")
    print("\nTop features (permutation importance on validation):")
    print(f"  regression ({best_tree_reg}): " + ", ".join(t_reg.feature.head(5)))
    print(f"  classification ({best_tree_clf}): " + ", ".join(t_clf.feature.head(5)))

    # ----------------------------------------------------- artifacts
    reg_pipe, clf_pipe = reg_fitted[best_reg], clf_fitted[best_clf]
    joblib.dump(reg_pipe, models_dir / "delay_regressor.pkl")      # full pipeline: prep + model
    joblib.dump(clf_pipe, models_dir / "delay_classifier.pkl")     # full pipeline: prep + model
    joblib.dump(reg_pipe.named_steps["prep"], models_dir / "preprocessor.pkl")
    # route segments seen in training: the planner uses the model only on these
    (tr[["train_no", "station_code", "next_station_code"]].astype(str)
     .assign(train_no=lambda d: d.train_no.str.lstrip("0")).drop_duplicates()
     .sort_values(["train_no", "station_code", "next_station_code"])
     .to_csv(models_dir / "coverage.csv", index=False))
    (models_dir / "feature_columns.json").write_text(json.dumps({
        "features": dc.FEATURE_COLUMNS, "numeric": dc.NUMERIC_FEATURES,
        "categorical": dc.CATEGORICAL_FEATURES}, indent=2))

    te = splits["test"]
    rng = lambda d: {"min": str(d.journey_date.min().date()), "max": str(d.journey_date.max().date())}
    metadata = {
        "model_type": {"regressor": best_reg, "classifier": best_clf},
        "target": {"regression": dc.REG_TARGET, "classification": dc.CLF_TARGET,
                   "risk_label": dc.RISK_TARGET},
        "features": dc.FEATURE_COLUMNS,
        "training_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_rows": len(tr), "validation_rows": len(va), "test_rows": len(te),
        "dataset_date_ranges": {"train": rng(tr), "validation": rng(va), "test": rng(te)},
        "selection": {"regression": "lowest validation MAE",
                      "classification": "highest validation macro-F1"},
        "imbalance_handling": {"method": "class_weight='balanced' (LR, RF); scale_pos_weight (XGB)",
                               "scale_pos_weight": spw, "decision_threshold": 0.5,
                               "positive_rate": {"train": float(pos_rate), "validation": float(y_va.mean()),
                                                 "test": float(te[dc.CLF_TARGET].mean())}},
        "metrics": {"validation": {"regression": reg_val, "classification": clf_val},
                    "test": None},     # filled by scripts/evaluate_models.py
        "best_tree_models_for_importance": {"regression": best_tree_reg, "classification": best_tree_clf},
        "library_versions": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
                             "pandas": pd.__version__, "numpy": np.__version__},
        "seed": args.seed,
    }
    (models_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nSaved artifacts to {models_dir}")
    print("Next: python scripts/evaluate_models.py")


if __name__ == "__main__":
    main()
