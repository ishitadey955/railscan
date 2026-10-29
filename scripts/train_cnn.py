#!/usr/bin/env python3
"""
RailScan - train the CNN script/language classifier on real signboard images.

Input : data/processed/RailScan_CNN_Metadata.csv (scripts/download_datasets.py)
Labels:
  --labels script    (default, used by the app)  Devanagari / Kannada / Telugu
  --labels language  (experiment)                Hindi / Kannada / Marathi / Telugu
Hindi and Marathi are written in the same Devanagari script, so telling them
apart from pixels alone is expected to be hard; the language experiment
measures how hard instead of assuming it.

Model (small CNN, 128x128x3 input):
  augmentation (training only) -> Conv32-Pool -> Conv64-Pool -> Conv128-Pool
  -> GlobalAveragePooling -> Dense128 -> Dropout0.3 -> Dense(n, softmax)
Adam, sparse categorical cross-entropy, early stopping on validation loss,
best-checkpoint saving, class weights from the training labels.

The split comes from the metadata file (stratified, deduplicated before
splitting). Augmentation happens on the fly inside the model, so no
augmented copy of an image can reach the validation or test set.

Outputs (script mode):
  models/language_classifier.keras, models/language_labels.json
  reports/cnn_evaluation.json, reports/figures/cnn_*.png
Language mode writes the *_4class variants instead.

Usage:  python scripts/train_cnn.py [--labels script|language] [--epochs 80]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.image.preprocessing import IMG_SIZE, preprocess_for_cnn  # noqa: E402


def load_split(meta: pd.DataFrame, split: str, label_col: str, classes: list, size: int = IMG_SIZE):
    part = meta[meta.split == split]
    X = np.stack([preprocess_for_cnn(ROOT / p, size) for p in part.image_path])
    y = np.array([classes.index(c) for c in part[label_col]], dtype=np.int64)
    return X, y


def build_model(n_classes: int, size: int = IMG_SIZE):
    from tensorflow import keras
    from tensorflow.keras import layers
    augment = keras.Sequential([
        layers.RandomRotation(0.03, fill_mode="constant", fill_value=0.5),
        layers.RandomZoom((-0.1, 0.1), fill_mode="constant", fill_value=0.5),
        layers.RandomTranslation(0.05, 0.05, fill_mode="constant", fill_value=0.5),
        layers.RandomContrast(0.2),
    ], name="augmentation")          # no flips: mirrored text is not real text
    inputs = keras.Input((size, size, 3))
    x = augment(inputs)
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="railscan_script_cnn")
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_transfer_model(n_classes: int, size: int = IMG_SIZE):
    """MobileNetV2 (ImageNet weights, frozen) + the same small head. Optional improvement."""
    from tensorflow import keras
    from tensorflow.keras import layers
    augment = keras.Sequential([
        layers.RandomRotation(0.03, fill_mode="constant", fill_value=0.5),
        layers.RandomZoom((-0.1, 0.1), fill_mode="constant", fill_value=0.5),
        layers.RandomTranslation(0.05, 0.05, fill_mode="constant", fill_value=0.5),
        layers.RandomContrast(0.2),
    ], name="augmentation")
    base = keras.applications.MobileNetV2(input_shape=(size, size, 3), include_top=False, weights="imagenet")
    base.trainable = False
    inputs = keras.Input((size, size, 3))
    x = augment(inputs)
    x = layers.Rescaling(2.0, offset=-1.0)(x)          # our [0, 1] input -> MobileNetV2's [-1, 1]
    x = base(x, training=False)                         # keep BatchNorm statistics frozen
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="railscan_script_mobilenetv2")
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model, base


def unfreeze_top(model, base, n_layers: int = 30) -> None:
    """Fine-tuning stage: train the last ``n_layers`` of the base with a small learning rate."""
    from tensorflow import keras
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:                            # BatchNorm stays frozen while fine-tuning
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(1e-5),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])


def plot_history(hist: dict, path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key in zip(axes, ("loss", "accuracy")):
        ax.plot(hist[key], label="train")
        ax.plot(hist[f"val_{key}"], label="validation")
        ax.set_xlabel("epoch")
        ax.set_ylabel(key)
        ax.set_title(f"{title}: {key}")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, classes: list, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(1.4 * len(classes) + 2.5, 1.2 * len(classes) + 2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Train the RailScan CNN.")
    ap.add_argument("--labels", choices=["script", "language"], default="script")
    ap.add_argument("--arch", choices=["small", "mobilenetv2"], default="small",
                    help="small CNN from scratch, or MobileNetV2 transfer learning (downloads ~9 MB weights)")
    ap.add_argument("--img-size", type=int, default=IMG_SIZE, help="square CNN input size (e.g. 128, 224)")
    ap.add_argument("--fine-tune-epochs", type=int, default=30, help="mobilenetv2 only: second stage")
    ap.add_argument("--metadata", default=str(ROOT / "data" / "processed" / "RailScan_CNN_Metadata.csv"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    import tensorflow as tf
    from tensorflow import keras
    keras.utils.set_random_seed(args.seed)

    meta_path = Path(args.metadata)
    if not meta_path.exists():
        raise SystemExit(f"{meta_path} not found. Run: python scripts/download_datasets.py")
    meta = pd.read_csv(meta_path, encoding="utf-8")
    label_col = "script_label" if args.labels == "script" else "language_label"
    classes = sorted(meta[label_col].unique())
    missing = [p for p in meta.image_path if not (ROOT / p).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images listed in the metadata are missing, e.g. {missing[0]}. "
                         "Run: python scripts/download_datasets.py")

    suffix = "" if args.labels == "script" else "_4class"
    models_dir, reports_dir, fig_dir = ROOT / "models", ROOT / "reports", ROOT / "reports" / "figures"
    for d in (models_dir, reports_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"language_classifier{suffix}.keras"

    t0 = time.time()
    size = args.img_size
    X_tr, y_tr = load_split(meta, "train", label_col, classes, size)
    X_va, y_va = load_split(meta, "validation", label_col, classes, size)
    X_te, y_te = load_split(meta, "test", label_col, classes, size)
    print(f"Classes: {classes}")
    print(f"Images  train={len(y_tr)}  validation={len(y_va)}  test={len(y_te)}  "
          f"(loaded in {time.time() - t0:.1f}s)")
    counts = np.bincount(y_tr, minlength=len(classes))
    weights = compute_class_weight("balanced", classes=np.arange(len(classes)), y=y_tr)
    class_weight = {i: float(w) for i, w in enumerate(weights)}
    print("Train counts:", dict(zip(classes, counts.tolist())), "class weights:",
          {classes[i]: round(w, 3) for i, w in class_weight.items()})

    if args.arch == "small":
        model, base = build_model(len(classes), size), None
    else:
        model, base = build_transfer_model(len(classes), size)
    model.summary(print_fn=lambda s: print("  " + s))
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(str(model_path), monitor="val_loss", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
    ]
    t0 = time.time()
    hist = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=args.epochs,
                     batch_size=args.batch_size, class_weight=class_weight, callbacks=callbacks, verbose=2)
    history = {k: list(v) for k, v in hist.history.items()}
    if base is not None and args.fine_tune_epochs > 0:
        print("\nFine-tuning the top of MobileNetV2 (learning rate 1e-5)")
        model = keras.models.load_model(model_path)          # continue from the best stage-1 weights
        base = next(l for l in model.layers if l.name.startswith("mobilenetv2"))
        unfreeze_top(model, base)
        callbacks[1] = keras.callbacks.ModelCheckpoint(str(model_path), monitor="val_loss", save_best_only=True,
                                                       initial_value_threshold=min(history["val_loss"]))
        h2 = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=args.fine_tune_epochs,
                       batch_size=args.batch_size, class_weight=class_weight, callbacks=callbacks, verbose=2)
        for k in history:
            history[k] += list(h2.history.get(k, []))
    train_seconds = time.time() - t0

    best = keras.models.load_model(model_path)         # the checkpoint with the lowest val loss
    results = {}
    for name, X, y in (("validation", X_va, y_va), ("test", X_te, y_te)):
        pred = np.argmax(best.predict(X, verbose=0), axis=1)
        rep = classification_report(y, pred, labels=list(range(len(classes))), target_names=classes,
                                    output_dict=True, zero_division=0)
        cm = confusion_matrix(y, pred, labels=list(range(len(classes))))
        results[name] = {"accuracy": rep["accuracy"], "n": int(len(y)),
                         "per_class": {c: {k: rep[c][k] for k in ("precision", "recall", "f1-score", "support")}
                                       for c in classes},
                         "macro_avg": rep["macro avg"], "weighted_avg": rep["weighted avg"],
                         "confusion_matrix": cm.tolist()}
        print(f"\n{name.upper()} accuracy = {rep['accuracy']:.4f}")
        print(classification_report(y, pred, labels=list(range(len(classes))), target_names=classes,
                                    zero_division=0))
        if name == "test":
            plot_confusion(cm, classes, fig_dir / f"cnn_confusion_matrix{suffix}.png",
                           f"CNN test confusion matrix ({args.labels})")

    h = {k: [float(v) for v in vals] for k, vals in history.items() if k != "learning_rate"}
    plot_history(h, fig_dir / f"cnn_training_curves{suffix}.png", f"CNN ({args.labels})")
    best_epoch = int(np.argmin(h["val_loss"])) + 1
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label_mode": args.labels, "architecture": args.arch, "classes": classes, "model_file": model_path.relative_to(ROOT).as_posix(),
        "input_shape": [size, size, 3], "parameters": int(best.count_params()),
        "data": {"metadata": meta_path.relative_to(ROOT).as_posix(),
                 "train": int(len(y_tr)), "validation": int(len(y_va)), "test": int(len(y_te)),
                 "train_class_counts": dict(zip(classes, counts.tolist()))},
        "training": {"epochs_run": len(h["loss"]), "best_epoch": best_epoch, "batch_size": args.batch_size,
                     "class_weight": {classes[i]: w for i, w in class_weight.items()},
                     "seconds": round(train_seconds, 1), "seed": args.seed,
                     "tensorflow": tf.__version__},
        "metrics": results, "history": h,
    }
    (reports_dir / f"cnn_evaluation{suffix}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    labels = {"classes": classes, "label_mode": args.labels, "image_size": size,
              "preprocessing": f"RGB, letterbox to {size}x{size} (grey padding), float32 / 255",
              "script_to_languages": {"Devanagari": ["Hindi", "Marathi"], "Bengali": ["Bengali"], "Kannada": ["Kannada"],
                                      "Telugu": ["Telugu"]}}
    (models_dir / f"language_labels{suffix}.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    print(f"\nSaved {model_path.relative_to(ROOT)} (best epoch {best_epoch}), "
          f"reports/cnn_evaluation{suffix}.json, figures in reports/figures/")


if __name__ == "__main__":
    main()
