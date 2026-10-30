"""Script prediction with the trained CNN.

``predict_script(image)`` returns the dict the OCR engine expects::

    {"predicted_script": "Kannada", "confidence": 0.91,
     "class_probabilities": {"Devanagari": 0.05, "Kannada": 0.91, "Telugu": 0.04},
     "low_confidence": False}

The model and its label file are produced by ``scripts/train_cnn.py`` (TensorFlow).
The app runs the saved model with Keras on the PyTorch backend: PyTorch is loaded for
EasyOCR anyway, so the app does not need TensorFlow at all (~250 MB less memory). Both
backends give the same probabilities (max difference 3e-6 on 53 test images).
"""
from __future__ import annotations

import json
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .preprocessing import IMG_SIZE, ImageSource, to_batch

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "language_classifier.keras"
LABELS_PATH = ROOT / "models" / "language_labels.json"
DEFAULT_THRESHOLD = 0.60


class ModelNotAvailable(RuntimeError):
    """The CNN model file (or TensorFlow) is missing."""


def load_labels(path: Path = LABELS_PATH) -> Dict:
    if not path.exists():
        raise ModelNotAvailable(f"label file not found: {path}. Run: python scripts/train_cnn.py")
    return json.loads(path.read_text(encoding="utf-8"))


_LOAD_LOCK = threading.Lock()      # the app may load the model in a background thread


def load_model(model_path: str = str(MODEL_PATH), labels_path: str = str(LABELS_PATH)) -> Tuple[object, List[str], int]:
    """Load the Keras model once per process. Returns (model, class_names, image_size)."""
    with _LOAD_LOCK:
        return _load_model(model_path, labels_path)


@lru_cache(maxsize=2)
def _load_model(model_path: str, labels_path: str) -> Tuple[object, List[str], int]:
    if not Path(model_path).exists():
        raise ModelNotAvailable(f"CNN model not found: {model_path}. Run: python scripts/train_cnn.py")
    # the backend is fixed when keras is first imported; scripts that already use TensorFlow keep it
    os.environ.setdefault("KERAS_BACKEND", "torch")
    try:
        import keras
    except ImportError as exc:
        raise ModelNotAvailable("Keras is not installed (pip install -r requirements.txt)") from exc
    model = keras.models.load_model(model_path)
    labels = load_labels(Path(labels_path))
    return model, list(labels["classes"]), int(labels.get("image_size", IMG_SIZE))


def predict_script(image: ImageSource, threshold: float = DEFAULT_THRESHOLD,
                   model_path: Optional[str] = None) -> Dict:
    """Classify the script of one image. Raises ``InvalidImageError`` / ``ModelNotAvailable``."""
    model, classes, size = load_model(model_path or str(MODEL_PATH))
    probs = np.asarray(model.predict(to_batch(image, size), verbose=0)[0], dtype=float)
    best = int(np.argmax(probs))
    return {
        "predicted_script": classes[best],
        "confidence": float(probs[best]),
        "class_probabilities": {c: float(p) for c, p in zip(classes, probs)},
        "low_confidence": bool(probs[best] < threshold),
    }
