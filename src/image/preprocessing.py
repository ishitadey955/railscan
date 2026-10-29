"""CNN input preprocessing for the RailScan script classifier.

This is deliberately separate from ``src/ocr/preprocessing.py``: the CNN needs
a fixed-size, colour, normalised tensor, while OCR works best on the image at
full resolution with only the clean-ups that help text recognition. No
thresholding or blurring is applied here.

Steps
    1. decode safely (path, bytes, PIL image or array) -> RGB uint8
    2. letterbox: scale the longest side to ``size`` keeping the aspect ratio,
       pad the rest with a neutral grey, so glyph shapes are not stretched
    3. scale pixel values to float32 in [0, 1]
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

IMG_SIZE = 128
PAD_VALUE = 128          # neutral grey padding
MIN_SIDE = 8             # anything smaller is not a usable photo

ImageSource = Union[str, Path, bytes, bytearray, Image.Image, np.ndarray]


class InvalidImageError(ValueError):
    """The input could not be decoded as an image."""


def load_rgb(source: ImageSource) -> np.ndarray:
    """Decode any supported input into an RGB ``uint8`` array (H, W, 3).

    Arrays are assumed to be RGB already (as produced by PIL / Streamlit);
    convert OpenCV BGR arrays with ``cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`` first.
    EXIF orientation from phone cameras is applied.
    """
    if isinstance(source, np.ndarray):
        arr = source
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise InvalidImageError(f"unsupported array shape {source.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr * (255.0 if arr.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    else:
        try:
            if isinstance(source, Image.Image):
                img = source
            elif isinstance(source, (bytes, bytearray)):
                img = Image.open(io.BytesIO(source))
            else:
                path = Path(source)
                if not path.is_file():
                    raise InvalidImageError(f"file not found: {path}")
                img = Image.open(path)
            img = ImageOps.exif_transpose(img).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageError(f"not a readable image ({exc})") from exc
        arr = np.asarray(img, dtype=np.uint8)
    if min(arr.shape[:2]) < MIN_SIDE:
        raise InvalidImageError(f"image too small: {arr.shape[1]}x{arr.shape[0]} px")
    return np.ascontiguousarray(arr)


def letterbox(rgb: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Resize keeping the aspect ratio and pad to ``size x size`` (uint8 RGB)."""
    h, w = rgb.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(rgb, (nw, nh), interpolation=interp)
    out = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top + nh, left:left + nw] = resized
    return out


def preprocess_for_cnn(source: ImageSource, size: int = IMG_SIZE) -> np.ndarray:
    """Image -> float32 array (size, size, 3) with values in [0, 1]."""
    return letterbox(load_rgb(source), size).astype(np.float32) / 255.0


def to_batch(source: ImageSource, size: int = IMG_SIZE) -> np.ndarray:
    """Same as :func:`preprocess_for_cnn` with a leading batch axis: (1, size, size, 3)."""
    return preprocess_for_cnn(source, size)[np.newaxis, ...]
