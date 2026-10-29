"""Image preprocessing strategies for OCR of railway-sign photos.

No single preprocessing method works for every sign: white-on-blue boards,
tilted photos, tiny crops and noisy phone images all behave differently.
This module therefore provides a *registry* of strategies plus a small
image-analysis helper that proposes a sensible **order** in which to try
them. The OCR engine tries them in that order and keeps the best result.

Strategies (name -> description)
    original            the image as loaded (colour)
    grayscale           single-channel grayscale
    denoise             grayscale + non-local-means denoising
    contrast            grayscale + CLAHE contrast enhancement
    adaptive_threshold  polarity-fixed grayscale + Gaussian adaptive threshold
    otsu_threshold      polarity-fixed grayscale + Otsu global threshold
    upscale             grayscale enlarged 2x (bicubic)
    deskew              grayscale rotated to level the text lines (optional)
    upscale_otsu        upscale followed by Otsu threshold (small text)
    inverted            grayscale with light-on-dark text flipped to dark-on-light

All functions take and return ``numpy`` ``uint8`` arrays (OpenCV layout, BGR
for colour). ``load_image`` accepts a path, a PIL image or an array.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

import cv2
import numpy as np

try:  # PIL is only needed to accept PIL images as input
    from PIL import Image as _PILImage
except Exception:  # pragma: no cover
    _PILImage = None

ImageInput = Union[str, Path, np.ndarray, "object"]

DEFAULT_MAX_SIDE = 1600       # long side cap applied once, at load time
UPSCALE_FACTOR = 2.0
UPSCALE_MAX_SIDE = 3600       # never upscale beyond this long side
SKEW_SEARCH_DEG = 15.0        # skew search range, +/- degrees
AUTO_DESKEW_MIN_DEG = 1.0     # "auto" mode only deskews at/above this tilt


class ImageLoadError(ValueError):
    """The input could not be turned into an image."""


# --------------------------------------------------------------------------
# Loading / basic helpers
# --------------------------------------------------------------------------
def load_image(src: ImageInput, max_side: Optional[int] = DEFAULT_MAX_SIDE) -> np.ndarray:
    """Return a ``uint8`` BGR image from a path, PIL image or ndarray.

    Images whose long side exceeds ``max_side`` are shrunk (INTER_AREA) so
    that later steps (denoising, deskew, OCR) have bounded cost.
    Raises :class:`ImageLoadError` for unreadable or empty input.
    """
    if src is None:
        raise ImageLoadError("no image given")

    if isinstance(src, (str, Path)):
        path = Path(src)
        if not path.is_file():
            raise ImageLoadError(f"image file not found: {path}")
        try:
            # np.fromfile + imdecode also works for non-ASCII Windows paths
            buf = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception as exc:  # pragma: no cover - defensive
            raise ImageLoadError(f"could not read {path}: {exc}") from exc
        if img is None:
            raise ImageLoadError(f"could not decode image: {path}")
    elif isinstance(src, (bytes, bytearray)):            # e.g. a Streamlit upload
        img = cv2.imdecode(np.frombuffer(bytes(src), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ImageLoadError("could not decode image bytes")
    elif _PILImage is not None and isinstance(src, _PILImage.Image):
        img = np.asarray(src.convert("RGB"))[:, :, ::-1].copy()
    elif isinstance(src, np.ndarray):
        img = _ndarray_to_bgr(src)
    else:
        raise ImageLoadError(f"unsupported image input type: {type(src).__name__}")

    if img.size == 0 or img.shape[0] < 2 or img.shape[1] < 2:
        raise ImageLoadError("image is empty")

    if max_side:
        h, w = img.shape[:2]
        long_side = max(h, w)
        if long_side > max_side:
            s = max_side / long_side
            img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img)


def _ndarray_to_bgr(arr: np.ndarray) -> np.ndarray:
    a = arr
    if a.dtype != np.uint8:
        a = a.astype(np.float64)
        a = a * 255.0 if a.size and a.max() <= 1.0 else a
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    if a.ndim == 2:
        return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    if a.ndim == 3 and a.shape[2] == 3:
        return a.copy()
    if a.ndim == 3 and a.shape[2] == 4:
        return cv2.cvtColor(a, cv2.COLOR_BGRA2BGR)
    raise ImageLoadError(f"unsupported array shape: {arr.shape}")


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.copy()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def background_is_dark(gray: np.ndarray) -> bool:
    """True if the border ring (usually background) is darker than mid-grey.

    OCR reads dark text on a light background most reliably. Many Indian
    railway boards are the opposite (white text on blue/green/black), so the
    threshold strategies flip those first.
    """
    h, w = gray.shape[:2]
    t = max(1, int(round(min(h, w) * 0.05)))
    ring = np.concatenate([
        gray[:t, :].ravel(), gray[-t:, :].ravel(),
        gray[:, :t].ravel(), gray[:, -t:].ravel(),
    ])
    return float(np.median(ring)) < 127.0


def _polarity_fixed_gray(img: np.ndarray) -> np.ndarray:
    g = to_gray(img)
    return cv2.bitwise_not(g) if background_is_dark(g) else g


def estimate_skew_angle(img: np.ndarray, search_deg: float = SKEW_SEARCH_DEG) -> float:
    """Estimate the rotation (degrees, OpenCV sign: + = counter-clockwise)
    that must be applied to make text lines horizontal.

    Uses a projection-profile search: the correct angle maximises the
    variance of the row sums of the binarised text. Returns 0.0 when there
    is too little foreground to judge.
    """
    g = _polarity_fixed_gray(img)
    h, w = g.shape[:2]
    scale = 600.0 / max(h, w) if max(h, w) > 600 else 1.0
    if scale != 1.0:
        g = cv2.resize(g, (max(2, int(w * scale)), max(2, int(h * scale))), interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    _, binary = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)  # text = 255
    if binary.mean() / 255.0 < 0.005 or binary.mean() / 255.0 > 0.6:
        return 0.0

    bh, bw = binary.shape
    # pad to a square so rotation does not clip text corners
    side = int(np.ceil(np.hypot(bh, bw)))
    canvas = np.zeros((side, side), np.uint8)
    y0, x0 = (side - bh) // 2, (side - bw) // 2
    canvas[y0:y0 + bh, x0:x0 + bw] = binary
    centre = (side / 2.0, side / 2.0)

    def score(angle: float) -> float:
        m = cv2.getRotationMatrix2D(centre, angle, 1.0)
        rot = cv2.warpAffine(canvas, m, (side, side), flags=cv2.INTER_NEAREST)
        prof = rot.sum(axis=1, dtype=np.float64)
        return float(np.sum(np.diff(prof) ** 2))

    coarse = np.arange(-search_deg, search_deg + 1e-9, 0.5)
    scores = [score(a) for a in coarse]
    best = float(coarse[int(np.argmax(scores))])
    fine = np.arange(best - 0.5, best + 0.5 + 1e-9, 0.1)
    best = float(fine[int(np.argmax([score(a) for a in fine]))])
    # if the best angle is barely better than level, treat as level
    if score(best) <= score(0.0) * 1.02:
        return 0.0
    return round(best, 2)


def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate about the centre keeping the full frame; fill with edge colour."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2.0 - w / 2.0
    m[1, 2] += nh / 2.0 - h / 2.0
    return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


# --------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------
def original(img: np.ndarray) -> np.ndarray:
    return img.copy()


def grayscale(img: np.ndarray) -> np.ndarray:
    return to_gray(img)


def denoise(img: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(to_gray(img), None, h=10, templateWindowSize=7, searchWindowSize=21)


def contrast(img: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(to_gray(img))


def adaptive_threshold(img: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(_polarity_fixed_gray(img), (3, 3), 0)
    block = max(15, (min(g.shape[:2]) // 8) | 1)  # odd, scales with image size
    block = min(block, 101)
    block += 1 - block % 2
    return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 10)


def otsu_threshold(img: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(_polarity_fixed_gray(img), (5, 5), 0)
    _, out = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out


def upscale(img: np.ndarray, factor: float = UPSCALE_FACTOR) -> np.ndarray:
    g = to_gray(img)
    h, w = g.shape[:2]
    f = min(factor, UPSCALE_MAX_SIDE / max(h, w))
    if f <= 1.0:
        return g
    return cv2.resize(g, (int(w * f), int(h * f)), interpolation=cv2.INTER_CUBIC)


def deskew(img: np.ndarray) -> np.ndarray:
    g = to_gray(img)
    angle = estimate_skew_angle(img)
    return g if angle == 0.0 else rotate_image(g, angle)


def upscale_otsu(img: np.ndarray) -> np.ndarray:
    return otsu_threshold(upscale(img))


def inverted(img: np.ndarray) -> np.ndarray:
    return _polarity_fixed_gray(img)


STRATEGIES: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "original": original,
    "grayscale": grayscale,
    "denoise": denoise,
    "contrast": contrast,
    "adaptive_threshold": adaptive_threshold,
    "otsu_threshold": otsu_threshold,
    "upscale": upscale,
    "deskew": deskew,
    "upscale_otsu": upscale_otsu,
    "inverted": inverted,
}


def apply_strategy(img: np.ndarray, name: str) -> np.ndarray:
    try:
        fn = STRATEGIES[name]
    except KeyError:
        raise KeyError(f"unknown preprocessing strategy {name!r}; choose from {sorted(STRATEGIES)}") from None
    return fn(img)


# --------------------------------------------------------------------------
# Image analysis -> strategy order
# --------------------------------------------------------------------------
@dataclass
class ImageAnalysis:
    height: int
    width: int
    contrast_std: float       # std-dev of grayscale intensities
    mean_brightness: float
    dark_background: bool
    noise_sigma: float        # Immerkaer fast noise estimate (rough; text edges inflate it)
    skew_deg: float           # rotation needed to level text; 0.0 if undetermined

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_noise_sigma(gray: np.ndarray) -> float:
    h, w = gray.shape[:2]
    if h < 3 or w < 3:
        return 0.0
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    conv = cv2.filter2D(gray.astype(np.float32), -1, k)
    return float(np.sqrt(np.pi / 2.0) * np.abs(conv[1:-1, 1:-1]).sum() / (6.0 * (w - 2) * (h - 2)))


def analyze_image(img: np.ndarray) -> ImageAnalysis:
    g = to_gray(img)
    return ImageAnalysis(
        height=int(g.shape[0]),
        width=int(g.shape[1]),
        contrast_std=round(float(g.std()), 2),
        mean_brightness=round(float(g.mean()), 2),
        dark_background=background_is_dark(g),
        noise_sigma=round(estimate_noise_sigma(g), 2),
        skew_deg=estimate_skew_angle(img),
    )


def select_strategies(
    analysis: ImageAnalysis,
    deskew_mode: Union[bool, str] = "auto",
    allowed: Optional[Sequence[str]] = None,
) -> List[str]:
    """Order strategies by how likely they are to help for this image.

    This only decides the *order*; later strategies are still tried when the
    early ones give weak OCR, so a wrong guess costs time, not accuracy.
    ``deskew_mode``: ``True`` always include deskew, ``False`` never,
    ``"auto"`` include (first) only when tilt >= ``AUTO_DESKEW_MIN_DEG``.
    ``allowed`` restricts the result to the named strategies (and, if given,
    their order is used as-is).
    """
    if allowed is not None:
        unknown = [n for n in allowed if n not in STRATEGIES]
        if unknown:
            raise KeyError(f"unknown preprocessing strategies: {unknown}")
        return list(allowed)

    order: List[str] = []

    def add(name: str) -> None:
        if name not in order:
            order.append(name)

    tilted = abs(analysis.skew_deg) >= AUTO_DESKEW_MIN_DEG
    if deskew_mode is True or (deskew_mode == "auto" and tilted):
        add("deskew")
    if analysis.dark_background:
        add("inverted")
    if analysis.height < 120 or analysis.width < 300:
        add("upscale")
    add("original")
    add("grayscale")
    if analysis.contrast_std < 45:
        add("contrast")
    if analysis.noise_sigma > 6:
        add("denoise")
    add("otsu_threshold")
    add("adaptive_threshold")
    add("contrast")
    add("upscale")
    add("upscale_otsu")
    add("denoise")
    return order
