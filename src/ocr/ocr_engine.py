"""EasyOCR engine for RailScan.

Pipeline for one sign image
    1. detect the script with the existing CNN (``src.image.prediction``),
       unless a script / language is supplied by the caller
    2. map the script to language codes: Hindi->hin, Kannada->kan,
       Marathi->mar, Telugu->tel, Devanagari->hin+mar; bilingual signs use
       e.g. ``eng+hin``
    3. choose preprocessing strategies for the image (see ``preprocessing``)
    4. run EasyOCR for each (language, strategy) until a confident result
       is found, otherwise keep the best-scoring one
    5. clean the text, split train numbers from station text
    6. return a plain dict - this module never raises for bad input

Nothing here invents numbers: ``ocr_confidence`` is the mean box confidence
(0-100) reported by EasyOCR for the chosen attempt, or ``None`` when OCR failed.

Language strings use three-letter codes (``eng+hin``) and are mapped to
EasyOCR's codes (``["hi", "en"]``). EasyOCR can combine Hindi and Marathi
(shared Devanagari model) with English, but Kannada and Telugu each need their
own reader, so every language combination gets its own cached reader.
"""
from __future__ import annotations

import logging
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from .preprocessing import (
    ImageInput,
    ImageLoadError,
    analyze_image,
    apply_strategy,
    load_image,
    select_strategies,
)
from .text_cleaning import clean_text
from .train_numbers import split_train_numbers

try:
    import easyocr  # type: ignore
except Exception:  # pragma: no cover - reported at run time as a failed OCR
    easyocr = None  # type: ignore

log = logging.getLogger(__name__)

STATUS_OK = "ok"
STATUS_FAILED = "OCR failed"

# CNN class name -> language code(s)
SCRIPT_TO_LANG: Dict[str, str] = {
    "English": "eng",
    "Hindi": "hin",
    "Kannada": "kan",
    "Marathi": "mar",
    "Telugu": "tel",
    # The CNN classifies the SCRIPT. Hindi and Marathi share Devanagari,
    # so a Devanagari prediction tries both language packs.
    "Devanagari": "hin+mar",
}
LANG_TO_SCRIPT = {v: k for k, v in SCRIPT_TO_LANG.items()}
LANG_TO_EASYOCR = {"eng": "en", "hin": "hi", "mar": "mr", "kan": "kn", "tel": "te"}
SUPPORTED_LANGUAGES = tuple(SCRIPT_TO_LANG.values())
NATIVE_SCRIPTS = ("Hindi", "Kannada", "Marathi", "Telugu")


# --------------------------------------------------------------------------
# Language helpers
# --------------------------------------------------------------------------
def normalize_script_name(value: str) -> str:
    """'hindi' / 'HIN' / 'Hindi' -> 'Hindi'. Raises ValueError if unknown."""
    v = (value or "").strip()
    for script, lang in SCRIPT_TO_LANG.items():
        if v.casefold() in (script.casefold(), lang):
            return script
    raise ValueError(f"unknown script {value!r}; expected one of {sorted(SCRIPT_TO_LANG)}")


def language_candidates(script: str, mode: str = "bilingual") -> List[str]:
    """Language strings to try for a script.

    ``mode``: ``"bilingual"`` -> ``["eng+hin"]`` (default; boards almost always
    carry English or Latin digits too), ``"native"`` -> ``["hin"]``,
    ``"both"`` -> ``["eng+hin", "hin"]``.
    """
    script = normalize_script_name(script)
    lang = SCRIPT_TO_LANG[script]
    if lang == "eng":
        return ["eng"]
    if mode == "native":
        return [lang]
    if mode == "both":
        return [f"eng+{lang}", lang]
    if mode == "bilingual":
        return [f"eng+{lang}"]
    raise ValueError(f"unknown language mode {mode!r}")


def _split_lang(lang: str) -> List[str]:
    return [p for p in lang.split("+") if p]


# --------------------------------------------------------------------------
# Scoring helpers
# --------------------------------------------------------------------------
def _alnum_count(text: str) -> int:
    return sum(1 for c in text if unicodedata.category(c)[0] in "LMN" and not c.isspace())


def _quality(text: str) -> float:
    """Fraction of non-space characters that are letters/marks/numbers.
    Output full of symbols is more likely garbage than text."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if unicodedata.category(c)[0] in "LMN") / len(chars)


@dataclass
class OCRConfig:
    languages_mode: str = "bilingual"       # "bilingual" | "native" | "both"
    deskew: Union[bool, str] = "auto"       # True | False | "auto"
    strategies: Optional[Sequence[str]] = None  # force a strategy list / order
    early_stop_confidence: float = 85.0     # stop trying once an attempt reaches this
    min_confidence: float = 60.0            # below this the result is flagged low_confidence
    min_chars: int = 2                      # fewer letters/digits than this = unusable
    min_accept_confidence: float = 30.0     # attempts below this are treated as noise (OCR failed)
    time_budget: float = 20.0               # stop trying new (language, strategy) pairs after this many seconds
    secondary_script_prob: float = 0.25     # CNN runner-up tried too if >= this prob
    script_threshold: float = 0.60          # passed to the CNN predictor
    max_image_side: int = 1600
    tie_margin: float = 3.0                 # scores this close prefer the longer text
    max_strategies: int = 2                 # EasyOCR is slow on CPU; preprocessing variants tried per language


@dataclass
class _Attempt:
    language: str
    method: str
    raw_text: str = ""
    confidence: Optional[float] = None
    n_chars: int = 0
    score: float = 0.0
    error: Optional[str] = None

    def summary(self) -> dict:
        return {
            "language": self.language, "preprocessing_method": self.method,
            "ocr_confidence": self.confidence, "chars": self.n_chars,
            "score": round(self.score, 2), "error": self.error,
        }


class OCRBackendUnavailable(RuntimeError):
    """EasyOCR is not installed or its models cannot be loaded."""


_EASYOCR_READERS: Dict[Tuple[str, ...], Any] = {}


def easyocr_languages(lang: str) -> List[str]:
    """'eng+hin+mar' -> ['hi', 'mr', 'en'] (native first, English last)."""
    codes = [LANG_TO_EASYOCR[p] for p in _split_lang(lang) if p in LANG_TO_EASYOCR]
    return sorted(set(codes), key=lambda c: (c == "en", codes.index(c)))


def _easyocr_reader(langs: Sequence[str]):
    """One cached reader per language combination (model files load once per process)."""
    key = tuple(langs)
    if key not in _EASYOCR_READERS:
        if easyocr is None:
            raise OCRBackendUnavailable("easyocr is not installed (pip install -r requirements.txt)")
        _EASYOCR_READERS[key] = easyocr.Reader(list(langs), gpu=False, verbose=False)
    return _EASYOCR_READERS[key]


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------
class OCREngine:
    """Reusable OCR engine. Create once, call :meth:`recognize` per image.

    Parameters
    ----------
    config : OCRConfig, optional
    script_predictor : callable, optional
        ``f(image) -> dict`` with keys like the project's
        ``predict_script`` (``predicted_script``/``script``, ``confidence``,
        ``class_probabilities``, ``low_confidence``). Defaults to
        ``src.image.prediction.predict_script``, imported lazily.
    """

    backend = "easyocr"

    def __init__(self, config: Optional[OCRConfig] = None,
                 script_predictor: Optional[Callable[..., dict]] = None):
        self.config = config or OCRConfig()
        self._predictor = script_predictor
        self._langs_cache: Optional[set] = None     # tests may restrict the languages

    def available_languages(self) -> set:
        """Language codes EasyOCR can read here (empty set if easyocr is not installed)."""
        if self._langs_cache is not None:
            return self._langs_cache
        return set(LANG_TO_EASYOCR) if easyocr is not None else set()

    # ---- script detection ------------------------------------------------
    def _get_predictor(self) -> Callable[..., dict]:
        if self._predictor is None:
            try:
                from src.image.prediction import predict_script
            except ImportError:
                root = str(Path(__file__).resolve().parents[2])
                if root not in sys.path:
                    sys.path.insert(0, root)
                from src.image.prediction import predict_script  # raises ImportError again if absent
            self._predictor = predict_script
        return self._predictor

    def detect_script(self, source: ImageInput, image_bgr: Optional[np.ndarray] = None) -> dict:
        """Run the CNN. Returns ``{"script", "confidence", "class_probabilities",
        "low_confidence"}``. Raises if the CNN is unavailable."""
        predictor = self._get_predictor()
        if isinstance(source, (str, Path)):
            arg: Any = source
        elif image_bgr is not None:
            from PIL import Image
            arg = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        else:
            arg = source
        try:
            out = predictor(arg, threshold=self.config.script_threshold)
        except TypeError:  # custom predictor without a threshold argument
            out = predictor(arg)
        script = out.get("predicted_script") or out.get("script")
        return {
            "script": normalize_script_name(script) if script else None,
            "confidence": out.get("confidence"),
            "class_probabilities": out.get("class_probabilities") or {},
            "low_confidence": bool(out.get("low_confidence", False)),
        }

    # ---- low level OCR call --------------------------------------------
    def _run_easyocr(self, image: np.ndarray, lang: str) -> Tuple[str, Optional[float]]:
        """EasyOCR on one image: boxes grouped into lines top-to-bottom, mean confidence 0-100."""
        reader = _easyocr_reader(easyocr_languages(lang))
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        boxes = reader.readtext(image, detail=1, paragraph=False)
        items = []
        for bbox, text, conf in boxes:
            text = (text or "").strip()
            if text:
                ys = [pt[1] for pt in bbox]
                xs = [pt[0] for pt in bbox]
                items.append((min(ys), max(ys), min(xs), text, float(conf)))
        if not items:
            return "", None
        items.sort(key=lambda t: (t[0], t[2]))
        lines: List[List[Tuple[float, str]]] = []
        last_bottom = None
        for top, bottom, left, text, _ in items:
            # a box whose top lies above the middle of the previous box continues that line
            if lines and last_bottom is not None and top < last_bottom - (bottom - top) / 2:
                lines[-1].append((left, text))
            else:
                lines.append([(left, text)])
            last_bottom = bottom
        out = "\n".join(" ".join(t for _, t in sorted(line)) for line in lines)
        return out, round(100.0 * sum(i[4] for i in items) / len(items), 1)

    # ---- helpers ---------------------------------------------------------
    def _make_attempt(self, image: np.ndarray, lang: str, method: str) -> _Attempt:
        att = _Attempt(language=lang, method=method)
        try:
            text, conf = self._run_easyocr(image, lang)
        except OCRBackendUnavailable:
            raise
        except Exception as exc:  # model download / runtime error in one attempt
            att.error = f"{type(exc).__name__}: {exc}"[:300]
            return att
        att.raw_text = text
        att.confidence = conf
        cleaned = clean_text(text)
        att.n_chars = _alnum_count(cleaned)
        if conf is not None and att.n_chars >= 1:
            att.score = conf * _quality(cleaned)
        return att

    def _usable(self, att: _Attempt) -> bool:
        return (att.confidence is not None and att.n_chars >= self.config.min_chars
                and att.confidence >= self.config.min_accept_confidence)

    def _better(self, a: _Attempt, b: Optional[_Attempt]) -> bool:
        if b is None:
            return True
        if abs(a.score - b.score) <= self.config.tie_margin:
            return a.n_chars > b.n_chars
        return a.score > b.score

    # ---- main entry point ------------------------------------------------
    def recognize(self, image: ImageInput, script: Optional[str] = None,
                  languages: Optional[str] = None, return_attempts: bool = False) -> dict:
        """OCR one sign image. Never raises; failures are reported in the dict.

        ``script`` (e.g. ``"Hindi"``) skips the CNN; ``languages`` (e.g.
        ``"eng+hin"``) skips both the CNN and the script-to-language mapping.
        """
        result = _blank_result()
        if return_attempts:
            result["attempts"] = []
        try:
            self._recognize(image, script, languages, result, return_attempts)
        except OCRBackendUnavailable as exc:
            _fail(result, str(exc))
        except Exception as exc:  # last-resort guard: OCR must never crash the pipeline
            log.exception("unexpected OCR error")
            _fail(result, f"unexpected error: {type(exc).__name__}: {exc}")
        return result

    def _recognize(self, source: ImageInput, script: Optional[str], languages: Optional[str],
                   result: dict, return_attempts: bool) -> None:
        cfg = self.config
        try:
            img = load_image(source, max_side=cfg.max_image_side)
        except ImageLoadError as exc:
            _fail(result, str(exc))
            return

        # ---- 1/2: script detection and language selection -------------
        candidates: List[str] = []
        if languages:
            candidates = [languages]
            result["script_source"] = "language_override"
        elif script:
            sc = normalize_script_name(script)
            candidates = language_candidates(sc, cfg.languages_mode)
            result.update(script=sc, script_source="override")
        else:
            try:
                info = self.detect_script(source, img)
                if not info["script"]:
                    raise RuntimeError("CNN returned no script")
                result.update(script=info["script"], script_confidence=info["confidence"],
                              script_source="cnn", script_low_confidence=info["low_confidence"])
                scripts = [info["script"]]
                probs = sorted(info["class_probabilities"].items(), key=lambda kv: kv[1], reverse=True)
                for name, p in probs:
                    try:
                        nm = normalize_script_name(name)
                    except ValueError:
                        continue
                    if nm in scripts:
                        continue
                    if len(scripts) < 2 and (info["low_confidence"] or p >= cfg.secondary_script_prob):
                        scripts.append(nm)
                for sc in scripts:
                    candidates += [c for c in language_candidates(sc, cfg.languages_mode) if c not in candidates]
            except Exception as exc:
                result["warnings"].append(f"script detection unavailable ({type(exc).__name__}: {exc}); "
                                          "trying all supported scripts")
                result["script_source"] = "unavailable"
                candidates = ["eng"]  # Latin-only boards first, then every supported script
                for sc in NATIVE_SCRIPTS:
                    candidates += [c for c in language_candidates(sc, cfg.languages_mode) if c not in candidates]

        installed = self.available_languages()
        usable_langs = [c for c in candidates if all(p in installed for p in _split_lang(c))]
        missing = sorted({p for c in candidates for p in _split_lang(c) if p not in installed})
        if missing:
            result["warnings"].append(f"OCR languages not available: {', '.join(missing)}")
        if not usable_langs:
            if not installed:
                _fail(result, "EasyOCR is not installed: pip install -r requirements.txt")
            else:
                _fail(result, f"no available OCR language for {candidates}; missing: {', '.join(missing)}")
            return

        # ---- 3: preprocessing plan ------------------------------------
        analysis = analyze_image(img)
        result["image_analysis"] = analysis.to_dict()
        strategies = select_strategies(analysis, cfg.deskew, cfg.strategies)
        strategies = strategies[:max(cfg.max_strategies, 1)]
        result["ocr_backend"] = self.backend

        # ---- 4: run OCR ------------------------------------------------
        cache: Dict[str, np.ndarray] = {}

        def variant(name: str) -> np.ndarray:
            if name not in cache:
                cache[name] = apply_strategy(img, name)
            return cache[name]

        attempts: List[_Attempt] = []
        best: Optional[_Attempt] = None
        started = time.monotonic()
        out_of_time = False
        for lang in usable_langs:
            for name in strategies:
                if time.monotonic() - started > cfg.time_budget:
                    out_of_time = True
                    break
                att = self._make_attempt(variant(name), lang, name)
                attempts.append(att)
                if self._usable(att) and self._better(att, best):
                    best = att
                if self._usable(att) and att.confidence >= cfg.early_stop_confidence:
                    break
            if out_of_time:
                result["warnings"].append(f"time budget of {cfg.time_budget:.0f}s reached; remaining attempts skipped")
                break

        if return_attempts:
            result["attempts"] = [a.summary() for a in attempts]
        errors = sorted({a.error for a in attempts if a.error})
        if errors:
            result["warnings"].extend(errors[:3])

        # ---- 5/6: clean, split train numbers, report -------------------
        if best is None:
            _fail(result, None)
            result["raw_text"] = max((a.raw_text for a in attempts), key=len, default="")
            return
        clean = clean_text(best.raw_text)
        numbers, station_text = split_train_numbers(clean)
        result.update(
            status=STATUS_OK, text=clean, raw_text=best.raw_text, clean_text=clean,
            ocr_confidence=best.confidence, ocr_language=best.language,
            preprocessing_method=best.method,
            low_confidence=best.confidence < cfg.min_confidence,
            train_numbers=numbers, station_text=station_text,
        )


def _blank_result() -> dict:
    return {
        "status": STATUS_FAILED,
        "text": "",
        "raw_text": "",
        "clean_text": "",
        "ocr_confidence": None,
        "ocr_language": None,
        "preprocessing_method": None,
        "ocr_backend": None,
        "script": None,
        "script_confidence": None,
        "script_source": None,
        "script_low_confidence": None,
        "low_confidence": None,
        "train_numbers": [],
        "station_text": "",
        "image_analysis": None,
        "error": None,
        "warnings": [],
    }


def _fail(result: dict, error: Optional[str]) -> None:
    result.update(status=STATUS_FAILED, text="", clean_text="", ocr_confidence=None,
                  ocr_language=None, preprocessing_method=None, low_confidence=None,
                  train_numbers=[], station_text="")
    if error:
        result["error"] = error


_default_engine: Optional[OCREngine] = None


def extract_text(image: ImageInput, script: Optional[str] = None, languages: Optional[str] = None,
                 engine: Optional[OCREngine] = None, **kwargs: Any) -> dict:
    """Convenience wrapper: ``extract_text("sign.jpg")`` -> result dict."""
    global _default_engine
    if engine is None:
        if _default_engine is None:
            _default_engine = OCREngine()
        engine = _default_engine
    return engine.recognize(image, script=script, languages=languages, **kwargs)
