"""RailScan OCR component (EasyOCR)."""
from .ocr_engine import (
    OCRConfig,
    OCREngine,
    SCRIPT_TO_LANG,
    STATUS_FAILED,
    STATUS_OK,
    extract_text,
    language_candidates,
)
from .preprocessing import STRATEGIES, analyze_image, apply_strategy, load_image, select_strategies
from .text_cleaning import clean_text
from .train_numbers import extract_train_numbers, split_train_numbers

__all__ = [
    "OCRConfig", "OCREngine", "SCRIPT_TO_LANG", "STATUS_FAILED", "STATUS_OK", "extract_text",
    "language_candidates", "STRATEGIES", "analyze_image", "apply_strategy", "load_image",
    "select_strategies", "clean_text", "extract_train_numbers", "split_train_numbers",
]
