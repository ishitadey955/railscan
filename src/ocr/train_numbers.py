"""Detect candidate Indian train numbers in OCR text.

Indian train numbers are (generally) five digits, e.g. ``12301``. This is a
*candidate* detector based on a regular expression only; it does not check
a number against any railway database. Native-script digits (Devanagari
``१२३०१``, Kannada ``೧೨೩೦೧``, Telugu ``౧౨౩౦౧`` ...) are converted to ASCII
before matching, and returned as ASCII.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Tuple

from .text_cleaning import clean_text

# exactly five digits, not part of a longer digit run (so PIN codes / phone
# numbers / 6+ digit codes are not split into fake train numbers)
TRAIN_NUMBER_RE = re.compile(r"(?<!\d)\d{5}(?!\d)", re.ASCII)


@dataclass(frozen=True)
class TrainNumberMatch:
    number: str   # ASCII digits
    start: int    # span in the *input* text
    end: int


def to_ascii_digits(text: str) -> str:
    """Map every Unicode decimal digit to its ASCII digit, 1 char -> 1 char
    so that match offsets stay valid for the original text."""
    out = []
    for ch in text:
        if not ch.isascii() and ch.isdecimal():
            out.append(str(unicodedata.decimal(ch)))
        else:
            out.append(ch)
    return "".join(out)


def find_train_numbers(text: str) -> List[TrainNumberMatch]:
    """All five-digit candidates with their positions, in order of appearance."""
    if not text:
        return []
    ascii_text = to_ascii_digits(text)
    return [
        TrainNumberMatch(m.group(), m.start(), m.end())
        for m in TRAIN_NUMBER_RE.finditer(ascii_text)
        if m.group() != "00000"
    ]


def extract_train_numbers(text: str, unique: bool = True) -> List[str]:
    """Candidate train numbers as ASCII strings (order kept, duplicates removed
    unless ``unique=False``)."""
    nums = [m.number for m in find_train_numbers(text)]
    if unique:
        nums = list(dict.fromkeys(nums))
    return nums


def split_train_numbers(text: str) -> Tuple[List[str], str]:
    """Return ``(train_numbers, station_text)``.

    ``station_text`` is the input with the detected train-number spans removed
    and re-cleaned, so the next pipeline stage can match station names
    without numeric noise.
    """
    matches = find_train_numbers(text)
    if not matches:
        return [], clean_text(text)
    pieces, last = [], 0
    for m in matches:
        pieces.append(text[last:m.start])
        pieces.append(" ")
        last = m.end
    pieces.append(text[last:])
    numbers = list(dict.fromkeys(m.number for m in matches))
    return numbers, clean_text("".join(pieces))
