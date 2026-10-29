"""OCR text clean-up that preserves Indian-language characters.

Rules of thumb
* Keep every Unicode letter (L*), combining mark (M*: matras, virama, nukta,
  anusvara...) and number (N*, including Devanagari/Kannada/Telugu digits).
* Normalise to NFC (not NFKC: compatibility folding can alter Indic text).
* ZWJ / ZWNJ (U+200D / U+200C) are removed by default: OCR models for Indic
  scripts often insert a ZWJ inside conjuncts (e.g. "दिल्\u200dली"), which
  would break exact station-name matching although the text looks identical.
  Pass ``keep_joiners=True`` to keep them (only between two letters/marks).
  ``raw_text`` in the OCR result always keeps them untouched.
* Only symbols that are almost always OCR noise (| ~ ` ^ _ \\ { } [ ] < > * #
  @ = + etc.) are removed.
"""
from __future__ import annotations

import re
import unicodedata
from typing import List

ZWNJ = "‌"
ZWJ = "‍"

# invisible / formatting characters that are never meaningful in a sign
_DROP_INVISIBLE = set(
    "​⁠﻿­‎‏"
    "‪‫‬‭‮⁦⁧⁨⁩"
)
# punctuation worth keeping inside station / train text
_KEEP_PUNCT = set(".,-/&():'।॥")  # includes danda U+0964 and double danda U+0965
_DASHES = "‐‑‒–—―−"
_APOSTROPHES = "‘’ʼ´"
_LEADING_STRIP = ".,-:/'" + "।॥"
_TRAILING_STRIP = ",-:/'"  # a trailing "." is kept ("Jn.", "Nzm.")


def normalize_unicode(text: str, keep_joiners: bool = False) -> str:
    """NFC-normalise, unify dash/apostrophe variants, drop invisible junk and
    control characters (newlines are kept). ZWJ/ZWNJ are dropped unless
    ``keep_joiners``."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    out: List[str] = []
    for ch in text:
        if ch in _DROP_INVISIBLE or (ch in (ZWJ, ZWNJ) and not keep_joiners):
            continue
        if ch in _DASHES:
            out.append("-")
        elif ch in _APOSTROPHES:
            out.append("'")
        elif ch in ("\r\f\v"):
            out.append("\n")
        elif ch == " " or ch == " ":
            out.append("\n")
        elif ch == "\t" or ch == " ":
            out.append(" ")
        elif unicodedata.category(ch) == "Cc" and ch != "\n":
            continue
        else:
            out.append(ch)
    return "".join(out)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces, trim every line, drop empty lines."""
    lines = (re.sub(r"[  -   　]+", " ", ln).strip() for ln in text.split("\n"))
    return "\n".join(ln for ln in lines if ln)


def _is_word_char(ch: str) -> bool:
    return unicodedata.category(ch)[0] in "LMN"


def _clean_line_punct(line: str) -> str:
    # 1) per-character: keep letters/marks/numbers, safe punctuation, joiners
    chars = list(line)
    out: List[str] = []
    for i, ch in enumerate(chars):
        cat = unicodedata.category(ch)
        if cat[0] in "LMN":
            out.append(ch)
        elif ch in (ZWJ, ZWNJ):
            prev_ok = bool(out) and unicodedata.category(out[-1])[0] in "LM"
            next_ok = i + 1 < len(chars) and unicodedata.category(chars[i + 1])[0] in "LM"
            if prev_ok and next_ok:
                out.append(ch)
        elif ch == " " or cat == "Zs":
            out.append(" ")
        elif ch in _KEEP_PUNCT:
            out.append(ch)
        else:
            out.append(" ")  # stray symbol -> separator
    s = "".join(out)

    # 2) collapse repeated punctuation: "--" -> "-", ".." -> "."
    s = re.sub(r"([.,\-:/'।॥])\1+", r"\1", s)

    # 3) token level: drop pure-punctuation tokens, trim edge punctuation
    tokens: List[str] = []
    for tok in s.split(" "):
        if not tok:
            continue
        if not any(_is_word_char(c) for c in tok):
            if tok == "&":
                tokens.append(tok)
            continue
        tok = tok.lstrip(_LEADING_STRIP)
        tok = tok.rstrip(_TRAILING_STRIP)
        # an unbalanced bracket is almost certainly noise
        if tok.count("(") != tok.count(")"):
            tok = tok.replace("(", "").replace(")", "")
        if tok and any(_is_word_char(c) for c in tok):
            tokens.append(tok)
    return " ".join(tokens)


def cleanup_punctuation(text: str) -> str:
    """Remove OCR noise symbols and edge punctuation, keep Indic characters."""
    lines = (_clean_line_punct(ln) for ln in text.split("\n"))
    return "\n".join(ln for ln in lines if ln)


def remove_duplicate_lines(text: str) -> str:
    """Drop repeated lines (case-insensitive, whitespace-insensitive), keeping
    the first occurrence and the original order."""
    seen = set()
    kept: List[str] = []
    for ln in text.split("\n"):
        key = re.sub(r"\s+", " ", ln).strip().casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append(ln)
    return "\n".join(kept)


def clean_text(raw: str, keep_joiners: bool = False) -> str:
    """Full pipeline: Unicode -> whitespace -> punctuation -> duplicates."""
    if not raw or not raw.strip():
        return ""
    t = normalize_unicode(raw, keep_joiners)
    t = normalize_whitespace(t)
    t = cleanup_punctuation(t)
    t = normalize_whitespace(t)
    t = remove_duplicate_lines(t)
    return t.strip()
