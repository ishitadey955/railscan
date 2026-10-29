"""Text normalisation for station-name matching.

Two levels, both of which keep Indian-script characters intact:

* ``normalize_text``   - Unicode NFC, invisible-character removal, case
  folding (Latin only), punctuation -> space, whitespace collapse.
* ``matching_key``     - ``normalize_text`` plus railway abbreviation
  expansion (JN -> JUNCTION, CNTL -> CENTRAL, ...) and a few script-specific
  spelling folds (Devanagari nukta / chandrabindu) used ONLY to build lookup
  keys, never to rewrite what is shown to the user.

``transliterate_to_latin`` / ``phonetic_skeleton`` give an optional, lossy
English rendering of Indic text (needs ``indic-transliteration``); the matcher
uses it only as a low-confidence fallback.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, Optional

_INVISIBLE = set("​‌‍⁠﻿­‎‏")
_DASHES = "‐‑‒–—―−"

# Indian-railways style abbreviations -> canonical word (applied token-wise on
# upper-cased Latin text; both the master names and the OCR text go through this).
ABBREVIATIONS: Dict[str, str] = {
    "JN": "JUNCTION", "JUNC": "JUNCTION", "JCT": "JUNCTION", "JUNCTN": "JUNCTION",
    "CNTL": "CENTRAL", "CENTRL": "CENTRAL", "CTRL": "CENTRAL", "CNTRL": "CENTRAL",
    "TERM": "TERMINUS", "TMNS": "TERMINUS", "TRMS": "TERMINUS",
    "CANT": "CANTT", "CANTONMENT": "CANTT", "CANTONMENT": "CANTT",
    "RD": "ROAD", "NR": "NORTH", "STN": "STATION",
}
# Words that carry no identity and are dropped when building the *core* key.
GENERIC_SUFFIXES = {
    "JUNCTION", "STATION",
    # the same generic words in the supported scripts (so "ಮೈಸೂರು ಜಂಕ್ಷನ್" is compared on "ಮೈಸೂರು")
    "जंक्शन", "जंकशन", "स्टेशन", "स्टेसन", "रेलवे",
    "ಜಂಕ್ಷನ್", "ಜಂಕ್ಷನ", "ನಿಲ್ದಾಣ", "ರೈಲ್ವೆ",
    "జంక్షన్", "స్టేషన్", "రైల్వే",
}
# Whole trailing phrases removed from OCR text ("... RAILWAY STATION").
_NOISE_PHRASES = re.compile(r"\b(RAILWAY STATION|RLY STATION|RLY STN|RAILWAY STN)\b")

_SCRIPT_RANGES = {
    "devanagari": (0x0900, 0x097F),
    "kannada": (0x0C80, 0x0CFF),
    "telugu": (0x0C00, 0x0C7F),
}


def detect_script(text: str) -> str:
    """'latin' | 'devanagari' | 'kannada' | 'telugu' | 'mixed' | 'other' | 'none'.

    Digits/punctuation are ignored; 'none' means no letters at all.
    """
    found = set()
    for ch in text or "":
        if not unicodedata.category(ch).startswith(("L", "M")):
            continue
        cp = ord(ch)
        for name, (lo, hi) in _SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                found.add(name)
                break
        else:
            found.add("latin" if cp < 0x250 else "other")
    if not found:
        return "none"
    return found.pop() if len(found) == 1 else "mixed"


def normalize_text(text: Optional[str]) -> str:
    """NFC, drop invisible characters, case-fold, punctuation -> space,
    collapse whitespace. Letters, vowel signs, virama, nukta and digits of
    every script are kept. Newlines become spaces."""
    if not text:
        return ""
    t = unicodedata.normalize("NFC", str(text))
    out = []
    for ch in t:
        if ch in _INVISIBLE:
            continue
        cat = unicodedata.category(ch)
        if cat[0] in "LMN":
            out.append(ch)
        elif ch in _DASHES or cat[0] in "PSZC":
            out.append(" ")  # punctuation, symbols, spaces, controls -> separator
        else:
            out.append(" ")
    s = "".join(out).casefold()
    return re.sub(r"\s+", " ", s).strip()


def _fold_indic(s: str) -> str:
    """Spelling folds for lookup keys only."""
    s = s.replace("़", "")            # Devanagari nukta: निज़ामुद्दीन == निजामुद्दीन
    s = s.replace("ँ", "ं")      # chandrabindu -> anusvara
    s = s.replace("಼", "")            # Kannada nukta
    return s


def matching_key(text: Optional[str]) -> str:
    """Canonical key used for exact / normalised / fuzzy comparison."""
    s = normalize_text(text)
    if not s:
        return ""
    s = _fold_indic(s)
    up = s.upper()
    up = _NOISE_PHRASES.sub(" ", up)
    toks = [ABBREVIATIONS.get(t, t) for t in up.split()]
    return " ".join(toks).casefold()


def core_key(text: Optional[str]) -> str:
    """``matching_key`` without generic suffix words (JUNCTION, STATION)."""
    toks = [t for t in matching_key(text).split() if t.upper() not in GENERIC_SUFFIXES]
    return " ".join(toks)


def compact_key(text: Optional[str]) -> str:
    """Key with all spaces removed (OCR often drops or adds spaces)."""
    return matching_key(text).replace(" ", "")


# ------------------------------------------------------------------ transliteration
def _transliterator():
    try:
        from indic_transliteration import sanscript
    except Exception:
        return None
    return sanscript


def transliterate_to_latin(text: str) -> Optional[str]:
    """Rough Latin rendering (Harvard-Kyoto scheme) of Devanagari / Kannada /
    Telugu text, or ``None`` if the optional library is missing or the text
    is not one of those scripts."""
    sc = _transliterator()
    script = detect_script(text)
    if sc is None or script not in ("devanagari", "kannada", "telugu"):
        return None
    src = {"devanagari": sc.DEVANAGARI, "kannada": sc.KANNADA, "telugu": sc.TELUGU}[script]
    try:
        return sc.transliterate(normalize_text(text), src, sc.HK)
    except Exception:
        return None


_VOWELS = set("aeiou")


def phonetic_skeleton(latin: Optional[str]) -> str:
    """Very lossy consonant skeleton ('nAgapura' and 'NAGPUR' -> 'ngpr').

    Used to compare an Indic name, transliterated to Latin, with English
    station names. Because it discards vowels it is only ever used as a
    low-confidence fallback.
    """
    if not latin:
        return ""
    s = latin.lower()
    s = s.replace("z", "s").replace("ph", "f")
    s = re.sub(r"[^a-z ]", "", s)
    s = "".join(c for c in s if c not in _VOWELS and c not in "hyw")
    s = re.sub(r"(.)\1+", r"\1", s.replace(" ", ""))
    return s


def display_name(name: str) -> str:
    """'NEW DELHI' -> 'New Delhi', 'HOWRAH JN' -> 'Howrah Jn'. Mixed-case
    source names and non-Latin names are returned unchanged."""
    if name and name.isupper():
        return " ".join(w.capitalize() for w in name.split(" "))
    return name
