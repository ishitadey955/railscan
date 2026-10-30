"""Identify the most likely railway station from OCR text.

Methods, tried in this order (exact methods first; fuzzy only when nothing
matched exactly):

    station_code          text is exactly a station code           score 100
    exact_name            identical to the master name             score 100
    case_insensitive      same ignoring case                       score  98
    alias                 listed alias (any language), normalised  score  97
    normalized_name       same after normalisation / abbreviations score  96
                          (JN = JUNCTION, CNTL = CENTRAL, spacing)         95 for spacing-only
    core_name / core_alias same without JUNCTION / STATION (any script) score  92
    partial_name          every word of the text is a whole word of     score  85 (never "high")
                          the station name, e.g. "DELHI" -> several
                          stations => ambiguous, "SHAHDARA" -> one => possible
    fuzzy / alias_fuzzy   RapidFuzz token-sort ratio               score = ratio
    transliteration_fuzzy Indic text -> Latin: consonant skeleton  score = mean of skeleton and full
                          and full spelling (vowels too) vs the   spelling ratios x 0.85 (never "high")
                          English name, so RATNAGIRI beats RATANGARH

Confidence tiers (configurable in :class:`MatchConfig`):
    score >= 90        high      -> accepted automatically
    75 <= score < 90   possible  -> returned, but NOT accepted (needs confirmation)
    score < 75         low       -> "Station could not be confidently identified."

When two different stations are equally plausible the result status is
"ambiguous" and no station is forced.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

from rapidfuzz import fuzz, process

from .station_master import StationMaster
from .text_normalizer import (
    GENERIC_SUFFIXES, compact_key, core_key, detect_script, display_name, matching_key,
    phonetic_skeleton, transliterate_to_latin,
)

NOT_IDENTIFIED_MESSAGE = "Station could not be confidently identified."

METHOD_SCORES = {
    "station_code": 100.0,
    "exact_name": 100.0,
    "case_insensitive": 98.0,
    "alias": 97.0,
    "normalized_name": 96.0,
    "compact_name": 95.0,
    "core_name": 92.0,
    "partial_name": 85.0,
}


@dataclass
class MatchConfig:
    high_threshold: float = 90.0          # >= : high-confidence, accepted
    possible_threshold: float = 75.0      # >= : possible match, needs confirmation
    min_candidate_score: float = 60.0     # fuzzy candidates below this are not even listed
    ambiguity_margin: float = 2.0         # two stations within this many points = ambiguous
    core_fuzzy_penalty: float = 0.95      # fuzzy match against a "core" name is slightly discounted
    transliteration_cap: float = 85.0     # transliteration matches never exceed this
    min_skeleton_len: int = 3
    max_candidates: int = 5
    exact_core_min_len: int = 3
    generic_core_cap: float = 85.0        # query equals a name minus JUNCTION/STATION: never "high", ties with partial matches
    partial_name_score: float = 85.0      # query = whole words inside a longer station name (never "high")
    partial_min_token_len: int = 3
    partial_max_stations: int = 40        # more candidates than this = too generic, ignore


@dataclass
class _Cand:
    code: str
    score: float
    method: str
    text: str = ""


# Words of stock-photo watermarks ("Adobe Stock" tiled over a preview image). OCR misreads them
# ("Stodk", "Moobe"), so they are compared fuzzily; see strip_watermarks().
WATERMARK_WORDS = ("adobe", "stock", "shutterstock", "alamy", "dreamstime", "istock", "getty", "images",
                   "depositphotos", "freepik", "vectorstock", "bigstock", "123rf", "preview", "watermark")


def strip_watermarks(text: str, min_repeats: int = 3) -> str:
    """Remove watermark words from OCR text before station matching.

    Dropped: Latin words that look like a stock-photo watermark word (fuzzy, as OCR misreads them),
    Latin words repeated ``min_repeats`` times or more (a tiled watermark repeats; a station name on
    a board does not), and tokens with no letters at all (digit noise). Indic words are kept.
    """
    lines = [ln.split() for ln in re.split(r"[\r\n|]+", text or "")]
    latin = [t.casefold() for ln in lines for t in ln if detect_script(t) == "latin"]
    counts = {t: latin.count(t) for t in set(latin)}

    def keep(tok: str) -> bool:
        if not re.search(r"[^\W\d_]", tok):                          # no letters: digits, symbols
            return False
        if detect_script(tok) != "latin":
            return True
        low = tok.casefold()
        if counts.get(low, 0) >= min_repeats:
            return False
        return len(low) < 3 or max(fuzz.ratio(low, w) for w in WATERMARK_WORDS) < 75

    return "\n".join(" ".join(t for t in ln if keep(t)) for ln in lines if any(keep(t) for t in ln))


def script_runs(text: str) -> List[str]:
    """Each line split into runs of one script: 'Stock रत्नागिरी tock' -> ['Stock', 'रत्नागिरी', 'tock'].
    Boards print each script on its own; noise of another script then cannot hide the name."""
    runs: List[str] = []
    for ln in re.split(r"[\r\n|]+", text or ""):
        cur, cur_script = [], None
        for tok in ln.split():
            sc = detect_script(tok)
            if cur and sc != cur_script:
                runs.append(" ".join(cur)); cur = []
            cur.append(tok); cur_script = sc
        if cur:
            runs.append(" ".join(cur))
    return [r for r in runs if r.strip()]


def _letters(text: Optional[str]) -> str:
    """Lower-case Latin letters only: 'ratnAgirI' -> 'ratnagiri', 'RATNAGIRI JN' -> 'ratnagirijn'."""
    return re.sub(r"[^a-z]", "", (text or "").lower())


class StationMatcher:
    """Match free text (or an OCR result dict) to a station in a :class:`StationMaster`."""

    def __init__(self, master: StationMaster, config: Optional[MatchConfig] = None):
        self.master = master
        self.cfg = config or MatchConfig()
        self._build_indexes()

    # ------------------------------------------------------------------ indexes
    def _build_indexes(self) -> None:
        self._raw: Dict[str, Set[str]] = defaultdict(set)         # exact master name
        self._ci: Dict[str, Set[str]] = defaultdict(set)          # casefolded master name
        self._alias: Dict[str, Set[str]] = defaultdict(set)       # matching_key(alias)
        self._alias_core: Dict[str, Set[str]] = defaultdict(set)  # core_key(alias), generic words dropped
        self._norm: Dict[str, Set[str]] = defaultdict(set)        # matching_key(name)
        self._compact: Dict[str, Set[str]] = defaultdict(set)     # compact_key(name)
        self._core: Dict[str, Set[str]] = defaultdict(set)        # core_key(name)
        corpora: Dict[str, Tuple[List[str], List[Tuple[str, str]]]] = {
            s: ([], []) for s in ("latin", "devanagari", "kannada", "telugu", "bengali")}

        def corpus_add(key: str, code: str, kind: str) -> None:
            script = detect_script(key)
            if script in corpora and key:
                corpora[script][0].append(key)
                corpora[script][1].append((code, kind))

        for st in self.master.stations.values():
            self._raw[st.name.strip()].add(st.code)
            self._ci[st.name.strip().casefold()].add(st.code)
            key, core, comp = matching_key(st.name), core_key(st.name), compact_key(st.name)
            if key:
                self._norm[key].add(st.code)
                self._compact[comp].add(st.code)
                corpus_add(key, st.code, "name")
            if core and len(core) >= self.cfg.exact_core_min_len:
                self._core[core].add(st.code)
                if core != key:
                    corpus_add(core, st.code, "core")
        for al in self.master.aliases:
            key = matching_key(al.alias)
            if key:
                self._alias[key].add(al.code)
                corpus_add(key, al.code, "alias")
                ack = core_key(al.alias)
                if ack and ack != key and len(ack) >= self.cfg.exact_core_min_len:
                    self._alias_core[ack].add(al.code)
                    corpus_add(ack, al.code, "core")
        self._corpora = corpora

        # whole-word index for the partial_name method
        tok: Dict[str, Set[str]] = defaultdict(set)
        for table in (self._norm, self._alias):
            for key, codes in table.items():
                for t in key.split():
                    if t.upper() not in GENERIC_SUFFIXES:
                        tok[t].update(codes)
        self._tok_index = tok

        # consonant skeletons of English names/aliases for the transliteration fallback
        sk: Dict[str, Set[str]] = defaultdict(set)
        latin: Dict[str, Set[str]] = defaultdict(set)      # code -> full Latin spellings (no spaces)
        for key, codes in list(self._norm.items()) + list(self._alias.items()) + list(self._core.items()):
            if detect_script(key) == "latin":
                s = phonetic_skeleton(key)
                if len(s) >= self.cfg.min_skeleton_len:
                    sk[s].update(codes)
                for code in codes:
                    latin[code].add(_letters(key))
        self._skeletons = sk
        self._skeleton_keys = list(sk)
        self._latin = latin

    # ------------------------------------------------------------------ segments
    @staticmethod
    def _segments(text: str) -> List[str]:
        lines = [ln.strip() for ln in re.split(r"[\r\n]+", text) if ln.strip()]
        segs: List[str] = []
        for ln in lines:
            for code in re.findall(r"[(\[]\s*([A-Z0-9]{1,6})\s*[)\]]", ln):
                segs.append(code)                                 # "(NDLS)" style codes
            stripped = re.sub(r"[(\[]\s*[A-Z0-9]{1,6}\s*[)\]]", " ", ln).strip()
            segs.append(stripped or ln)
        if len(lines) > 1:
            segs.append(" ".join(lines))                          # a name split over lines
        seen, out = set(), []
        for s in segs:
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out

    # ------------------------------------------------------------------ one segment
    def _exact_candidates(self, raw: str) -> List[_Cand]:
        out: List[_Cand] = []
        # exact station code: case-sensitive, so lower-case words are not read as codes
        if re.fullmatch(r"[A-Z0-9]{1,6}", raw) and raw in self.master:
            out.append(_Cand(raw, METHOD_SCORES["station_code"], "station_code", raw))
        for code in self._raw.get(raw, ()):
            out.append(_Cand(code, METHOD_SCORES["exact_name"], "exact_name", raw))
        for code in self._ci.get(raw.casefold(), ()):
            out.append(_Cand(code, METHOD_SCORES["case_insensitive"], "case_insensitive", raw))
        key = matching_key(raw)
        if not key:
            return out
        for code in self._alias.get(key, ()):
            out.append(_Cand(code, METHOD_SCORES["alias"], "alias", raw))
        for code in self._norm.get(key, ()):
            out.append(_Cand(code, METHOD_SCORES["normalized_name"], "normalized_name", raw))
        for code in self._compact.get(compact_key(raw), ()):
            out.append(_Cand(code, METHOD_SCORES["compact_name"], "compact_name", raw))
        ck = core_key(raw)
        if len(ck) >= self.cfg.exact_core_min_len:
            for code in self._core.get(ck, ()):
                out.append(_Cand(code, METHOD_SCORES["core_name"], "core_name", raw))
            if ck != key:  # the text carried a generic word (JUNCTION / जंक्शन ...): compare aliases without it too
                for table in (self._alias, self._alias_core):
                    for code in table.get(ck, ()):
                        out.append(_Cand(code, METHOD_SCORES["core_name"], "core_alias", raw))
        return out

    def _fuzzy_candidates(self, raw: str) -> List[_Cand]:
        key = matching_key(raw)
        script = detect_script(key)
        if not key or script not in self._corpora:
            return []
        keys, meta = self._corpora[script]
        if not keys:
            return []
        # probe with the full key and with the key minus JUNCTION/STATION, so
        # "NEW DEHLI JN" can still reach "NEW DELHI"
        probes = [key]
        ck = core_key(raw)
        query_stripped = bool(ck) and ck != key
        if ck and ck != key and len(ck) >= self.cfg.exact_core_min_len:
            probes.append(ck)
        best: Dict[Tuple[str, str], float] = {}
        for probe in probes:
            for choice, score, idx in process.extract(
                    probe, keys, scorer=fuzz.token_sort_ratio, limit=12, score_cutoff=self.cfg.min_candidate_score):
                code, kind = meta[idx]
                if kind == "core":
                    score *= self.cfg.core_fuzzy_penalty
                    if not query_stripped:
                        # "DELHI" vs the core of "DELHI JUNCTION": plausible but not certain
                        score = min(score, self.cfg.generic_core_cap)
                k = (code, kind)
                best[k] = max(best.get(k, 0.0), float(score))
        return [_Cand(code, round(sc, 1), "alias_fuzzy" if kind == "alias" else "fuzzy", raw)
                for (code, kind), sc in best.items()]

    def _partial_candidates(self, raw: str) -> List[_Cand]:
        toks = [t for t in matching_key(raw).split() if t.upper() not in GENERIC_SUFFIXES]
        if not toks or any(len(t) < self.cfg.partial_min_token_len for t in toks):
            return []
        sets = [self._tok_index.get(t) for t in toks]
        if any(s is None for s in sets):
            return []
        codes = set.intersection(*sets)
        if not codes or len(codes) > self.cfg.partial_max_stations:
            return []
        return [_Cand(c, self.cfg.partial_name_score, "partial_name", raw) for c in codes]

    def _transliteration_candidates(self, raw: str) -> List[_Cand]:
        if detect_script(raw) not in ("devanagari", "kannada", "telugu", "bengali"):
            return []
        latin = transliterate_to_latin(core_key(raw) or raw)
        sk = phonetic_skeleton(latin)
        if len(sk) < self.cfg.min_skeleton_len:
            return []
        full = _letters(latin)
        out: List[_Cand] = []
        for choice, score, _ in process.extract(sk, self._skeleton_keys, scorer=fuzz.ratio, limit=10,
                                                score_cutoff=self.cfg.min_candidate_score):
            for code in self._skeletons[choice]:
                # the skeleton drops vowels (RATNAGIRI and RATANGARH are both "rtngr"); the full spelling
                # tells them apart. Scaled so the maximum is the cap: never "high" on its own.
                spelled = max((fuzz.ratio(full, k) for k in self._latin.get(code, ())), default=0.0)
                capped = round((float(score) + spelled) / 2 * self.cfg.transliteration_cap / 100.0, 1)
                out.append(_Cand(code, capped, "transliteration_fuzzy", raw))
        return out

    def _segment_candidates(self, raw: str) -> List[_Cand]:
        cands = self._exact_candidates(raw)
        if cands:
            return cands
        cands = self._fuzzy_candidates(raw) + self._partial_candidates(raw)
        if not any(c.score >= self.cfg.possible_threshold for c in cands):
            cands += self._transliteration_candidates(raw)
        return cands

    # ------------------------------------------------------------------ public API
    def match(self, query: Union[str, Dict[str, Any], None],
              context_codes: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        """Identify the station in ``query`` (text, or an OCR result dict from
        ``src.ocr``). ``context_codes`` (e.g. the stations a detected train
        calls at) is only used to break ties between equally good stations.

        Always returns a dict; see module docstring for the ``status`` values:
        matched | possible | ambiguous | not_identified | no_text | ocr_failed.
        """
        warnings: List[str] = []
        text = ""
        if isinstance(query, dict):
            if query.get("status") not in (None, "ok") or (query.get("status") is None and not query.get("clean_text")):
                return self._result("ocr_failed", message="OCR failed; no text to match.",
                                    warnings=[query["error"]] if query.get("error") else [])
            text = query.get("station_text") or query.get("clean_text") or query.get("text") or ""
            if query.get("low_confidence"):
                warnings.append(f"OCR confidence is low ({query.get('ocr_confidence')}); verify the match.")
        elif isinstance(query, str):
            text = query
        if not text.strip():
            return self._result("no_text", message="No station text to match.", warnings=warnings)

        # best candidate per station across all text segments
        best: Dict[str, _Cand] = {}
        self._collect(text, best)
        return self._decide(best, warnings, context_codes)

    def match_readings(self, primary: Union[str, Dict[str, Any], None], others: Iterable[str],
                       context_codes: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        """Like :meth:`match` on ``primary`` (the reading the app chose), but stations found only in
        ``others`` (the sign read with the other scripts, or faint readings the OCR rejected as too
        uncertain) are added as candidates. Their scores are scaled by transliteration_cap / 100 (0.85),
        so they are suggested for the user to confirm, never accepted automatically.

        A blurry or watermarked photo often has the station name in a reading other than the chosen
        one (e.g. the chosen reading is mostly "Adobe Stock", the Devanagari one says रत्नागिरी).
        """
        result = self.match(primary, context_codes)
        scale = self.cfg.transliteration_cap / 100.0        # other readings: score x 0.85, never "high"
        extra: Dict[str, _Cand] = {}
        for text in others:
            if text and text.strip():
                self._collect_cleaned(text, extra, scale=scale, tag="+other_reading")
        if result.get("accepted") or result.get("status") == "possible":
            # the chosen reading already names a station: keep that decision exactly, and only list
            # stations from the other readings as further suggestions after its own candidates
            have = {c["station_code"] for c in result.get("candidates") or []}
            more = sorted((c for c in extra.values() if c.code not in have), key=lambda c: (-c.score, c.code))
            room = max(self.cfg.max_candidates - len(have), 0)
            result["candidates"] = list(result.get("candidates") or []) + [self._cand_dict(c) for c in more[:room]]
            return result
        # the chosen reading found nothing usable: decide on all readings together. (Tried: a bonus
        # for stations several readings agree on. It made results worse: a tiled watermark repeats
        # the same junk in every reading, so the junk "agreed" too.)
        best: Dict[str, _Cand] = dict(extra)
        if isinstance(primary, dict) and primary.get("status") == "ok":
            self._collect_cleaned(primary.get("station_text") or primary.get("clean_text") or "", best)
        elif isinstance(primary, str):
            self._collect_cleaned(primary, best)
        if not best:                    # nothing anywhere: keep the original verdict (e.g. ocr_failed)
            return result
        return self._decide(best, list(result.get("warnings") or []), context_codes)

    def _collect_cleaned(self, text: str, best: Dict[str, _Cand], scale: float = 1.0, tag: str = "") -> None:
        """_collect on the text without watermark words, line by line and script run by script run."""
        cleaned = strip_watermarks(text)
        self._collect(cleaned, best, scale=scale, tag=tag)
        for run in script_runs(cleaned):
            self._collect(run, best, scale=scale, tag=tag)

    def _collect(self, text: str, best: Dict[str, _Cand], scale: float = 1.0, tag: str = "") -> None:
        """Add the best candidate per station found in ``text`` to ``best`` (scores times ``scale``)."""
        for seg in self._segments(text):
            for c in self._segment_candidates(seg):
                if scale != 1.0 or tag:
                    c = _Cand(c.code, round(c.score * scale, 1), c.method + tag, c.text)
                if c.code not in best or c.score > best[c.code].score:
                    best[c.code] = c

    def _decide(self, best: Dict[str, "_Cand"], warnings: List[str],
                context_codes: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        ranked = sorted(best.values(), key=lambda c: (-c.score, c.code))
        cand_list = [self._cand_dict(c) for c in ranked[: self.cfg.max_candidates]]
        if not ranked or ranked[0].score < self.cfg.possible_threshold:
            return self._result("not_identified", message=NOT_IDENTIFIED_MESSAGE, level="low",
                                candidates=cand_list, warnings=warnings)

        top = ranked[0]
        rivals = [c for c in ranked[1:]
                  if c.score >= self.cfg.possible_threshold and top.score - c.score < self.cfg.ambiguity_margin]
        if rivals:
            tied = [top] + rivals
            ctx = {(c or "").strip().upper() for c in (context_codes or [])}
            in_ctx = [c for c in tied if c.code in ctx]
            if len(in_ctx) == 1:
                chosen = in_ctx[0]
                chosen = _Cand(chosen.code, chosen.score, chosen.method + "+train_route", chosen.text)
                return self._accept(chosen, cand_list, warnings)
            names = ", ".join(f"{self._name(c.code)} ({c.code})" for c in tied[:4])
            return self._result("ambiguous", message=f"Multiple possible stations: {names}. Not selecting one.",
                                level="ambiguous", candidates=cand_list, warnings=warnings)
        return self._accept(top, cand_list, warnings)

    def match_text(self, text: str, **kw) -> Dict[str, Any]:
        return self.match(text, **kw)

    # ------------------------------------------------------------------ result helpers
    def _name(self, code: str) -> str:
        st = self.master.get(code)
        return display_name(st.name) if st else code

    def _cand_dict(self, c: _Cand) -> Dict[str, Any]:
        return {"station_name": self._name(c.code), "station_code": c.code,
                "match_score": c.score, "match_method": c.method}

    def _accept(self, c: _Cand, candidates: List[dict], warnings: List[str]) -> Dict[str, Any]:
        st = self.master.get(c.code)
        high = c.score >= self.cfg.high_threshold
        res = self._result(
            "matched" if high else "possible",
            message=("High-confidence match." if high else
                     "Possible match - please confirm; not accepted automatically."),
            level="high" if high else "possible", candidates=candidates, warnings=warnings)
        res.update(
            station_name=display_name(st.name), station_name_raw=st.name, station_code=st.code,
            match_score=c.score, match_method=c.method, matched_text=c.text,
            latitude=st.latitude, longitude=st.longitude, accepted=high, needs_confirmation=not high)
        return res

    @staticmethod
    def _result(status: str, message: str, level: Optional[str] = None,
                candidates: Optional[List[dict]] = None, warnings: Optional[List[str]] = None) -> Dict[str, Any]:
        return {
            "status": status, "message": message, "confidence_level": level,
            "accepted": False, "needs_confirmation": False,
            "station_name": None, "station_name_raw": None, "station_code": None,
            "match_score": None, "match_method": None, "matched_text": None,
            "latitude": None, "longitude": None,
            "candidates": candidates or [], "warnings": warnings or [],
        }
