"""OCR result -> station -> train -> train state.

``RailwayPipeline.process(ocr_result, journey_date=None)`` consumes the dict
returned by ``src.ocr`` (or a plain string) and chains

    station matching  ->  train lookup (five-digit number)  ->  train state

It never raises and never forces an answer. ``status`` says where the chain
stopped:

    ok / insufficient_history / no_next_station / ...   passed through from get_state
    ocr_failed                no usable OCR text
    station_not_identified    "Station could not be confidently identified."
    station_ambiguous         several equally likely stations
    station_needs_confirmation  a "possible" (75-89) match; not used automatically
    train_unavailable         a train number was read but is not in the dataset
    train_ambiguous           the number maps to several trains
    journey_date_required     train + station known, date unknown and not inferable
    no_train_number           station identified, but no train number on the sign
    data_unavailable          processed dataset missing
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .railway_data import RailwayData, parse_date
from .station_master import StationMaster
from .station_matcher import MatchConfig, StationMatcher
from .train_state import TRAIN_UNAVAILABLE_MESSAGE, TrainStateConfig, TrainStateService

try:  # train-number regex lives with the OCR component; keep the layer usable without it
    from src.ocr.train_numbers import extract_train_numbers, split_train_numbers
except Exception:  # pragma: no cover
    import re as _re

    def extract_train_numbers(text: str, unique: bool = True) -> List[str]:
        nums = _re.findall(r"(?<!\d)\d{5}(?!\d)", text or "")
        return list(dict.fromkeys(nums)) if unique else nums

    def split_train_numbers(text: str):
        return extract_train_numbers(text), _re.sub(r"(?<!\d)\d{5}(?!\d)", " ", text or "").strip()


class RailwayPipeline:
    def __init__(self, station_master: Optional[StationMaster] = None,
                 data: Union[RailwayData, pd.DataFrame, str, Path, None] = None,
                 match_config: Optional[MatchConfig] = None,
                 state_config: Optional[TrainStateConfig] = None,
                 accept_possible_station: bool = False):
        self.master = station_master or StationMaster.load()   # SQLite if populated, else CSV
        self.matcher = StationMatcher(self.master, match_config)
        self.states = TrainStateService(data, self.master, state_config)
        self.accept_possible = accept_possible_station

    def process(self, ocr: Union[str, Dict[str, Any], None], journey_date: Any = None,
                train_no: Optional[str] = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "status": "", "message": "", "ocr_text": None, "station": None, "train": None,
            "train_state": None, "journey_date": None, "warnings": [],
        }

        # -- OCR output ----------------------------------------------------
        if isinstance(ocr, dict):
            if ocr.get("status") not in (None, "ok") or not (ocr.get("clean_text") or ocr.get("text")):
                out.update(status="ocr_failed", message="OCR failed; no text to identify a station or train.")
                if ocr.get("error"):
                    out["warnings"].append(ocr["error"])
                return out
            text = ocr.get("clean_text") or ocr.get("text") or ""
            numbers = list(ocr.get("train_numbers") or extract_train_numbers(text))
        elif isinstance(ocr, str) and ocr.strip():
            text = ocr
            numbers, plain_station_text = split_train_numbers(ocr)
        else:
            out.update(status="ocr_failed", message="OCR failed; no text to identify a station or train.")
            return out
        out["ocr_text"] = text
        if train_no:
            numbers = [train_no] + [n for n in numbers if n != train_no]

        # -- train(s) --------------------------------------------------------
        trains = [self.states.lookup_train(n, journey_date) for n in numbers]
        found = [t for t in trains if t["status"] == "ok"]
        if len(found) == 1:
            out["train"] = found[0]
        elif len(trains) == 1:
            out["train"] = trains[0]
        elif trains:
            out["train"] = {"status": "multiple_numbers", "message": "More than one train number on the sign.",
                            "candidates": trains}
        if len(found) > 1:
            out["warnings"].append("several train numbers found in the data; using none automatically")

        # -- station ---------------------------------------------------------
        context = self.states.data.route_codes(found[0]["train_no"]) if (len(found) == 1 and self.states.data) else None
        station_query = (ocr.get("station_text") or text) if isinstance(ocr, dict) else plain_station_text
        match = self.matcher.match(station_query, context_codes=context)
        out["station"] = match
        out["warnings"].extend(match.get("warnings", []))
        if match["status"] in ("not_identified", "no_text"):
            out.update(status="station_not_identified", message=match["message"])
            return out
        if match["status"] == "ambiguous":
            out.update(status="station_ambiguous", message=match["message"])
            return out
        if match["status"] == "possible" and not self.accept_possible:
            out.update(status="station_needs_confirmation",
                       message=f"Possible station: {match['station_name']} ({match['station_code']}, "
                               f"score {match['match_score']}); needs confirmation.")
            return out
        station_code = match["station_code"]

        # -- train must exist -----------------------------------------------
        if not numbers:
            out.update(status="no_train_number",
                       message=f"Station identified as {match['station_name']} ({station_code}); "
                               "no five-digit train number found on the sign.")
            return out
        if self.states.data is None:
            out.update(status="data_unavailable", message=self.states.load_error or "Railway dataset not loaded.")
            return out
        if len(found) == 0:
            kinds = {t["status"] for t in trains}
            out.update(status="train_ambiguous" if "ambiguous_train" in kinds else "train_unavailable",
                       message=trains[0]["message"] if trains else TRAIN_UNAVAILABLE_MESSAGE)
            return out
        if len(found) > 1:
            out.update(status="train_ambiguous", message="More than one known train number on the sign.")
            return out
        tno = found[0]["train_no"]

        # -- journey date: given, or the only one on record ----------------
        jd = parse_date(journey_date)
        if jd is None:
            dates = self.states.data.available_dates(tno, station_code)
            if len(dates) == 1:
                jd = pd.Timestamp(dates[0])
                out["warnings"].append(f"journey_date not given; using {jd.date()}, the only date on record "
                                       f"for train {tno} at {station_code}")
            else:
                out.update(status="journey_date_required",
                           message="Journey date needed to look up the train state.",
                           available_dates=[pd.Timestamp(d).date().isoformat() for d in dates[-10:]])
                return out
        out["journey_date"] = jd.date().isoformat()

        state = self.states.get_state(tno, jd, station_code)
        out["train_state"] = state
        out.update(status=state["status"], message=state["message"])
        return out
