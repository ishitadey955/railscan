# RailScan railway information layer

OCR text → **station** → **train** → **train state**. No ETA yet (next stage).

```
src/ocr result ─► StationMatcher ─► station_name / station_code / match_score / match_method
              └─► train number (5 digits) ─► processed dataset ─► train_no / train_name / train_type
                                           └─► TrainStateService ─► delay, times, next station, history
```

Install: `pip install -r requirements-railway.txt`  (rapidfuzz, pandas, indic-transliteration).

## Use
```python
from src.railway import RailwayPipeline, StationMatcher, StationMaster, TrainStateService

# 1. station only
m = StationMatcher(StationMaster.from_csv())
m.match("नई दिल्ली")   # {'status': 'matched', 'station_name': 'New Delhi', 'station_code': 'NDLS',
                        #  'match_score': 97.0, 'match_method': 'alias', ...}

# 2. train state (needs data/processed/RailScan_Final_Dataset.csv)
svc = TrainStateService("data/processed/RailScan_Final_Dataset.csv", StationMaster.from_csv())
svc.get_state("12301", "2025-03-08", "NDLS")
svc.lookup_train("12301")

# 3. whole chain from an OCR result dict (src.ocr) or plain text
RailwayPipeline().process(ocr_result, journey_date="2025-03-08")
```
Demo: `python scripts/demo_railway_pipeline.py --image sign.jpg --journey-date YYYY-MM-DD`
(`--fixture` runs on a labelled SYNTHETIC fixture when you have no dataset yet).

## Station matching (`station_matcher.py`)
Order: exact **station code** (case-sensitive, so "pune" is not a code) → exact name → case-insensitive →
**alias** (any script) → normalised name (JN = JUNCTION, CNTL = CENTRAL, spacing, "RAILWAY STATION") →
name without JUNCTION/STATION → whole-word partial → **RapidFuzz** token-sort ratio → Indic→Latin
transliteration fallback (consonant skeleton, scaled ×0.85 so it can never be "high").

| score | status | |
|---|---|---|
| ≥ 90 | `matched` | accepted automatically |
| 75–89 | `possible` | returned, **not accepted** (`needs_confirmation`) |
| < 75 | `not_identified` | "Station could not be confidently identified." — no station returned |
| equal rivals | `ambiguous` | e.g. "DELHI", "ಮಂಗಳೂರು": none is picked |

Thresholds: `MatchConfig(high_threshold=90, possible_threshold=75, ...)`. A train's route can break a tie
(`context_codes`) but never raises a score or forces a station. Other statuses: `no_text`, `ocr_failed`.
Result keys: `station_name, station_code, match_score, match_method` (+ `status, message, candidates, latitude, longitude`).

Normaliser (`text_normalizer.py`): NFC, invisible-char removal, case-folding (Latin), punctuation → space,
whitespace, railway abbreviations; Indic letters/matras/virama kept. Nukta/chandrabindu folds are used for
lookup keys only, never to rewrite text shown to the user.

## Train state (`train_state.py`)
Uses only `RailScan_Final_Dataset.csv`. Missing values are `None` and listed in `missing_fields`; target
columns are never returned (no future leakage). `status`: `ok`, `insufficient_history` (state returned, < 5
earlier journeys), `no_next_station`, `unknown_train` ("Train information unavailable"), `no_record_for_date`,
`unknown_station_for_train`, `ambiguous_train`, `ambiguous_stop`, `invalid_input`, `data_unavailable`.

Points to know:
* The dataset stores **delays**, not raw actual timestamps, so `actual_arrival` / `actual_departure` are
  *scheduled time + observed delay* (labelled "derived" in `field_sources`).
* `current_delay_min` = departure delay at the station, else arrival delay (`current_observed_delay_min`).
* `historical.*` describes this train's arrival at its **next** station on **earlier** journeys that are in the
  processed sample (strictly earlier; built leak-free by `build_dataset.py`). `reliability_pct` = share of those
  journeys under 15 min late. With a ~10k-row sample many trains have little or no history.
* The dataset only has rows whose next stop was also observed, so the **last observed stop of a journey has no
  row** → `no_next_station`.
* One train number with several names/types is narrowed by journey date, then station; if that does not
  decide, `ambiguous_train` is returned with all candidates.

## Pipeline statuses (`pipeline.py`)
Everything above plus `ocr_failed`, `station_not_identified`, `station_ambiguous`,
`station_needs_confirmation` (a "possible" match is not used unless `accept_possible_station=True`),
`train_unavailable`, `train_ambiguous`, `journey_date_required` (date unknown and not unique in the data),
`no_train_number`, `data_unavailable`.

## Limits
* Station coverage = DataMeet snapshot + aliases (see `data/stations/README.md`).
* Bare city names that cover several stations ("Delhi", "Mangaluru") are deliberately not resolved.
* Tests use a synthetic fixture for train state; run the pipeline on your real dataset to see real values.
