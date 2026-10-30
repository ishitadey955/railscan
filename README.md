# RailScan

**CNN-Based Indian Railway Sign Recognition with Delay Risk and Next-Station Time Prediction**

RailScan is a Streamlit application for Indian railway passengers and a student ML project. Upload a photo of a station board or sign and RailScan:

1. identifies the script (Devanagari, Bengali, Kannada or Telugu) with a CNN and OCR evidence;
2. reads the text with OCR;
3. looks for a train number, station name and platform number, and checks them against reference data.

**Coverage: northern India and West Bengal.** RailScan covers the **5,300 trains that stop in northern India or West Bengal** (Jammu & Kashmir, Ladakh, Himachal Pradesh, Punjab, Chandigarh, Haryana, Delhi, Rajasthan, Uttar Pradesh, Uttarakhand, West Bengal), with 924,706 recorded stop observations.

**Where is my train?** For a planned journey, RailScan also shows on a map where the train **should be now**, estimated from the timetable and the expected delays. It is labelled as an estimate: there is no live feed.

The **Journey planner** works for **any date** (today, a future date or a past one). Choose a train, the travel date and the station the train is at. If you enter its current delay, a trained model estimates the delay at the next stop, and RailScan gives a validated estimate with a likely range for every later stop. If you don't know the delay, it shows how late the train usually was at each stop.

> **No live data.** RailScan has no access to live running status. Estimates come from the delay you enter, the train's timetable, and patterns in recorded journeys (11 Aug – 1 Sep 2026). They are estimates, not official information.

---

## Contents
1. [Problem statement](#1-problem-statement)
2. [Features](#2-features)
3. [Architecture](#3-architecture)
4. [Datasets and licenses](#4-datasets-and-licenses)
5. [Data preparation](#5-data-preparation)
6. [CNN script classifier](#6-cnn-script-classifier)
7. [OCR and railway matching](#7-ocr-and-railway-matching)
8. [Delay prediction](#8-delay-prediction)
9. [Feature engineering and leakage prevention](#9-feature-engineering-and-leakage-prevention)
10. [Evaluation results](#10-evaluation-results)
11. [Installation (Windows PowerShell)](#11-installation-windows-powershell)
12. [Running the app](#12-running-the-app)
13. [Retraining](#13-retraining)
14. [Tests](#14-tests)
15. [Deployment (Streamlit Community Cloud)](#15-deployment-streamlit-community-cloud)
16. [Limitations](#16-limitations)
17. [Future improvements](#17-future-improvements)
18. [Project layout](#18-project-layout)

---

## 1. Problem statement
Station boards in India are often written in a regional script that a traveller may not read. Delay information is spread across several services. RailScan asks two separate questions, answered by separate models:

- **What does this sign say, and which station or train does it refer to?** Computer vision answers this: a CNN detects the script, OCR reads the text, and fuzzy matching finds the station.
- **Given where the train is now and how late it is, how late will it be at the next station?** A tabular ML model trained on recorded stop-level running data answers this.

## 2. Features
| Feature | Status |
|---|---|
| Image upload with original and processed (CNN input) preview | Working |
| CNN script classification with per-class probabilities | Working (accuracy in §10) |
| OCR text extraction, raw and cleaned output, confidence | Working (EasyOCR; accuracy in §7) |
| Train-number detection, validated against the dataset | Working; numbers not in the dataset are shown but not used |
| Station matching (exact, alias, fuzzy, transliteration) with match score | Working; DataMeet station list plus newer/renamed codes (§4) |
| Platform-number extraction ("PF 3", "Platform No. 2") | Working; needs the word *platform* or *PF*; shown as text on the sign, never as an assignment |
| Photo check with retake advice | Working; based on what was measured to matter (§7) |
| Station candidates with scores, search and user confirmation | Working |
| Station information and scheduled calls from the timetable, link to the planner | Working; scheduled data only, no live status |
| Accounts (sign up / log in) and per-user history | Working; SQLite locally, PostgreSQL when configured |
| Journey planner for **any date**: train, date, current station, current delay | Working |
| Next-stop delay estimate, risk label, P(delay of 15 min or more), likely range | Working (accuracy in §10) |
| Every later stop: estimated delay, likely range, expected arrival, time after departure / from now | Working; validated method and ranges (§10) |
| Typical recorded delays per stop when the current delay is unknown | Working; history, labelled as such |
| Check the model against a recorded journey (prediction vs what happened) | Working |
| Scheduled halt and stop category | Working |
| Historical stop statistics (average, median, std, reliability) | Working; recorded data only, past observations only |
| Dataset and model-evaluation page | Working |
| Estimated current position on a map (timetable + expected delays) | Working; labelled "estimate, not live" (`src/position.py`) |
| Live running status | **Not available.** Indian Railways offers no public live API; no unofficial source is scraped |

## 3. Architecture
```
                 ┌──────────── Scan a sign ────────────┐
 image upload ─► src/image (load, letterbox) ─► CNN (models/language_classifier.keras)
                         │                         │ script
                         └─► src/ocr (EasyOCR, per-script languages) ─► text
                                       │
                                       ├─► train numbers (5 digits) ─► checked against dataset
                                       ├─► src/railway StationMatcher ─► station code + score
                                       └─► platform regex
                 ┌────────── Journey planner ──────────┐
 train/any date/station/delay ─► src/planner.forecast
                         ├─ src/timetable  (current timetable placed on the chosen date)
                         ├─ past-only history from data/processed/RailScan_Master_Dataset.csv
                         ├─ src/delay_prediction (models/delay/*.pkl, same make_features as training)
                         ├─ later stops: method + likely range from reports/planner_evaluation.json
                         └─ src/next_station, src/stop_analysis (arrival times, halts, stop statistics)
 no delay known ─► src/planner.typical_delays (recorded history per stop)
 recorded-journey check ─► src/railscan_service.analyze
 app.py: presentation only (Streamlit)
```
Each kind of value is labelled in the UI: **recorded**, **scheduled**, **model estimate**, **estimate**, or **your input**.

## 4. Datasets and licenses
Full details are in [`data/processed/dataset_sources.md`](data/processed/dataset_sources.md).

| Purpose | Dataset | License |
|---|---|---|
| Delay models, train lookup, stop statistics | [yadavsaurabh/indian-railways-runs](https://huggingface.co/datasets/yadavsaurabh/indian-railways-runs): stop-level scheduled and actual times | GPL-3.0 |
| CNN training, OCR evaluation | [jai23023/indicstr12-signboards](https://huggingface.co/datasets/jai23023/indicstr12-signboards): 975 real Indic signboard photos, a subset of INDICSTR12 | CC BY 4.0 (cite INDICSTR12) |
| Station names and codes | [DataMeet railways](https://github.com/datameet/railways) `stations.json` | CC0 |
| Station states (which stations are in northern India) | [geoBoundaries IND ADM1](https://www.geoboundaries.org/) (from DataMeet maps) | CC BY 2.5 IN |
| Names of newer / renamed station codes (e.g. PRYJ, VGLJ, DDU) | English Wikipedia station articles, accepted only when the infobox states that exact code (`scripts/build_station_supplement.py`) | CC BY-SA 4.0 |

Which dataset supports which component:

| Component | Source |
|---|---|
| CNN script classification | INDICSTR12 signboards, `language` label mapped to script |
| OCR evaluation | INDICSTR12 transcriptions (see §7) |
| Timetable for any date, scheduled halt | Running records: the most common scheduled times over the last 60 recorded days (`scripts/build_timetable.py`) plus the DataMeet station master |
| Current delay, delay risk, next-station delay | Running records: actual vs scheduled arrival and departure |
| Actual stop duration and historical reliability | Running records: actual departure minus actual arrival, repeated over many journeys |

Sources considered but not used:
- **prasenjit-27/Indian-Railway-Data** and **DataMeet schedules**: timetables only, with no recorded actual times.
- **meshramanirudh/indian-railways-delay-analysis**: journey-level only, so it cannot support station-level prediction or stop statistics.

`sample_images/station_sign.jpg` (a New Delhi station board) was supplied by an earlier project stage. Its original source was not recorded, so it is used only as a local demo image.

## 5. Data preparation
```
12,935 train files (34.7 GB, Hugging Face)
  ──select_northern_trains.py──► 5,331 trains with a halt in a covered state (end of each file read only)
  ──download_northern.py --days 60──► data/raw/delay_north/ (5,300 still-running trains, 4,831,683 rows)
  ──build_northern_dataset.py --max-rows 950000──► data/processed/
        RailScan_Final_Dataset.csv     924,706 rows x 40 cols (model-ready)   + .parquet copy (45 MB, committed)
        RailScan_Master_Dataset.csv    924,706 rows x 50 cols (+ names, derived times, stop category)
        train / validation / test.csv  655,243 / 131,475 / 137,988 (chronological)
        dataset_summary.json, dataset_manifest.csv, filtering_report.csv
  ──build_timetable.py──► RailScan_Timetable.csv (5,300 trains, 113,498 stops, full routes, stop coordinates)
```
- **Which trains.** The source has one file per train. Only the end of each file was read (its latest journeys), to learn the current route.
  - Every station was placed in a state, using the DataMeet state where given, otherwise its coordinates inside the official state boundaries.
  - Codes missing from the 2016 station list (PRYJ, VGLJ, AYC, ...) were placed by the coordinates in the running records themselves.
  - A train is northern if it has a scheduled halt in northern India.
- **One row** = a train observed at stop *S* on one journey, paired with its arrival at the next stop *N*. Only rows whose **current** stop is in a covered state are kept; the next stop may lie across the border.
- **Next stop.** The source lists only the stops where a train halts, and its `seq` numbers positions along the line, so consecutive stops can be numbered 10 → 13. The next stop is the train's next **regular** stop (a seq present on at least half of its journeys). If that stop is missing from a day's record, the row is dropped.
- **Under a million rows.** The 60 downloaded days gave 2,572,458 observations; West Bengal's suburban trains run many times a day. The oldest journey dates were dropped (whole dates) to fit 950,000 rows, which leaves **924,706 observations from 5,068 trains at 3,267 stations, journeys 11 Aug – 1 Sep 2026** (three weeks).
- **History** (`historical_*`) is computed inside these rows with strictly past-only rules.
- **Midnight and date handling.** Scheduled and actual datetimes carry their own calendar dates from the source. All times are naive IST (Asia/Kolkata).
- **Derived values:** `scheduled_halt_min = sch_departure − sch_arrival`, `actual_stop_min = act_departure − act_arrival`, and `delay = actual − scheduled`. A consistency check recomputes each of these from the timestamps and finds 0 mismatches.
  - 58 source rows whose own delay field contradicted their own timestamps were dropped, not corrected.
- **Validation:** 0 duplicate keys, 0 negative durations, every row has a target, and the history features are reproduced by an independent brute-force audit. Every dropped row is counted by reason in `filtering_report.csv`.
- **Timetable.** For each train, the most common scheduled times over the 60 days, from journeys on the train's **latest** route. Some specials changed route, and some stations were renamed during the window; the latest code is kept.
- **Targets:**
  - `next_station_arrival_delay_min` (regression)
  - `next_station_delayed_15` = 1 if the delay is **15 min or more**
  - `next_delay_risk_label`: Low < 15, Medium 15–59, High ≥ 60 min
  - Class balance: 400,181 delayed vs 524,525 not delayed.
- **Stop category** uses the **scheduled** halt: Short < 5 min, Moderate 5–10 min (inclusive), Long > 10 min.
- **Stop reliability** = 100 × (earlier observed stops of this train at this station with |actual stop − scheduled halt| ≤ 2 min) / (earlier observed stops). It counts only stops completed strictly before the current observation.

## 6. CNN script classifier
- **Task:** classify the script of a sign photo. The CNN does not read text; OCR does that.
- **Classes:** Devanagari (Hindi + Marathi), Bengali, Kannada, Telugu.
- **Data:** 1,310 deduplicated IndicSTR12 images (347 of them Bengali), split 916 / 197 / 197 (train / validation / test), stratified by language.
- **Imbalance:** class weights computed from the training labels (Devanagari 260, Kannada 205, Telugu 208).
- **Preprocessing** (`src/image/preprocessing.py`):
  - safe decoding, with EXIF rotation applied, into RGB
  - letterbox to a square (aspect ratio kept, grey padding)
  - scaling to [0, 1]
  - no thresholding or blur
- **Augmentation** happens on the fly, only during training: small rotation, zoom, shift and contrast changes. There are no flips, because mirrored text is not real text. No augmented copy can reach validation or test.
- **Training:** Adam, sparse categorical cross-entropy, early stopping on validation loss, and checkpointing of the best epoch.

**Architectures tried**, in order. The final model was selected on **validation** accuracy.

| Model | Input | Val acc | **Test acc** | Test macro-F1 | Notes |
|---|---|---|---|---|---|
| Small CNN: Conv32-64-128 → GAP → Dense128 → Dropout 0.3 → softmax | 128 | 0.428 | 0.455 | 0.387 | Underfits (train acc ≈ 0.48); never predicts Devanagari |
| MobileNetV2 (ImageNet, frozen) + same head, then top 30 layers fine-tuned | 128 | 0.572 | 0.628 | 0.596 | |
| **MobileNetV2, as above (used by the app)** | **224** | **0.607** | **0.586** | **0.582** | Highest validation accuracy |
| 4-language experiment (Hindi/Kannada/Marathi/Telugu), MobileNetV2 | 224 | 0.531 | 0.566 | 0.551 | Hindi↔Marathi confusions: 8 of 145 |

App model, test-set results (145 images):

| Script | Precision | Recall | F1 | Images |
|---|---|---|---|---|
| Devanagari | 0.60 | 0.61 | 0.60 | 56 |
| Kannada | 0.50 | 0.45 | 0.48 | 44 |
| Telugu | 0.65 | 0.69 | 0.67 | 45 |

**Four scripts (Bengali added, used by the app).** The same MobileNetV2 network retrained on 1,310 images: test accuracy **49.2%** on four classes (chance 25%; Bengali recall 31%). The previous three-script model (58.6% test) is kept in `data/interim/backup_1500_pune_kolhapur/`.

**Image model + OCR evidence (used by the app).** `src/script_evidence.py`, evaluated with `scripts/evaluate_script_fusion.py`:
- The sign is read once with each script's EasyOCR recogniser (now including Bengali). A script's evidence = OCR confidence × the share of the recognised letters that belong to that script's Unicode block.
- The evidence is combined with the CNN probabilities: `softmax(log P_cnn + w × evidence)`.
- The weight `w` (60, the largest value in the grid) and the "confident" threshold (0.9) were chosen on the 197 validation images only.

| On the 197 test images | Image model alone | Image model + OCR evidence |
|---|---|---|
| Accuracy | 49.2% | **79.2%** |
| Share marked confident | – | 72% |
| Accuracy when confident | – | **92.9%** |
| Accuracy when not confident | – | 45% |

- A scan reads the photo with four recognisers, so it takes about 30 s on CPU.

**Interpretation of the image model alone.** About 59% test accuracy on three classes (chance is 33%) is modest.
- The images are whole scenes in which the text is often a small region, and many signs also carry English.
- In the 4-language experiment, the main errors were Kannada images classified as Hindi, not Hindi↔Marathi.
- The bottleneck is therefore locating the text, not similar scripts. Cropping text regions before classification is the obvious next step (§17).
- The 128-px model scored higher on test (0.628) but lower on validation, so it was not selected. With 145 test images, one standard error is about ±4 points, so the two are statistically close.
- Because the CNN is unreliable, the app shows the probabilities for every script and warns when confidence is below 0.60.

Outputs:
- `models/language_classifier.keras`
- `models/language_labels.json`
- `reports/cnn_evaluation.json` (the app model)
- `reports/cnn_evaluation_{small_cnn,mobilenetv2_128,mobilenetv2_224,4class}.json`
- `reports/figures/cnn_*` (confusion matrices, loss and accuracy curves)

## 7. OCR and railway matching
- **Engine.** `src/ocr/ocr_engine.py` uses **EasyOCR** (PyTorch, CPU), installed by `requirements.txt`. No separate OCR program is needed. Its models download once, to `~/.EasyOCR/model`, the first time each script is read.
- **Language selection.** The CNN's script picks the OCR languages: Devanagari → `hi+mr+en`, Kannada → `kn+en`, Telugu → `te+en`. EasyOCR cannot load Kannada or Telugu together with Devanagari, so each combination gets its own cached reader.
- **Preprocessing.** OCR uses its own preprocessing (`src/ocr/preprocessing.py`). It chooses strategies per image, such as contrast or upscaling, and applies thresholding or deskewing only when the image analysis calls for it. CNN preprocessing is separate and never thresholds or blurs.
- **Output.** The raw text is preserved for debugging, and a cleaned copy is used for matching.
- **Train numbers.** Five-digit sequences count as train numbers only if they exist in the dataset. Other numbers are shown but ignored.
- **Station matching** tries, in order: exact code, exact name, aliases in any script, normalised name, RapidFuzz similarity, then transliteration.
  - A score of 90 or more is accepted automatically. Scores of 75–89 need confirmation.
  - If two stations score about the same, RailScan refuses to pick one.
  - You can always choose the train and station yourself in the Journey planner.

**OCR evaluation** (`scripts/evaluate_ocr.py`): EasyOCR 1.7.2 on CPU, run on the 145 held-out test images. The true language was supplied, so this measures text reading alone. The run took 605 s, about 4 s per image.

| Language | Images | No usable text | Word recall (all) | Word recall (when text read) | Median CER (when text read) |
|---|---|---|---|---|---|
| Hindi | 26 | 9 | 0.42 | 0.65 | 0.89 |
| Marathi | 30 | 14 | 0.34 | 0.64 | 0.83 |
| Kannada | 44 | 17 | 0.18 | 0.29 | 1.10 |
| Telugu | 45 | 12 | 0.28 | 0.38 | 0.75 |
| **All** | **145** | **52** | **0.29** | **0.45** | **0.87** |

- *Word recall* = share of ground-truth words that appear exactly in the OCR output.
- *CER* = character edit distance divided by reference length. It can exceed 1 because the OCR also reads English text and numbers that the INDICSTR12 transcriptions leave out. A failed read counts as CER 1.

**Interpretation.**
- OCR on cluttered scene photos is the weakest part of RailScan.
- On a clear station board it works: the bundled New Delhi board was read as "नई दिल्ली / NEW DELHI" and matched to NDLS with a score of 97.
- This is why the app never requires a successful scan: you can always choose the train and station manually.

**Photo-to-railway assistant** (Scan page, `src/railway_assistant.py`, `src/image/quality.py`):
- **Photo check.** Retake advice comes mainly from what OCR and matching produced: no text, OCR confidence ≤ 50, or no station matched. I measured what predicts OCR failure on the 145 test photos:
  - blur, brightness and contrast did **not** predict failure, so they are flagged only at extremes beyond anything in the set;
  - images with a short side under about 300 px had 15% word recall, versus 34–39% for larger ones;
  - OCR confidence ≤ 50 went with 39% word recall, versus 57% above 70.
- **Station.** The ranked candidates are shown with their scores, and there is a search box to correct the match. A station is preselected only when the matcher accepted it with high confidence; otherwise you choose. Your choice is saved to your history.
- **Railway information.** Shows the station code and name (from the SQLite station table) and the scheduled calls of the trains in RailScan's timetable, with a button to plan a journey from there.
  - Platform numbers are shown as text printed on the sign.
  - RailScan never infers a platform assignment, train position or running status from a photo.


## 8. Delay prediction
- **Two tasks:**
  - **Regression**: next-station arrival delay in minutes.
  - **Classification**: delayed by 15 min or more at the next station.
- **Candidates:**
  - Linear or Logistic Regression
  - Random Forest (100 trees, each on a 10% bootstrap sample of the 651k training rows, min. 10 rows per leaf, to keep training practical)
  - XGBoost (class weights or `scale_pos_weight` from the training labels only)
- **Selection** happens on the **validation** split: lowest MAE for regression, highest macro-F1 for classification. The test split is used once, by `evaluate_models.py`.
- **Saved models** are full scikit-learn pipelines (imputer + scaler + one-hot encoder + model) in `models/delay/`. The app builds its inputs with the same `make_features` function used in training, so the preprocessing cannot drift. A test confirms that the app's predictions equal the training pipeline's.
- **Any date.** The planner places the current timetable on the chosen date, including day offsets for stops after midnight. The model input is built from:
  - the delay you entered;
  - the calendar of the **journey start date** (as in training);
  - history from recorded journeys strictly before the prediction moment.

  On recorded rows the planner reproduces the training data's history features exactly (`tests/test_planner.py`).
- **Expected arrival** = scheduled arrival + estimated delay. Times are shown after departure, and **from now** when the journey is within a day.
- **Model coverage.** The model is used only on route segments (train, station, next station) that appear in its training rows (35,593 segments, `models/delay/coverage.csv`). Elsewhere, for example south of the northern states, the delay is carried forward and the row is labelled.
- **Later stops.** Four methods were compared on recorded journeys after the training period (`scripts/evaluate_planner.py`): the model applied stop by stop, its next-stop estimate carried forward, the current delay carried forward, and the current delay plus the typical recorded change. The method for stops beyond the next one was chosen on the validation period. Carrying the current delay forward won, so that is what the app shows. Each estimate has a likely range: the 10th–90th percentile of past errors at that distance.
- **Typical delays.** Without a current delay, no prediction is made. The planner shows the recorded median delay per stop, the median for the same weekday, and the share of arrivals 15+ minutes late.

## 9. Feature engineering and leakage prevention
**Features** (all known at the moment the train is at the current station):
- current arrival and departure delay
- scheduled hour and minute
- scheduled halt
- scheduled running time and distance to the next stop
- day of week, month, weekend flag
- station position (`seq`)
- train number, train type, current and next station code
- past-only history:
  - average, std and count of earlier delays at the next station
  - on-time rate
  - average and std of earlier actual stops
  - stop reliability

A departure that has not been recorded is filled with the arrival delay, the latest observed delay. It is not filled with the median, because that turned a 294-minute arrival delay into about 30 minutes. That bug was found while inspecting test errors; §10 reports the numbers before and after the fix.

**Leakage checks** run automatically, and training stops if any fails:
- No target, `target_*`, next-station actual time or "future" column may be a feature (regex audit in `scripts/delay_common.py`).
- `observation_time ≤ next actual arrival` in every row.
- History counts a past row only if its outcome was known **strictly before** the current observation time. This excludes overlapping multi-day runs of the same train. The builder checks this against an independent brute-force recomputation, and `tests/test_delay_prediction.py` checks it again.
- The split is chronological and cut at journey-date boundaries, so no date is shared between splits.

## 10. Evaluation results
All numbers below were measured by the training scripts on held-out data. Models were chosen on the validation days (26–28 Aug 2026, 131,475 rows) and measured **once** on the test days (29 Aug – 1 Sep 2026, 137,988 rows from 4,279 trains).

**Next-station delay (regression), test set**

| Model | MAE (min) | RMSE (min) | R² |
|---|---|---|---|
| **Random Forest** (selected on validation: MAE 3.25 vs Linear 3.35, XGBoost 4.08) | **2.99** | **7.19** | 0.969 |
| Persistence baseline (next delay = current delay) | 4.38 | 9.60 | 0.945 |

**Delayed by 15 min or more (classification), test set**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **XGBoost** (selected on validation: macro-F1 0.945 vs Logistic 0.936, Random Forest 0.918) | **0.948** | 0.940 | 0.935 | **0.937** | 0.988 |
| Persistence baseline | 0.921 | 0.887 | 0.929 | 0.907 | – |

Confusion matrix (test): TN 76,783, FP 3,448, FN 3,771, TP 53,986. Risk label (Low/Medium/High) derived from the regressor: accuracy 0.940, macro-F1 0.937.

**How to read these numbers.**
- Delays persist from one stop to the next, so even the baseline scores high (R² 0.945).
- The model cuts the average error by 32%, from 4.38 to 2.99 minutes, across 4,279 trains.
- 42% of test rows are "delayed". Macro-F1 is 0.946 for the model vs 0.919 for the baseline.
- The most important input by far is the current delay; next come the scheduled running time and distance to the next stop, then this train's past delays at that stop (`reports/results/feature_importance.csv`).
- The test days directly follow the training weeks. Accuracy further in the future, or in other seasons, is not measured.
- Earlier versions (northern India only: test MAE 4.18 vs 6.47; Pune–Kolhapur, 1,500 rows: 3.45 vs 5.95) are not comparable: different trains, period and size.

**Journey planner, measured on recorded journeys** (`reports/planner_evaluation.json`). The evaluation replayed 1,000 journeys after the training period (11,683 comparisons), using the planner exactly as the app does. The method for later stops was chosen on the validation days (26–28 Aug) and is reported on the test days (29 Aug – 1 Sep):

| Stops ahead | Method shown in the app | Test MAE (min) | Carry-forward MAE | Likely range (actual − estimate) | Test values inside the range |
|---|---|---|---|---|---|
| 1 | model | **4.07** | 4.93 | −5.7 to +5.5 | 83% |
| 2–3 | current delay + typical recorded change | **5.44** | 7.39 | −6 to +10 | 81% |
| 4–6 | current delay + typical recorded change | **8.33** | 11.04 | −7 to +14 | 77% |
| 7–10 | current delay + typical recorded change | **9.14** | 13.57 | −10 to +19 | 81% |
| 11+ | current delay + typical recorded change | **12.31** | 18.76 | −14 to +34 | 83% |

- "Typical recorded change" = the median change of this train's delay between the starting station and that stop, over its earlier journeys only. Where the train has no such history, the delay is carried forward (shown in the *Basis* column).
- Applying the next-stop model stop by stop compounds its small errors and was not chosen.
- The ranges aim for 80% coverage and achieved 77–83% on test.

Validation results for every candidate, feature importances and plots are in `reports/results/` and `models/delay/metadata.json`.

CNN and OCR results are in §6 and §7.

## 11. Installation (Windows PowerShell)
Tested with **Python 3.11.9** on Windows 11, CPU only.
```powershell
cd C:\path\to\railscan
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt       # the app: EasyOCR + Keras on CPU-only PyTorch
pip install -r requirements-dev.txt   # also TensorFlow (only to retrain the CNN) and test tools
```
**OCR** needs nothing beyond `requirements.txt`. The file points pip at PyTorch's CPU-only index, so the multi-GB CUDA build is never downloaded. Check the OCR setup with `python scripts/test_ocr.py --check`.

The trained models and processed CSVs are already in the repository, so the app runs without downloading anything else. EasyOCR downloads its models (about 330 MB in total for all four scripts) the first time each script is read.

## 12. Running the app
```powershell
.\.venv\Scripts\Activate.ps1
python scripts/setup_database.py all      # once: builds data/database/railscan.db (railway data, timetable, account tables)
streamlit run app.py
```
Open http://localhost:8501. If that port is busy, add `--server.port 8502`. After retraining or rebuilding data, restart the app (Ctrl + C, then run it again): models and data are cached while it runs.

### Accounts and history
The app opens on a **Log in / Create account** screen.
- **Usernames** are 3–30 characters (letters, numbers, `.`, `_`, `-`) and are not case-sensitive.
- **Passwords** need at least 8 characters, including a letter and a number.

After signing in:
- every **Scan image** and every **Show estimate** / **Show typical delays** is saved to your history;
- **My history** lets you view each entry, delete it or clear everything, change your password, or delete your account (password required).

Accounts live in the same SQLite file as the railway data (`data/database/railscan.db`; set `RAILSCAN_DB_PATH` to use another file). Alternatively, set `RAILSCAN_ACCOUNTS_DB_URL` to keep them in an external **PostgreSQL** database ([docs/HOSTING.md](docs/HOSTING.md)). They use three tables:

| Table | Contents |
|---|---|
| `users` | username, display name, salted **scrypt** password hash (the password itself is never stored), creation and last-login time |
| `user_history` | one row per saved result: time, type (`scan`, `forecast`, `typical`), a one-line summary and JSON details. Deleted with the account |
| `login_attempts` | time and outcome of each login. After 5 failures within 15 minutes, the username is locked for the rest of that window |

**Privacy and limits.**
- History stores the results RailScan showed, not your photos.
- Login messages never reveal whether a username exists.
- The database file is git-ignored, so accounts stay on the machine running the app.
- Signing in lasts for the browser session: refreshing the page asks you to log in again.
- On hosts without persistent disk (such as Streamlit Community Cloud), use the PostgreSQL option. Otherwise accounts are lost when the app restarts.

## 13. Retraining
```powershell
python scripts/download_datasets.py --skip-delay   # signboard images for the CNN (without the flag: also the old 2-train files)
python scripts/add_bengali_signboards.py      # 347 Bengali signboard photos (IndicSTR12), appended to the CNN metadata
python scripts/select_northern_trains.py      # which trains halt in a covered state (~30-60 min, reads file ends only)
python scripts/download_northern.py --days 60 # their last 60 days (~45 min, ~860 MB)
python scripts/build_northern_dataset.py --max-rows 950000   # dataset + splits + parquet (~20 min)
python scripts/build_timetable.py --window-days 60            # current timetable for the planner (~4 min)
python scripts/build_station_supplement.py   # names of newer / renamed station codes (Wikipedia, verified)
python scripts/train_delay_models.py         # selection on validation (~15 min)
python scripts/evaluate_models.py            # one-time test evaluation
python scripts/evaluate_planner.py --max-journeys 1000   # later-stop method + likely ranges
python scripts/setup_database.py all         # then: import-data --replace, import-timetable (keeps accounts)
python scripts/train_cnn.py --arch mobilenetv2 --img-size 224 --epochs 40   # CNN app model, 4 scripts (~5 min CPU)
python scripts/evaluate_script_fusion.py --reuse-ocr <earlier rows CSV>   # CNN + OCR evidence; reuses OCR readings
python scripts/evaluate_ocr.py               # OCR on held-out signboards
```
Hugging Face limits anonymous downloads to 3,000 requests per 5 minutes; the scripts wait when asked to and can be re-run to resume. The previous Pune–Kolhapur pipeline (`build_dataset.py --local-dir data/raw/delay`, `prepare_dataset.py`) still works and its outputs are kept in `data/interim/backup_1500_pune_kolhapur/`.

## 14. Tests
```powershell
python -m pytest tests -q
```
Last run (2026-10-30, Windows 11, Python 3.11.9): **349 passed, 6 skipped**.
- Real EasyOCR runs on rendered English test images, so OCR is exercised for real, not only mocked.
- The 6 skipped tests draw Hindi, Kannada and Telugu test images, and need a Pillow build with **libraqm** to shape Indic text. The Windows Pillow wheel on the test machine lacks it.

| File | Covers |
|---|---|
| `test_preprocessing.py` | CNN input shape and range, aspect ratio, RGB/BGR order, invalid and tiny uploads, upload bytes in OCR, platform regex, train numbers checked against the dataset |
| `test_language_classifier.py` | saved model vs label file, predictions on real held-out images, disjoint CNN splits, EasyOCR line grouping and empty or missing backend (stub reader) |
| `test_ocr.py` | OCR language mapping, strategies, text cleaning, train-number extraction, failure handling (stubbed EasyOCR), real EasyOCR on rendered text |
| `test_station_matching.py`, `test_train_state.py` | exact, alias, fuzzy and ambiguous station matching; unknown train or station; train-state lookup |
| `test_stop_analysis.py` | scheduled halt, actual stop, delay, midnight crossings, stop categories, past-only stop statistics, missing data |
| `test_planner.py` | timetable on any date (overnight day offsets, origin and destination), history parity with training, calendar features from the journey date, future-date forecasts, labelled uncovered segments, unknown train or station, last stop, typical delays, builder next-regular-stop rule |
| `test_next_station.py` | next-stop selection, repeated stations, expected arrival across midnight, negative time remaining |
| `test_delay_prediction.py` | leakage regex audit, saved vs code feature list, real dataset under 1 million rows, target definition, brute-force past-only history, chronological split, app vs training predictions identical, unknown categories reported |
| `test_users.py` | sign-up validation, scrypt hashing (no plain-text passwords), case-insensitive unique usernames, identical errors for wrong password and unknown user, lockout and reset, SQL-injection strings, private per-user history, password change, account deletion with cascade, upgrade of a v1 database |
| `test_database.py` | SQLite import, NULL handling, rollback, SQL injection, SQLite-vs-CSV parity |

## 15. Deployment (external database and hosting)
Full step-by-step guide: **[docs/HOSTING.md](docs/HOSTING.md)**. In short:
1. **Database.** Create a PostgreSQL database (for example Supabase or Neon) and copy its connection string (`postgresql://...?sslmode=require`). RailScan creates its account tables itself.
2. **Railway data.** Stays in SQLite where it is built (`python scripts/setup_database.py all`, which also exports `data/processed/dataset_overview.json`). A new server does not build it: each train is imported from the committed `RailScan_Final_Dataset.parquet` (45 MB) when first needed (about 0.4 s, same rows), and the Data page uses the exported overview. After changing the dataset, run `setup_database.py all` and commit the new overview file (a test checks that it matches the Parquet file). The large CSVs are git-ignored (above GitHub's 100 MB file limit).
3. **Local test.** Run `$env:RAILSCAN_ACCOUNTS_DB_URL = "postgresql://..."` and then `streamlit run app.py`. My history should say "stored in PostgreSQL at …".
4. **GitHub.** Push the repository. `.gitignore` keeps the SQLite file, large CSVs, raw data and secrets out.
5. **Host.**
   - Streamlit Community Cloud: put `RAILSCAN_ACCOUNTS_DB_URL = "..."` under *Secrets* and choose Python 3.11.
   - Any other Python host: run `pip install -r requirements.txt`, then `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`, with the same environment variable.
   - Plan for about **2 GB of RAM** and **1 GB of disk** for the database (590 MB) and models.

**Tested:** the PostgreSQL account path against a real local PostgreSQL 16, and a database build from the Parquet file that is identical row by row to the one built from the CSV.
**Not tested:** a hosted provider or an actual Streamlit Cloud deployment.

## 16. Limitations
- **Northern India and West Bengal, three recent weeks.** Observations cover 5,068 trains at 3,267 stations in J&K, Ladakh, Himachal Pradesh, Punjab, Chandigarh, Haryana, Delhi, Rajasthan, Uttar Pradesh, Uttarakhand and West Bengal, with journeys from 11 Aug to 1 Sep 2026. Keeping under a million rows with West Bengal's suburban trains shortened the period.
  - Seasonal effects such as winter fog are not in the data, so estimates may be too optimistic then.
  - Stations outside the north appear only in timetables; there the delay is carried forward and labelled.
- **No live data.** Indian Railways offers no public live-status API. For a real journey you must enter the current delay. The map position is an estimate from the timetable and that delay, placed on a straight line between stations. The planner assumes the latest recorded timetable (Jul–Sep 2026) still applies on the date you choose; it cannot know about later timetable changes or cancellations.
- **Estimates further ahead are rough.** Beyond the next stop the typical error grows from about 5 min (2–3 stops) to 12 min (11+ stops); each estimate has a likely range (§10).
- **Delay convention of the source.** The source reports 0 for on-time and early arrivals; where its delay field is blank, the delay is computed from timestamps and can be negative. The data mixes both.
- **Station names.** Some newer station codes have no name in any source used; the app then shows the code. About 300 halt stations have no coordinates and no state.
- **Signboard images are not railway-specific.** INDICSTR12 contains general shop and street signs. Accuracy on railway boards, which are often bilingual with English, is not measured.
- **Scripts.** The CNN knows Devanagari, Bengali, Kannada and Telugu. Signs in Gurmukhi (Punjabi) or Urdu script are not recognised; Hindi, Bengali and English, the most common scripts on boards in the covered states, are. Bengali station matching uses curated Bengali spellings for 17 major West Bengal stations plus transliteration.
- **Modest OCR accuracy, slow scans.** Script identification reaches 82.8% when the image model is combined with OCR evidence (58.6% for the image model alone), and OCR finds 29% of reference words on scene photos (§6–7). A scan takes about 20–30 s on CPU. The planner never depends on a scan.
- **Derived actual times.** Actual times in the processed data are scheduled time + recorded delay.
- **Missing columns.** `journey_day`, `distance_from_source_km` and platform are not in the processed data. Platform numbers come only from OCR.

## 17. Future improvements
- Extend the period (more weeks, other seasons) or the region; the pipeline is parameterised by states, days and row budget.
- Classify the script on **detected text regions** (EasyOCR's CRAFT detector already finds them) instead of the whole photo. The CNN error analysis points to this as the main improvement.
- Add Gurmukhi and Urdu script classes (IndicSTR12 has both) and more Bengali station spellings.
- Collect labelled railway-board photos to measure, and fine-tune, the CNN and OCR on the real target domain.
- Train a dedicated model for stops further ahead (predicting the delay k stops ahead directly), and compare it with the current method using `scripts/evaluate_planner.py`.
- Use calibrated probabilities (Platt or isotonic) for the risk score.
- Connect a live source only if an authorised API becomes available.

## 18. Project layout
```
app.py                     Streamlit UI (presentation only)
src/image/                 CNN preprocessing + prediction
src/ocr/                   OCR engine (EasyOCR), text cleaning, train numbers
src/railway/               station master, matcher, train-state lookup
src/database/              SQLite layer: railway data, accounts and history (users.py) - docs/DATABASE.md
src/delay_prediction.py    loads delay pipelines, builds features with training code
src/next_station.py        next stop, expected arrival, time remaining
src/stop_analysis.py       halts, actual stops, categories, past-only stop statistics
src/timetable.py           current timetable placed on any date
src/planner.py             journey planner: next-stop model estimate, later stops, typical delays
src/railscan_service.py    recorded-journey check and dataset helpers used by the app
scripts/                   download, prepare, train, evaluate
data/processed/            master/final CSVs, splits, CNN metadata, summary, sources
models/                    language_classifier.keras, language_labels.json, delay/*.pkl
reports/                   cnn_evaluation*.json, ocr_evaluation.json, figures/, results/
tests/                     pytest suite
docs/                      DATABASE.md, PROJECT_REPORT.md
```

## Operations and design docs

- `CLAUDE.md` / `AGENTS.md`: project rules and commands for coding agents (and people)
- `docs/system-design.md`, `docs/architecture.md`, `docs/decisions/`: what RailScan is and why it is built this way
- `docs/runbooks/`: deploy, roll back, back up and restore accounts, handle an incident
- `docs/security/threat-model.md`, `docs/monitoring.md`
- CI (`.github/workflows/ci.yml`): lint, tests and a secret scan on every push; deploy only when it is green
