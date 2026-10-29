# RailScan

**CNN-Based Indian Railway Sign Recognition with Delay Risk and Next-Station Time Prediction**

RailScan is a Streamlit application for Indian railway passengers and a student ML project. Upload a photo of a station board or sign and RailScan:

1. identifies the script (Devanagari, Kannada or Telugu) with a CNN;
2. reads the text with OCR;
3. looks for a train number, station name and platform number, and checks them against reference data.

The **Journey planner** works for **any date** (today, a future date or a past one). Choose a train, the travel date and the station the train is at. If you enter its current delay, a trained model estimates the delay at the next stop, and RailScan gives a validated estimate with a likely range for every later stop. If you don't know the delay, it shows how late the train usually was at each stop.

> **No live data.** RailScan has no access to live running status. Estimates come from the delay you enter, the train's timetable, and patterns in recorded journeys (Nov 2023 – Aug 2026). They are estimates, not official information.

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
| Station matching (exact, alias, fuzzy, transliteration) with match score | Working; 8,965 stations |
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
| Live running status | **Not available.** No authorised live source is used |

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

Which dataset supports which component:

| Component | Source |
|---|---|
| CNN script classification | INDICSTR12 signboards, `language` label mapped to script |
| OCR evaluation | INDICSTR12 transcriptions (see §7) |
| Timetable for any date, scheduled halt | Running records: the most common scheduled times over the last 90 recorded days (`scripts/build_timetable.py`) plus the DataMeet station master |
| Current delay, delay risk, next-station delay | Running records: actual vs scheduled arrival and departure |
| Actual stop duration and historical reliability | Running records: actual departure minus actual arrival, repeated over many journeys |

Sources considered but not used:
- **prasenjit-27/Indian-Railway-Data** and **DataMeet schedules**: timetables only, with no recorded actual times.
- **meshramanirudh/indian-railways-delay-analysis**: journey-level only, so it cannot support station-level prediction or stop statistics.

`sample_images/station_sign.jpg` (a New Delhi station board) was supplied by an earlier project stage. Its original source was not recorded, so it is used only as a local demo image.

## 5. Data preparation
```
raw per-train CSVs ──build_dataset.py --local-dir──► 35,273 observations (data/interim/RailScan_Full_Dataset.csv)
                   ──prepare_dataset.py --target-rows 1500──► data/processed/
                       RailScan_Final_Dataset.csv   1,500 rows x 40 cols (model-ready)
                       RailScan_Master_Dataset.csv  1,500 rows x 50 cols (+ names, derived times, stop category)
                       train.csv / validation.csv / test.csv  1,055 / 223 / 222 (chronological)
                       dataset_summary.json, dataset_manifest.csv, filtering_report.csv
                   ──build_timetable.py──► RailScan_Timetable.csv (current timetable, 21-22 stops per train)
```
- **One row** = a train observed at stop *S* on one journey, paired with its arrival at the next stop *N*.
- **Next stop.** The source lists only the stops where a train halts, and its `seq` numbers positions along the line, so consecutive stops can be numbered 10 → 13. The next stop is the train's next **regular** stop (a seq present on at least half of its journeys). If that stop is missing from a day's record, the row is dropped (459 rows).
  - *Fix made in this version:* the original builder required `next seq = seq + 1`. That wrongly discarded about 18,000 valid rows and left 8 of the 22 stations without any training data. The rebuilt data covers all 22 stations; the earlier 11,697-row build is kept as `data/interim/RailScan_Full_Dataset_v1.csv`.
- **The 1,500-row reduction** keeps whole journeys only (85 of 1,960). They are chosen in a golden-ratio order over the date-sorted journeys, which spreads the sample evenly over Nov 2023 to Aug 2026 and over both trains. The remaining gap is then filled to reach exactly 1,500 rows.
- **History is recomputed** for every `historical_*` feature after the reduction, because the old values referred to journeys that are no longer in the sample.
- **Midnight and date handling.** Scheduled and actual datetimes carry their own calendar dates from the source. Where an explicit date was missing, the builder applied a midnight rollover (96 cases, logged). All times are naive IST (Asia/Kolkata).
- **Derived values:** `scheduled_halt_min = sch_departure − sch_arrival`, `actual_stop_min = act_departure − act_arrival`, and `delay = actual − scheduled`. A consistency check recomputes each of these from the timestamps and finds 0 mismatches.
- **Validation:** 0 duplicate keys, 0 negative durations, and every row has a target. All 1,500 rows match a raw source record: scheduled arrival agrees 100%, and arrival delay agrees within 1 minute in 100% of the 1,280 rows that have a source delay value.
- **Targets:**
  - `next_station_arrival_delay_min` (regression)
  - `next_station_delayed_15` = 1 if the delay is **15 min or more**
  - `next_delay_risk_label`: Low < 15, Medium 15–59, High ≥ 60 min
  - Class balance: 892 delayed vs 608 not delayed.
- **Stop category** uses the **scheduled** halt: Short < 5 min, Moderate 5–10 min (inclusive), Long > 10 min. In this data, 1,460 rows are Short and 40 are Moderate (the 5-minute Miraj halts).
- **Stop reliability** = 100 × (earlier observed stops of this train at this station with |actual stop − scheduled halt| ≤ 2 min) / (earlier observed stops). It counts only stops completed strictly before the current observation.

## 6. CNN script classifier
- **Task:** classify the script of a sign photo. The CNN does not read text; OCR does that.
- **Classes:** Devanagari (Hindi + Marathi), Kannada, Telugu.
- **Data:** 963 deduplicated INDICSTR12 images, split 673 / 145 / 145 (train / validation / test), stratified by language.
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

**Image model + OCR evidence (used by the app).** `src/script_evidence.py`, evaluated with `scripts/evaluate_script_fusion.py`:
- The sign is read once with each script's EasyOCR recogniser. A script's evidence = OCR confidence × the share of the recognised letters that belong to that script's Unicode block.
- The evidence is combined with the CNN probabilities: `softmax(log P_cnn + w × evidence)`.
- The weight `w` (35) and the "confident" threshold (0.7) were chosen on the 145 validation images only.

| On the 145 test images | Image model alone | Image model + OCR evidence |
|---|---|---|
| Accuracy | 58.6% | **82.8%** |
| Share marked confident | 58% | 83% |
| Accuracy when confident | – | **92.6%** |
| Accuracy when not confident | – | 33% |

- On the New Delhi board, the image model said Kannada (0.47), while the combination says Devanagari (0.999).
- The cost is speed: three OCR passes take about 20–30 s per photo on CPU.

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
  - Random Forest
  - XGBoost (class weights or `scale_pos_weight` from the training labels only)
- **Selection** happens on the **validation** split: lowest MAE for regression, highest macro-F1 for classification. The test split is used once, by `evaluate_models.py`.
- **Saved models** are full scikit-learn pipelines (imputer + scaler + one-hot encoder + model) in `models/delay/`. The app builds its inputs with the same `make_features` function used in training, so the preprocessing cannot drift. A test confirms that the app's predictions equal the training pipeline's.
- **Any date.** The planner places the current timetable on the chosen date, including day offsets for stops after midnight. The model input is built from:
  - the delay you entered;
  - the calendar of the **journey start date** (as in training);
  - history from recorded journeys strictly before the prediction moment.

  On all 1,500 recorded rows the planner reproduces the training data's eight history features exactly.
- **Expected arrival** = scheduled arrival + estimated delay. Times are shown after departure, and **from now** when the journey is within a day.
- **Model coverage.** The model is used only on the 37 of 41 route segments that appear in its training rows. On the other 4, the delay is carried forward and the row is labelled.
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
All numbers below were measured by the training scripts on held-out data. The test period is 2026-04-06 to 2026-08-28 (222 rows).

**Next-station delay (regression), test set**

| Model | MAE (min) | RMSE (min) | R² |
|---|---|---|---|
| **Linear Regression** (selected on validation) | **3.45** | **7.81** | 0.926 |
| Persistence baseline (next delay = current delay) | 5.95 | 10.90 | 0.856 |

**Delayed by 15 min or more (classification), test set**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** (selected on validation) | **0.932** | 0.975 | 0.908 | **0.940** | 0.982 |
| Persistence baseline | 0.896 | 0.896 | 0.931 | 0.913 | – |

Confusion matrix (test): TN 89, FP 3, FN 12, TP 118. Risk label (Low/Medium/High) derived from the regressor: accuracy 0.941, macro-F1 0.950.

**How to read these numbers.**
- Delays persist from one stop to the next, so even the baseline scores fairly high (R² 0.856).
- The model cuts the average error by 42%, from 5.95 to 3.45 minutes.
- 59% of test rows are "delayed". Macro-F1 is 0.931 for the model vs 0.892 for the baseline.
- **Earlier versions.**
  - An earlier dataset build skipped every segment whose stops were not numbered consecutively (§5). On that easier subset the model scored MAE 2.09 vs a baseline of 3.68. Those numbers are not comparable with the table above.
  - A missing departure delay used to be filled with the median. Fixing that changed test RMSE from 13.31 to 2.96 on the old data, while validation MAE stayed at 2.40 → 2.39.

**Journey planner, measured on recorded journeys** (`reports/planner_evaluation.json`). The evaluation used 400 journeys after the training period (7,078 comparisons). The method was chosen on 27 Oct 2025 – 20 Mar 2026 and is reported here on the test period after 20 Mar 2026:

| Stops ahead | Method shown in the app | Test MAE (min) | Carry-forward MAE | Likely range (actual − estimate) | Test values inside the range |
|---|---|---|---|---|---|
| 1 | model | **2.66** | 3.25 | −2.1 to +4.0 | 66% |
| 2–3 | current delay carried forward | 5.38 | 5.38 | −7 to +4 | 72% |
| 4–6 | current delay carried forward | 9.31 | 9.31 | −7 to +15 | 80% |
| 7–10 | current delay carried forward | 20.27 | 20.27 | +5 to +28 | 74% |
| 11+ | current delay carried forward | 16.50 | 16.50 | −9 to +31 | 82% |

- Applying the model stop by stop was **worse** than the baseline beyond the next stop (for example, 16.5 vs 9.3 MAE at 4–6 stops), because small errors compound. The app therefore does not do it.
- "Current delay + typical recorded change" did better on the test period at 4–10 stops (7.9 and 14.1 MAE), but validation did not choose it. Switching to it after seeing the test results would be test-set tuning.
- The ranges aim for 80% coverage and achieved 66–82%, so they are somewhat narrow. The app states this.

Validation results for every candidate, feature importances and plots are in `reports/results/` and `models/delay/metadata.json`.

CNN and OCR results are in §6 and §7.

## 11. Installation (Windows PowerShell)
Tested with **Python 3.11.9** on Windows 11, CPU only.
```powershell
cd C:\path\to\railscan
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt       # ~3.2 GB: TensorFlow + EasyOCR with CPU-only PyTorch
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
python scripts/download_datasets.py                 # raw files (~190 MB parquet + 2 CSVs), checksum-verified
python scripts/build_dataset.py --local-dir data/raw/delay --target-rows 1000000 --output-dir data/interim/build_v2
#   then copy data/interim/build_v2/RailScan_Final_Dataset.csv to data/interim/RailScan_Full_Dataset.csv
python scripts/prepare_dataset.py --target-rows 1500   # exactly 1,500 rows of whole journeys
python scripts/build_timetable.py                    # current timetable for the planner
python scripts/train_delay_models.py                 # selection on validation
python scripts/evaluate_models.py                    # one-time test evaluation
python scripts/evaluate_planner.py --max-journeys 400   # later-stop method + likely ranges (~5 min)
python scripts/train_cnn.py --arch mobilenetv2 --img-size 224 --epochs 40   # app model (~3 min CPU)
python scripts/train_cnn.py --arch small     # small CNN baseline (overwrites the app model; copy it first)
python scripts/train_cnn.py --labels language --arch mobilenetv2 --img-size 224 --epochs 40   # 4-language experiment
python scripts/evaluate_ocr.py                       # OCR on held-out signboards (needs an OCR backend)
python scripts/setup_database.py all --replace       # optional SQLite copy (docs/DATABASE.md)
```
Without `--local-dir`, `build_dataset.py` streams the whole Hugging Face dataset instead (much larger). In that mode it has no stop list, so it treats consecutive recorded stops as consecutive.
Then rename its `RailScan_Final_Dataset.csv` to `RailScan_Full_Dataset.csv`.

## 14. Tests
```powershell
python -m pytest tests -q
```
Last run (2026-10-29, Windows 11, Python 3.11.9): **294 passed, 6 skipped**.
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
| `test_delay_prediction.py` | leakage regex audit, saved vs code feature list, exact 1,500 real rows, target definition, brute-force past-only history, chronological split, app vs training predictions identical, unknown categories reported |
| `test_users.py` | sign-up validation, scrypt hashing (no plain-text passwords), case-insensitive unique usernames, identical errors for wrong password and unknown user, lockout and reset, SQL-injection strings, private per-user history, password change, account deletion with cascade, upgrade of a v1 database |
| `test_database.py` | SQLite import, NULL handling, rollback, SQL injection, SQLite-vs-CSV parity |

## 15. Deployment (external database and hosting)
Full step-by-step guide: **[docs/HOSTING.md](docs/HOSTING.md)**. In short:
1. **Database.** Create a PostgreSQL database (for example Neon or Supabase) and copy its connection string (`postgresql://...?sslmode=require`). RailScan creates its account tables itself. Railway data stays in SQLite and is rebuilt from the committed CSVs on a new server.
2. **Local test.** Run `$env:RAILSCAN_ACCOUNTS_DB_URL = "postgresql://..."` and then `streamlit run app.py`. My history should say "stored in PostgreSQL at …".
3. **GitHub.** Push the repository. `.gitignore` keeps the SQLite file, raw data and secrets out.
4. **Host.**
   - Streamlit Community Cloud: put `RAILSCAN_ACCOUNTS_DB_URL = "..."` under *Secrets* and choose Python 3.11.
   - Any other Python host: run `pip install -r requirements.txt`, then `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`, with the same environment variable.
   - Plan for about **2 GB of RAM**: a sign scan peaked at about 1.8 GB locally.

**Tested:** the PostgreSQL account path, against a real local PostgreSQL 16 (all account tests plus a full app run).
**Not tested:** a hosted provider or an actual Streamlit Cloud deployment.

## 16. Limitations
- **Narrow railway coverage.** 1,500 observations of **two trains** (01023/01024, Pune–Kolhapur specials) at 22 stations. The models have not seen other trains or routes, and the app warns when an input category is unknown.
- **No live data.** For a real journey you must enter the current delay. The planner assumes the latest recorded timetable (Jun–Aug 2026) still applies on the date you choose; it cannot know about later timetable changes or cancellations.
- **Estimates further ahead are rough.** Beyond the next stop, the app carries the current delay forward with a likely range. On test journeys the typical error was 5 min at 2–3 stops and 16–20 min at 7+ stops.
- **Signboard images are not railway-specific.** INDICSTR12 contains general shop and street signs. Accuracy on railway boards, which are often bilingual with English, is not measured.
- **Hindi vs Marathi.** Both use Devanagari, so the app classifies the script, not the language. See §6 for the 4-language experiment.
- **Modest OCR accuracy, slow scans.** Script identification reaches 82.8% when the image model is combined with OCR evidence (58.6% for the image model alone), and OCR finds 29% of reference words on scene photos (§6–7). A scan takes about 20–30 s on CPU. The planner never depends on a scan.
- **Derived actual times.** Actual times in the processed data are scheduled time + recorded delay. The source has both, so they agree within 1 minute.
- **Gaps in the timetable.** A stop with no sequence number in recent records is left out of the timetable (RKD for 01023; SSV and FSG for 01024). Four route-end segments have no training rows, and there the delay is carried forward and labelled.
- **Missing columns.** `journey_day`, `distance_from_source_km` and platform are not in the processed data. Platform numbers come only from OCR.

## 17. Future improvements
- Add more trains from the same source by running `build_dataset.py` with more trains, then retrain. The code is not limited to two trains.
- Classify the script on **detected text regions** (EasyOCR's CRAFT detector already finds them) instead of the whole photo. The CNN error analysis points to this as the main improvement.
- Collect labelled railway-board photos to measure, and fine-tune, the CNN and OCR on the real target domain.
- Train a dedicated model for stops further ahead (predicting the delay k stops ahead directly), and compare it with the carried-forward delay using `scripts/evaluate_planner.py`.
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
