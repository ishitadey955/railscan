# RailScan: Technical Project Report

**CNN-Based Indian Railway Sign Recognition with Delay Risk and Next-Station Time Prediction**

All figures in this report were measured by the project's own scripts on held-out data (October 2026). None are estimated or copied from other work.

---

## 1. Objective
The goal was to build a working application that helps Indian railway passengers in two ways:
- **Read a sign:** identify the script of a photographed sign, read its text, and find the station, train number or platform it mentions.
- **Anticipate delay:** given a train's current position and delay, estimate its delay and arrival time at the next station.

These are two separate machine-learning problems, solved by two separate models: a **CNN** for images and a **tabular model** for delays.

## 2. Data
| Component | Dataset | Size used | License |
|---|---|---|---|
| Delays, timetable, stop statistics | `yadavsaurabh/indian-railways-runs` (Hugging Face): stop-level scheduled and actual times | **924,706 observations** at stations in northern India and West Bengal, 73,935 journeys, 5,068 trains, 3,267 stations, 11 Aug – 1 Sep 2026 | GPL-3.0 |
| Script CNN, OCR evaluation | IndicSTR12 signboards: `jai23023/indicstr12-signboards` (Hindi, Marathi, Kannada, Telugu) and `darknight054/indicstr12-full` (Bengali) | 1,310 images after removing duplicates (347 Bengali) | CC BY 4.0 (Bengali copy: licence not stated by the original source; used for research with attribution) |
| Station names | DataMeet `stations.json` | 8,965 stations | CC0 |

**Unit of observation.** Each row records a train observed at station *S*, together with what happened at the next station *N*.

**Building the observations.** The source lists only the stations where a train halts, while its sequence numbers also count stations passed without stopping. The next stop of an observation is therefore the train's next *regular* stop, not "sequence + 1".

**Coverage: northern India and West Bengal.** J&K, Ladakh, Himachal Pradesh, Punjab, Chandigarh, Haryana, Delhi, Rajasthan, Uttar Pradesh, Uttarakhand and West Bengal.
- Every station was placed in a state, using its listed state or its coordinates inside official state boundaries.
- 5,331 of the source's 12,935 trains halt in these states. The 5,300 still running were downloaded for their last 60 days (4.83 million source rows).
- Observations whose current station is in a covered state were kept: 2,572,458.
- To stay under one million rows, the oldest journey dates were dropped, which leaves 924,706 rows (three weeks, because West Bengal's suburban trains run many times a day).
- 58 source rows whose delay field contradicted their own timestamps were dropped rather than corrected.

An earlier version of RailScan used 1,500 rows of two Pune–Kolhapur trains; it is kept for reference in `data/interim/backup_1500_pune_kolhapur/`.

**Derived quantities.**
- scheduled halt = scheduled departure − scheduled arrival
- actual stop = actual departure − actual arrival
- delay = actual time − scheduled time

Recomputing each of these from the timestamps gives 0 mismatches. Calendar dates are carried with every timestamp, so stops that cross midnight are handled correctly.

**Targets.**
- next-station arrival delay in minutes (regression);
- delayed by 15 minutes or more (binary; 59% of rows are positive);
- risk label: Low < 15 min, Medium 15–59 min, High ≥ 60 min.

## 3. Method

### 3.1 Script classification (CNN)
- **Input:** RGB image, letterboxed to a square with the aspect ratio kept, values scaled to [0, 1].
- **Augmentation:** only during training (rotation, zoom, shift, contrast). There are no flips, because mirrored text is not valid text.
- **Classes:** Bengali, Devanagari, Kannada, Telugu. Hindi and Marathi share the Devanagari script.
- **Split:** stratified 916 / 197 / 197 (train / validation / test), made after deduplication so that no image appears in two splits.
- **Training:** class weights, Adam, early stopping on validation loss.

Experiments, with the final model selected on **validation** accuracy:

| Model | Val acc | Test acc |
|---|---|---|
| Small CNN (3 conv blocks, GAP, Dense 128, Dropout 0.3) at 128 px | 0.428 | 0.455 |
| MobileNetV2 transfer learning, 128 px | 0.572 | 0.628 |
| **MobileNetV2 transfer learning, 224 px (selected)** | **0.607** | **0.586** |
| 4-language variant, 224 px | 0.531 | 0.566 |

**Finding 1: the small CNN underfits.** Its training accuracy stalled at about 48%, and it never predicted Devanagari. Transfer learning from ImageNet features added 13–17 points of test accuracy.

**Finding 2: the problem is locating the text, not similar scripts.** In the 4-language experiment, Hindi↔Marathi confusions were rare (8 of 145). The main errors came from the script occupying only a small part of a cluttered scene photo.

**Finding 3: OCR evidence fixes most of the CNN's mistakes.** The app reads the sign with each script's OCR recogniser and combines the image model's probabilities with the OCR evidence: OCR confidence × share of letters in that script's Unicode block. The weight and the "confident" threshold were chosen on validation images only.
- On the test images (four scripts), accuracy rose from **49.2% to 86.3%**.
- 80% of test images were marked confident, and **94.3%** of those were right.
- A reading the OCR rejects as noise still counts through its best attempt. Counting it as 0 gave 79.2% and let
  tiny numeric differences between computers flip the decision (one reading at 30.4 counted, at 29.9 it did not).
- The cost is four OCR passes, about 20–30 s per photo on CPU.

### 3.2 OCR and matching
- **Engine:** EasyOCR (CPU).
- **Languages:** chosen from the CNN's prediction. When the CNN is unsure, the runner-up script is also tried, and the best-scoring reading is kept.
- **Station matching** tries, in order: exact code, exact name, aliases in Indian scripts, fuzzy similarity (RapidFuzz), then transliteration.
  - A score of 90 or more is accepted automatically; 75–89 needs confirmation.
  - If two stations score about the same, the matcher refuses to choose.
- **Train numbers** are accepted only if they exist in the dataset.

| OCR result on 145 test images (true language given) | Value |
|---|---|
| Images with no usable text | 52 (36%) |
| Ground-truth words found, all images | 29% |
| Ground-truth words found, images where text was read | 45% (Hindi 65%, Marathi 64%, Telugu 38%, Kannada 29%) |

**Worked example.** On a clear New Delhi station board, the CNN's top guess was wrong (Kannada 0.47 vs Devanagari 0.42). The runner-up fallback still read "नई दिल्ली / NEW DELHI" and matched New Delhi (NDLS) with a score of 97.

### 3.3 Delay prediction
**Features**, all known at the current station:
- current arrival and departure delay
- timetable (scheduled times, halt, running time and distance to the next stop)
- calendar (day of week, month, weekend flag)
- train, train type, current station and next station identities
- past-only history:
  - this train's earlier delays at the next station (average, spread, count)
  - its on-time rate there
  - its earlier actual stop durations at this station
  - stop reliability: the share of earlier stops within ±2 minutes of the scheduled halt

**Leakage control.**
- A regex audit blocks any target, future-time or next-station actual-time column from being used as a feature.
- A history value counts an earlier row only if that row's outcome was known *strictly before* the current observation. This is verified by an independent brute-force recomputation.
- The split is chronological, with cut points on journey-date boundaries:
  - train: 655,243 rows, 11 – 25 Aug 2026
  - validation: 131,475 rows, 26 – 28 Aug 2026
  - test: 137,988 rows, 29 Aug – 1 Sep 2026

**Model selection.** Linear/Logistic Regression, Random Forest and XGBoost were compared on validation data. Random Forest won the regression (validation MAE 3.25 vs 3.35 linear, 4.08 XGBoost); XGBoost won the classification (macro-F1 0.945 vs 0.936 logistic, 0.918 forest).

## 4. Results (test set, 137,988 rows, 4,279 trains)
| Next-station delay | MAE | RMSE | R² |
|---|---|---|---|
| **Random Forest** | **2.99 min** | **7.19 min** | 0.969 |
| Baseline: next delay = current delay | 4.38 min | 9.60 min | 0.945 |

| Delayed by 15 min or more | Precision | Recall | F1 | Macro-F1 | ROC-AUC |
|---|---|---|---|---|---|
| **XGBoost** | 0.940 | 0.935 | **0.937** | 0.946 | 0.988 |
| Baseline | 0.887 | 0.929 | 0.907 | 0.919 | – |

**Interpretation.** Delay persists between neighbouring stations, so even the naive baseline is strong. The model cuts the average error by 32% (4.38 → 2.99 minutes) and raises macro-F1 from 0.919 to 0.946, across thousands of trains. The test days directly follow the training weeks; other seasons are not covered.

**Where is my train? (estimate).** Indian Railways offers no public live-status API, so RailScan does not show live positions. For a planned journey it estimates where the train should be now: expected arrival = scheduled arrival + estimated delay, departure = arrival + scheduled halt. Between stations the point is interpolated by time on a straight line between the stops' recorded coordinates. The map is labelled "estimate, not live" (`src/position.py`, 9 tests).

**Journey planner (any date).** The planner places the current timetable on any chosen date. It builds the model input from the delay the user enters, the journey date's calendar and past-only history, and reproduces the training features exactly on recorded rows. It was evaluated by replaying 1,000 recorded journeys after the training period (11,683 comparisons). The method for later stops was chosen on the validation days and is reported on the test days:

| Stops ahead | Shown in the app | Test MAE | Carry-forward MAE |
|---|---|---|---|
| 1 | model | **4.07 min** | 4.93 min |
| 2–3 | current delay + typical recorded change | **5.44 min** | 7.39 min |
| 4–6 | current delay + typical recorded change | **8.33 min** | 11.04 min |
| 7–10 | current delay + typical recorded change | **9.14 min** | 13.57 min |
| 11+ | current delay + typical recorded change | **12.31 min** | 18.76 min |

- Applying the next-stop model stop by stop compounded its small errors and was worse, so the app does not do it.
- Each estimate carries a likely range from past errors at the same distance. These ranges target 80% coverage and achieved 77–83% on test.

**Bugs found and fixed.**
- **Missing departure delay.** It used to be filled with the training median (about 30 minutes); it is now taken from the arrival delay, the latest observed value. This was found while inspecting test errors, and is reported for transparency.
- **The "sequence + 1" rule** in the dataset builder, described in §2.
- **Planner inputs.** The first planner version built inputs that differed from training: a missing origin arrival time and calendar features taken from the wrong date. A feature-by-feature comparison against the recorded rows found and fixed both.

## 5. Application
The Streamlit app has three pages: **Scan a sign**, **Journey planner** and **Data and models**. Every value is labelled as *recorded*, *scheduled*, *model estimate*, *estimate* or *your input*.

The Journey planner works for **any date** (today, future or past). The user picks a train, a date and the station the train is at.
- **With a current delay entered,** it shows:
  - the next-stop estimate with its likely range, risk label and probability of a 15+ minute delay;
  - expected arrival times along the whole route (after departure, and from now when the journey is within a day);
  - historical stop statistics.
- **Without a delay,** it shows the recorded median delay at every stop, including for the same weekday, labelled as history rather than a forecast.
- **A "check against a recorded journey" panel** shows a prediction next to what actually happened. There is **no live feed**.

**Testing.** 355 automated tests: 349 pass, 6 skipped. They cover preprocessing, OCR handling (including real EasyOCR runs), matching, time and date arithmetic, the planner on future dates, leakage, feature consistency, and app-vs-training prediction parity. The 6 skipped tests need a Pillow build with libraqm to draw Indic test text.

## 6. Limitations
- **Northern India and West Bengal, three recent weeks.** Observations cover stations in these states from 11 Aug to 1 Sep 2026. Other regions appear only in timetables, and seasonal effects (winter fog) are not in the data.
- **Modest OCR accuracy** on general scene photos (§3.1–3.2), and scans take about 20–30 s on CPU. Script identification reaches 82.8% only with the OCR evidence. Accuracy on real railway boards has not been measured with a labelled set.
- **Later-stop estimates are rough.** Beyond the next stop the error grows from about 5 minutes (2–3 stops) to 12 minutes (11+ stops). Users must enter the current delay themselves, and the timetable is assumed unchanged.
- **Not deployed.** Deployment to Streamlit Cloud is prepared (CPU-only requirements, instructions) but was not performed in this work.

## 7. Future work
- Classify the script on detected text regions rather than the whole photo.
- Collect a labelled railway-board test set.
- Extend the dataset to more trains; the pipeline already supports this.
- Calibrate the delay probabilities.
- Train a dedicated multi-stop model to replace the iterative estimate.
