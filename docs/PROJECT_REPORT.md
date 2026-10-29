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
| Delays, timetable, stop statistics | `yadavsaurabh/indian-railways-runs` (Hugging Face): stop-level scheduled and actual times | **1,500 observations**, 85 journeys, 2 trains, 22 stations, Nov 2023 – Aug 2026 | GPL-3.0 |
| Script CNN, OCR evaluation | `jai23023/indicstr12-signboards` (INDICSTR12 subset) | 963 images after removing 12 duplicates | CC BY 4.0 |
| Station names | DataMeet `stations.json` | 8,965 stations | CC0 |

**Unit of observation.** Each row records a train observed at station *S*, together with what happened at the next station *N*.

**Building the observations.** The source lists only the stations where a train halts, while its sequence numbers also count stations passed without stopping. The next stop of an observation is therefore the train's next *regular* stop, not "sequence + 1".

An earlier version used "sequence + 1". That discarded about 18,000 valid rows and left 8 of the 22 stations without training data. It was found when the journey planner showed implausible estimates on those stretches. With the fix there are 35,273 valid observations.

**Reduction to 1,500 rows.** The project requirement was a dataset of 1,500 records. These were selected as **whole journeys** (85 of 1,960) in an evenly spaced (golden-ratio) order over time, so the sample still covers the full period and both trains. Every row was cross-checked against the raw source files:
- scheduled times agree in 100% of rows;
- recorded delays agree within 1 minute in 100% of rows.

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
- **Classes:** Devanagari, Kannada, Telugu. Hindi and Marathi share the Devanagari script.
- **Split:** stratified 673 / 145 / 145 (train / validation / test), made after deduplication so that no image appears in two splits.
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
- On the test images, accuracy rose from **58.6% to 82.8%**.
- 83% of test images were marked confident, and **92.6%** of those were right.
- The cost is three OCR passes, about 20–30 s per photo on CPU.

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
  - train: 1,055 rows, up to Oct 2025
  - validation: 223 rows, Nov 2025 – Mar 2026
  - test: 222 rows, Apr – Aug 2026

**Model selection.** Linear/Logistic Regression, Random Forest and XGBoost were compared on validation data. The linear models won both tasks.

## 4. Results (test set, 222 rows)
| Next-station delay | MAE | RMSE | R² |
|---|---|---|---|
| **Linear Regression** | **3.45 min** | **7.81 min** | 0.926 |
| Baseline: next delay = current delay | 5.95 min | 10.90 min | 0.856 |

| Delayed by 15 min or more | Precision | Recall | F1 | Macro-F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.975 | 0.908 | **0.940** | 0.931 | 0.982 |
| Baseline | 0.896 | 0.931 | 0.913 | 0.892 | – |

**Interpretation.** Delay persists between neighbouring stations, so even the naive baseline is fairly strong. The model cuts the average error by 42% (5.95 → 3.45 minutes) and raises macro-F1 from 0.892 to 0.931.

**Journey planner (any date).** The planner places the current timetable on any chosen date. It builds the model input from the delay the user enters, the journey date's calendar and past-only history, and reproduces the training features exactly on all 1,500 recorded rows. It was evaluated by replaying 400 recorded journeys after the training period (7,078 comparisons). The method for later stops was chosen on the validation period and is reported on the test period:

| Stops ahead | Shown in the app | Test MAE | Carry-forward MAE |
|---|---|---|---|
| 1 | model | **2.66 min** | 3.25 min |
| 2–3 | current delay carried forward | 5.38 min | 5.38 min |
| 4–6 | current delay carried forward | 9.31 min | 9.31 min |
| 7–10 | current delay carried forward | 20.27 min | 20.27 min |
| 11+ | current delay carried forward | 16.50 min | 16.50 min |

- Applying the next-stop model stop by stop compounded its small errors and was clearly worse than the baseline, so the app does not do it.
- Each estimate carries a likely range from past errors at the same distance. These ranges target 80% coverage and achieved 66–82% on test.

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

**Testing.** 272 automated tests pass. They cover preprocessing, OCR handling (including real EasyOCR runs), matching, time and date arithmetic, the planner on future dates, leakage, feature consistency, and app-vs-training prediction parity. The 6 skipped tests need a Pillow build with libraqm to draw Indic test text.

## 6. Limitations
- **Narrow delay data.** Two trains on one route. The models are not validated for other trains, and the app reports inputs it has never seen.
- **Modest OCR accuracy** on general scene photos (§3.1–3.2), and scans take about 20–30 s on CPU. Script identification reaches 82.8% only with the OCR evidence. Accuracy on real railway boards has not been measured with a labelled set.
- **Later-stop estimates are rough.** Beyond the next stop the app carries the current delay forward, with an error of 5–20 minutes. Users must enter the current delay themselves, and the timetable is assumed unchanged.
- **Not deployed.** Deployment to Streamlit Cloud is prepared (CPU-only requirements, instructions) but was not performed in this work.

## 7. Future work
- Classify the script on detected text regions rather than the whole photo.
- Collect a labelled railway-board test set.
- Extend the dataset to more trains; the pipeline already supports this.
- Calibrate the delay probabilities.
- Train a dedicated multi-stop model to replace the iterative estimate.
