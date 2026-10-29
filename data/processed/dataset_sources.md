# RailScan dataset sources

Every row and image in RailScan comes from the sources below. Nothing is synthetic.

## 1. Railway running records (delay models, train lookup, stop statistics)

| | |
|---|---|
| Dataset | `yadavsaurabh/indian-railways-runs` |
| URL | https://huggingface.co/datasets/yadavsaurabh/indian-railways-runs |
| License | GPL-3.0 (stated on the dataset card) |
| Railway system | Indian Railways. Third-party historical records, not an official feed |
| Granularity | One row per train, station and journey date: scheduled and actual arrival/departure, delays, distance, coordinates |
| Files used | `trains/01023.csv`, `trains/01024.csv` (raw copies in `data/raw/delay/`) |

**Processing.**
1. `scripts/build_dataset.py --local-dir data/raw/delay` built **35,273 observations** for trains 01023 and 01024 (Pune-Kolhapur specials), dated 2023-11-05 to 2026-08-31. Each observation is "train observed at stop S; what happened at the next stop N?".
   - The next stop is the train's next **regular** stop: a seq present on at least half of its journeys. The source lists only halts, and its seq numbers skip stations that are passed without stopping.
   - Every dropped row is counted in `filtering_report.csv`. This full file is kept in `data/interim/RailScan_Full_Dataset.csv`.
   - An earlier build required `next seq = seq + 1`. That kept only 11,697 rows and 14 stations, and is kept for reference as `RailScan_Full_Dataset_v1.csv`.
2. `scripts/prepare_dataset.py --target-rows 1500` reduced it to **exactly 1,500 rows**, using these rules:
   - It keeps whole journeys only (85 of 1,960 journeys), chosen in a golden-ratio order over the date-sorted journeys, so the sample spans the full period and both trains.
   - It recomputes all `historical_*` features inside the 1,500-row sample, with strictly past-only rules.
   - It splits the data chronologically, about 70/15/15 by rows, with cuts on journey-date boundaries.
3. Cross-check against the raw files: all 1,500 rows match a raw record. Scheduled arrival agrees in 100% of rows. Arrival delay agrees within 1 minute in 100% of the 1,280 rows where the source delay field is filled.
4. `scripts/build_timetable.py` writes `RailScan_Timetable.csv`: the most common scheduled arrival and departure (with day offset) of every stop over the last 90 recorded days. The journey planner uses it to place a journey on any date.

**Limitations.**
- The sample covers only 2 trains and 22 stations on one route.
- The timetable is assumed to stay as last recorded (Jun - Aug 2026).
- Actual arrival and departure times in the master CSV are derived as scheduled time + recorded delay.
- The last stop of each journey has no row, because it has no next stop.

## 2. Signboard images (CNN script classifier)

| | |
|---|---|
| Dataset | `jai23023/indicstr12-signboards`, a curated subset of INDICSTR12 (verified-twice split) |
| URL | https://huggingface.co/datasets/jai23023/indicstr12-signboards |
| License | CC BY 4.0. Attribution required; the dataset card also asks you to cite the INDICSTR12 paper |
| Content | 975 real signboard photos with columns `image`, `language` (hindi/kannada/marathi/telugu), `prompt`, `transcription` |
| Files | `data/raw/signboards/*.parquet`, verified against the Hugging Face LFS SHA-256 checksums |

**Processing.** `scripts/download_datasets.py` removes 12 byte-identical duplicates, which leaves 963 images. It saves the original bytes to `data/sign_images/<Script>/`. It then makes a stratified 70/15/15 split by language (seed 42) and writes `RailScan_CNN_Metadata.csv`.

**Limitations.** These are general Indian signboards (shops, streets, offices), not only railway boards. Hindi and Marathi are both written in Devanagari, so the app's CNN classifies the **script**: Devanagari, Kannada or Telugu.

## 3. Station master (station names, OCR matching)

| | |
|---|---|
| Source | DataMeet `stations.json`, https://github.com/datameet/railways |
| License | CC0 |
| File | `data/stations/station_master.csv` (8,965 stations, a roughly 2016 snapshot) plus `station_master_supplement.csv` and `station_aliases.csv` |

## Sources considered but not used

- `github.com/prasenjit-27/Indian-Railway-Data` and DataMeet `schedules.json`: timetables only, with no recorded actual times. The running-records dataset already carries the schedule of every observed stop, so a second, unmatched timetable was not merged.
- `github.com/meshramanirudh/indian-railways-delay-analysis`: journey-level analysis. It cannot support station-level next-stop prediction or stop-duration statistics.

## Column groups in `RailScan_Master_Dataset.csv`

- **Source (recorded or scheduled):** `train_no`, `train_name`, `train_type`, `journey_date`, `station_code`, `seq`, `next_station_code`, `current_scheduled_arrival`/`departure`, `next_scheduled_arrival`, `current_arrival_delay_min`, `current_departure_delay_min`, `distance_to_next_km`, `station_lat`/`lng`
- **Derived from the source:** `scheduled_stop_min` (= `scheduled_halt_min`), `actual_stop_min`, `stop_variation_min`, `scheduled_travel_to_next_min`, `actual_arrival`/`departure`, calendar fields, `stop_category` (based on the scheduled halt), `station_name`/`next_station_name` (joined from the DataMeet master)
- **Past-only history:** `historical_avg_delay_min`, `historical_delay_std_min`, `historical_observation_count`, `historical_on_time_rate_pct`, `historical_avg_actual_stop_min`, `historical_stop_std_min`, `historical_stop_observation_count`, `historical_stop_reliability_pct`
- **Targets (never model inputs):** `next_station_arrival_delay_min`, `next_station_delayed_15` (>= 15 min), `next_delay_risk_label`, `target_next_actual_arrival`

Exact definitions are in `dataset_summary.json` under `definitions`.

**Columns not provided:** `journey_day`, `distance_from_source_km` and platform. These exist in the raw source but were not carried through the stage-1 builder. Prediction-time fields (`predicted_next_station_delay_min`, `expected_next_station_arrival`, `minutes_until_next_station`) are produced by the app at run time and are not stored, so the CSV contains no model output presented as data.
