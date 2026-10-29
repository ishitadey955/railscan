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
| Files used | the 5,300 still-running trains that halt in a covered state (raw copies of their last 60 days in `data/raw/delay_north/`) |

**Coverage: northern India and West Bengal.** Jammu and Kashmir, Ladakh, Himachal Pradesh, Punjab, Chandigarh, Haryana, Delhi, Rajasthan (the Northern Zonal Council), plus Uttar Pradesh, Uttarakhand and West Bengal (`COVERED_STATES` in `scripts/select_northern_trains.py`).

**Processing.**
1. **Which trains** (`scripts/select_northern_trains.py`). The source has 12,935 train files (34.7 GB). Only the end of each file was read, which holds its latest journeys, to get the current route.
   - Each station was placed in a state as follows:
     - **4,397** stations by the state in the DataMeet station list;
     - **4,483** by their coordinates inside the state boundaries (geoBoundaries IND ADM1, from DataMeet maps, CC BY 2.5 IN);
     - **728** newer or renamed codes that are missing from the station list (e.g. PRYJ Prayagraj, VGLJ Jhansi, AYC Ayodhya Cantt) by the coordinates in the running records themselves.
   - About 300 halt stations have no coordinates anywhere and stay unplaced.
   - **5,331 trains** have at least one scheduled halt in a covered state. 5,300 of them ran in the last 30 days of the source and were kept.
2. **Download** (`scripts/download_northern.py`): every journey of those trains that started from 2026-07-04 to 2026-09-01 (60 days), as whole journeys with all source columns. 4,831,683 source rows.
3. **Observations** (`scripts/build_northern_dataset.py`, which reuses `build_dataset.process_journey`). Each observation is "train observed at stop S; what happened at the next stop N?".
   - The next stop is the train's next **regular** stop, a seq present on at least half of its journeys.
   - Only observations whose **current** station is in a covered state are kept. The next stop may lie just across the border.
   - 58 rows were dropped because the source's delay field contradicts its own timestamps.
   - Every dropped row is counted by reason in `filtering_report.csv`.
   - This gives 2,572,458 observations from 5,104 trains.
4. **Size limit (fewer than a million rows).** West Bengal's suburban (EMU) services run many times a day, so the oldest journey dates were dropped, whole dates at a time, to fit 950,000 rows. That leaves **924,706 observations, 5,068 trains and 3,267 stations, with journeys from 2026-08-11 to 2026-09-01** (three weeks).
   - The `historical_*` features are computed inside these rows with strictly past-only rules.
   - The split is chronological, cut on journey-date boundaries: train 655,243 rows (08-11 to 08-25), validation 131,475 (08-26 to 08-28), test 137,988 (08-29 to 09-01).
5. **Timetable** (`scripts/build_timetable.py`): `RailScan_Timetable.csv` holds the most common scheduled arrival and departure (with day offset) of every stop over the 60 downloaded days. It uses journeys on each train's latest route, keeps the newest code where a station was renamed, and stores the median recorded coordinates of every stop. That is 5,300 trains and 113,498 stops, with full routes including stops outside the covered states. The journey planner and the position estimate use it.

**Limitations.**
- **Time span.** The data covers three weeks (August 2026), so seasonal effects such as winter fog or the monsoon peak are not in it, and each train has only a few weeks of history.
- **Region.** Only stations in the covered states have observations.
- **Delay values.**
  - The source reports 0 for both on-time and early arrivals.
  - Where its delay field is blank, the delay is computed from the timestamps and can be negative (early). Delays therefore mix both conventions.
- **Station names.** Codes missing from the 2016 station list get a name from Wikipedia only when the article's infobox states exactly that code (`scripts/build_station_supplement.py`, 123 stations). 2% of timetable stops still have no name, and the app shows their code.
- **Timetable.** The timetable is assumed to stay as last recorded.
- **Derived times.** Actual arrival and departure times in the master CSV are derived as scheduled time + recorded delay.
- **Last stop.** The last stop of each journey has no row, because it has no next stop.

Earlier versions are kept in `data/interim/backup_1500_pune_kolhapur/`: the 1,500-row Pune-Kolhapur version and the 3-script CNN.

## 2. Signboard images (CNN script classifier)

| | |
|---|---|
| Dataset | `jai23023/indicstr12-signboards`, a curated subset of INDICSTR12 (verified-twice split) |
| URL | https://huggingface.co/datasets/jai23023/indicstr12-signboards |
| License | CC BY 4.0. Attribution required; the dataset card also asks you to cite the INDICSTR12 paper |
| Content | 975 real signboard photos with columns `image`, `language` (hindi/kannada/marathi/telugu), `prompt`, `transcription` |
| Files | `data/raw/signboards/*.parquet`, verified against the Hugging Face LFS SHA-256 checksums |

**Bengali** (added 2026-10-30): 347 photos from `darknight054/indicstr12-full` (subset `bengali`, pinned revision `b3c5620`, SHA-256-checked), the full-image part of the same IndicSTR12 real dataset (CVIT, IIIT Hyderabad; Lunia, Mondal and Jawahar, ICDAR 2023). The original project page states no licence; the photos are used for research with attribution. `scripts/add_bengali_signboards.py` keeps each distinct image once (10 duplicates removed), saves the original bytes to `data/sign_images/Bengali/` and splits them 70/15/15 with the same rule and seed (243 / 52 / 52).

**Processing.** `scripts/download_datasets.py` removes 12 byte-identical duplicates, which leaves 963 images. It saves the original bytes to `data/sign_images/<Script>/`. It then makes a stratified 70/15/15 split by language (seed 42) and writes `RailScan_CNN_Metadata.csv`.

**Limitations.** These are general Indian signboards (shops, streets, offices), not only railway boards. Hindi and Marathi are both written in Devanagari, so the app's CNN classifies the **script**: Devanagari, Bengali, Kannada or Telugu.

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
