# RailScan database (SQLite, optional)

The Streamlit app reads `data/processed/RailScan_Master_Dataset.csv` directly; this database is an optional indexed copy used by `src/railway` and `src/database`.

### Why SQLite

RailScan's data is small (1,500 observation rows and about 9k stations), read far more often than it is written, and used by one process at a time. SQLite gives indexed lookups, transactions and constraints with no server to install: the whole database is one file that ships with the project, and Python's built-in `sqlite3` module is the only dependency. Before this stage, every train-state lookup loaded the whole processed CSV into memory; now a lookup reads one journey through an index.

### Location

`data/database/railscan.db` (created automatically). To use another file, pass `--db PATH` to the setup script or set the environment variable `RAILSCAN_DB_PATH`.

The CSV files stay the source of truth. The database is built from them and can be rebuilt at any time. Images and model files (`.pkl`, `.keras`) are not stored in the database.

### Schema

| Table | Key | Contents |
|---|---|---|
| `stations` | `station_code` | Station master: name, latitude, longitude, source. Built from DataMeet's CC0 list (`data/stations/station_master.csv`) plus `station_master_supplement.csv`. |
| `train_runs` | `id`; unique `(train_key, journey_date, seq)` | One row per observed stop of a journey, from `RailScan_Final_Dataset.csv`: schedule, current delays, stop durations, next stop, leakage-safe `historical_*` features, and the target columns (next-stop outcomes). |
| `train_metadata` | `train_no` | One row per train: name, type, first/last journey date, row count, and `name_type_history` (JSON). |
| `database_metadata` | `id` | One audit row per import attempt: source file, SHA-256, time, rows imported, status (`success` / `failed`), and notes (JSON with rejected rows, values set to NULL, duplicates and warnings). |
| `users` | `id`; unique `username` (case-insensitive) | Accounts (schema v2): display name, salted scrypt password hash, created and last-login time. |
| `user_history` | `id`; `user_id` → `users` (ON DELETE CASCADE) | One row per result a user saved: time, `kind` (`scan` / `forecast` / `typical`), summary, JSON details. |
| `login_attempts` | `id` | Time and success of each login, used for the lockout after 5 failures in 15 minutes. |

Schema v2 adds the three account tables. `python scripts/setup_database.py init` adds them to an existing v1 file without touching other data. The account functions are in `src/database/users.py` and are tested in `tests/test_users.py`.

Column mapping from the dataset to `train_runs`: `current_scheduled_arrival` → `scheduled_arrival` and `current_scheduled_departure` → `scheduled_departure`. Every other dataset column keeps its name. The importer adds four columns:

- `train_key`: the train number without leading zeros, so `"01023"` and `"1023"` are one train.
- `actual_arrival` and `actual_departure`: derived as scheduled time + observed delay, because the dataset stores delays, not raw actual timestamps. They are NULL when either part is missing.
- `actual_time_basis`: says that the two columns above are derived.

Indexes: `stations(station_name COLLATE NOCASE)`, and on `train_runs`: `train_no`, `journey_date`, `station_code`, `(train_no, journey_date, seq)`, `(train_key, journey_date)` and `next_station_code`.

Constraints: foreign keys are switched on for every connection, and `train_runs.train_no` references `train_metadata`. CHECK constraints limit `is_weekend` and `next_station_delayed_15` to 0/1 and coordinates to valid ranges.

Two design decisions, both based on the real data:

- **`train_type` is not constant per train.** Trains 01023 and 01024 appear as `MAIL/EXPRESS` on some journey dates and `TOD` on others. `train_metadata.train_type` is therefore set only when a train has exactly one type, and is NULL otherwise. `name_type_history` lists every type with its date range, and the per-journey value stays in `train_runs`.
- **Station codes are not foreign keys.** The station master is a roughly 2016 snapshot, and newer stations such as PRYJ are missing from it. A foreign key would force real observations to be dropped. The importer and `stats` report unknown codes instead (there are none in the current dataset).

### Setup and import

Run these from the project root:

```bash
python scripts/setup_database.py init              # create tables and indexes (safe to repeat)
python scripts/setup_database.py import-stations   # data/stations/station_master.csv (+ supplement)
python scripts/setup_database.py import-data       # data/processed/RailScan_Final_Dataset.csv
python scripts/setup_database.py stats             # real row counts and last imports
python scripts/setup_database.py all               # all four steps in order
```

Options:

- `--csv PATH`, `--master PATH`, `--supplement PATH`: use files from another location. If the default file is missing, the importer also searches the project for it.
- `--replace`: replace the table contents instead of upserting.
- `--duplicates keep-first`: keep the first of conflicting duplicate rows (by default these stop the import).
- `--json`: machine-readable output.

The exit code is 1 when a step fails.

How imports behave:

- Required columns are checked, and missing optional columns are reported and stored as NULL.
- Empty cells become NULL. Unparseable or out-of-range values become NULL and are counted per column; no value is ever invented.
- Rows with an invalid key (train number, journey date, station code or seq) are rejected and counted.
- Identical duplicate rows are collapsed. Conflicting duplicates stop the import unless you pass `--duplicates keep-first`.
- Each import is one transaction with batched inserts. Any error rolls everything back, and the failure is then recorded in `database_metadata`.
- The default mode is an upsert on the natural key, so re-running an import is safe and deletes nothing. Rows that are in the database but no longer in the CSV are reported, not removed.
- `train.csv`, `validation.csv` and `test.csv` are refused. They are splits of `RailScan_Final_Dataset.csv`, so importing them would duplicate every observation.

### Refreshing after regenerating the CSV

The `historical_*` features depend on which rows are in the sample, so a rebuilt dataset changes existing rows. Replace the table instead of upserting into it:

```bash
python scripts/prepare_dataset.py --target-rows 1500   # regenerate data/processed/
python scripts/setup_database.py import-data --replace
python scripts/setup_database.py stats
```

The report shows how many previous rows were replaced. If the station master changed, also run `import-stations --replace`.

### Repository queries

```python
from src.database import repository as repo

repo.get_station_by_code("KOP")                     # dict or None
repo.search_stations("kolhapur", limit=5)           # exact code > exact name > prefix > substring; no fuzzy matching
repo.get_train_metadata("01023")                    # "1023" works too
repo.get_train_observations("01023", "2025-03-10")  # target columns only with include_targets=True
repo.get_train_state("01023", "2025-03-10", "KRG")  # state row without target columns, or None
repo.get_next_station("01023", "2025-03-10", "KRG") # next stop: schedule data only
repo.get_historical_station_reliability("RMP", "01023", before="2025-03-10 23:00:00")
repo.get_database_statistics()
```

`get_train_state` raises `repository.AmbiguousRecord` when a train calls at the same station twice on one day; pass `seq=` to choose one. Every function also accepts `db_path=` or an open `conn=`. All SQL is parameterised.

**Leakage.** Model features must use the precomputed, leakage-safe `historical_*` columns of `train_runs`. `get_historical_station_reliability` aggregates real outcomes, so only use it for prediction with `before=<observation time>`, which counts only outcomes that happened strictly earlier. Without a cutoff, its result is marked `leakage_safe_for_prediction: False`.

### How the existing modules use the database

- **Station matcher.** `StationMaster.load()` reads the `stations` table when it is populated, and falls back to the CSV files otherwise. Aliases still come from `data/stations/station_aliases.csv`. The matcher's logic (exact, alias, RapidFuzz and transliteration matching, its thresholds and result format) is unchanged. Fuzzy matching needs every name in memory, so the station list (about 9k rows) is loaded once. `RailwayPipeline()` uses `StationMaster.load()` by default.
- **Train-state lookup.** `TrainStateService(data=None)` and `RailwayPipeline()` use SQLite when `train_runs` has rows, and fall back to `RailScan_Final_Dataset.csv` otherwise. A path ending in `.db`, `.sqlite` or `.sqlite3` selects SQLite explicitly; a DataFrame or CSV path works as before. `service.backend` reports which source is in use. All function signatures and result dictionaries are unchanged, and lookups query one journey instead of loading the whole dataset.
- **Delay models.** Nothing was retrained. `src/database/model_inputs.py` turns database rows into the model input by calling the training code's own `delay_common.make_features`, and checks the column order against `models/delay/feature_columns.json`. One detail matters here: training read the CSV without dtypes, so the saved encoder learned train number `"1023"`, not `"01023"`. The adapter reproduces that form, and `unknown_categories()` lists any value the model has never seen, because the encoder would otherwise ignore it silently.

### Verification

Re-import of the 1,500-row dataset on 2026-10-29 (corrected next-stop rule, 85 journeys): 1,500 rows and 8,965 stations imported, 0 rejected, 0 values set to NULL, 0 foreign-key violations. The table below was measured on 2026-10-14 against the earlier 11,697-row dataset.

| Check | Result |
|---|---|
| Import | 11,697 dataset rows and 8,965 stations imported; 0 rejected; 0 values set to NULL; 0 foreign-key violations |
| Round trip | All 34 mapped columns identical to the CSV in every row; NULL counts preserved |
| Re-import | 0 inserted and 11,697 updated; row count unchanged |
| Train state | All 11,697 real `(train, date, station)` lookups give identical results from SQLite and CSV; about 20 ms per lookup vs about 25 ms from the CSV |
| Delay models | Predictions from database rows identical to the CSV training path on all 11,697 rows (max difference 0.0) |

`tests/test_database.py` covers initialisation, repeated initialisation, import, NULL handling, duplicates, rollback, SQL injection, every repository function, statistics, and SQLite-vs-CSV parity. It uses temporary databases only.

### Known limitations

- **History is sample-limited.** Historical statistics only see journeys inside the processed sample. The current dataset has 2 trains on one route (22 station codes, 85 journeys), so the database covers only that route.
- **Actual times are derived.** `actual_arrival` and `actual_departure` are scheduled time + delay, not raw timestamps.
- **No row for a journey's last stop.** The dataset only keeps stops whose next stop was also observed, so the last stop of a journey has no `train_runs` row.
- **`train_type` changes between journeys**, so `train_metadata.train_type` is NULL for both current trains. The train lookup in `src/railway` returns `ambiguous_train` when no journey date is given (this behaviour comes from stage 5).
- **The station master is a roughly 2016 snapshot.** Add newer stations to `station_master_supplement.csv` and re-run `import-stations`.
- **Aliases are not in the database.** They stay in `station_aliases.csv`.
- **The saved delay models need scikit-learn 1.9.1** (the version they were pickled with). Older versions fail to load them.
- **Single-user design.** There is one writer at a time, with a 10 s busy timeout. That is fine for this project, but not for a multi-user server.
