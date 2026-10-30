# 0004 Railway data from Parquet, one train on demand

**Context.** The full SQLite database (924,706 rows, ~590 MB) is too big for GitHub and took 4–5 minutes of
full CPU to build after every server restart, which blocked pages and contributed to CPU throttling.

**Decision.** On a server without the full database, each train's rows are imported from the committed
Parquet file when first needed (same import code, ~0.4 s); the Data page uses figures exported from the full
database (`data/processed/dataset_overview.json`, tagged with the Parquet file's sha256). Where the full
database exists (development), it is used.

**Evidence.** Identical rows for 31 trains compared; identical train list, overview, risk split, missing
values and delays (tests/test_on_demand.py). Planner estimate 26 s -> 1.8 s overall (with a separate fix in
src/position.py).

**Consequences.** After changing the dataset, `python scripts/setup_database.py all` must be run and the new
overview file committed; a test fails if they do not match.
