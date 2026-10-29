# data/stations/

| file | what | source |
|---|---|---|
| `station_master.csv` | `station_code, station_name, latitude, longitude` — 8,965 stations | DataMeet "Indian Railways" `stations.json` (CC0), https://github.com/datameet/railways — built by `scripts/build_station_master.py`; see `SOURCE.txt` |
| `station_aliases.csv` | `alias, station_code, language, source` — alternate English names and Hindi / Marathi / Kannada / Telugu spellings | hand-curated seed (`curated_seed`); every `station_code` is checked against the master at load time |
| `station_master_supplement.csv` | same columns as the master (+ `source`) — **empty** | put newer / renamed stations here |

**No station code is invented.** Codes, names and coordinates come from the DataMeet file;
268 stations in it have no coordinates and keep them empty. The source is a ~2016 snapshot:
stations opened or renamed since then may be missing (e.g. `PRYJ`), and a few names are stale
(`ALLAHABAD JN`, `MUGHAL SARAI`). Add such rows to `station_master_supplement.csv` from an
official list — Indian Railways / NTES station list or data.gov.in — in the same four columns
(+ `source`), e.g. `PRYJ,PRAYAGRAJ JN,,,NTES`. Rebuild the master with
`python scripts/build_station_master.py` (downloads) or `--input stations.json` (local copy).

Aliases are a convenience for names the master spells differently (`CSTM` is "MUMBAI CST",
`NDLS` is also "नई दिल्ली"). They are **not** an official multilingual list: Hindi/Marathi/Kannada/Telugu
coverage is limited to ~120 major stations. Extend the file with the spellings you see on real
signs (one row per spelling).
