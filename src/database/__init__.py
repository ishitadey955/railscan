"""RailScan SQLite persistence layer (``data/database/railscan.db``).

    from src.database import initialize_database, import_dataset, import_stations
    from src.database import repository as repo
    repo.get_station_by_code("KOP")

See README.md ("Database") for the schema and commands.
"""
from .connection import (DEFAULT_DB_PATH, DatabaseUnavailable, connect, database_ready, open_connection,
                         resolve_db_path, transaction)
from .schema import SCHEMA_VERSION, initialize_database
from .import_data import ImportReport, import_dataset, import_stations
from .repository import (AmbiguousRecord, get_all_stations, get_database_statistics,
                         get_historical_station_reliability, get_next_station, get_station_by_code,
                         get_train_metadata, get_train_observations, get_train_state, search_stations)

__all__ = [
    "DEFAULT_DB_PATH", "DatabaseUnavailable", "connect", "database_ready", "open_connection", "resolve_db_path",
    "transaction", "SCHEMA_VERSION", "initialize_database", "ImportReport", "import_dataset", "import_stations",
    "AmbiguousRecord", "get_all_stations", "get_database_statistics", "get_historical_station_reliability",
    "get_next_station", "get_station_by_code", "get_train_metadata", "get_train_observations", "get_train_state",
    "search_stations",
]
