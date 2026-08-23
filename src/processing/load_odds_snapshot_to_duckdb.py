"""
NFL Analytics Platform
Odds Snapshot DuckDB Loader

Purpose:
    Normalize a raw Odds API JSON snapshot into
    relational DuckDB tables.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import duckdb


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_FILE = PROJECT_ROOT / "data" / "nfl_analytics.duckdb"

TARGET_SCHEMA = "raw"

SNAPSHOTS_TABLE = "odds_snapshots"
EVENTS_TABLE = "odds_events"
MARKETS_TABLE = "odds_markets"

SNAPSHOTS_FULL_NAME = f"{TARGET_SCHEMA}.{SNAPSHOTS_TABLE}"
EVENTS_FULL_NAME = f"{TARGET_SCHEMA}.{EVENTS_TABLE}"
MARKETS_FULL_NAME = f"{TARGET_SCHEMA}.{MARKETS_TABLE}"


def validate_snapshot_file(snapshot_file: Path) -> None:
    """Validate that the supplied snapshot file exists."""

    if not snapshot_file.exists():
        raise FileNotFoundError(
            f"Odds snapshot file not found: {snapshot_file}"
        )

    if not snapshot_file.is_file():
        raise ValueError(
            f"Odds snapshot path is not a file: {snapshot_file}"
        )

    if snapshot_file.suffix.lower() != ".json":
        raise ValueError(
            "Odds snapshot file must use the .json extension."
        )


def load_snapshot_json(
    snapshot_file: Path,
) -> dict[str, Any]:
    """Load and validate the top-level snapshot structure."""

    validate_snapshot_file(snapshot_file)

    try:
        snapshot = json.loads(
            snapshot_file.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError:
        raise ValueError(
            f"Odds snapshot contains invalid JSON: {snapshot_file}"
        ) from None

    if not isinstance(snapshot, dict):
        raise ValueError(
            "Odds snapshot must contain a JSON object."
        )

    if not isinstance(snapshot.get("metadata"), dict):
        raise ValueError(
            "Odds snapshot is missing metadata."
        )

    if not isinstance(snapshot.get("events"), list):
        raise ValueError(
            "Odds snapshot is missing the events list."
        )

    return snapshot


def create_target_tables(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create the raw Odds API target tables."""

    connection.execute(
        f"CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA}"
    )

    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {SNAPSHOTS_FULL_NAME} (
            snapshot_id VARCHAR PRIMARY KEY,
            fetched_at TIMESTAMPTZ NOT NULL,
            sport_key VARCHAR NOT NULL,
            regions VARCHAR NOT NULL,
            markets VARCHAR[] NOT NULL,
            odds_format VARCHAR NOT NULL,
            event_count INTEGER NOT NULL,
            requests_remaining INTEGER,
            requests_used INTEGER,
            requests_last INTEGER,
            source_file VARCHAR NOT NULL,
            loaded_at TIMESTAMPTZ NOT NULL
                DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {EVENTS_FULL_NAME} (
            snapshot_id VARCHAR NOT NULL,
            event_id VARCHAR NOT NULL,
            sport_key VARCHAR NOT NULL,
            sport_title VARCHAR,
            commence_time TIMESTAMPTZ NOT NULL,
            home_team VARCHAR NOT NULL,
            away_team VARCHAR NOT NULL,
            PRIMARY KEY (snapshot_id, event_id)
        )
        """
    )

    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {MARKETS_FULL_NAME} (
            snapshot_id VARCHAR NOT NULL,
            event_id VARCHAR NOT NULL,
            bookmaker_key VARCHAR NOT NULL,
            bookmaker_title VARCHAR,
            bookmaker_last_update TIMESTAMPTZ,
            market_key VARCHAR NOT NULL,
            outcome_name VARCHAR NOT NULL,
            price INTEGER NOT NULL,
            point DOUBLE,
            PRIMARY KEY (
                snapshot_id,
                event_id,
                bookmaker_key,
                market_key,
                outcome_name
            )
        )
        """
    )

    logger.info(
        "Odds target tables validated: %s, %s and %s.",
        SNAPSHOTS_FULL_NAME,
        EVENTS_FULL_NAME,
        MARKETS_FULL_NAME,
    )


def require_field(
    record: dict[str, Any],
    field_name: str,
    context: str,
) -> Any:
    """Return a required field or fail with context."""

    value = record.get(field_name)

    if value is None:
        raise ValueError(
            f"Missing required field '{field_name}' in {context}."
        )

    return value


def build_normalized_records(
    snapshot: dict[str, Any],
    snapshot_file: Path,
) -> tuple[
    tuple[Any, ...],
    list[tuple[Any, ...]],
    list[tuple[Any, ...]],
]:
    """Transform nested snapshot JSON into relational records."""

    metadata = snapshot["metadata"]
    events = snapshot["events"]

    snapshot_id = snapshot_file.stem

    snapshot_record = (
        snapshot_id,
        require_field(
            metadata,
            "fetched_at",
            "snapshot metadata",
        ),
        require_field(
            metadata,
            "sport_key",
            "snapshot metadata",
        ),
        require_field(
            metadata,
            "regions",
            "snapshot metadata",
        ),
        require_field(
            metadata,
            "markets",
            "snapshot metadata",
        ),
        require_field(
            metadata,
            "odds_format",
            "snapshot metadata",
        ),
        len(events),
        metadata.get("requests_remaining"),
        metadata.get("requests_used"),
        metadata.get("requests_last"),
        str(snapshot_file.resolve()),
    )

    event_records: list[tuple[Any, ...]] = []
    market_records: list[tuple[Any, ...]] = []

    for event_index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(
                f"Event at index {event_index} must be an object."
            )

        event_context = f"event at index {event_index}"

        event_id = require_field(
            event,
            "id",
            event_context,
        )

        event_records.append(
            (
                snapshot_id,
                event_id,
                require_field(
                    event,
                    "sport_key",
                    event_context,
                ),
                event.get("sport_title"),
                require_field(
                    event,
                    "commence_time",
                    event_context,
                ),
                require_field(
                    event,
                    "home_team",
                    event_context,
                ),
                require_field(
                    event,
                    "away_team",
                    event_context,
                ),
            )
        )

        bookmakers = event.get("bookmakers", [])

        if not isinstance(bookmakers, list):
            raise ValueError(
                f"Bookmakers must be a list in {event_context}."
            )

        for bookmaker_index, bookmaker in enumerate(bookmakers):
            bookmaker_context = (
                f"bookmaker at index {bookmaker_index} "
                f"in {event_context}"
            )

            bookmaker_key = require_field(
                bookmaker,
                "key",
                bookmaker_context,
            )

            markets = bookmaker.get("markets", [])

            if not isinstance(markets, list):
                raise ValueError(
                    f"Markets must be a list in {bookmaker_context}."
                )

            for market_index, market in enumerate(markets):
                market_context = (
                    f"market at index {market_index} "
                    f"in {bookmaker_context}"
                )

                market_key = require_field(
                    market,
                    "key",
                    market_context,
                )
                outcomes = market.get("outcomes", [])

                if not isinstance(outcomes, list):
                    raise ValueError(
                        f"Outcomes must be a list in {market_context}."
                    )

                for outcome_index, outcome in enumerate(outcomes):
                    outcome_context = (
                        f"outcome at index {outcome_index} "
                        f"in {market_context}"
                    )

                    market_records.append(
                        (
                            snapshot_id,
                            event_id,
                            bookmaker_key,
                            bookmaker.get("title"),
                            bookmaker.get("last_update"),
                            market_key,
                            require_field(
                                outcome,
                                "name",
                                outcome_context,
                            ),
                            require_field(
                                outcome,
                                "price",
                                outcome_context,
                            ),
                            outcome.get("point"),
                        )
                    )

    return (
        snapshot_record,
        event_records,
        market_records,
    )


def validate_loaded_snapshot(
    connection: duckdb.DuckDBPyConnection,
    snapshot_id: str,
    expected_event_count: int,
    expected_market_count: int,
) -> None:
    """Validate the inserted snapshot row counts."""

    event_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {EVENTS_FULL_NAME}
        WHERE snapshot_id = ?
        """,
        [snapshot_id],
    ).fetchone()[0]

    market_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {MARKETS_FULL_NAME}
        WHERE snapshot_id = ?
        """,
        [snapshot_id],
    ).fetchone()[0]

    if event_count != expected_event_count:
        raise RuntimeError(
            "Loaded odds event count does not match "
            "the normalized record count."
        )

    if market_count != expected_market_count:
        raise RuntimeError(
            "Loaded odds market count does not match "
            "the normalized record count."
        )

    logger.info(
        "Loaded odds snapshot validated: "
        "%s events and %s market outcomes.",
        event_count,
        market_count,
    )


def load_odds_snapshot_to_duckdb(
    snapshot_file: Path,
    database_file: Path = DATABASE_FILE,
) -> tuple[str, int, int]:
    """Load one raw Odds API snapshot into DuckDB."""

    resolved_snapshot_file = (
        snapshot_file
        if snapshot_file.is_absolute()
        else PROJECT_ROOT / snapshot_file
    )

    if not database_file.exists():
        raise FileNotFoundError(
            f"DuckDB database not found: {database_file}"
        )

    snapshot = load_snapshot_json(
        resolved_snapshot_file
    )

    (
        snapshot_record,
        event_records,
        market_records,
    ) = build_normalized_records(
        snapshot=snapshot,
        snapshot_file=resolved_snapshot_file,
    )

    snapshot_id = snapshot_record[0]

    with duckdb.connect(str(database_file)) as connection:
        create_target_tables(connection)

        try:
            connection.execute("BEGIN TRANSACTION")

            connection.execute(
                f"""
                DELETE FROM {MARKETS_FULL_NAME}
                WHERE snapshot_id = ?
                """,
                [snapshot_id],
            )
            connection.execute(
                f"""
                DELETE FROM {EVENTS_FULL_NAME}
                WHERE snapshot_id = ?
                """,
                [snapshot_id],
            )
            connection.execute(
                f"""
                DELETE FROM {SNAPSHOTS_FULL_NAME}
                WHERE snapshot_id = ?
                """,
                [snapshot_id],
            )

            connection.execute(
                f"""
                INSERT INTO {SNAPSHOTS_FULL_NAME} (
                    snapshot_id,
                    fetched_at,
                    sport_key,
                    regions,
                    markets,
                    odds_format,
                    event_count,
                    requests_remaining,
                    requests_used,
                    requests_last,
                    source_file
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                snapshot_record,
            )

            if event_records:
                connection.executemany(
                    f"""
                    INSERT INTO {EVENTS_FULL_NAME} (
                        snapshot_id,
                        event_id,
                        sport_key,
                        sport_title,
                        commence_time,
                        home_team,
                        away_team
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    event_records,
                )

            if market_records:
                connection.executemany(
                    f"""
                    INSERT INTO {MARKETS_FULL_NAME} (
                        snapshot_id,
                        event_id,
                        bookmaker_key,
                        bookmaker_title,
                        bookmaker_last_update,
                        market_key,
                        outcome_name,
                        price,
                        point
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    market_records,
                )

            validate_loaded_snapshot(
                connection=connection,
                snapshot_id=snapshot_id,
                expected_event_count=len(event_records),
                expected_market_count=len(market_records),
            )

            connection.execute("COMMIT")

        except Exception:
            connection.execute("ROLLBACK")
            logger.exception(
                "Odds snapshot transaction rolled back."
            )
            raise

    logger.info(
        "Odds snapshot loaded into DuckDB: "
        "%s, %s events and %s market outcomes.",
        snapshot_id,
        len(event_records),
        len(market_records),
    )

    return (
        snapshot_id,
        len(event_records),
        len(market_records),
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Load a raw Odds API JSON snapshot into DuckDB."
        )
    )

    parser.add_argument(
        "snapshot_file",
        type=Path,
        help=(
            "Path to an Odds API snapshot relative "
            "to the project root."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run the odds snapshot DuckDB load workflow."""

    args = parse_arguments()

    try:
        load_odds_snapshot_to_duckdb(
            snapshot_file=args.snapshot_file,
        )
    except Exception:
        logger.exception(
            "Odds snapshot DuckDB load failed."
        )
        raise


if __name__ == "__main__":
    main()