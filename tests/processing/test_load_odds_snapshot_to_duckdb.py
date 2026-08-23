"""Tests for the Odds API snapshot DuckDB loader."""

import json
from pathlib import Path

import duckdb
import pytest

from src.processing.load_odds_snapshot_to_duckdb import (
    build_normalized_records,
    load_odds_snapshot_to_duckdb,
    load_snapshot_json,
)


def write_test_snapshot(
    snapshot_file: Path,
    duplicate_outcome: bool = False,
) -> None:
    """Write a minimal Odds API snapshot for testing."""

    h2h_outcomes = [
        {
            "name": "Away Team",
            "price": 120,
        },
        {
            "name": "Home Team",
            "price": -140,
        },
    ]

    if duplicate_outcome:
        h2h_outcomes.append(
            {
                "name": "Home Team",
                "price": -145,
            }
        )

    snapshot = {
        "metadata": {
            "fetched_at": "2026-07-19T08:50:44+00:00",
            "sport_key": "americanfootball_nfl",
            "regions": "us",
            "markets": [
                "h2h",
                "spreads",
                "totals",
            ],
            "odds_format": "american",
            "event_count": 1,
            "requests_remaining": 497,
            "requests_used": 3,
            "requests_last": 3,
        },
        "events": [
            {
                "id": "event-1",
                "sport_key": "americanfootball_nfl",
                "sport_title": "NFL",
                "commence_time": "2026-09-10T00:20:00Z",
                "home_team": "Home Team",
                "away_team": "Away Team",
                "bookmakers": [
                    {
                        "key": "testbook",
                        "title": "Test Book",
                        "last_update": (
                            "2026-07-19T08:49:00Z"
                        ),
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": h2h_outcomes,
                            },
                            {
                                "key": "spreads",
                                "outcomes": [
                                    {
                                        "name": "Away Team",
                                        "price": -110,
                                        "point": 3.5,
                                    },
                                    {
                                        "name": "Home Team",
                                        "price": -110,
                                        "point": -3.5,
                                    },
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {
                                        "name": "Over",
                                        "price": -105,
                                        "point": 47.5,
                                    },
                                    {
                                        "name": "Under",
                                        "price": -115,
                                        "point": 47.5,
                                    },
                                ],
                            },
                        ],
                    }
                ],
            }
        ],
    }

    snapshot_file.write_text(
        json.dumps(snapshot),
        encoding="utf-8",
    )


def create_empty_database(database_file: Path) -> None:
    """Create an empty temporary DuckDB database."""

    with duckdb.connect(str(database_file)):
        pass


def test_load_snapshot_json_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """Fail when the snapshot file does not exist."""

    missing_file = tmp_path / "missing.json"

    with pytest.raises(
        FileNotFoundError,
        match="Odds snapshot file not found",
    ):
        load_snapshot_json(missing_file)


def test_load_snapshot_json_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    """Fail when the snapshot is not valid JSON."""

    snapshot_file = tmp_path / "invalid.json"
    snapshot_file.write_text(
        "{invalid-json",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="contains invalid JSON",
    ):
        load_snapshot_json(snapshot_file)


def test_build_normalized_records_creates_expected_rows(
    tmp_path: Path,
) -> None:
    """Flatten the snapshot into relational records."""

    snapshot_file = tmp_path / "test_snapshot.json"
    write_test_snapshot(snapshot_file)

    snapshot = load_snapshot_json(snapshot_file)

    (
        snapshot_record,
        event_records,
        market_records,
    ) = build_normalized_records(
        snapshot=snapshot,
        snapshot_file=snapshot_file,
    )

    assert snapshot_record[0] == "test_snapshot"
    assert snapshot_record[6] == 1
    assert len(event_records) == 1
    assert len(market_records) == 6

    assert event_records[0][1] == "event-1"
    assert market_records[0][2] == "testbook"
    assert market_records[0][5] == "h2h"


def test_load_odds_snapshot_creates_raw_tables(
    tmp_path: Path,
) -> None:
    """Load a normalized snapshot into temporary DuckDB tables."""

    snapshot_file = tmp_path / "test_snapshot.json"
    database_file = tmp_path / "test.duckdb"

    write_test_snapshot(snapshot_file)
    create_empty_database(database_file)

    result = load_odds_snapshot_to_duckdb(
        snapshot_file=snapshot_file,
        database_file=database_file,
    )

    assert result == (
        "test_snapshot",
        1,
        6,
    )

    with duckdb.connect(str(database_file)) as connection:
        snapshot_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM raw.odds_snapshots
            """
        ).fetchone()[0]

        event_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM raw.odds_events
            """
        ).fetchone()[0]

        market_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM raw.odds_markets
            """
        ).fetchone()[0]

    assert snapshot_count == 1
    assert event_count == 1
    assert market_count == 6


def test_repeated_snapshot_load_is_idempotent(
    tmp_path: Path,
) -> None:
    """Replace an existing snapshot without duplicating rows."""

    snapshot_file = tmp_path / "test_snapshot.json"
    database_file = tmp_path / "test.duckdb"

    write_test_snapshot(snapshot_file)
    create_empty_database(database_file)

    load_odds_snapshot_to_duckdb(
        snapshot_file=snapshot_file,
        database_file=database_file,
    )
    load_odds_snapshot_to_duckdb(
        snapshot_file=snapshot_file,
        database_file=database_file,
    )

    with duckdb.connect(str(database_file)) as connection:
        counts = connection.execute(
            """
            SELECT
                (
                    SELECT COUNT(*)
                    FROM raw.odds_snapshots
                ),
                (
                    SELECT COUNT(*)
                    FROM raw.odds_events
                ),
                (
                    SELECT COUNT(*)
                    FROM raw.odds_markets
                )
            """
        ).fetchone()

    assert counts == (
        1,
        1,
        6,
    )


def test_failed_reload_preserves_previous_snapshot(
    tmp_path: Path,
) -> None:
    """Roll back a failed replacement of an existing snapshot."""

    snapshot_file = tmp_path / "test_snapshot.json"
    database_file = tmp_path / "test.duckdb"

    write_test_snapshot(snapshot_file)
    create_empty_database(database_file)

    load_odds_snapshot_to_duckdb(
        snapshot_file=snapshot_file,
        database_file=database_file,
    )

    write_test_snapshot(
        snapshot_file,
        duplicate_outcome=True,
    )

    with pytest.raises(duckdb.ConstraintException):
        load_odds_snapshot_to_duckdb(
            snapshot_file=snapshot_file,
            database_file=database_file,
        )

    with duckdb.connect(str(database_file)) as connection:
        market_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM raw.odds_markets
            WHERE snapshot_id = 'test_snapshot'
            """
        ).fetchone()[0]

    assert market_count == 6
