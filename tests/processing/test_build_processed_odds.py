"""Tests for the processed NFL odds builder."""

from pathlib import Path

import duckdb
import pytest

from src.processing.build_processed_odds import (
    build_processed_odds,
    validate_source_tables,
)


def create_raw_odds_database(
    database_file: Path,
) -> None:
    """Create minimal raw odds tables and test records."""

    with duckdb.connect(str(database_file)) as connection:
        connection.execute("CREATE SCHEMA raw")

        connection.execute(
            """
            CREATE TABLE raw.odds_snapshots (
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
                loaded_at TIMESTAMPTZ
                    DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE raw.odds_events (
                snapshot_id VARCHAR NOT NULL,
                event_id VARCHAR NOT NULL,
                sport_key VARCHAR NOT NULL,
                sport_title VARCHAR,
                commence_time TIMESTAMPTZ NOT NULL,
                home_team VARCHAR NOT NULL,
                away_team VARCHAR NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE raw.odds_markets (
                snapshot_id VARCHAR NOT NULL,
                event_id VARCHAR NOT NULL,
                bookmaker_key VARCHAR NOT NULL,
                bookmaker_title VARCHAR,
                bookmaker_last_update TIMESTAMPTZ,
                market_key VARCHAR NOT NULL,
                outcome_name VARCHAR NOT NULL,
                price INTEGER NOT NULL,
                point DOUBLE
            )
            """
        )

        connection.execute(
            """
            INSERT INTO raw.odds_snapshots (
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
            VALUES (
                'snapshot-1',
                '2026-07-19T08:50:44Z',
                'americanfootball_nfl',
                'us',
                ['h2h', 'spreads', 'totals'],
                'american',
                1,
                497,
                3,
                3,
                'test_snapshot.json'
            )
            """
        )

        connection.execute(
            """
            INSERT INTO raw.odds_events
            VALUES (
                'snapshot-1',
                'event-1',
                'americanfootball_nfl',
                'NFL',
                '2026-09-10T00:20:00Z',
                'Home Team',
                'Away Team'
            )
            """
        )

        connection.executemany(
            """
            INSERT INTO raw.odds_markets
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "snapshot-1",
                    "event-1",
                    "testbook",
                    "Test Book",
                    "2026-07-19T08:49:00Z",
                    "h2h",
                    "Home Team",
                    -150,
                    None,
                ),
                (
                    "snapshot-1",
                    "event-1",
                    "testbook",
                    "Test Book",
                    "2026-07-19T08:49:00Z",
                    "h2h",
                    "Away Team",
                    130,
                    None,
                ),
                (
                    "snapshot-1",
                    "event-1",
                    "testbook",
                    "Test Book",
                    "2026-07-19T08:49:00Z",
                    "spreads",
                    "Home Team",
                    -110,
                    -3.5,
                ),
                (
                    "snapshot-1",
                    "event-1",
                    "testbook",
                    "Test Book",
                    "2026-07-19T08:49:00Z",
                    "spreads",
                    "Away Team",
                    -110,
                    3.5,
                ),
                (
                    "snapshot-1",
                    "event-1",
                    "testbook",
                    "Test Book",
                    "2026-07-19T08:49:00Z",
                    "totals",
                    "Over",
                    -105,
                    47.5,
                ),
                (
                    "snapshot-1",
                    "event-1",
                    "testbook",
                    "Test Book",
                    "2026-07-19T08:49:00Z",
                    "totals",
                    "Under",
                    -115,
                    47.5,
                ),
            ],
        )


def test_validate_source_tables_rejects_missing_tables() -> None:
    """Fail when the raw odds tables do not exist."""

    with duckdb.connect(":memory:") as connection:
        with pytest.raises(
            RuntimeError,
            match="Missing raw odds source tables",
        ):
            validate_source_tables(connection)


def test_build_processed_odds_rejects_missing_database(
    tmp_path: Path,
) -> None:
    """Fail when the DuckDB database does not exist."""

    missing_database = tmp_path / "missing.duckdb"

    with pytest.raises(
        FileNotFoundError,
        match="DuckDB database not found",
    ):
        build_processed_odds(missing_database)


def test_build_processed_odds_calculates_probabilities(
    tmp_path: Path,
) -> None:
    """Calculate decimal, implied and no-vig probabilities."""

    database_file = tmp_path / "test.duckdb"
    create_raw_odds_database(database_file)

    build_processed_odds(database_file)

    with duckdb.connect(str(database_file)) as connection:
        home_row = connection.execute(
            """
            SELECT
                outcome_type,
                decimal_odds,
                implied_probability,
                bookmaker_margin,
                no_vig_probability
            FROM processed.odds_market_outcomes
            WHERE market_key = 'h2h'
              AND outcome_type = 'home'
            """
        ).fetchone()

    assert home_row[0] == "home"
    assert home_row[1] == pytest.approx(
        1.6666666667
    )
    assert home_row[2] == pytest.approx(0.6)
    assert home_row[3] == pytest.approx(
        0.0347826087
    )
    assert home_row[4] == pytest.approx(
        0.5798319328
    )


def test_no_vig_probabilities_sum_to_one(
    tmp_path: Path,
) -> None:
    """Normalize each bookmaker market to probability one."""

    database_file = tmp_path / "test.duckdb"
    create_raw_odds_database(database_file)

    build_processed_odds(database_file)

    with duckdb.connect(str(database_file)) as connection:
        probability_sums = connection.execute(
            """
            SELECT
                market_key,
                SUM(no_vig_probability)
            FROM processed.odds_market_outcomes
            GROUP BY market_key
            ORDER BY market_key
            """
        ).fetchall()

    assert probability_sums == pytest.approx(
        [
            ("h2h", 1.0),
            ("spreads", 1.0),
            ("totals", 1.0),
        ]
    )


def test_failed_build_preserves_previous_processed_table(
    tmp_path: Path,
) -> None:
    """Roll back a failed replacement of processed odds."""

    database_file = tmp_path / "test.duckdb"
    create_raw_odds_database(database_file)

    build_processed_odds(database_file)

    with duckdb.connect(str(database_file)) as connection:
        connection.execute(
            """
            UPDATE raw.odds_markets
            SET price = 50
            WHERE market_key = 'h2h'
              AND outcome_name = 'Home Team'
            """
        )

    with pytest.raises(
        RuntimeError,
        match="invalid probability or price",
    ):
        build_processed_odds(database_file)

    with duckdb.connect(str(database_file)) as connection:
        preserved_price = connection.execute(
            """
            SELECT american_price
            FROM processed.odds_market_outcomes
            WHERE market_key = 'h2h'
              AND outcome_type = 'home'
            """
        ).fetchone()[0]

    assert preserved_price == -150