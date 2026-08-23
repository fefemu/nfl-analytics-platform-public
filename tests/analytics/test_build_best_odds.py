"""Tests for the best available NFL odds builder."""

from pathlib import Path

import duckdb
import pytest

from src.analytics.build_best_odds import (
    build_best_odds,
    validate_source_table,
)


def create_processed_odds_database(
    database_file: Path,
) -> None:
    """Create minimal processed odds test data."""

    with duckdb.connect(str(database_file)) as connection:
        connection.execute("CREATE SCHEMA processed")

        connection.execute(
            """
            CREATE TABLE processed.odds_market_outcomes (
                snapshot_id VARCHAR,
                fetched_at TIMESTAMPTZ,
                event_id VARCHAR,
                commence_time TIMESTAMPTZ,
                home_team VARCHAR,
                away_team VARCHAR,
                bookmaker_key VARCHAR,
                bookmaker_title VARCHAR,
                bookmaker_last_update TIMESTAMPTZ,
                market_key VARCHAR,
                outcome_name VARCHAR,
                outcome_type VARCHAR,
                american_price INTEGER,
                point DOUBLE,
                market_line DOUBLE,
                decimal_odds DOUBLE,
                implied_probability DOUBLE,
                bookmaker_margin DOUBLE,
                no_vig_probability DOUBLE
            )
            """
        )

        connection.executemany(
            """
            INSERT INTO processed.odds_market_outcomes
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                (
                    "snapshot-1",
                    "2026-07-19T08:50:44Z",
                    "event-1",
                    "2026-09-10T00:20:00Z",
                    "Home Team",
                    "Away Team",
                    "book-a",
                    "Book A",
                    "2026-07-19T08:49:00Z",
                    "h2h",
                    "Home Team",
                    "home",
                    -150,
                    None,
                    None,
                    1.6666667,
                    0.6000,
                    0.0348,
                    0.5800,
                ),
                (
                    "snapshot-1",
                    "2026-07-19T08:50:44Z",
                    "event-1",
                    "2026-09-10T00:20:00Z",
                    "Home Team",
                    "Away Team",
                    "book-a",
                    "Book A",
                    "2026-07-19T08:49:00Z",
                    "h2h",
                    "Away Team",
                    "away",
                    130,
                    None,
                    None,
                    2.3000,
                    0.4348,
                    0.0348,
                    0.4200,
                ),
                (
                    "snapshot-1",
                    "2026-07-19T08:50:44Z",
                    "event-1",
                    "2026-09-10T00:20:00Z",
                    "Home Team",
                    "Away Team",
                    "book-b",
                    "Book B",
                    "2026-07-19T08:49:10Z",
                    "h2h",
                    "Home Team",
                    "home",
                    -140,
                    None,
                    None,
                    1.7142857,
                    0.5833,
                    0.0277,
                    0.5700,
                ),
                (
                    "snapshot-1",
                    "2026-07-19T08:50:44Z",
                    "event-1",
                    "2026-09-10T00:20:00Z",
                    "Home Team",
                    "Away Team",
                    "book-b",
                    "Book B",
                    "2026-07-19T08:49:10Z",
                    "h2h",
                    "Away Team",
                    "away",
                    125,
                    None,
                    None,
                    2.2500,
                    0.4444,
                    0.0277,
                    0.4300,
                ),
                (
                    "snapshot-1",
                    "2026-07-19T08:50:44Z",
                    "event-1",
                    "2026-09-10T00:20:00Z",
                    "Home Team",
                    "Away Team",
                    "book-a",
                    "Book A",
                    "2026-07-19T08:49:00Z",
                    "spreads",
                    "Home Team",
                    "home",
                    -110,
                    -3.5,
                    3.5,
                    1.9090909,
                    0.5238,
                    0.0476,
                    0.5000,
                ),
                (
                    "snapshot-1",
                    "2026-07-19T08:50:44Z",
                    "event-1",
                    "2026-09-10T00:20:00Z",
                    "Home Team",
                    "Away Team",
                    "book-b",
                    "Book B",
                    "2026-07-19T08:49:10Z",
                    "spreads",
                    "Home Team",
                    "home",
                    -115,
                    -4.0,
                    4.0,
                    1.8695652,
                    0.5349,
                    0.0698,
                    0.5000,
                ),
            ],
        )


def test_validate_source_table_rejects_missing_table() -> None:
    """Fail when the processed odds source is missing."""

    with duckdb.connect(":memory:") as connection:
        with pytest.raises(
            RuntimeError,
            match="Source table does not exist",
        ):
            validate_source_table(connection)


def test_build_best_odds_selects_highest_price(
    tmp_path: Path,
) -> None:
    """Select the largest decimal odds for an equal line."""

    database_file = tmp_path / "test.duckdb"
    create_processed_odds_database(database_file)

    build_best_odds(database_file)

    with duckdb.connect(str(database_file)) as connection:
        home_offer = connection.execute(
            """
            SELECT
                best_bookmaker_key,
                best_american_price,
                best_decimal_odds,
                bookmaker_count,
                consensus_no_vig_probability,
                decimal_price_improvement
            FROM analytics.best_odds_by_line
            WHERE market_key = 'h2h'
              AND outcome_type = 'home'
            """
        ).fetchone()

    assert home_offer[0] == "book-b"
    assert home_offer[1] == -140
    assert home_offer[2] == pytest.approx(1.7142857)
    assert home_offer[3] == 2
    assert home_offer[4] == pytest.approx(0.575)
    assert home_offer[5] > 0.0


def test_build_best_odds_keeps_different_lines_separate(
    tmp_path: Path,
) -> None:
    """Do not compare prices attached to different spreads."""

    database_file = tmp_path / "test.duckdb"
    create_processed_odds_database(database_file)

    build_best_odds(database_file)

    with duckdb.connect(str(database_file)) as connection:
        spread_lines = connection.execute(
            """
            SELECT point
            FROM analytics.best_odds_by_line
            WHERE market_key = 'spreads'
              AND outcome_type = 'home'
            ORDER BY point
            """
        ).fetchall()

    assert spread_lines == [
        (-4.0,),
        (-3.5,),
    ]


def test_failed_build_preserves_previous_best_odds(
    tmp_path: Path,
) -> None:
    """Roll back a failed replacement of best odds."""

    database_file = tmp_path / "test.duckdb"
    create_processed_odds_database(database_file)

    build_best_odds(database_file)

    with duckdb.connect(str(database_file)) as connection:
        connection.execute(
            """
            UPDATE processed.odds_market_outcomes
            SET no_vig_probability = 1.5
            WHERE market_key = 'h2h'
              AND outcome_type = 'home'
            """
        )

    with pytest.raises(
        RuntimeError,
        match="invalid consensus probability",
    ):
        build_best_odds(database_file)

    with duckdb.connect(str(database_file)) as connection:
        preserved_bookmaker = connection.execute(
            """
            SELECT best_bookmaker_key
            FROM analytics.best_odds_by_line
            WHERE market_key = 'h2h'
              AND outcome_type = 'home'
            """
        ).fetchone()[0]

    assert preserved_bookmaker == "book-b"