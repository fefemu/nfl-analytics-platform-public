"""Tests for the current NFL market board builder."""

from pathlib import Path

import duckdb
import pytest

from src.analytics.build_current_market_board import (
    build_current_market_board,
    validate_source_tables,
)


def create_market_board_database(
    database_file: Path,
) -> None:
    """Create minimal best-odds and bridge test data."""

    with duckdb.connect(str(database_file)) as connection:
        connection.execute("CREATE SCHEMA analytics")

        connection.execute(
            """
            CREATE TABLE analytics.best_odds_by_line (
                snapshot_id VARCHAR,
                fetched_at TIMESTAMPTZ,
                event_id VARCHAR,
                commence_time TIMESTAMPTZ,
                home_team VARCHAR,
                away_team VARCHAR,
                market_key VARCHAR,
                outcome_name VARCHAR,
                outcome_type VARCHAR,
                point DOUBLE,
                market_line DOUBLE,
                best_bookmaker_key VARCHAR,
                best_bookmaker_title VARCHAR,
                best_american_price INTEGER,
                best_decimal_odds DOUBLE,
                best_implied_probability DOUBLE,
                best_bookmaker_margin DOUBLE,
                best_bookmaker_no_vig_probability DOUBLE,
                bookmaker_count INTEGER,
                consensus_no_vig_probability DOUBLE,
                minimum_no_vig_probability DOUBLE,
                maximum_no_vig_probability DOUBLE,
                probability_dispersion DOUBLE,
                average_decimal_odds DOUBLE,
                decimal_price_improvement DOUBLE
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE analytics.odds_event_schedule_bridge (
                snapshot_id VARCHAR,
                odds_event_id VARCHAR,
                commence_time TIMESTAMPTZ,
                eastern_game_date DATE,
                odds_home_team VARCHAR,
                odds_away_team VARCHAR,
                home_team_code VARCHAR,
                away_team_code VARCHAR,
                game_id VARCHAR,
                season INTEGER,
                game_type VARCHAR,
                week INTEGER,
                gameday DATE,
                gametime TIME,
                match_status VARCHAR
            )
            """
        )

        connection.executemany(
            """
            INSERT INTO analytics.best_odds_by_line
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?
            )
            """,
            [
                (
                    "snapshot-old",
                    "2026-07-18T08:00:00Z",
                    "event-old",
                    "2026-09-10T00:20:00Z",
                    "Home Team",
                    "Away Team",
                    "h2h",
                    "Home Team",
                    "home",
                    None,
                    None,
                    "book-a",
                    "Book A",
                    -120,
                    1.8333333,
                    0.5455,
                    0.0455,
                    0.5220,
                    2,
                    0.5200,
                    0.5100,
                    0.5300,
                    0.0200,
                    1.8000,
                    0.0333333,
                ),
                (
                    "snapshot-latest",
                    "2026-07-19T08:00:00Z",
                    "event-latest",
                    "2026-09-17T00:20:00Z",
                    "Latest Home",
                    "Latest Away",
                    "h2h",
                    "Latest Home",
                    "home",
                    None,
                    None,
                    "book-b",
                    "Book B",
                    -105,
                    1.9523810,
                    0.5122,
                    0.0244,
                    0.5000,
                    3,
                    0.5050,
                    0.4950,
                    0.5150,
                    0.0200,
                    1.9000,
                    0.0523810,
                ),
            ],
        )

        connection.executemany(
            """
            INSERT INTO analytics.odds_event_schedule_bridge
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                (
                    "snapshot-old",
                    "event-old",
                    "2026-09-10T00:20:00Z",
                    "2026-09-09",
                    "Home Team",
                    "Away Team",
                    "BUF",
                    "NYJ",
                    "2026_01_NYJ_BUF",
                    2026,
                    "REG",
                    1,
                    "2026-09-09",
                    "20:20:00",
                    "MATCHED",
                ),
                (
                    "snapshot-latest",
                    "event-latest",
                    "2026-09-17T00:20:00Z",
                    "2026-09-16",
                    "Latest Home",
                    "Latest Away",
                    "PHI",
                    "DAL",
                    "2026_02_DAL_PHI",
                    2026,
                    "REG",
                    2,
                    "2026-09-16",
                    "20:20:00",
                    "MATCHED",
                ),
            ],
        )


def test_validate_source_tables_rejects_missing_tables() -> None:
    """Fail when market board sources are missing."""

    with duckdb.connect(":memory:") as connection:
        with pytest.raises(
            RuntimeError,
            match="Missing market board source tables",
        ):
            validate_source_tables(connection)


def test_build_market_board_selects_latest_snapshot(
    tmp_path: Path,
) -> None:
    """Expose only the latest matched market snapshot."""

    database_file = tmp_path / "test.duckdb"
    create_market_board_database(database_file)

    build_current_market_board(database_file)

    with duckdb.connect(str(database_file)) as connection:
        market_row = connection.execute(
            """
            SELECT
                snapshot_id,
                game_id,
                market_name,
                best_bookmaker_key,
                best_decimal_odds
            FROM analytics.current_market_board
            """
        ).fetchone()

        row_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM analytics.current_market_board
            """
        ).fetchone()[0]

    assert row_count == 1
    assert market_row == pytest.approx(
        (
            "snapshot-latest",
            "2026_02_DAL_PHI",
            "Moneyline",
            "book-b",
            1.9523810,
        )
    )


def test_failed_market_board_build_preserves_previous_table(
    tmp_path: Path,
) -> None:
    """Roll back a failed current market board replacement."""

    database_file = tmp_path / "test.duckdb"
    create_market_board_database(database_file)

    build_current_market_board(database_file)

    with duckdb.connect(str(database_file)) as connection:
        connection.execute(
            """
            UPDATE analytics.best_odds_by_line
            SET best_decimal_odds = 1.0
            WHERE snapshot_id = 'snapshot-latest'
            """
        )

    with pytest.raises(
        RuntimeError,
        match="invalid price or probability",
    ):
        build_current_market_board(database_file)

    with duckdb.connect(str(database_file)) as connection:
        preserved_price = connection.execute(
            """
            SELECT best_decimal_odds
            FROM analytics.current_market_board
            """
        ).fetchone()[0]

    assert preserved_price == pytest.approx(1.9523810)