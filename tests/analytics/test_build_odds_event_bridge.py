"""Tests for the Odds API event schedule bridge."""

from pathlib import Path

import duckdb
import pytest

from src.analytics.build_odds_event_bridge import (
    build_odds_event_bridge,
    validate_source_tables,
)


def create_event_bridge_database(
    database_file: Path,
) -> None:
    """Create minimal raw odds and processed schedule data."""

    with duckdb.connect(str(database_file)) as connection:
        connection.execute("CREATE SCHEMA raw")
        connection.execute("CREATE SCHEMA processed")

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
            CREATE TABLE processed.schedule (
                game_id VARCHAR NOT NULL,
                season INTEGER NOT NULL,
                game_type VARCHAR NOT NULL,
                week INTEGER NOT NULL,
                gameday DATE NOT NULL,
                gametime TIME,
                home_team VARCHAR NOT NULL,
                away_team VARCHAR NOT NULL
            )
            """
        )

        connection.execute(
            """
            INSERT INTO raw.odds_events
            VALUES (
                'snapshot-1',
                'odds-event-1',
                'americanfootball_nfl',
                'NFL',
                '2026-09-15T00:15:00Z',
                'Buffalo Bills',
                'New York Jets'
            )
            """
        )

        connection.execute(
            """
            INSERT INTO processed.schedule
            VALUES (
                '2026_01_NYJ_BUF',
                2026,
                'REG',
                1,
                '2026-09-14',
                '20:15:00',
                'BUF',
                'NYJ'
            )
            """
        )


def test_validate_source_tables_rejects_missing_tables() -> None:
    """Fail when bridge source tables are missing."""

    with duckdb.connect(":memory:") as connection:
        with pytest.raises(
            RuntimeError,
            match="Missing event bridge source tables",
        ):
            validate_source_tables(connection)


def test_build_event_bridge_matches_eastern_game_date(
    tmp_path: Path,
) -> None:
    """Match a UTC Tuesday event to a Monday NFL game."""

    database_file = tmp_path / "test.duckdb"
    create_event_bridge_database(database_file)

    build_odds_event_bridge(database_file)

    with duckdb.connect(str(database_file)) as connection:
        bridge_row = connection.execute(
            """
            SELECT
                odds_event_id,
                eastern_game_date,
                home_team_code,
                away_team_code,
                game_id,
                match_status
            FROM analytics.odds_event_schedule_bridge
            """
        ).fetchone()

    assert bridge_row == (
        "odds-event-1",
        duckdb.execute(
            "SELECT DATE '2026-09-14'"
        ).fetchone()[0],
        "BUF",
        "NYJ",
        "2026_01_NYJ_BUF",
        "MATCHED",
    )


def test_build_event_bridge_rejects_unmatched_game(
    tmp_path: Path,
) -> None:
    """Fail when an Odds API event has no schedule match."""

    database_file = tmp_path / "test.duckdb"
    create_event_bridge_database(database_file)

    with duckdb.connect(str(database_file)) as connection:
        connection.execute(
            """
            UPDATE processed.schedule
            SET gameday = '2026-09-15'
            """
        )

    with pytest.raises(
        RuntimeError,
        match="without a schedule match",
    ):
        build_odds_event_bridge(database_file)


def test_failed_bridge_build_preserves_previous_table(
    tmp_path: Path,
) -> None:
    """Roll back a failed replacement of the bridge."""

    database_file = tmp_path / "test.duckdb"
    create_event_bridge_database(database_file)

    build_odds_event_bridge(database_file)

    with duckdb.connect(str(database_file)) as connection:
        connection.execute(
            """
            UPDATE processed.schedule
            SET gameday = '2026-09-15'
            """
        )

    with pytest.raises(
        RuntimeError,
        match="without a schedule match",
    ):
        build_odds_event_bridge(database_file)

    with duckdb.connect(str(database_file)) as connection:
        preserved_game_id = connection.execute(
            """
            SELECT game_id
            FROM analytics.odds_event_schedule_bridge
            """
        ).fetchone()[0]

    assert preserved_game_id == "2026_01_NYJ_BUF"