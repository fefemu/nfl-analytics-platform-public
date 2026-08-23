"""Tests for the current season simulation runner."""

import duckdb
import pandas as pd
import pytest

from src.simulation.run_current_season_simulation import (
    evaluate_starting_rating_dispersion,
    load_current_team_records,
    load_latest_regular_season_schedule,
    validate_prediction_source,
    validate_simulation_schedule,
)


@pytest.fixture
def connection() -> duckdb.DuckDBPyConnection:
    """Create an in-memory simulation source."""

    database = duckdb.connect(":memory:")

    database.execute(
        """
        CREATE SCHEMA analytics;
        CREATE SCHEMA processed;

        CREATE TABLE processed.schedule (
            game_id VARCHAR,
            season INTEGER,
            game_type VARCHAR,
            home_team VARCHAR,
            away_team VARCHAR,
            home_score INTEGER,
            away_score INTEGER,
            is_completed BOOLEAN
        );

        CREATE TABLE analytics.current_game_predictions (
            game_id VARCHAR,
            season INTEGER,
            game_type VARCHAR,
            week INTEGER,
            gameday DATE,
            gametime VARCHAR,
            home_team VARCHAR,
            away_team VARCHAR,
            is_neutral BOOLEAN,
            home_win_probability DOUBLE,
            home_rating_pregame DOUBLE,
            away_rating_pregame DOUBLE
        );

        CREATE TABLE processed.external_nfelo_game_ratings (
            source_season INTEGER,
            source_week INTEGER,
            home_team VARCHAR,
            away_team VARCHAR,
            starting_nfelo_home DOUBLE,
            starting_nfelo_away DOUBLE
        );

        INSERT INTO processed.external_nfelo_game_ratings
        VALUES (2026, 1, 'NE', 'NYJ', 1600.0, 1400.0);

        INSERT INTO analytics.current_game_predictions
        VALUES
            (
                '2025_01_NE_NYJ',
                2025,
                'REG',
                1,
                DATE '2025-09-10',
                '20:20',
                'NE',
                'NYJ',
                FALSE,
                0.60,
                1510.0,
                1490.0
            ),
            (
                '2026_01_NE_NYJ',
                2026,
                'REG',
                1,
                DATE '2026-09-10',
                '20:20',
                'NE',
                'NYJ',
                FALSE,
                0.70,
                1550.0,
                1450.0
            ),
            (
                '2026_02_NYJ_NE',
                2026,
                'REG',
                2,
                DATE '2026-09-17',
                '20:20',
                'NYJ',
                'NE',
                FALSE,
                0.30,
                1450.0,
                1550.0
            ),
            (
                '2026_WC_NE_BUF',
                2026,
                'POST',
                19,
                DATE '2027-01-10',
                '20:20',
                'NE',
                'BUF',
                FALSE,
                0.55,
                1550.0,
                1520.0
            );
        INSERT INTO processed.schedule
        VALUES
            (
                '2026_COMPLETED_NE_NYJ',
                2026,
                'REG',
                'NE',
                'NYJ',
                24,
                17,
                TRUE
            ),
            (
                '2026_COMPLETED_BUF_KC',
                2026,
                'REG',
                'BUF',
                'KC',
                20,
                20,
                TRUE
            ),
            (
                '2026_UNPLAYED_NE_BUF',
                2026,
                'REG',
                'NE',
                'BUF',
                NULL,
                NULL,
                FALSE
            );
        """
    )

    yield database

    database.close()


def test_validate_prediction_source(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Accept the complete prediction source."""

    validate_prediction_source(connection)


def test_load_latest_regular_season_schedule(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Load only the latest regular season."""

    schedule = load_latest_regular_season_schedule(
        connection
    )

    assert list(schedule["game_id"]) == [
        "2026_01_NE_NYJ",
        "2026_02_NYJ_NE",
    ]

    assert set(schedule["season"]) == {
        2026
    }

    assert set(schedule["game_type"]) == {
        "REG"
    }


def test_starting_rating_dispersion_reports_compression(
    connection: duckdb.DuckDBPyConnection,
    caplog: pytest.LogCaptureFixture,
) -> None:
    schedule = load_latest_regular_season_schedule(connection)
    metrics = evaluate_starting_rating_dispersion(
        schedule,
        warning_threshold=150.0,
    )

    assert metrics["team_count"] == 2.0
    assert metrics["rating_standard_deviation"] == pytest.approx(100.0)
    assert "unusually compressed" in caplog.text


def test_starting_rating_dispersion_rejects_invalid_threshold(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    schedule = load_latest_regular_season_schedule(connection)
    with pytest.raises(ValueError, match="positive"):
        evaluate_starting_rating_dispersion(schedule, warning_threshold=0.0)


def test_validate_schedule_rejects_duplicates() -> None:
    """Reject duplicate scheduled games."""

    schedule = pd.DataFrame(
        [
            {
                "game_id": "duplicate",
                "season": 2026,
                "game_type": "REG",
                "week": 1,
                "gameday": pd.Timestamp(
                    "2026-09-10"
                ),
                "gametime": "20:20",
                "home_team": "NE",
                "away_team": "NYJ",
                "is_neutral": False,
                "home_rating_pregame": 1550.0,
                "away_rating_pregame": 1450.0,
            },
            {
                "game_id": "duplicate",
                "season": 2026,
                "game_type": "REG",
                "week": 2,
                "gameday": pd.Timestamp(
                    "2026-09-17"
                ),
                "gametime": "20:20",
                "home_team": "NYJ",
                "away_team": "NE",
                "is_neutral": False,
                "home_rating_pregame": 1450.0,
                "away_rating_pregame": 1550.0,
            },
        ]
    )

    with pytest.raises(
        ValueError,
        match="duplicate game identifiers",
    ):
        validate_simulation_schedule(schedule)


def test_validate_source_rejects_missing_table() -> None:
    """Reject a missing prediction source."""

    database = duckdb.connect(":memory:")

    with pytest.raises(
        RuntimeError,
        match="does not exist",
    ):
        validate_prediction_source(database)

    database.close()



def test_load_current_team_records(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Aggregate completed wins, losses, and ties."""

    records = load_current_team_records(
        connection=connection,
        season=2026,
    )

    records_by_team = records.set_index(
        "team"
    ).to_dict(orient="index")

    assert records_by_team["NE"] == {
        "wins": 1,
        "losses": 0,
        "ties": 0,
    }

    assert records_by_team["NYJ"] == {
        "wins": 0,
        "losses": 1,
        "ties": 0,
    }

    assert records_by_team["BUF"] == {
        "wins": 0,
        "losses": 0,
        "ties": 1,
    }

    assert records_by_team["KC"] == {
        "wins": 0,
        "losses": 0,
        "ties": 1,
    }
