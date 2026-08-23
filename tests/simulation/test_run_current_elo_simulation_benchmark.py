"""Tests for the current Elo benchmark runner."""

from pathlib import Path

import duckdb

from src.simulation.run_current_elo_simulation_benchmark import (
    run_current_elo_simulation_benchmark,
)
from src.simulation.run_elo_monte_carlo import (
    DYNAMIC_ELO_MODE,
    FROZEN_ELO_MODE,
)


def create_database(
    database_file: Path,
) -> None:
    """Create a minimal current simulation database."""

    with duckdb.connect(
        str(database_file)
    ) as connection:
        connection.execute(
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
                );
            """
        )


def test_run_current_paired_benchmark(
    tmp_path: Path,
) -> None:
    """Load current inputs and run both Elo modes."""

    database_file = (
        tmp_path
        / "simulation.duckdb"
    )

    create_database(database_file)

    result = (
        run_current_elo_simulation_benchmark(
            database_file=database_file,
            simulation_count=100,
            random_seed=42,
        )
    )

    assert (
        result.dynamic_result.simulation_mode
        == DYNAMIC_ELO_MODE
    )

    assert (
        result.frozen_result.simulation_mode
        == FROZEN_ELO_MODE
    )

    assert (
        result.dynamic_result.random_seed
        == result.frozen_result.random_seed
        == 42
    )

    assert set(
        result.team_comparison["team"]
    ) == {
        "NE",
        "NYJ",
    }
