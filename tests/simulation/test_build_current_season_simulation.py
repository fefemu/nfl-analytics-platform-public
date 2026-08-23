"""Tests for persisted season simulation outputs."""

from datetime import datetime

import duckdb
import pandas as pd
import pytest

from src.simulation.build_current_season_simulation import (
    DISTRIBUTION_FULL_NAME,
    SUMMARY_FULL_NAME,
    create_simulation_tables,
    prepare_simulation_tables,
    validate_simulation_tables,
)
from src.simulation.run_elo_monte_carlo import (
    DYNAMIC_ELO_MODE,
    INTERNAL_ELO_PROBABILITY_SOURCE,
    run_elo_monte_carlo,
)


def create_schedule() -> pd.DataFrame:
    """Create a small simulation schedule."""

    return pd.DataFrame(
        [
            {
                "game_id": "game_1",
                "season": 2026,
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
                "game_id": "game_2",
                "season": 2026,
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


def create_result():
    """Create a deterministic Monte Carlo result."""

    return run_elo_monte_carlo(
        schedule=create_schedule(),
        simulation_count=100,
        random_seed=42,
    )


def test_prepare_simulation_tables() -> None:
    """Add stable model and run metadata."""

    generated_at = datetime(
        2026,
        8,
        2,
        14,
        0,
        0,
    )

    summary, distribution = (
        prepare_simulation_tables(
            result=create_result(),
            generated_at=generated_at,
        )
    )

    assert set(summary["season"]) == {
        2026
    }
    assert set(summary["simulation_count"]) == {
        100
    }
    assert set(summary["model_name"]) == {
        "elo"
    }
    assert set(summary["simulation_mode"]) == {
        DYNAMIC_ELO_MODE
    }
    assert set(summary["probability_source"]) == {
        INTERNAL_ELO_PROBABILITY_SOURCE
    }
    assert set(distribution["simulation_mode"]) == {
        DYNAMIC_ELO_MODE
    }
    assert set(distribution["total_simulations"]) == {
        100
    }
    assert set(
        distribution["simulation_generated_at"]
    ) == {
        generated_at
    }


def test_create_and_validate_simulation_tables() -> None:
    """Persist valid summary and distribution tables."""

    connection = duckdb.connect(":memory:")

    summary, distribution = (
        prepare_simulation_tables(
            create_result()
        )
    )

    create_simulation_tables(
        connection=connection,
        summary=summary,
        distribution=distribution,
    )

    validate_simulation_tables(
        connection=connection,
        expected_team_count=len(summary),
        expected_distribution_count=len(
            distribution
        ),
    )

    assert connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {SUMMARY_FULL_NAME}
        """
    ).fetchone()[0] == 2

    assert connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {DISTRIBUTION_FULL_NAME}
        """
    ).fetchone()[0] > 0

    connection.close()


def test_validation_rejects_invalid_probability_sum() -> None:
    """Reject incomplete team win distributions."""

    connection = duckdb.connect(":memory:")

    summary, distribution = (
        prepare_simulation_tables(
            create_result()
        )
    )

    create_simulation_tables(
        connection=connection,
        summary=summary,
        distribution=distribution,
    )

    connection.execute(
        f"""
        DELETE FROM {DISTRIBUTION_FULL_NAME}
        WHERE team = 'NE'
          AND wins = (
              SELECT MIN(wins)
              FROM {DISTRIBUTION_FULL_NAME}
              WHERE team = 'NE'
          )
        """
    )

    remaining_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {DISTRIBUTION_FULL_NAME}
        """
    ).fetchone()[0]

    with pytest.raises(
        RuntimeError,
        match="probabilities do not sum to one",
    ):
        validate_simulation_tables(
            connection=connection,
            expected_team_count=len(summary),
            expected_distribution_count=(
                remaining_count
            ),
        )

    connection.close()
