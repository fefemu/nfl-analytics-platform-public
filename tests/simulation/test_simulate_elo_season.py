"""Tests for dynamic Elo season simulation."""

import numpy as np
import pandas as pd
import pytest

from src.simulation.simulate_elo_season import (
    extract_initial_ratings,
    simulate_regular_season_once,
)


def create_simulation_schedule() -> pd.DataFrame:
    """Create a small repeated-matchup schedule."""

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


def test_extract_initial_ratings() -> None:
    """Extract one stable rating for every team."""

    ratings = extract_initial_ratings(
        create_simulation_schedule()
    )

    assert ratings == {
        "NE": 1550.0,
        "NYJ": 1450.0,
    }


def test_extract_initial_ratings_rejects_inconsistency() -> None:
    """Reject static inputs with changing initial ratings."""

    schedule = create_simulation_schedule()

    schedule.loc[
        schedule["game_id"] == "game_2",
        "away_rating_pregame",
    ] = 1600.0

    with pytest.raises(
        ValueError,
        match="inconsistent initial ratings for NE",
    ):
        extract_initial_ratings(schedule)


def test_simulation_is_reproducible() -> None:
    """Produce identical results with the same seed."""

    schedule = create_simulation_schedule()

    first_result = simulate_regular_season_once(
        schedule=schedule,
        random_generator=np.random.default_rng(42),
    )

    second_result = simulate_regular_season_once(
        schedule=schedule,
        random_generator=np.random.default_rng(42),
    )

    pd.testing.assert_frame_equal(
        first_result.game_results,
        second_result.game_results,
    )

    pd.testing.assert_frame_equal(
        first_result.team_records,
        second_result.team_records,
    )


def test_simulation_updates_elo_between_games() -> None:
    """Use the first simulated result in the next game."""

    result = simulate_regular_season_once(
        schedule=create_simulation_schedule(),
        random_generator=np.random.default_rng(42),
    )

    first_game = result.game_results.iloc[0]
    second_game = result.game_results.iloc[1]

    assert second_game[
        "away_rating_pre"
    ] == pytest.approx(
        first_game["home_rating_post"]
    )

    assert second_game[
        "home_rating_pre"
    ] == pytest.approx(
        first_game["away_rating_post"]
    )


def test_simulation_preserves_total_elo() -> None:
    """Transfer Elo points without creating rating mass."""

    schedule = create_simulation_schedule()

    initial_ratings = extract_initial_ratings(
        schedule
    )

    result = simulate_regular_season_once(
        schedule=schedule,
        random_generator=np.random.default_rng(42),
    )

    assert sum(
        result.final_ratings.values()
    ) == pytest.approx(
        sum(initial_ratings.values())
    )


def test_simulation_creates_complete_records() -> None:
    """Record one win and loss for every simulated game."""

    result = simulate_regular_season_once(
        schedule=create_simulation_schedule(),
        random_generator=np.random.default_rng(42),
    )

    assert (
        result.team_records["wins"].sum()
        == 2
    )
    assert (
        result.team_records["losses"].sum()
        == 2
    )
    assert (
        result.team_records["games"].sum()
        == 4
    )