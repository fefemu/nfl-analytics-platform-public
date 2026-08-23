import pandas as pd
import pytest

from src.simulation.backtest_simulation_rating_inputs import (
    attach_nfelounits_ratings,
    normalize_nfelounits_elo,
)


def test_normalize_historical_oakland_by_season() -> None:
    source = pd.DataFrame(
        {
            "season": [2019, 2026],
            "week": [1, 1],
            "team": ["OAK", "OAK"],
            "elo": [1400.0, 1450.0],
        }
    )
    result = normalize_nfelounits_elo(source)
    assert list(result["team"]) == ["OAK", "LV"]


def test_attach_ratings_uses_exact_week_and_team() -> None:
    games = pd.DataFrame(
        {
            "game_id": ["g1"],
            "season": [2024],
            "week": [1],
            "home_team": ["KC"],
            "away_team": ["LV"],
        }
    )
    ratings = pd.DataFrame(
        {
            "season": [2024, 2024],
            "week": [1, 1],
            "team": ["KC", "LV"],
            "elo": [1600.0, 1400.0],
        }
    )
    result = attach_nfelounits_ratings(games, ratings)
    assert result.loc[0, "unit_home_rating"] == 1600.0
    assert result.loc[0, "unit_away_rating"] == 1400.0


def test_attach_ratings_rejects_incomplete_coverage() -> None:
    games = pd.DataFrame(
        {
            "game_id": ["g1"],
            "season": [2024],
            "week": [1],
            "home_team": ["KC"],
            "away_team": ["LV"],
        }
    )
    ratings = pd.DataFrame(
        {"season": [2024], "week": [1], "team": ["KC"], "elo": [1600.0]}
    )
    with pytest.raises(RuntimeError, match="0/1"):
        attach_nfelounits_ratings(games, ratings)
