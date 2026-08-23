import pandas as pd
import pytest

from src.simulation.benchmark_simulation_rating_inputs import (
    prepare_latest_nfelounits_ratings,
    replace_schedule_ratings,
)


def test_prepare_latest_ratings_requires_all_teams() -> None:
    source = pd.DataFrame(
        {"season": [2026], "week": [1], "team": ["KC"], "elo": [1600.0]}
    )
    with pytest.raises(RuntimeError, match="32"):
        prepare_latest_nfelounits_ratings(source, 2026)


def test_replace_schedule_ratings_maps_both_teams() -> None:
    schedule = pd.DataFrame(
        {
            "home_team": ["KC"],
            "away_team": ["LV"],
            "home_rating_pregame": [1500.0],
            "away_rating_pregame": [1500.0],
        }
    )
    ratings = pd.DataFrame(
        {"team": ["KC", "LV"], "rating": [1600.0, 1400.0]}
    )
    result = replace_schedule_ratings(schedule, ratings)

    assert result.loc[0, "home_rating_pregame"] == 1600.0
    assert result.loc[0, "away_rating_pregame"] == 1400.0


def test_replace_schedule_rejects_missing_team() -> None:
    schedule = pd.DataFrame(
        {"home_team": ["KC"], "away_team": ["LV"]}
    )
    ratings = pd.DataFrame({"team": ["KC"], "rating": [1600.0]})
    with pytest.raises(ValueError, match="LV"):
        replace_schedule_ratings(schedule, ratings)
