import pandas as pd
import pytest

from src.simulation.backtest_season_rating_priors import (
    SUMMARY_COLUMNS,
    create_season_schedule,
    blend_rating_sources,
    summarize_season_prior_results,
)


def test_create_season_schedule_repeats_initial_ratings() -> None:
    schedule = pd.DataFrame(
        {
            "season": [2024, 2024],
            "home_team": ["KC", "LV"],
            "away_team": ["LV", "KC"],
        }
    )
    ratings = pd.DataFrame(
        {"season": [2024, 2024], "team": ["KC", "LV"], "rating": [1600, 1400]}
    )
    result = create_season_schedule(schedule, ratings, 2024)
    assert list(result["home_rating_pregame"]) == [1600, 1400]
    assert list(result["away_rating_pregame"]) == [1400, 1600]


def test_create_season_schedule_rejects_missing_team() -> None:
    schedule = pd.DataFrame(
        {"season": [2024], "home_team": ["KC"], "away_team": ["LV"]}
    )
    ratings = pd.DataFrame(
        {"season": [2024], "team": ["KC"], "rating": [1600]}
    )
    with pytest.raises(RuntimeError, match="LV"):
        create_season_schedule(schedule, ratings, 2024)


def test_create_season_schedule_requires_unique_team_ratings() -> None:
    schedule = pd.DataFrame(
        {"season": [2024], "home_team": ["KC"], "away_team": ["LV"]}
    )
    ratings = pd.DataFrame(
        {
            "season": [2024, 2024, 2024],
            "team": ["KC", "KC", "LV"],
            "rating": [1600, 1610, 1400],
        }
    )
    with pytest.raises(ValueError):
        create_season_schedule(schedule, ratings, 2024)


def test_summary_schema_and_mae() -> None:
    results = pd.DataFrame(
        {
            "candidate_name": ["a", "a", "a", "a"],
            "simulation_mode": ["F"] * 4,
            "season": [2023, 2023, 2024, 2024],
            "expected_wins": [10.0, 7.0, 9.0, 8.0],
            "actual_wins": [11.0, 6.0, 8.0, 9.0],
        }
    )
    summary = summarize_season_prior_results(results)
    assert tuple(summary.columns) == SUMMARY_COLUMNS
    assert summary.loc[0, "expected_wins_mae"] == pytest.approx(1.0)


def test_actual_season_slice_can_merge_without_duplicate_season() -> None:
    actual = pd.DataFrame(
        {"season": [2024], "team": ["KC"], "actual_wins": [12.0]}
    )
    season_actual = actual.loc[
        actual["season"] == 2024, ["team", "actual_wins"]
    ]
    prediction = pd.DataFrame({"team": ["KC"], "expected_wins": [11.0]})
    comparison = prediction.merge(season_actual, on="team")
    comparison.insert(0, "season", 2024)
    assert list(comparison.columns)[0] == "season"


def test_blend_rating_sources_uses_requested_weight() -> None:
    base = pd.DataFrame(
        {"season": [2024], "team": ["KC"], "rating": [1500.0]}
    )
    challenger = pd.DataFrame(
        {"season": [2024], "team": ["KC"], "rating": [1700.0]}
    )
    result = blend_rating_sources(base, challenger, 0.25)
    assert result.loc[0, "rating"] == pytest.approx(1550.0)


def test_blend_rating_sources_rejects_invalid_weight() -> None:
    empty = pd.DataFrame(columns=["season", "team", "rating"])
    with pytest.raises(ValueError, match="between"):
        blend_rating_sources(empty, empty, 1.1)
