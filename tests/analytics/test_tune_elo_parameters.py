"""Tests for Elo parameter tuning."""

from datetime import date

import pytest

from src.analytics.tune_elo_parameters import (
    EloParameterSet,
    EloPeriodMetrics,
    EloTuningResult,
    create_parameter_grid,
    evaluate_parameter_set,
    select_best_result,
    select_tuning_games,
)
from src.models.elo_history import HistoricalGame


def create_game_for_season(
    season: int,
) -> HistoricalGame:
    """Create one minimal historical game for a season."""

    return HistoricalGame(
        game_id=f"{season}_game",
        season=season,
        game_type="REG",
        week=1,
        gameday=date(season, 9, 1),
        gametime=None,
        home_team="BUF",
        away_team="MIA",
        home_score=24,
        away_score=17,
        is_neutral=False,
    )


def test_create_parameter_grid_contains_200_combinations() -> None:
    """Create all expected Elo parameter combinations."""

    parameter_grid = create_parameter_grid()

    assert len(parameter_grid) == 200
    assert len(set(parameter_grid)) == 200 


def test_create_parameter_grid_contains_baseline() -> None:
    """Include the current baseline parameters."""

    parameter_grid = create_parameter_grid()

    baseline = EloParameterSet(
        k_factor=20.0,
        home_advantage=50.0,
        season_retention=0.70,
    )

    assert baseline in parameter_grid


def test_select_tuning_games_keeps_required_period() -> None:
    """Keep seasons from burn-in through final holdout."""

    games = [
        create_game_for_season(season)
        for season in range(1998, 2027)
    ]

    selected_games = select_tuning_games(games)

    selected_seasons = {
        game.season
        for game in selected_games
    }

    assert min(selected_seasons) == 1999
    assert max(selected_seasons) == 2025
    assert 1998 not in selected_seasons
    assert 2026 not in selected_seasons


def test_select_tuning_games_rejects_missing_season() -> None:
    """Reject tuning data when a required season is absent."""

    games = [
        create_game_for_season(season)
        for season in range(1999, 2026)
        if season != 2010
    ]

    with pytest.raises(
        RuntimeError,
        match="Missing seasons required for Elo tuning: 2010",
    ):
        select_tuning_games(games)


def test_evaluate_parameter_set_separates_time_periods() -> None:
    """Evaluate development, validation, and holdout separately."""

    games = [
        create_game_for_season(season)
        for season in range(1999, 2026)
    ]
    parameters = EloParameterSet(
        k_factor=20.0,
        home_advantage=50.0,
        season_retention=0.70,
    )

    result = evaluate_parameter_set(
        games=games,
        parameters=parameters,
    )

    assert result.parameters == parameters

    assert result.development.game_count == 23
    assert result.validation.game_count == 2
    assert result.holdout.game_count == 1

    assert 0.0 <= result.development.brier_score <= 1.0
    assert 0.0 <= result.validation.brier_score <= 1.0
    assert 0.0 <= result.holdout.brier_score <= 1.0


def test_select_best_result_uses_development_brier_score() -> None:
    """Select parameters using development performance only."""

    better_development = EloTuningResult(
        parameters=EloParameterSet(
            k_factor=15.0,
            home_advantage=45.0,
            season_retention=0.70,
        ),
        development=EloPeriodMetrics(
            game_count=100,
            brier_score=0.220,
            log_loss=0.630,
            accuracy=0.63,
        ),
        validation=EloPeriodMetrics(
            game_count=20,
            brier_score=0.240,
            log_loss=0.680,
            accuracy=0.55,
        ),
        holdout=EloPeriodMetrics(
            game_count=10,
            brier_score=0.250,
            log_loss=0.690,
            accuracy=0.50,
        ),
    )

    better_validation = EloTuningResult(
        parameters=EloParameterSet(
            k_factor=25.0,
            home_advantage=35.0,
            season_retention=0.80,
        ),
        development=EloPeriodMetrics(
            game_count=100,
            brier_score=0.230,
            log_loss=0.640,
            accuracy=0.62,
        ),
        validation=EloPeriodMetrics(
            game_count=20,
            brier_score=0.200,
            log_loss=0.590,
            accuracy=0.70,
        ),
        holdout=EloPeriodMetrics(
            game_count=10,
            brier_score=0.190,
            log_loss=0.570,
            accuracy=0.80,
        ),
    )

    best_result = select_best_result(
        [
            better_validation,
            better_development,
        ]
    )

    assert best_result == better_development