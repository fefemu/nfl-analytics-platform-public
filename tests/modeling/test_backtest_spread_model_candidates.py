"""Tests for spread expanding-window backtests."""

import pandas as pd
import pytest

from src.modeling.backtest_spread_model_candidates import (
    FOLD_RESULT_COLUMNS,
    SUMMARY_RESULT_COLUMNS,
    create_backtest_candidates,
    evaluate_spread_expanding_window,
)
from src.modeling.evaluate_spread_model_candidates import (
    SPREAD_CORE_FEATURES,
)


def create_backtest_data() -> pd.DataFrame:
    """Create chronological synthetic spread data."""

    rows: list[dict[str, object]] = []

    for season in range(2018, 2025):
        for game_index in range(12):
            elo_difference = float(
                -110 + game_index * 20
            )

            qb_difference = float(
                -4 + game_index * 0.8
            )

            injury_difference = float(
                -1
                if game_index % 2 == 0
                else 1
            )

            rows.append(
                {
                    "game_id": (
                        f"{season}_{game_index}"
                    ),
                    "season": season,
                    "split_name": (
                        "train"
                        if season <= 2022
                        else "validation"
                    ),
                    "has_complete_injury_data": True,
                    "target_point_differential": (
                        0.04 * elo_difference
                        + 1.1 * qb_difference
                        + 0.5 * injury_difference
                    ),
                    "elo_rating_difference": (
                        elo_difference
                    ),
                    "listed_qb_rating_difference": (
                        qb_difference
                    ),
                    "offense_injury_burden_difference": (
                        injury_difference
                    ),
                    "defense_injury_burden_difference": (
                        0.0
                    ),
                    "special_teams_injury_burden_difference": (
                        0.0
                    ),
                }
            )

    return pd.DataFrame(rows)


def test_create_candidate_grid() -> None:
    """Create every feature-set and alpha combination."""

    candidates = create_backtest_candidates(
        alpha_grid=(
            0.0,
            1.0,
        )
    )

    assert len(candidates) == 6

    assert {
        candidate.candidate_name
        for candidate in candidates
    } == {
        "ridge_elo",
        "ridge_elo_qb",
        "ridge_elo_qb_injury",
    }

    assert {
        candidate.ridge_alpha
        for candidate in candidates
    } == {
        0.0,
        1.0,
    }


def test_backtest_uses_only_earlier_training_seasons(
) -> None:
    """Prevent future seasons from entering training."""

    _, fold_results = (
        evaluate_spread_expanding_window(
            development_data=(
                create_backtest_data()
            ),
            validation_seasons=(
                2021,
                2022,
                2023,
                2024,
            ),
            alpha_grid=(
                1.0,
            ),
        )
    )

    assert tuple(
        fold_results.columns
    ) == FOLD_RESULT_COLUMNS

    assert (
        fold_results["train_last_season"]
        < fold_results["validation_season"]
    ).all()


def test_backtest_returns_expected_candidate_rows(
) -> None:
    """Return one summary row per model setting."""

    summary, fold_results = (
        evaluate_spread_expanding_window(
            development_data=(
                create_backtest_data()
            ),
            validation_seasons=(
                2022,
                2023,
                2024,
            ),
            alpha_grid=(
                0.0,
                1.0,
            ),
        )
    )

    assert tuple(
        summary.columns
    ) == SUMMARY_RESULT_COLUMNS

    assert len(summary) == 7
    assert len(fold_results) == 21

    assert set(
        summary["fold_count"]
    ) == {
        3,
    }

    assert set(
        summary["validation_game_count"]
    ) == {
        36,
    }


def test_signal_model_beats_constant_baseline(
) -> None:
    """Recover the synthetic chronological signal."""

    summary, _ = (
        evaluate_spread_expanding_window(
            development_data=(
                create_backtest_data()
            ),
            validation_seasons=(
                2021,
                2022,
                2023,
                2024,
            ),
            alpha_grid=(
                0.0,
                1.0,
            ),
        )
    )

    constant_mae = summary.loc[
        summary["candidate_name"]
        == "constant_train_mean",
        "pooled_validation_mae",
    ].iloc[0]

    best_model_mae = summary.loc[
        summary["candidate_name"]
        != "constant_train_mean",
        "pooled_validation_mae",
    ].min()

    assert best_model_mae < constant_mae


def test_common_features_are_complete() -> None:
    """Synthetic data supplies the full spread core."""

    data = create_backtest_data()

    assert data[
        list(SPREAD_CORE_FEATURES)
    ].notna().all().all()


def test_invalid_alpha_grid_is_rejected() -> None:
    """Reject duplicate and negative alpha values."""

    with pytest.raises(
        ValueError,
        match="unique",
    ):
        create_backtest_candidates(
            alpha_grid=(
                1.0,
                1.0,
            )
        )

    with pytest.raises(
        ValueError,
        match="negative",
    ):
        create_backtest_candidates(
            alpha_grid=(
                -1.0,
            )
        )