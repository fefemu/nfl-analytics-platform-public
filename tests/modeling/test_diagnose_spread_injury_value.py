"""Tests for spread injury feature diagnostics."""

import numpy as np
import pandas as pd
import pytest

from src.modeling.diagnose_spread_injury_value import (
    COEFFICIENT_RESULT_COLUMNS,
    FOLD_RESULT_COLUMNS,
    PAIRED_RESULT_COLUMNS,
    SUMMARY_RESULT_COLUMNS,
    bootstrap_paired_mean_delta,
    diagnose_spread_injury_value,
)


def create_injury_signal_data() -> pd.DataFrame:
    """Create chronological data with injury signal."""

    rows: list[dict[str, object]] = []

    for season in range(2018, 2025):
        for game_index in range(20):
            elo_difference = float(
                -120 + game_index * 12
            )

            qb_difference = float(
                -5 + game_index * 0.5
            )

            injury_difference = float(
                -3 + game_index % 7
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
                        + 0.8 * qb_difference
                        + 2.0 * injury_difference
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


def test_diagnostics_return_expected_schemas() -> None:
    """Return summary, fold, coefficient and game rows."""

    summary, folds, coefficients, paired = (
        diagnose_spread_injury_value(
            development_data=(
                create_injury_signal_data()
            ),
            validation_seasons=(
                2022,
                2023,
                2024,
            ),
            ridge_alpha=1.0,
            bootstrap_iterations=500,
        )
    )

    assert tuple(
        summary.columns
    ) == SUMMARY_RESULT_COLUMNS

    assert tuple(
        folds.columns
    ) == FOLD_RESULT_COLUMNS

    assert tuple(
        coefficients.columns
    ) == COEFFICIENT_RESULT_COLUMNS

    assert tuple(
        paired.columns
    ) == PAIRED_RESULT_COLUMNS


def test_injury_signal_improves_paired_mae() -> None:
    """Detect useful injury information."""

    summary, _, _, _ = (
        diagnose_spread_injury_value(
            development_data=(
                create_injury_signal_data()
            ),
            validation_seasons=(
                2022,
                2023,
                2024,
            ),
            ridge_alpha=1.0,
            bootstrap_iterations=500,
        )
    )

    result = summary.iloc[0]

    assert result["injury_mae"] < result["base_mae"]
    assert result["injury_mae_delta"] < 0.0
    assert (
        result["bootstrap_95_percent_upper"]
        < 0.0
    )


def test_each_fold_uses_paired_games() -> None:
    """Compare models on identical validation games."""

    _, folds, _, paired = (
        diagnose_spread_injury_value(
            development_data=(
                create_injury_signal_data()
            ),
            validation_seasons=(
                2021,
                2022,
                2023,
                2024,
            ),
            ridge_alpha=100.0,
            bootstrap_iterations=100,
        )
    )

    assert len(folds) == 4
    assert len(paired) == 80

    assert (
        folds["validation_game_count"]
        == 20
    ).all()


def test_coefficients_cover_both_models() -> None:
    """Extract fold-level standardized coefficients."""

    _, _, coefficients, _ = (
        diagnose_spread_injury_value(
            development_data=(
                create_injury_signal_data()
            ),
            validation_seasons=(
                2023,
                2024,
            ),
            ridge_alpha=1.0,
            bootstrap_iterations=100,
        )
    )

    assert set(
        coefficients["candidate_name"]
    ) == {
        "ridge_elo_qb",
        "ridge_elo_qb_injury",
    }

    assert set(
        coefficients["validation_season"]
    ) == {
        2023,
        2024,
    }


def test_bootstrap_is_reproducible() -> None:
    """Use a deterministic bootstrap seed."""

    deltas = np.array(
        [
            -2.0,
            -1.0,
            0.5,
            1.0,
        ]
    )

    first = bootstrap_paired_mean_delta(
        paired_deltas=deltas,
        iteration_count=500,
        random_seed=42,
    )

    second = bootstrap_paired_mean_delta(
        paired_deltas=deltas,
        iteration_count=500,
        random_seed=42,
    )

    assert first == second


def test_invalid_bootstrap_is_rejected() -> None:
    """Reject empty data and invalid iterations."""

    with pytest.raises(
        ValueError,
        match="non-empty",
    ):
        bootstrap_paired_mean_delta(
            paired_deltas=np.array([]),
        )

    with pytest.raises(
        ValueError,
        match="positive",
    ):
        bootstrap_paired_mean_delta(
            paired_deltas=np.array(
                [
                    1.0,
                ]
            ),
            iteration_count=0,
        )