"""Tests for totals expanding-window backtests."""

import pandas as pd
import pytest

from src.modeling.backtest_totals_model_candidates import (
    FOLD_RESULT_COLUMNS,
    SUMMARY_RESULT_COLUMNS,
    create_backtest_candidates,
    evaluate_totals_expanding_window,
)
from src.modeling.evaluate_totals_model_candidates import (
    RAW_TOTALS_FEATURE_COLUMNS,
    TOTALS_CANDIDATE_SAMPLE_FEATURES,
    create_totals_aggregate_features,
)
from src.modeling.diagnose_totals_scoring_window import (
    ALTERNATIVE_MODEL_NAME,
    BASE_MODEL_NAME,
    COEFFICIENT_RESULT_COLUMNS,
    FOLD_RESULT_COLUMNS as DIAGNOSTIC_FOLD_COLUMNS,
    PAIRED_RESULT_COLUMNS,
    SUMMARY_RESULT_COLUMNS as DIAGNOSTIC_SUMMARY_COLUMNS,
    bootstrap_paired_mean_delta,
    diagnose_totals_scoring_window,
)


def create_backtest_data() -> pd.DataFrame:
    """Create chronological synthetic totals data."""

    rows: list[dict[str, object]] = []

    for season in range(2018, 2025):
        for game_index in range(12):
            row: dict[str, object] = {
                column_name: 0.0
                for column_name
                in RAW_TOTALS_FEATURE_COLUMNS
            }

            home_offensive_epa = (
                -0.10 + game_index * 0.02
            )

            away_offensive_epa = (
                0.08 - game_index * 0.01
            )

            home_defensive_epa = (
                -0.04 + game_index * 0.01
            )

            away_defensive_epa = (
                0.05 - game_index * 0.005
            )

            home_qb_rating = float(
                -2 + game_index * 0.5
            )

            away_qb_rating = float(
                3 - game_index * 0.2
            )

            league_average_64 = (
                44.0
                + 0.4 * (season - 2018)
            )

            league_average_128 = (
                44.5
                + 0.35 * (season - 2018)
            )

            row.update(
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
                    "both_short_windows_complete": True,
                    "target_total_points": (
                        8.0
                        + 16.0
                        * (
                            home_offensive_epa
                            + away_offensive_epa
                        )
                        + 7.0
                        * (
                            home_defensive_epa
                            + away_defensive_epa
                        )
                        + 0.5
                        * (
                            home_qb_rating
                            + away_qb_rating
                        )
                        + 0.8
                        * league_average_128
                    ),
                    "home_offensive_epa_per_play_last_4": (
                        home_offensive_epa
                    ),
                    "away_offensive_epa_per_play_last_4": (
                        away_offensive_epa
                    ),
                    (
                        "home_defensive_epa_allowed_"
                        "per_play_last_4"
                    ): home_defensive_epa,
                    (
                        "away_defensive_epa_allowed_"
                        "per_play_last_4"
                    ): away_defensive_epa,
                    "home_listed_qb_rating": (
                        home_qb_rating
                    ),
                    "away_listed_qb_rating": (
                        away_qb_rating
                    ),
                    "is_indoor": (
                        game_index % 4 == 0
                    ),
                    "has_game_weather": True,
                    "cold_degrees_below_50": float(
                        game_index % 3
                    ),
                    "heat_degrees_above_80": 0.0,
                    "wind_mph_above_10": float(
                        game_index % 2
                    ),
                    "league_average_total_last_32": (
                        43.5
                        + 0.45 * (season - 2018)
                    ),
                    "league_average_total_last_64": (
                        league_average_64
                    ),
                    "league_average_total_last_128": (
                        league_average_128
                    ),
                }
            )

            rows.append(row)

    return pd.DataFrame(rows)


def test_create_candidate_grid() -> None:
    """Create both window and alpha combinations."""

    candidates = create_backtest_candidates(
        alpha_grid=(
            0.0,
            1.0,
        )
    )

    assert len(candidates) == 4

    assert {
        candidate.candidate_name
        for candidate in candidates
    } == {
        "ridge_epa_weather_qb_league_64",
        "ridge_epa_weather_qb_league_128",
    }

    assert {
        candidate.ridge_alpha
        for candidate in candidates
    } == {
        0.0,
        1.0,
    }


def test_backtest_uses_only_earlier_seasons(
) -> None:
    """Prevent future seasons from entering training."""

    _, fold_results = (
        evaluate_totals_expanding_window(
            development_data=create_backtest_data(),
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


def test_backtest_returns_expected_rows() -> None:
    """Return one row per candidate setting."""

    summary, fold_results = (
        evaluate_totals_expanding_window(
            development_data=create_backtest_data(),
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

    assert len(summary) == 5
    assert len(fold_results) == 15

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
    """Recover the synthetic totals signal."""

    summary, _ = (
        evaluate_totals_expanding_window(
            development_data=create_backtest_data(),
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
    """Supply every common candidate feature."""

    data = create_totals_aggregate_features(
        create_backtest_data()
    )

    assert data[
        list(TOTALS_CANDIDATE_SAMPLE_FEATURES)
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


def test_scoring_window_diagnostic_schema() -> None:
    """Return all paired diagnostic result tables."""

    (
        summary,
        folds,
        coefficients,
        paired,
    ) = diagnose_totals_scoring_window(
        development_data=create_backtest_data(),
        validation_seasons=(
            2022,
            2023,
            2024,
        ),
        base_ridge_alpha=100.0,
        alternative_ridge_alpha=300.0,
        bootstrap_iterations=200,
        random_seed=42,
    )

    assert tuple(
        summary.columns
    ) == DIAGNOSTIC_SUMMARY_COLUMNS

    assert tuple(
        folds.columns
    ) == DIAGNOSTIC_FOLD_COLUMNS

    assert tuple(
        coefficients.columns
    ) == COEFFICIENT_RESULT_COLUMNS

    assert tuple(
        paired.columns
    ) == PAIRED_RESULT_COLUMNS

    assert len(folds) == 3
    assert len(paired) == 36


def test_scoring_window_coefficients_cover_models(
) -> None:
    """Return coefficients for both scoring windows."""

    _, _, coefficients, _ = (
        diagnose_totals_scoring_window(
            development_data=create_backtest_data(),
            validation_seasons=(
                2023,
                2024,
            ),
            bootstrap_iterations=100,
        )
    )

    assert set(
        coefficients["candidate_name"]
    ) == {
        BASE_MODEL_NAME,
        ALTERNATIVE_MODEL_NAME,
    }

    assert set(
        coefficients["validation_season"]
    ) == {
        2023,
        2024,
    }


def test_paired_delta_matches_model_errors() -> None:
    """Define delta as alternative minus base error."""

    _, _, _, paired = (
        diagnose_totals_scoring_window(
            development_data=create_backtest_data(),
            validation_seasons=(
                2024,
            ),
            bootstrap_iterations=100,
        )
    )

    expected_delta = (
        paired["alternative_absolute_error"]
        - paired["base_absolute_error"]
    )

    assert paired[
        "alternative_absolute_error_delta"
    ].to_numpy(
        dtype=float
    ) == pytest.approx(
        expected_delta.to_numpy(
            dtype=float
        )
    )


def test_bootstrap_is_reproducible() -> None:
    """Return identical intervals for one seed."""

    deltas = pd.Series(
        [
            -0.5,
            0.2,
            -0.1,
            0.4,
        ]
    ).to_numpy()

    first = bootstrap_paired_mean_delta(
        paired_deltas=deltas,
        iteration_count=200,
        random_seed=42,
    )

    second = bootstrap_paired_mean_delta(
        paired_deltas=deltas,
        iteration_count=200,
        random_seed=42,
    )

    assert first == second