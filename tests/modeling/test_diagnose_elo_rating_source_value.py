"""Tests for paired Elo rating source diagnostics."""

import numpy as np
import pandas as pd
import pytest

from src.modeling.backtest_elo_rating_sources import (
    BLEND_25_INTERNAL_FEATURE,
    BLEND_75_INTERNAL_FEATURE,
    BLEND_FEATURE,
    EXTERNAL_FEATURE,
    EXTERNAL_QB_FEATURE,
    INTERNAL_FEATURE,
    LISTED_QB_FEATURE,
    PUBLISHED_NFELO_CANDIDATE,
    TRAINED_CANDIDATES,
)
from src.modeling.diagnose_elo_rating_source_value import (
    COMPARISONS,
    FOLD_RESULT_COLUMNS,
    PREDICTION_COLUMNS,
    SUMMARY_COLUMNS,
    bootstrap_paired_mean_delta,
    create_oof_candidate_predictions,
    create_paired_comparison,
    diagnose_elo_rating_source_value,
)
from src.modeling.train_logistic_baseline import (
    TARGET_COLUMN,
)


SPREAD_TARGET_COLUMN = (
    "target_point_differential"
)

TEST_VALIDATION_SEASONS = (
    2021,
    2022,
)


def create_test_data() -> pd.DataFrame:
    """Create chronological synthetic Elo and QB data."""

    raw_rows = [
        (
            "2020_01_A_B",
            2020,
            "train",
            1,
            10.0,
            80.0,
            100.0,
            20.0,
            15.0,
            0.70,
        ),
        (
            "2020_02_C_D",
            2020,
            "train",
            0,
            -7.0,
            -60.0,
            -90.0,
            -15.0,
            -12.0,
            0.30,
        ),
        (
            "2020_03_E_F",
            2020,
            "train",
            1,
            4.0,
            30.0,
            50.0,
            8.0,
            6.0,
            0.60,
        ),
        (
            "2020_04_G_H",
            2020,
            "train",
            0,
            -3.0,
            -20.0,
            -40.0,
            -6.0,
            -5.0,
            0.40,
        ),
        (
            "2021_01_A_C",
            2021,
            "validation",
            1,
            6.0,
            45.0,
            70.0,
            12.0,
            9.0,
            0.65,
        ),
        (
            "2021_02_D_B",
            2021,
            "validation",
            0,
            -5.0,
            -35.0,
            -55.0,
            -10.0,
            -8.0,
            0.35,
        ),
        (
            "2022_01_E_G",
            2022,
            "validation",
            1,
            8.0,
            55.0,
            85.0,
            14.0,
            11.0,
            0.68,
        ),
        (
            "2022_02_H_F",
            2022,
            "validation",
            0,
            -6.0,
            -50.0,
            -75.0,
            -13.0,
            -10.0,
            0.32,
        ),
    ]

    data = pd.DataFrame(
        raw_rows,
        columns=[
            "game_id",
            "season",
            "split_name",
            TARGET_COLUMN,
            SPREAD_TARGET_COLUMN,
            INTERNAL_FEATURE,
            EXTERNAL_FEATURE,
            LISTED_QB_FEATURE,
            EXTERNAL_QB_FEATURE,
            "published_nfelo_home_probability",
        ],
    )

    data[BLEND_25_INTERNAL_FEATURE] = (
        0.25 * data[INTERNAL_FEATURE]
        + 0.75 * data[EXTERNAL_FEATURE]
    )

    data[BLEND_FEATURE] = (
        0.50 * data[INTERNAL_FEATURE]
        + 0.50 * data[EXTERNAL_FEATURE]
    )

    data[BLEND_75_INTERNAL_FEATURE] = (
        0.75 * data[INTERNAL_FEATURE]
        + 0.25 * data[EXTERNAL_FEATURE]
    )

    return data


def test_bootstrap_is_deterministic():
    """The same seed must return the same interval."""

    deltas = np.array(
        [
            -1.0,
            -0.5,
            0.0,
            0.5,
            1.0,
        ]
    )

    first_result = bootstrap_paired_mean_delta(
        paired_deltas=deltas,
        iteration_count=500,
        random_seed=42,
    )

    second_result = bootstrap_paired_mean_delta(
        paired_deltas=deltas,
        iteration_count=500,
        random_seed=42,
    )

    assert first_result == second_result
    assert first_result[0] <= 0.0
    assert first_result[1] >= 0.0


def test_bootstrap_rejects_empty_input():
    """Bootstrap requires paired observations."""

    with pytest.raises(
        ValueError,
        match="non-empty",
    ):
        bootstrap_paired_mean_delta(
            paired_deltas=np.array([]),
            iteration_count=100,
        )


def test_bootstrap_rejects_invalid_iteration_count():
    """Bootstrap iteration count must be positive."""

    with pytest.raises(
        ValueError,
        match="must be positive",
    ):
        bootstrap_paired_mean_delta(
            paired_deltas=np.array(
                [
                    -0.1,
                    0.2,
                ]
            ),
            iteration_count=0,
        )


def test_oof_predictions_cover_every_candidate_and_game():
    """Each candidate must predict every validation game."""

    predictions = create_oof_candidate_predictions(
        development_data=create_test_data(),
        validation_seasons=(
            TEST_VALIDATION_SEASONS
        ),
    )

    expected_candidates = (
        set(TRAINED_CANDIDATES)
        | {
            PUBLISHED_NFELO_CANDIDATE,
        }
    )

    assert tuple(predictions.columns) == (
        PREDICTION_COLUMNS
    )

    assert set(
        predictions["candidate_name"]
    ) == expected_candidates

    assert len(predictions) == (
        4 * len(expected_candidates)
    )

    candidate_game_counts = (
        predictions.groupby(
            "candidate_name"
        )["game_id"].nunique()
    )

    assert (
        candidate_game_counts == 4
    ).all()

    assert predictions[
        [
            "candidate_name",
            "game_id",
        ]
    ].duplicated().sum() == 0


def test_oof_predictions_do_not_open_holdout():
    """The diagnostic must reject the 2025 holdout."""

    data = create_test_data()

    data.loc[
        data.index[-1],
        "season",
    ] = 2025

    with pytest.raises(
        ValueError,
        match="must end before the 2025 holdout",
    ):
        create_oof_candidate_predictions(
            development_data=data,
            validation_seasons=(
                TEST_VALIDATION_SEASONS
            ),
        )


def test_paired_delta_matches_candidate_losses():
    """Paired deltas must equal challenger minus base."""

    predictions = create_oof_candidate_predictions(
        development_data=create_test_data(),
        validation_seasons=(
            TEST_VALIDATION_SEASONS
        ),
    )

    paired = create_paired_comparison(
        predictions=predictions,
        base_candidate="internal_elo",
        challenger_candidate="external_nfelo",
    )

    expected_brier_delta = (
        paired["challenger_brier_loss"]
        - paired["base_brier_loss"]
    )

    expected_spread_delta = (
        paired[
            "challenger_spread_absolute_error"
        ]
        - paired[
            "base_spread_absolute_error"
        ]
    )

    np.testing.assert_allclose(
        paired["brier_loss_delta"],
        expected_brier_delta,
    )

    np.testing.assert_allclose(
        paired[
            "spread_absolute_error_delta"
        ],
        expected_spread_delta,
    )

    assert len(paired) == 4


def test_missing_comparison_candidate_is_rejected():
    """Both named candidates must exist."""

    predictions = create_oof_candidate_predictions(
        development_data=create_test_data(),
        validation_seasons=(
            TEST_VALIDATION_SEASONS
        ),
    )

    with pytest.raises(
        ValueError,
        match="candidates are missing",
    ):
        create_paired_comparison(
            predictions=predictions,
            base_candidate="internal_elo",
            challenger_candidate=(
                "missing_candidate"
            ),
        )


def test_diagnostic_returns_expected_comparisons():
    """Every planned paired comparison must be returned."""

    (
        summary,
        fold_results,
        predictions,
    ) = diagnose_elo_rating_source_value(
        development_data=create_test_data(),
        validation_seasons=(
            TEST_VALIDATION_SEASONS
        ),
        bootstrap_iterations=500,
        random_seed=42,
    )

    assert tuple(summary.columns) == (
        SUMMARY_COLUMNS
    )

    assert tuple(fold_results.columns) == (
        FOLD_RESULT_COLUMNS
    )

    assert tuple(predictions.columns) == (
        PREDICTION_COLUMNS
    )

    assert set(
        summary["comparison_name"]
    ) == set(COMPARISONS)

    assert len(summary) == len(COMPARISONS)

    assert len(fold_results) == (
        len(COMPARISONS)
        * len(TEST_VALIDATION_SEASONS)
    )

    assert (
        summary["fold_count"]
        == len(TEST_VALIDATION_SEASONS)
    ).all()

    assert (
        summary["validation_game_count"] == 4
    ).all()


def test_published_probability_comparison_has_no_spread():
    """The published probability has no spread output."""

    summary, fold_results, _ = (
        diagnose_elo_rating_source_value(
            development_data=create_test_data(),
            validation_seasons=(
                TEST_VALIDATION_SEASONS
            ),
            bootstrap_iterations=500,
            random_seed=42,
        )
    )

    comparison_name = (
        "published_probability_vs_"
        "external_both_qb"
    )

    summary_row = summary.loc[
        summary["comparison_name"]
        == comparison_name
    ].iloc[0]

    spread_columns = [
        "base_spread_mae",
        "challenger_spread_mae",
        "spread_mae_delta",
        "challenger_spread_win_rate",
        "spread_bootstrap_95_percent_lower",
        "spread_bootstrap_95_percent_upper",
    ]

    assert summary_row[
        spread_columns
    ].isna().all()

    fold_rows = fold_results.loc[
        fold_results["comparison_name"]
        == comparison_name
    ]

    assert fold_rows[
        [
            "base_spread_mae",
            "challenger_spread_mae",
            "spread_mae_delta",
            "challenger_spread_win_rate",
        ]
    ].isna().all().all()