"""Tests for the Elo rating source backtest."""

import pandas as pd
import pytest

from src.modeling.backtest_elo_rating_sources import (
    BLEND_25_INTERNAL_FEATURE,
    BLEND_75_INTERNAL_FEATURE,
    BLEND_FEATURE,
    EXTERNAL_FEATURE,
    EXTERNAL_QB_FEATURE,
    FOLD_RESULT_COLUMNS,
    INTERNAL_FEATURE,
    LISTED_QB_FEATURE,
    PUBLISHED_NFELO_CANDIDATE,
    SUMMARY_COLUMNS,
    TRAINED_CANDIDATES,
    evaluate_rating_source_backtest,
    prepare_common_backtest_sample,
    validate_backtest_seasons,
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

    base_rows = [
        {
            "game_id": "2020_01_A_B",
            "season": 2020,
            "split_name": "train",
            TARGET_COLUMN: 1,
            SPREAD_TARGET_COLUMN: 10.0,
            INTERNAL_FEATURE: 80.0,
            EXTERNAL_FEATURE: 100.0,
            LISTED_QB_FEATURE: 20.0,
            EXTERNAL_QB_FEATURE: 15.0,
            "published_nfelo_home_probability": 0.70,
        },
        {
            "game_id": "2020_02_C_D",
            "season": 2020,
            "split_name": "train",
            TARGET_COLUMN: 0,
            SPREAD_TARGET_COLUMN: -7.0,
            INTERNAL_FEATURE: -60.0,
            EXTERNAL_FEATURE: -90.0,
            LISTED_QB_FEATURE: -15.0,
            EXTERNAL_QB_FEATURE: -12.0,
            "published_nfelo_home_probability": 0.30,
        },
        {
            "game_id": "2020_03_E_F",
            "season": 2020,
            "split_name": "train",
            TARGET_COLUMN: 1,
            SPREAD_TARGET_COLUMN: 4.0,
            INTERNAL_FEATURE: 30.0,
            EXTERNAL_FEATURE: 50.0,
            LISTED_QB_FEATURE: 8.0,
            EXTERNAL_QB_FEATURE: 6.0,
            "published_nfelo_home_probability": 0.60,
        },
        {
            "game_id": "2020_04_G_H",
            "season": 2020,
            "split_name": "train",
            TARGET_COLUMN: 0,
            SPREAD_TARGET_COLUMN: -3.0,
            INTERNAL_FEATURE: -20.0,
            EXTERNAL_FEATURE: -40.0,
            LISTED_QB_FEATURE: -6.0,
            EXTERNAL_QB_FEATURE: -5.0,
            "published_nfelo_home_probability": 0.40,
        },
        {
            "game_id": "2021_01_A_C",
            "season": 2021,
            "split_name": "validation",
            TARGET_COLUMN: 1,
            SPREAD_TARGET_COLUMN: 6.0,
            INTERNAL_FEATURE: 45.0,
            EXTERNAL_FEATURE: 70.0,
            LISTED_QB_FEATURE: 12.0,
            EXTERNAL_QB_FEATURE: 9.0,
            "published_nfelo_home_probability": 0.65,
        },
        {
            "game_id": "2021_02_D_B",
            "season": 2021,
            "split_name": "validation",
            TARGET_COLUMN: 0,
            SPREAD_TARGET_COLUMN: -5.0,
            INTERNAL_FEATURE: -35.0,
            EXTERNAL_FEATURE: -55.0,
            LISTED_QB_FEATURE: -10.0,
            EXTERNAL_QB_FEATURE: -8.0,
            "published_nfelo_home_probability": 0.35,
        },
        {
            "game_id": "2022_01_E_G",
            "season": 2022,
            "split_name": "validation",
            TARGET_COLUMN: 1,
            SPREAD_TARGET_COLUMN: 8.0,
            INTERNAL_FEATURE: 55.0,
            EXTERNAL_FEATURE: 85.0,
            LISTED_QB_FEATURE: 14.0,
            EXTERNAL_QB_FEATURE: 11.0,
            "published_nfelo_home_probability": 0.68,
        },
        {
            "game_id": "2022_02_H_F",
            "season": 2022,
            "split_name": "validation",
            TARGET_COLUMN: 0,
            SPREAD_TARGET_COLUMN: -6.0,
            INTERNAL_FEATURE: -50.0,
            EXTERNAL_FEATURE: -75.0,
            LISTED_QB_FEATURE: -13.0,
            EXTERNAL_QB_FEATURE: -10.0,
            "published_nfelo_home_probability": 0.32,
        },
    ]

    data = pd.DataFrame(base_rows)

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


def test_prepare_common_sample_preserves_complete_rows():
    """Complete development games should remain."""

    data = create_test_data()

    sample = prepare_common_backtest_sample(
        data
    )

    assert len(sample) == len(data)
    assert sample["game_id"].tolist() == (
        data["game_id"].tolist()
    )


def test_prepare_common_sample_removes_incomplete_rows():
    """All candidates must use the same complete sample."""

    data = create_test_data()

    data.loc[
        data["game_id"] == "2021_01_A_C",
        EXTERNAL_QB_FEATURE,
    ] = None

    sample = prepare_common_backtest_sample(
        data
    )

    assert len(sample) == len(data) - 1
    assert (
        "2021_01_A_C"
        not in set(sample["game_id"])
    )


def test_missing_required_column_is_rejected():
    """Required comparison columns must exist."""

    data = create_test_data().drop(
        columns=[EXTERNAL_QB_FEATURE]
    )

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        prepare_common_backtest_sample(data)


def test_duplicate_game_id_is_rejected():
    """A game must occur only once in the sample."""

    data = create_test_data()

    data.loc[
        data.index[-1],
        "game_id",
    ] = data.loc[
        data.index[0],
        "game_id",
    ]

    with pytest.raises(
        ValueError,
        match="duplicate game identifiers",
    ):
        prepare_common_backtest_sample(data)


def test_holdout_split_is_rejected():
    """The protected holdout split must stay closed."""

    data = create_test_data()

    data.loc[
        data.index[-1],
        "split_name",
    ] = "holdout"

    with pytest.raises(
        ValueError,
        match="must not contain holdout",
    ):
        prepare_common_backtest_sample(data)


def test_2025_season_is_rejected():
    """No 2025 row may enter development evaluation."""

    data = create_test_data()

    data.loc[
        data.index[-1],
        "season",
    ] = 2025

    with pytest.raises(
        ValueError,
        match="must end before the 2025 holdout",
    ):
        prepare_common_backtest_sample(data)


def test_validation_seasons_require_earlier_training():
    """The first fold must have prior-season data."""

    data = create_test_data()

    sample = data.loc[
        data["season"] >= 2021
    ].copy()

    with pytest.raises(
        ValueError,
        match="requires earlier training data",
    ):
        validate_backtest_seasons(
            sample=sample,
            validation_seasons=(
                2021,
                2022,
            ),
        )


def test_candidate_grid_contains_elo_and_qb_models():
    """The comparison must cover Elo and QB variants."""

    expected_candidates = {
        "internal_elo",
        "external_nfelo",
        "blend_25_internal_75_external",
        "internal_external_blend_50",
        "blend_75_internal_25_external",
        "internal_elo_listed_qb",
        "external_nfelo_listed_qb",
        "external_nfelo_external_qb",
        "external_nfelo_both_qb",
        (
            "blend_25_internal_75_external_"
            "listed_qb"
        ),
    }

    assert set(TRAINED_CANDIDATES) == (
        expected_candidates
    )

    assert TRAINED_CANDIDATES[
        "external_nfelo_listed_qb"
    ] == (
        EXTERNAL_FEATURE,
        LISTED_QB_FEATURE,
    )

    assert TRAINED_CANDIDATES[
        "external_nfelo_external_qb"
    ] == (
        EXTERNAL_FEATURE,
        EXTERNAL_QB_FEATURE,
    )


def test_backtest_returns_expected_candidate_rows():
    """Every Elo source should cover every fold."""

    summary, fold_results = (
        evaluate_rating_source_backtest(
            data=create_test_data(),
            validation_seasons=(
                TEST_VALIDATION_SEASONS
            ),
        )
    )

    expected_candidates = (
        set(TRAINED_CANDIDATES)
        | {
            PUBLISHED_NFELO_CANDIDATE,
        }
    )

    assert tuple(summary.columns) == (
        SUMMARY_COLUMNS
    )

    assert tuple(fold_results.columns) == (
        FOLD_RESULT_COLUMNS
    )

    assert set(
        summary["candidate_name"]
    ) == expected_candidates

    assert len(summary) == 11
    assert len(fold_results) == 22

    assert set(
        fold_results["candidate_name"]
    ) == expected_candidates


def test_backtest_uses_expanding_training_window():
    """Later folds must include earlier seasons."""

    _, fold_results = (
        evaluate_rating_source_backtest(
            data=create_test_data(),
            validation_seasons=(
                TEST_VALIDATION_SEASONS
            ),
        )
    )

    training_counts = (
        fold_results.groupby(
            "validation_season"
        )["training_game_count"]
        .unique()
        .to_dict()
    )

    assert training_counts[2021].tolist() == [4]
    assert training_counts[2022].tolist() == [6]

    validation_counts = (
        fold_results.groupby(
            "validation_season"
        )["validation_game_count"]
        .unique()
        .to_dict()
    )

    assert validation_counts[2021].tolist() == [2]
    assert validation_counts[2022].tolist() == [2]


def test_summary_uses_pooled_validation_games():
    """Each candidate should summarize all folds."""

    summary, _ = (
        evaluate_rating_source_backtest(
            data=create_test_data(),
            validation_seasons=(
                TEST_VALIDATION_SEASONS
            ),
        )
    )

    assert (
        summary["fold_count"] == 2
    ).all()

    assert (
        summary["validation_game_count"] == 4
    ).all()

    assert summary["accuracy"].between(
        0.0,
        1.0,
    ).all()

    assert summary["brier_score"].between(
        0.0,
        1.0,
    ).all()

    assert (
        summary["log_loss"] >= 0.0
    ).all()


def test_published_probability_has_no_spread_metrics():
    """Published nfelo probability is probability-only."""

    summary, fold_results = (
        evaluate_rating_source_backtest(
            data=create_test_data(),
            validation_seasons=(
                TEST_VALIDATION_SEASONS
            ),
        )
    )

    spread_columns = [
        "spread_mae",
        "spread_rmse",
        "spread_bias",
        "spread_r_squared",
    ]

    published_summary = summary.loc[
        summary["candidate_name"]
        == PUBLISHED_NFELO_CANDIDATE
    ].iloc[0]

    assert published_summary[
        spread_columns
    ].isna().all()

    published_folds = fold_results.loc[
        fold_results["candidate_name"]
        == PUBLISHED_NFELO_CANDIDATE
    ]

    assert published_folds[
        spread_columns
    ].isna().all().all()

    trained_summary = summary.loc[
        summary["candidate_name"]
        != PUBLISHED_NFELO_CANDIDATE
    ]

    assert trained_summary[
        spread_columns
    ].notna().all().all()