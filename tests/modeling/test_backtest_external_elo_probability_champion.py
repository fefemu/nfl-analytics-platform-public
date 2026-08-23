"""Tests for the external Elo probability champion backtest."""

import numpy as np
import pandas as pd
import pytest

from src.modeling.backtest_external_elo_probability_champion import (
    CANDIDATE_NAMES,
    COMPARISONS,
    CURRENT_BLEND_CANDIDATE,
    EXTERNAL_BLEND_CANDIDATE,
    EXTERNAL_ELO_FEATURE,
    EXTERNAL_LOGISTIC_CANDIDATE,
    EXTERNAL_QB_FEATURE,
    FOLD_RESULT_COLUMNS,
    INTERNAL_ELO_FEATURE,
    INTERNAL_LOGISTIC_CANDIDATE,
    LISTED_QB_FEATURE,
    PAIRED_SUMMARY_COLUMNS,
    PREDICTION_COLUMNS,
    SUMMARY_COLUMNS,
    bootstrap_paired_mean_delta,
    create_candidate_summary,
    create_champion_oof_predictions,
    create_paired_summary,
    evaluate_probability_champions,
    prepare_common_champion_sample,
)
from src.modeling.production_probability_model import (
    PRODUCTION_PROBABILITY_MODEL,
)
from src.modeling.run_logistic_injury_time_cv import (
    INJURY_AVAILABILITY_COLUMN,
    UNIT_BURDEN_FEATURES,
)
from src.modeling.train_logistic_baseline import (
    TARGET_COLUMN,
)


TEST_VALIDATION_SEASONS = (
    2021,
    2022,
)


def create_test_data() -> pd.DataFrame:
    """Create chronological champion-level data."""

    definitions = [
        (
            "2020_01_A_B",
            2020,
            "train",
            1,
            80.0,
            100.0,
            20.0,
            15.0,
            0.68,
            0.72,
        ),
        (
            "2020_02_C_D",
            2020,
            "train",
            0,
            -60.0,
            -90.0,
            -15.0,
            -12.0,
            0.32,
            0.28,
        ),
        (
            "2020_03_E_F",
            2020,
            "train",
            1,
            30.0,
            50.0,
            8.0,
            6.0,
            0.58,
            0.62,
        ),
        (
            "2020_04_G_H",
            2020,
            "train",
            0,
            -20.0,
            -40.0,
            -6.0,
            -5.0,
            0.42,
            0.38,
        ),
        (
            "2021_01_A_C",
            2021,
            "validation",
            1,
            45.0,
            70.0,
            12.0,
            9.0,
            0.62,
            0.66,
        ),
        (
            "2021_02_D_B",
            2021,
            "validation",
            0,
            -35.0,
            -55.0,
            -10.0,
            -8.0,
            0.38,
            0.34,
        ),
        (
            "2022_01_E_G",
            2022,
            "validation",
            1,
            55.0,
            85.0,
            14.0,
            11.0,
            0.65,
            0.69,
        ),
        (
            "2022_02_H_F",
            2022,
            "validation",
            0,
            -50.0,
            -75.0,
            -13.0,
            -10.0,
            0.35,
            0.31,
        ),
    ]

    rows: list[dict[str, object]] = []

    for row_index, (
        game_id,
        season,
        split_name,
        target,
        internal_elo,
        external_elo,
        listed_qb,
        external_qb,
        internal_probability,
        external_probability,
    ) in enumerate(definitions):
        row: dict[str, object] = {
            "game_id": game_id,
            "season": season,
            "split_name": split_name,
            TARGET_COLUMN: target,
            INJURY_AVAILABILITY_COLUMN: True,
            INTERNAL_ELO_FEATURE: internal_elo,
            EXTERNAL_ELO_FEATURE: external_elo,
            LISTED_QB_FEATURE: listed_qb,
            EXTERNAL_QB_FEATURE: external_qb,
            "elo_home_win_probability": (
                internal_probability
            ),
            "published_nfelo_home_probability": (
                external_probability
            ),
        }

        for (
            feature_index,
            feature_name,
        ) in enumerate(
            UNIT_BURDEN_FEATURES
        ):
            direction = (
                1.0
                if target == 1
                else -1.0
            )

            row[feature_name] = (
                direction
                * (
                    feature_index + 1
                )
                + row_index * 0.01
            )

        rows.append(row)

    return pd.DataFrame(rows)


def test_common_sample_preserves_complete_games():
    """Complete injury and external games remain."""

    data = create_test_data()

    sample = prepare_common_champion_sample(
        data
    )

    assert len(sample) == len(data)

    assert sample[
        INJURY_AVAILABILITY_COLUMN
    ].all()


def test_common_sample_removes_incomplete_games():
    """Every candidate must use one common sample."""

    data = create_test_data()

    data.loc[
        data["game_id"] == "2021_01_A_C",
        EXTERNAL_QB_FEATURE,
    ] = None

    sample = prepare_common_champion_sample(
        data
    )

    assert len(sample) == len(data) - 1

    assert (
        "2021_01_A_C"
        not in set(sample["game_id"])
    )


def test_common_sample_rejects_holdout():
    """The 2025 holdout must remain closed."""

    data = create_test_data()

    data.loc[
        data.index[-1],
        "split_name",
    ] = "holdout"

    with pytest.raises(
        ValueError,
        match="must not contain holdout",
    ):
        prepare_common_champion_sample(data)


def test_common_sample_rejects_2025():
    """No 2025 game may enter development."""

    data = create_test_data()

    data.loc[
        data.index[-1],
        "season",
    ] = 2025

    with pytest.raises(
        ValueError,
        match="must end before the 2025 holdout",
    ):
        prepare_common_champion_sample(data)


def test_oof_predictions_cover_every_candidate():
    """Every champion candidate predicts every game."""

    (
        predictions,
        fold_results,
    ) = create_champion_oof_predictions(
        development_data=create_test_data(),
        validation_seasons=(
            TEST_VALIDATION_SEASONS
        ),
    )

    assert tuple(predictions.columns) == (
        PREDICTION_COLUMNS
    )

    assert tuple(fold_results.columns) == (
        FOLD_RESULT_COLUMNS
    )

    assert set(
        predictions["candidate_name"]
    ) == set(CANDIDATE_NAMES)

    assert len(predictions) == (
        4 * len(CANDIDATE_NAMES)
    )

    assert len(fold_results) == (
        2 * len(CANDIDATE_NAMES)
    )

    candidate_counts = (
        predictions.groupby(
            "candidate_name"
        )["game_id"].nunique()
    )

    assert (
        candidate_counts == 4
    ).all()


def test_blends_use_production_weights():
    """Both blends must use frozen production weights."""

    data = create_test_data()

    predictions, _ = (
        create_champion_oof_predictions(
            development_data=data,
            validation_seasons=(
                TEST_VALIDATION_SEASONS
            ),
        )
    )

    logistic_weight = (
        PRODUCTION_PROBABILITY_MODEL
        .logistic_weight
    )

    elo_weight = (
        PRODUCTION_PROBABILITY_MODEL
        .elo_weight
    )

    internal_logistic = predictions.loc[
        predictions["candidate_name"]
        == INTERNAL_LOGISTIC_CANDIDATE,
        [
            "game_id",
            "home_win_probability",
        ],
    ].rename(
        columns={
            "home_win_probability": (
                "logistic_probability"
            ),
        }
    )

    current_blend = predictions.loc[
        predictions["candidate_name"]
        == CURRENT_BLEND_CANDIDATE,
        [
            "game_id",
            "home_win_probability",
        ],
    ].rename(
        columns={
            "home_win_probability": (
                "blend_probability"
            ),
        }
    )

    internal_check = (
        internal_logistic.merge(
            current_blend,
            on="game_id",
            validate="one_to_one",
        )
        .merge(
            data[
                [
                    "game_id",
                    "elo_home_win_probability",
                ]
            ],
            on="game_id",
            validate="one_to_one",
        )
    )

    expected_internal_blend = (
        logistic_weight
        * internal_check[
            "logistic_probability"
        ]
        + elo_weight
        * internal_check[
            "elo_home_win_probability"
        ]
    )

    np.testing.assert_allclose(
        internal_check["blend_probability"],
        expected_internal_blend,
    )

    external_logistic = predictions.loc[
        predictions["candidate_name"]
        == EXTERNAL_LOGISTIC_CANDIDATE,
        [
            "game_id",
            "home_win_probability",
        ],
    ].rename(
        columns={
            "home_win_probability": (
                "logistic_probability"
            ),
        }
    )

    external_blend = predictions.loc[
        predictions["candidate_name"]
        == EXTERNAL_BLEND_CANDIDATE,
        [
            "game_id",
            "home_win_probability",
        ],
    ].rename(
        columns={
            "home_win_probability": (
                "blend_probability"
            ),
        }
    )

    external_check = (
        external_logistic.merge(
            external_blend,
            on="game_id",
            validate="one_to_one",
        )
        .merge(
            data[
                [
                    "game_id",
                    (
                        "published_nfelo_"
                        "home_probability"
                    ),
                ]
            ],
            on="game_id",
            validate="one_to_one",
        )
    )

    expected_external_blend = (
        logistic_weight
        * external_check[
            "logistic_probability"
        ]
        + elo_weight
        * external_check[
            (
                "published_nfelo_"
                "home_probability"
            )
        ]
    )

    np.testing.assert_allclose(
        external_check["blend_probability"],
        expected_external_blend,
    )


def test_candidate_summary_has_expected_schema():
    """Pooled summary must contain every candidate."""

    predictions, _ = (
        create_champion_oof_predictions(
            development_data=create_test_data(),
            validation_seasons=(
                TEST_VALIDATION_SEASONS
            ),
        )
    )

    summary = create_candidate_summary(
        predictions
    )

    assert tuple(summary.columns) == (
        SUMMARY_COLUMNS
    )

    assert set(
        summary["candidate_name"]
    ) == set(CANDIDATE_NAMES)

    assert len(summary) == 4

    assert (
        summary["fold_count"] == 2
    ).all()

    assert (
        summary["validation_game_count"] == 4
    ).all()

    assert summary["brier_score"].between(
        0.0,
        1.0,
    ).all()


def test_paired_summary_contains_every_comparison():
    """Every planned paired comparison is returned."""

    predictions, _ = (
        create_champion_oof_predictions(
            development_data=create_test_data(),
            validation_seasons=(
                TEST_VALIDATION_SEASONS
            ),
        )
    )

    paired_summary = create_paired_summary(
        predictions=predictions,
        bootstrap_iterations=500,
        random_seed=42,
    )

    assert tuple(
        paired_summary.columns
    ) == PAIRED_SUMMARY_COLUMNS

    assert set(
        paired_summary["comparison_name"]
    ) == set(COMPARISONS)

    assert len(paired_summary) == 3

    total_rate = (
        paired_summary["challenger_win_rate"]
        + paired_summary[
            "challenger_loss_rate"
        ]
        + paired_summary[
            "equal_loss_rate"
        ]
    )

    np.testing.assert_allclose(
        total_rate,
        1.0,
    )


def test_full_evaluation_returns_all_outputs():
    """The combined evaluation returns four outputs."""

    (
        candidate_summary,
        paired_summary,
        fold_results,
        predictions,
    ) = evaluate_probability_champions(
        development_data=create_test_data(),
        validation_seasons=(
            TEST_VALIDATION_SEASONS
        ),
        bootstrap_iterations=500,
        random_seed=42,
    )

    assert len(candidate_summary) == 4
    assert len(paired_summary) == 3
    assert len(fold_results) == 8
    assert len(predictions) == 16


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