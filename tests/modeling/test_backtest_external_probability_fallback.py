"""Tests for external probability fallback backtests."""

import numpy as np
import pandas as pd
import pytest

from src.modeling.backtest_external_elo_probability_champion import (
    EXTERNAL_ELO_FEATURE,
    EXTERNAL_QB_FEATURE,
)
from src.modeling.backtest_external_probability_fallback import (
    CANDIDATE_NAMES,
    COMPARISONS,
    CURRENT_FALLBACK_CANDIDATE,
    EXTERNAL_BLEND_CANDIDATE,
    EXTERNAL_LOGISTIC_CANDIDATE,
    FOLD_RESULT_COLUMNS,
    PAIRED_SUMMARY_COLUMNS,
    PREDICTION_COLUMNS,
    PRIMARY_ELIGIBILITY_COLUMN,
    PUBLISHED_NFELO_CANDIDATE,
    SUMMARY_COLUMNS,
    create_candidate_summary,
    create_fallback_oof_predictions,
    create_paired_summary,
    evaluate_probability_fallbacks,
    prepare_probability_fallback_sample,
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
    """Create chronological fallback routing data."""

    definitions = [
        (
            "2020_01_A_B",
            2020,
            "train",
            1,
            True,
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
            True,
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
            False,
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
            False,
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
            True,
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
            False,
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
            True,
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
            False,
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
        injury_complete,
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
            "is_core_model_eligible": (
                injury_complete
            ),
            TARGET_COLUMN: target,
            INJURY_AVAILABILITY_COLUMN: (
                injury_complete
            ),
            "elo_rating_difference": (
                internal_elo
            ),
            "listed_qb_rating_difference": (
                listed_qb
            ),
            EXTERNAL_ELO_FEATURE: (
                external_elo
            ),
            EXTERNAL_QB_FEATURE: (
                external_qb
            ),
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


def test_prepare_marks_primary_eligibility():
    """Injury coverage controls historical routing."""

    sample = prepare_probability_fallback_sample(
        create_test_data()
    )

    eligible_games = set(
        sample.loc[
            sample[
                PRIMARY_ELIGIBILITY_COLUMN
            ],
            "game_id",
        ]
    )

    fallback_games = set(
        sample.loc[
            ~sample[
                PRIMARY_ELIGIBILITY_COLUMN
            ],
            "game_id",
        ]
    )

    assert "2021_01_A_C" in eligible_games
    assert "2021_02_D_B" in fallback_games
    assert "2022_01_E_G" in eligible_games
    assert "2022_02_H_F" in fallback_games


def test_prepare_rejects_holdout_split():
    """The fallback backtest must not load holdout."""

    data = create_test_data()

    data.loc[
        data.index[-1],
        "split_name",
    ] = "holdout"

    with pytest.raises(
        ValueError,
        match="must not contain holdout",
    ):
        prepare_probability_fallback_sample(
            data
        )


def test_prepare_rejects_2025():
    """No 2025 game may enter fallback selection."""

    data = create_test_data()

    data.loc[
        data.index[-1],
        "season",
    ] = 2025

    with pytest.raises(
        ValueError,
        match="must end before the 2025 holdout",
    ):
        prepare_probability_fallback_sample(
            data
        )


def test_oof_predictions_include_only_fallback_games():
    """Evaluation must exclude primary-eligible games."""

    (
        predictions,
        fold_results,
    ) = create_fallback_oof_predictions(
        source_data=create_test_data(),
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

    assert set(
        predictions["game_id"]
    ) == {
        "2021_02_D_B",
        "2022_02_H_F",
    }

    assert len(predictions) == 8
    assert len(fold_results) == 8


def test_external_blend_uses_production_weights():
    """Fallback blend must use frozen 70/30 weights."""

    data = create_test_data()

    predictions, _ = (
        create_fallback_oof_predictions(
            source_data=data,
            validation_seasons=(
                TEST_VALIDATION_SEASONS
            ),
        )
    )

    logistic_predictions = predictions.loc[
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

    blend_predictions = predictions.loc[
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

    comparison = (
        logistic_predictions.merge(
            blend_predictions,
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

    expected_blend = (
        PRODUCTION_PROBABILITY_MODEL
        .logistic_weight
        * comparison[
            "logistic_probability"
        ]
        + PRODUCTION_PROBABILITY_MODEL
        .elo_weight
        * comparison[
            (
                "published_nfelo_"
                "home_probability"
            )
        ]
    )

    np.testing.assert_allclose(
        comparison["blend_probability"],
        expected_blend,
    )


def test_candidate_summary_contains_all_models():
    """Every fallback candidate is summarized."""

    predictions, _ = (
        create_fallback_oof_predictions(
            source_data=create_test_data(),
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
        summary[
            "fallback_validation_game_count"
        ] == 2
    ).all()


def test_paired_summary_contains_comparisons():
    """Every fallback challenger is compared."""

    predictions, _ = (
        create_fallback_oof_predictions(
            source_data=create_test_data(),
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

    assert set(
        paired_summary["base_candidate"]
    ) == {
        CURRENT_FALLBACK_CANDIDATE,
    }

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
    """Combined fallback evaluation returns four tables."""

    (
        candidate_summary,
        paired_summary,
        fold_results,
        predictions,
    ) = evaluate_probability_fallbacks(
        source_data=create_test_data(),
        validation_seasons=(
            TEST_VALIDATION_SEASONS
        ),
        bootstrap_iterations=500,
        random_seed=42,
    )

    assert len(candidate_summary) == 4
    assert len(paired_summary) == 3
    assert len(fold_results) == 8
    assert len(predictions) == 8


def test_published_candidate_uses_source_probability():
    """Published candidate must preserve source values."""

    data = create_test_data()

    predictions, _ = (
        create_fallback_oof_predictions(
            source_data=data,
            validation_seasons=(
                TEST_VALIDATION_SEASONS
            ),
        )
    )

    published = predictions.loc[
        predictions["candidate_name"]
        == PUBLISHED_NFELO_CANDIDATE,
        [
            "game_id",
            "home_win_probability",
        ],
    ]

    expected = data.loc[
        data["game_id"].isin(
            published["game_id"]
        ),
        [
            "game_id",
            (
                "published_nfelo_"
                "home_probability"
            ),
        ],
    ]

    comparison = published.merge(
        expected,
        on="game_id",
        validate="one_to_one",
    )

    np.testing.assert_allclose(
        comparison[
            "home_win_probability"
        ],
        comparison[
            (
                "published_nfelo_"
                "home_probability"
            )
        ],
    )