"""Tests for the external probability holdout component."""

import numpy as np
import pandas as pd
import pytest

from src.modeling.backtest_external_elo_probability_champion import (
    CURRENT_BLEND_CANDIDATE,
    EXTERNAL_BLEND_CANDIDATE,
    EXTERNAL_ELO_FEATURE,
    EXTERNAL_LOGISTIC_FEATURES,
    EXTERNAL_QB_FEATURE,
    INTERNAL_LOGISTIC_FEATURES,
    LISTED_QB_FEATURE,
)
from src.modeling.external_probability_holdout_component import (
    HOLDOUT_SEASON,
    HOLDOUT_SPLIT,
    PREDICTION_COLUMNS,
    SUMMARY_COLUMNS,
    evaluate_locked_probability_holdout,
    prepare_probability_holdout_data,
)
from src.modeling.run_logistic_injury_time_cv import (
    INJURY_AVAILABILITY_COLUMN,
    UNIT_BURDEN_FEATURES,
)
from src.modeling.train_logistic_baseline import (
    TARGET_COLUMN,
)


def create_test_data() -> pd.DataFrame:
    """Create synthetic training and holdout data."""

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
            "2021_01_E_F",
            2021,
            "validation",
            1,
            35.0,
            55.0,
            9.0,
            7.0,
            0.59,
            0.63,
        ),
        (
            "2021_02_G_H",
            2021,
            "validation",
            0,
            -30.0,
            -50.0,
            -8.0,
            -6.0,
            0.41,
            0.37,
        ),
        (
            "2025_01_A_C",
            HOLDOUT_SEASON,
            HOLDOUT_SPLIT,
            1,
            50.0,
            75.0,
            13.0,
            10.0,
            0.63,
            0.67,
        ),
        (
            "2025_02_D_B",
            HOLDOUT_SEASON,
            HOLDOUT_SPLIT,
            0,
            -45.0,
            -70.0,
            -11.0,
            -9.0,
            0.37,
            0.33,
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
            "elo_rating_difference": (
                internal_elo
            ),
            EXTERNAL_ELO_FEATURE: (
                external_elo
            ),
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


def test_feature_sets_are_available_in_fixture():
    """The fixture covers both champion feature sets."""

    data = create_test_data()

    assert set(
        INTERNAL_LOGISTIC_FEATURES
    ).issubset(data.columns)

    assert set(
        EXTERNAL_LOGISTIC_FEATURES
    ).issubset(data.columns)


def test_prepare_separates_training_and_holdout():
    """Pre-2025 and holdout rows must remain separate."""

    training_data, holdout_data = (
        prepare_probability_holdout_data(
            create_test_data()
        )
    )

    assert len(training_data) == 4
    assert len(holdout_data) == 2

    assert (
        training_data["season"]
        < HOLDOUT_SEASON
    ).all()

    assert (
        holdout_data["season"]
        == HOLDOUT_SEASON
    ).all()

    assert set(
        holdout_data["split_name"]
    ) == {
        HOLDOUT_SPLIT,
    }


def test_prepare_rejects_duplicate_games():
    """Each game may occur only once."""

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
        prepare_probability_holdout_data(
            data
        )


def test_prepare_rejects_post_holdout_games():
    """No season after 2025 may enter evaluation."""

    data = create_test_data()

    data.loc[
        data.index[-1],
        "season",
    ] = 2026

    with pytest.raises(
        ValueError,
        match="post-2025",
    ):
        prepare_probability_holdout_data(
            data
        )


def test_prepare_requires_holdout_rows():
    """A protected holdout sample is required."""

    data = create_test_data().loc[
        lambda frame:
        frame["season"] < HOLDOUT_SEASON
    ].copy()

    with pytest.raises(
        RuntimeError,
        match="No protected 2025",
    ):
        prepare_probability_holdout_data(
            data
        )


def test_incomplete_rows_use_common_sample():
    """Missing external data removes the row for both."""

    data = create_test_data()

    data.loc[
        data["game_id"] == "2025_01_A_C",
        EXTERNAL_QB_FEATURE,
    ] = None

    _, holdout_data = (
        prepare_probability_holdout_data(
            data
        )
    )

    assert len(holdout_data) == 1

    assert (
        "2025_01_A_C"
        not in set(holdout_data["game_id"])
    )


def test_evaluation_returns_both_locked_candidates():
    """Both locked champions use the same holdout."""

    summary, predictions = (
        evaluate_locked_probability_holdout(
            create_test_data()
        )
    )

    assert tuple(summary.columns) == (
        SUMMARY_COLUMNS
    )

    assert tuple(predictions.columns) == (
        PREDICTION_COLUMNS
    )

    assert set(
        summary["candidate_name"]
    ) == {
        CURRENT_BLEND_CANDIDATE,
        EXTERNAL_BLEND_CANDIDATE,
    }

    assert len(summary) == 2
    assert len(predictions) == 2

    assert (
        summary["training_game_count"] == 4
    ).all()

    assert (
        summary["holdout_game_count"] == 2
    ).all()

    assert summary["brier_score"].between(
        0.0,
        1.0,
    ).all()

    assert (
        summary["log_loss"] >= 0.0
    ).all()


def test_prediction_delta_matches_brier_losses():
    """Paired loss delta must be challenger minus current."""

    _, predictions = (
        evaluate_locked_probability_holdout(
            create_test_data()
        )
    )

    expected_delta = (
        predictions["external_brier_loss"]
        - predictions["current_brier_loss"]
    )

    np.testing.assert_allclose(
        predictions[
            "external_brier_loss_delta"
        ],
        expected_delta,
    )

    expected_current_loss = np.square(
        predictions[
            "current_home_win_probability"
        ]
        - predictions["actual_home_win"]
    )

    expected_external_loss = np.square(
        predictions[
            "external_home_win_probability"
        ]
        - predictions["actual_home_win"]
    )

    np.testing.assert_allclose(
        predictions["current_brier_loss"],
        expected_current_loss,
    )

    np.testing.assert_allclose(
        predictions["external_brier_loss"],
        expected_external_loss,
    )