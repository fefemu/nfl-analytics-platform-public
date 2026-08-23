"""Tests for the external Totals holdout component."""

import numpy as np
import pandas as pd
import pytest

from src.modeling.backtest_external_elo_totals_candidates import (
    EXTERNAL_ELO_SUM_FEATURE,
    EXTERNAL_QB_SUM_FEATURE,
)
from src.modeling.evaluate_totals_model_candidates import (
    TOTALS_TARGET_COLUMN,
)
from src.modeling.external_probability_holdout_component import (
    HOLDOUT_SEASON,
    HOLDOUT_SPLIT,
)
from src.modeling.external_totals_holdout_component import (
    CURRENT_CANDIDATE,
    CURRENT_FALLBACK_FEATURES,
    CURRENT_PRIMARY_FEATURES,
    EXTERNAL_CANDIDATE,
    EXTERNAL_FALLBACK_FEATURES,
    EXTERNAL_PRIMARY_FEATURES,
    PREDICTION_COLUMNS,
    SUMMARY_COLUMNS,
    evaluate_locked_totals_routing_holdout,
    prepare_totals_routing_data,
)
from src.modeling.production_totals_model import (
    PRODUCTION_TOTALS_MODEL,
)


def create_test_data() -> pd.DataFrame:
    """Create synthetic Totals routing data."""

    definitions = [
        (
            "2020_01_A_B",
            2020,
            "train",
            50.0,
            True,
            1.0,
        ),
        (
            "2020_02_C_D",
            2020,
            "train",
            38.0,
            True,
            -1.0,
        ),
        (
            "2021_01_E_F",
            2021,
            "validation",
            47.0,
            True,
            0.7,
        ),
        (
            "2021_02_G_H",
            2021,
            "validation",
            41.0,
            False,
            -0.6,
        ),
        (
            "2025_01_A_C",
            HOLDOUT_SEASON,
            HOLDOUT_SPLIT,
            49.0,
            True,
            0.8,
        ),
        (
            "2025_02_D_B",
            HOLDOUT_SEASON,
            HOLDOUT_SPLIT,
            40.0,
            False,
            -0.8,
        ),
    ]

    rows: list[dict[str, object]] = []

    for row_index, (
        game_id,
        season,
        split_name,
        target_total,
        short_windows_complete,
        signal,
    ) in enumerate(definitions):
        row: dict[str, object] = {
            "game_id": game_id,
            "season": season,
            "split_name": split_name,
            "both_short_windows_complete": (
                short_windows_complete
            ),
            TOTALS_TARGET_COLUMN: target_total,
        }

        for (
            feature_index,
            feature_name,
        ) in enumerate(
            CURRENT_PRIMARY_FEATURES
        ):
            row[feature_name] = (
                signal
                * (
                    feature_index + 1
                )
                + row_index * 0.01
            )

        row[
            "league_average_total_last_64"
        ] = 44.0 + row_index * 0.1

        row["is_indoor"] = float(
            row_index % 2
        )

        row["elo_rating_sum"] = (
            3000.0 + signal * 100.0
        )

        row[EXTERNAL_ELO_SUM_FEATURE] = (
            3050.0 + signal * 120.0
        )

        row[EXTERNAL_QB_SUM_FEATURE] = (
            signal * 15.0
        )

        rows.append(row)

    return pd.DataFrame(rows)


def test_fixture_covers_locked_feature_sets():
    """Synthetic data covers every routed model."""

    data = create_test_data()

    expected_features = {
        *CURRENT_PRIMARY_FEATURES,
        *EXTERNAL_PRIMARY_FEATURES,
        *CURRENT_FALLBACK_FEATURES,
        *EXTERNAL_FALLBACK_FEATURES,
    }

    assert expected_features.issubset(
        data.columns
    )


def test_prepare_creates_routing_samples():
    """Primary, fallback and holdout samples differ."""

    (
        primary_training,
        fallback_training,
        holdout,
    ) = prepare_totals_routing_data(
        create_test_data()
    )

    assert len(primary_training) == 3
    assert len(fallback_training) == 4
    assert len(holdout) == 2

    assert primary_training[
        "both_short_windows_complete"
    ].all()

    assert (
        fallback_training["season"]
        < HOLDOUT_SEASON
    ).all()

    assert set(
        holdout["split_name"]
    ) == {
        HOLDOUT_SPLIT,
    }


def test_incomplete_windows_remain_for_fallback():
    """Incomplete rolling windows must route fallback."""

    (
        primary_training,
        fallback_training,
        holdout,
    ) = prepare_totals_routing_data(
        create_test_data()
    )

    assert (
        "2021_02_G_H"
        not in set(
            primary_training["game_id"]
        )
    )

    assert (
        "2021_02_G_H"
        in set(
            fallback_training["game_id"]
        )
    )

    assert (
        "2025_02_D_B"
        in set(holdout["game_id"])
    )


def test_prepare_rejects_duplicate_games():
    """Game identifiers must be unique."""

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
        prepare_totals_routing_data(data)


def test_prepare_rejects_post_holdout_games():
    """No post-2025 game may enter evaluation."""

    data = create_test_data()

    data.loc[
        data.index[-1],
        "season",
    ] = 2026

    with pytest.raises(
        ValueError,
        match="post-2025",
    ):
        prepare_totals_routing_data(data)


def test_evaluation_returns_locked_candidates():
    """Current and external routing are evaluated."""

    summary, predictions = (
        evaluate_locked_totals_routing_holdout(
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
        CURRENT_CANDIDATE,
        EXTERNAL_CANDIDATE,
    }

    assert len(summary) == 2
    assert len(predictions) == 2

    assert (
        summary["holdout_game_count"] == 2
    ).all()

    assert (
        summary["holdout_mae"] >= 0.0
    ).all()

    assert (
        summary["holdout_rmse"] >= 0.0
    ).all()


def test_evaluation_uses_identical_routing():
    """Both systems must share the same route mask."""

    summary, predictions = (
        evaluate_locked_totals_routing_holdout(
            create_test_data()
        )
    )

    assert set(
        predictions["prediction_mode"]
    ) == {
        "PRIMARY",
        "FALLBACK",
    }

    assert (
        summary[
            "primary_holdout_game_count"
        ] == 1
    ).all()

    assert (
        summary[
            "fallback_holdout_game_count"
        ] == 1
    ).all()

    assert (
        summary["primary_ridge_alpha"]
        == PRODUCTION_TOTALS_MODEL.ridge_alpha
    ).all()

    assert (
        summary["fallback_ridge_alpha"]
        == (
            PRODUCTION_TOTALS_MODEL
            .fallback_ridge_alpha
        )
    ).all()


def test_prediction_delta_matches_absolute_errors():
    """Paired delta must equal external minus current."""

    _, predictions = (
        evaluate_locked_totals_routing_holdout(
            create_test_data()
        )
    )

    expected_delta = (
        predictions["external_absolute_error"]
        - predictions["current_absolute_error"]
    )

    np.testing.assert_allclose(
        predictions[
            "external_absolute_error_delta"
        ],
        expected_delta,
    )

    expected_current_error = np.abs(
        predictions[
            "current_predicted_total"
        ]
        - predictions["actual_total"]
    )

    expected_external_error = np.abs(
        predictions[
            "external_predicted_total"
        ]
        - predictions["actual_total"]
    )

    np.testing.assert_allclose(
        predictions["current_absolute_error"],
        expected_current_error,
    )

    np.testing.assert_allclose(
        predictions["external_absolute_error"],
        expected_external_error,
    )