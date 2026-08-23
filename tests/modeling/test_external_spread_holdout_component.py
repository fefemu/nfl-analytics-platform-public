"""Tests for the external Spread holdout component."""

import numpy as np
import pandas as pd
import pytest

from src.modeling.backtest_external_elo_probability_champion import (
    EXTERNAL_ELO_FEATURE,
    EXTERNAL_QB_FEATURE,
    LISTED_QB_FEATURE,
)
from src.modeling.backtest_elo_rating_sources import (
    SPREAD_RIDGE_ALPHA,
)
from src.modeling.external_probability_holdout_component import (
    HOLDOUT_SEASON,
    HOLDOUT_SPLIT,
)
from src.modeling.external_spread_holdout_component import (
    CURRENT_CANDIDATE,
    EXTERNAL_CANDIDATE,
    PREDICTION_COLUMNS,
    SPREAD_TARGET_COLUMN,
    SUMMARY_COLUMNS,
    evaluate_locked_spread_holdout,
    load_spread_holdout_data,
    prepare_spread_holdout_data,
)
from src.modeling.production_spread_model import (
    PRODUCTION_SPREAD_MODEL,
)


INTERNAL_ELO_FEATURE = (
    "elo_rating_difference"
)


def create_test_data() -> pd.DataFrame:
    """Create synthetic Spread training and holdout data."""

    rows = [
        {
            "game_id": "2020_01_A_B",
            "season": 2020,
            "split_name": "train",
            SPREAD_TARGET_COLUMN: 10.0,
            INTERNAL_ELO_FEATURE: 80.0,
            LISTED_QB_FEATURE: 20.0,
            EXTERNAL_ELO_FEATURE: 100.0,
            EXTERNAL_QB_FEATURE: 15.0,
        },
        {
            "game_id": "2020_02_C_D",
            "season": 2020,
            "split_name": "train",
            SPREAD_TARGET_COLUMN: -7.0,
            INTERNAL_ELO_FEATURE: -60.0,
            LISTED_QB_FEATURE: -15.0,
            EXTERNAL_ELO_FEATURE: -90.0,
            EXTERNAL_QB_FEATURE: -12.0,
        },
        {
            "game_id": "2021_01_E_F",
            "season": 2021,
            "split_name": "validation",
            SPREAD_TARGET_COLUMN: 4.0,
            INTERNAL_ELO_FEATURE: 30.0,
            LISTED_QB_FEATURE: 8.0,
            EXTERNAL_ELO_FEATURE: 50.0,
            EXTERNAL_QB_FEATURE: 6.0,
        },
        {
            "game_id": "2021_02_G_H",
            "season": 2021,
            "split_name": "validation",
            SPREAD_TARGET_COLUMN: -3.0,
            INTERNAL_ELO_FEATURE: -20.0,
            LISTED_QB_FEATURE: None,
            EXTERNAL_ELO_FEATURE: -40.0,
            EXTERNAL_QB_FEATURE: -5.0,
        },
        {
            "game_id": "2025_01_A_C",
            "season": HOLDOUT_SEASON,
            "split_name": HOLDOUT_SPLIT,
            SPREAD_TARGET_COLUMN: 6.0,
            INTERNAL_ELO_FEATURE: 45.0,
            LISTED_QB_FEATURE: 12.0,
            EXTERNAL_ELO_FEATURE: 70.0,
            EXTERNAL_QB_FEATURE: 9.0,
        },
        {
            "game_id": "2025_02_D_B",
            "season": HOLDOUT_SEASON,
            "split_name": HOLDOUT_SPLIT,
            SPREAD_TARGET_COLUMN: -5.0,
            INTERNAL_ELO_FEATURE: -35.0,
            LISTED_QB_FEATURE: None,
            EXTERNAL_ELO_FEATURE: -55.0,
            EXTERNAL_QB_FEATURE: -8.0,
        },
    ]

    return pd.DataFrame(rows)


def test_prepare_creates_three_samples():
    """Primary, fallback and holdout samples differ."""

    (
        primary_training,
        fallback_training,
        holdout,
    ) = prepare_spread_holdout_data(
        create_test_data()
    )

    assert len(primary_training) == 3
    assert len(fallback_training) == 4
    assert len(holdout) == 2

    assert primary_training[
        LISTED_QB_FEATURE
    ].notna().all()

    assert (
        fallback_training["season"]
        < HOLDOUT_SEASON
    ).all()

    assert set(
        holdout["split_name"]
    ) == {
        HOLDOUT_SPLIT,
    }


def test_missing_listed_qb_remains_in_fallback():
    """Missing listed-QB data must not remove games."""

    (
        primary_training,
        fallback_training,
        holdout,
    ) = prepare_spread_holdout_data(
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
    """Each game identifier must be unique."""

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
        prepare_spread_holdout_data(data)


def test_prepare_rejects_post_holdout_games():
    """No game after 2025 may enter evaluation."""

    data = create_test_data()

    data.loc[
        data.index[-1],
        "season",
    ] = 2026

    with pytest.raises(
        ValueError,
        match="post-2025",
    ):
        prepare_spread_holdout_data(data)


def test_prepare_requires_holdout_games():
    """The protected holdout sample is mandatory."""

    data = create_test_data().loc[
        lambda frame:
        frame["season"] < HOLDOUT_SEASON
    ].copy()

    with pytest.raises(
        RuntimeError,
        match="No protected 2025",
    ):
        prepare_spread_holdout_data(data)


def test_evaluation_returns_locked_candidates():
    """Current routing and external model are evaluated."""

    summary, predictions = (
        evaluate_locked_spread_holdout(
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


def test_current_routing_uses_primary_and_fallback():
    """Current model routes by listed-QB coverage."""

    summary, predictions = (
        evaluate_locked_spread_holdout(
            create_test_data()
        )
    )

    current_row = summary.loc[
        summary["candidate_name"]
        == CURRENT_CANDIDATE
    ].iloc[0]

    external_row = summary.loc[
        summary["candidate_name"]
        == EXTERNAL_CANDIDATE
    ].iloc[0]

    assert (
        current_row[
            "primary_holdout_game_count"
        ]
        == 1
    )

    assert (
        current_row[
            "fallback_holdout_game_count"
        ]
        == 1
    )

    assert set(
        predictions["current_prediction_mode"]
    ) == {
        "PRIMARY",
        "FALLBACK",
    }

    assert (
        current_row["ridge_alpha"]
        == PRODUCTION_SPREAD_MODEL.ridge_alpha
    )

    assert (
        current_row["fallback_ridge_alpha"]
        == (
            PRODUCTION_SPREAD_MODEL
            .fallback_ridge_alpha
        )
    )

    assert (
        external_row["ridge_alpha"]
        == SPREAD_RIDGE_ALPHA
    )


def test_prediction_delta_matches_absolute_errors():
    """Paired delta must equal external minus current."""

    _, predictions = (
        evaluate_locked_spread_holdout(
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
            "current_predicted_home_margin"
        ]
        - predictions["actual_home_margin"]
    )

    expected_external_error = np.abs(
        predictions[
            "external_predicted_home_margin"
        ]
        - predictions["actual_home_margin"]
    )

    np.testing.assert_allclose(
        predictions["current_absolute_error"],
        expected_current_error,
    )

    np.testing.assert_allclose(
        predictions["external_absolute_error"],
        expected_external_error,
    )


def test_loader_does_not_filter_core_eligibility():
    """Spread holdout must include every target game."""

    class RecordingConnection:
        """Record the submitted DuckDB query."""

        def __init__(self):
            self.query = ""

        def execute(self, query):
            self.query = query
            return self

        def fetchdf(self):
            return pd.DataFrame(
                {
                    "game_id": [
                        "synthetic_game",
                    ],
                }
            )

    connection = RecordingConnection()

    result = load_spread_holdout_data(
        connection
    )

    normalized_query = " ".join(
        connection.query.split()
    ).lower()

    assert not result.empty

    assert (
        "is_core_model_eligible"
        not in normalized_query
    )

    assert (
        "dataset.season <= 2025"
        in normalized_query
    )