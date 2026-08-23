"""
Tests for logistic calibration analysis.
"""

import pandas as pd
import pytest

from src.modeling.analyze_logistic_calibration import (
    build_calibration_table,
    calculate_expected_calibration_error,
    create_validation_predictions,
    evaluate_predictions_by_season,
)
from src.modeling.train_logistic_baseline import (
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    SCHEDULE_CONTEXT_FEATURE_COLUMNS,
)


def create_development_frame() -> pd.DataFrame:
    """Create deterministic train and validation data."""

    rows = []

    for index in range(40):
        target = index % 2

        row = {
            "game_id": f"game_{index}",
            "season": (
                2020
                if index < 20
                else (
                    2023
                    if index < 30
                    else 2024
                )
            ),
            "game_date": pd.Timestamp(
                "2020-01-01"
            )
            + pd.Timedelta(days=index),
            "split_name": (
                "train"
                if index < 20
                else "validation"
            ),
            TARGET_COLUMN: target,
            "elo_home_win_probability": (
                0.65 if target else 0.35
            ),
        }

        direction = (
            1.0 if target else -1.0
        )

        for feature_index, feature_name in enumerate(
            MODEL_FEATURE_COLUMNS
        ):
            row[feature_name] = (
                direction
                * (
                    1.0
                    + 0.01 * feature_index
                )
            )

        for (
            feature_index,
            feature_name,
        ) in enumerate(
            SCHEDULE_CONTEXT_FEATURE_COLUMNS
        ):
            row[feature_name] = (
                direction
                * (
                    0.10
                    + 0.01 * feature_index
                )
            )
        rows.append(row)

    return pd.DataFrame(rows)


def test_create_validation_predictions_uses_external_seasons() -> None:
    """Create predictions for 2023 and 2024 only."""

    data = create_development_frame()

    predictions = create_validation_predictions(
        data
    )

    assert set(predictions["season"]) == {
        2023,
        2024,
    }

    assert len(predictions) == 20


def test_evaluate_predictions_by_season_returns_models() -> None:
    """Evaluate logistic and Elo for each season."""

    data = create_development_frame()

    predictions = create_validation_predictions(
        data
    )

    results = evaluate_predictions_by_season(
        predictions
    )

    assert len(results) == 4

    assert set(results["model_name"]) == {
        "logistic_elo_qb_post_bye",
        "elo",
    }


def test_build_calibration_table_groups_probabilities() -> None:
    """Group predictions and calculate observed rates."""

    table = build_calibration_table(
        actual_values=pd.Series(
            [
                0,
                0,
                1,
                1,
            ]
        ),
        probabilities=pd.Series(
            [
                0.10,
                0.20,
                0.80,
                0.90,
            ]
        ),
        bin_count=5,
    )

    assert table["game_count"].sum() == 4

    assert set(
        table["observed_home_win_rate"]
    ) == {
        0.0,
        1.0,
    }


def test_calculate_expected_calibration_error() -> None:
    """Calculate weighted absolute calibration error."""

    calibration_table = pd.DataFrame(
        {
            "game_count": [
                2,
                2,
            ],
            "absolute_calibration_gap": [
                0.10,
                0.20,
            ],
        }
    )

    error = calculate_expected_calibration_error(
        calibration_table
    )

    assert error == pytest.approx(
        0.15
    )


def test_build_calibration_table_rejects_length_mismatch() -> None:
    """Reject unequal target and probability arrays."""

    with pytest.raises(
        ValueError,
        match="equal length",
    ):
        build_calibration_table(
            actual_values=pd.Series(
                [
                    0,
                    1,
                ]
            ),
            probabilities=pd.Series(
                [
                    0.5,
                ]
            ),
        )