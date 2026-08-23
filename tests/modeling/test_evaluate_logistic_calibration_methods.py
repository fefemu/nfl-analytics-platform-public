"""
Tests for leakage-safe calibration methods.
"""

import numpy as np
import pandas as pd

from src.modeling.evaluate_logistic_calibration_methods import (
    apply_probability_calibrators,
    create_calibrated_validation_predictions,
    create_time_cv_oof_predictions,
    evaluate_calibration_methods,
    fit_probability_calibrators,
    probabilities_to_logits,
)
from src.modeling.train_logistic_baseline import (
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    SCHEDULE_CONTEXT_FEATURE_COLUMNS,
)


def create_development_frame() -> pd.DataFrame:
    """Create deterministic multi-season model data."""

    rows = []

    for season in range(2018, 2025):
        for game_index in range(24):
            target = game_index % 2

            row = {
                "game_id": (
                    f"{season}_{game_index}_A_B"
                ),
                "season": season,
                "game_date": pd.Timestamp(
                    f"{season}-09-01"
                )
                + pd.Timedelta(
                    days=game_index
                ),
                "split_name": (
                    "train"
                    if season <= 2022
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

            for (
                feature_index,
                feature_name,
            ) in enumerate(
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


def test_probabilities_to_logits_returns_finite_values() -> None:
    """Clip boundary probabilities before logit conversion."""

    logits = probabilities_to_logits(
        np.array(
            [
                0.0,
                0.5,
                1.0,
            ]
        )
    )

    assert np.all(
        np.isfinite(logits)
    )

    assert logits[1] == 0.0


def test_create_oof_predictions_uses_internal_seasons() -> None:
    """Create predictions for 2020 through 2022."""

    data = create_development_frame()

    predictions = create_time_cv_oof_predictions(
        data
    )

    assert set(predictions["season"]) == {
        2020,
        2021,
        2022,
    }

    assert len(predictions) == 72


def test_fit_and_apply_probability_calibrators() -> None:
    """Fit sigmoid and isotonic calibrators."""

    oof_predictions = pd.DataFrame(
        {
            "actual_home_win": [
                0,
                0,
                1,
                1,
            ],
            "raw_probability": [
                0.20,
                0.40,
                0.60,
                0.80,
            ],
        }
    )

    calibrators = fit_probability_calibrators(
        oof_predictions
    )

    calibrated = apply_probability_calibrators(
        raw_probabilities=np.array(
            [
                0.30,
                0.70,
            ]
        ),
        calibrators=calibrators,
    )

    for probabilities in calibrated.values():
        assert np.all(
            probabilities >= 0.0
        )

        assert np.all(
            probabilities <= 1.0
        )


def test_create_calibrated_validation_predictions() -> None:
    """Create raw and calibrated external predictions."""

    data = create_development_frame()

    (
        oof_predictions,
        validation_predictions,
    ) = create_calibrated_validation_predictions(
        data
    )

    assert len(oof_predictions) == 72
    assert len(validation_predictions) == 48

    assert {
        "sigmoid_probability",
        "isotonic_probability",
    }.issubset(
        validation_predictions.columns
    )


def test_evaluate_calibration_methods_returns_all_models() -> None:
    """Compare raw, calibrated and Elo probabilities."""

    data = create_development_frame()

    (
        _,
        validation_predictions,
    ) = create_calibrated_validation_predictions(
        data
    )

    results = evaluate_calibration_methods(
        validation_predictions
    )

    assert set(results["model_name"]) == {
        "logistic_raw",
        "logistic_sigmoid",
        "logistic_isotonic",
        "elo",
    }

    assert set(results["game_count"]) == {
        48,
    }