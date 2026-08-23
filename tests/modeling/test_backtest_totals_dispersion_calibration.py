"""Tests for Totals dispersion calibration."""

import pandas as pd
import pytest

from src.modeling.backtest_totals_dispersion_calibration import evaluate_dispersion_factors


def sample_predictions() -> pd.DataFrame:
    return pd.DataFrame({
        "game_id": ["a", "b", "c", "d"],
        "season": [2023] * 4,
        "target_total_points": [30.0, 40.0, 50.0, 60.0],
        "training_mean_total": [45.0] * 4,
        "raw_predicted_total": [40.0, 43.0, 47.0, 50.0],
    })


def test_factor_one_preserves_raw_predictions() -> None:
    _, detail = evaluate_dispersion_factors(sample_predictions(), factors=(1.0,))
    assert detail["predicted_total"].tolist() == [40.0, 43.0, 47.0, 50.0]


def test_larger_factor_increases_prediction_dispersion() -> None:
    summary, _ = evaluate_dispersion_factors(sample_predictions(), factors=(1.0, 1.5))
    values = summary.set_index("dispersion_factor")["prediction_standard_deviation"]
    assert values.loc[1.5] > values.loc[1.0]


def test_summary_is_sorted_by_mae() -> None:
    summary, _ = evaluate_dispersion_factors(sample_predictions(), factors=(0.75, 1.0, 1.5))
    assert summary["validation_mae"].is_monotonic_increasing


def test_missing_columns_are_rejected() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        evaluate_dispersion_factors(pd.DataFrame({"game_id": ["a"]}))


def test_nonpositive_factor_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        evaluate_dispersion_factors(sample_predictions(), factors=(0.0,))
