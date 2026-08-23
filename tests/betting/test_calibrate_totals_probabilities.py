"""Tests for leakage-safe Totals residual calibration."""

import pandas as pd
import pytest

from src.betting.calibrate_totals_probabilities import (
    RESIDUAL_COLUMNS,
    create_totals_calibration_residuals,
)
from src.modeling.production_totals_model import PRODUCTION_TOTALS_MODEL


def create_data() -> pd.DataFrame:
    rows = []
    for season in (2020, 2021, 2022):
        for index in range(2):
            row = {
                "game_id": f"{season}_{index}", "season": season,
                "split_name": "train" if season < 2022 else "validation",
                "target_total_points": 40.0 + season % 3 + index,
                "both_short_windows_complete": True,
            }
            for offset, feature in enumerate(PRODUCTION_TOTALS_MODEL.feature_columns):
                row[feature] = float(index + offset + 1)
            for offset, feature in enumerate(PRODUCTION_TOTALS_MODEL.fallback_feature_columns):
                row[feature] = float(index + offset + 2)
            rows.append(row)
    return pd.DataFrame(rows)


def test_calibration_returns_both_modes() -> None:
    residuals = create_totals_calibration_residuals(create_data(), (2021, 2022))
    assert tuple(residuals.columns) == RESIDUAL_COLUMNS
    assert set(residuals["prediction_mode"]) == {
        "RIDGE_TOTALS_PRIMARY", "RIDGE_TOTALS_FALLBACK"
    }
    assert len(residuals) == 8
    assert residuals["absolute_error"].ge(0.0).all()


def test_calibration_is_chronological() -> None:
    residuals = create_totals_calibration_residuals(create_data(), (2021, 2022))
    assert residuals.loc[residuals.validation_season == 2021, "training_game_count"].eq(2).all()
    assert residuals.loc[residuals.validation_season == 2022, "training_game_count"].eq(4).all()


def test_holdout_is_rejected() -> None:
    data = create_data()
    data.loc[0, "split_name"] = "holdout"
    with pytest.raises(ValueError, match="holdout"):
        create_totals_calibration_residuals(data, (2021, 2022))


def test_duplicate_game_is_rejected() -> None:
    data = pd.concat([create_data(), create_data().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        create_totals_calibration_residuals(data, (2021, 2022))
