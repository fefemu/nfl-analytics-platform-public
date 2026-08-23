"""Tests for the production totals specification."""

import pytest

from src.modeling.production_totals_model import (
    PRODUCTION_TOTALS_MODEL,
    ProductionTotalsModel,
)


def create_model(
    **overrides,
) -> ProductionTotalsModel:
    """Create a valid specification with overrides."""

    values = {
        "model_name": "primary_totals",
        "model_version": "0.1.0",
        "feature_columns": (
            "primary_feature",
        ),
        "ridge_alpha": 100.0,
        "fallback_model_name": (
            "fallback_totals"
        ),
        "fallback_feature_columns": (
            "fallback_feature",
        ),
        "fallback_ridge_alpha": 1.0,
        "target_column": (
            "target_total_points"
        ),
        "scoring_environment_window": 64,
        "forward_test_season": 2026,
    }

    values.update(overrides)

    return ProductionTotalsModel(**values)


def test_production_totals_models_are_frozen() -> None:
    """Expose the locked production settings."""

    assert PRODUCTION_TOTALS_MODEL.feature_columns == (
        "offensive_epa_sum_last_4",
        "defensive_epa_allowed_sum_last_4",
        "is_indoor",
        "has_game_weather",
        "cold_degrees_below_50",
        "heat_degrees_above_80",
        "wind_mph_above_10",
        "listed_qb_rating_sum",
        "league_average_total_last_64",
    )

    assert PRODUCTION_TOTALS_MODEL.ridge_alpha == 100.0

    assert (
        PRODUCTION_TOTALS_MODEL
        .fallback_feature_columns
        == (
            "league_average_total_last_64",
            "is_indoor",
            "elo_rating_sum",
        )
    )

    assert (
        PRODUCTION_TOTALS_MODEL
        .fallback_ridge_alpha
        == 1.0
    )

    assert (
        PRODUCTION_TOTALS_MODEL
        .scoring_environment_window
        == 64
    )


def test_negative_primary_alpha_is_rejected() -> None:
    """Reject an invalid primary alpha."""

    with pytest.raises(
        ValueError,
        match="must not be negative",
    ):
        create_model(
            ridge_alpha=-1.0
        )


def test_negative_fallback_alpha_is_rejected() -> None:
    """Reject an invalid fallback alpha."""

    with pytest.raises(
        ValueError,
        match="must not be negative",
    ):
        create_model(
            fallback_ridge_alpha=-1.0
        )


def test_duplicate_features_are_rejected() -> None:
    """Require unique primary and fallback schemas."""

    with pytest.raises(
        ValueError,
        match="must be unique",
    ):
        create_model(
            feature_columns=(
                "feature",
                "feature",
            )
        )

    with pytest.raises(
        ValueError,
        match="must be unique",
    ):
        create_model(
            fallback_feature_columns=(
                "feature",
                "feature",
            )
        )