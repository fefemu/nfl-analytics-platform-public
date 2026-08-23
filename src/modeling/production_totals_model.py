"""
NFL Analytics Platform
Production Totals Model Specification

Purpose:
    Define the primary and fallback totals models selected
    through chronological validation.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProductionTotalsModel:
    """Describe the frozen production totals models."""

    model_name: str
    model_version: str
    feature_columns: tuple[str, ...]
    ridge_alpha: float

    fallback_model_name: str
    fallback_feature_columns: tuple[str, ...]
    fallback_ridge_alpha: float

    target_column: str
    scoring_environment_window: int
    forward_test_season: int

    def __post_init__(self) -> None:
        """Validate the frozen specification."""

        if not self.model_name.strip():
            raise ValueError(
                "Totals model name must not be empty."
            )

        if not self.model_version.strip():
            raise ValueError(
                "Totals model version must not be empty."
            )

        if not self.feature_columns:
            raise ValueError(
                "Totals model features must not be empty."
            )

        if len(self.feature_columns) != len(
            set(self.feature_columns)
        ):
            raise ValueError(
                "Totals model features must be unique."
            )

        if self.ridge_alpha < 0.0:
            raise ValueError(
                "Totals Ridge alpha must not be negative."
            )

        if not self.fallback_model_name.strip():
            raise ValueError(
                "Totals fallback model name must not "
                "be empty."
            )

        if not self.fallback_feature_columns:
            raise ValueError(
                "Totals fallback features must not "
                "be empty."
            )

        if len(
            self.fallback_feature_columns
        ) != len(
            set(self.fallback_feature_columns)
        ):
            raise ValueError(
                "Totals fallback features must be unique."
            )

        if self.fallback_ridge_alpha < 0.0:
            raise ValueError(
                "Totals fallback Ridge alpha must not "
                "be negative."
            )

        if not self.target_column.strip():
            raise ValueError(
                "Totals target column must not be empty."
            )

        if self.scoring_environment_window <= 0:
            raise ValueError(
                "Scoring environment window must "
                "be positive."
            )

        if self.forward_test_season <= 0:
            raise ValueError(
                "Forward-test season must be positive."
            )


PRODUCTION_TOTALS_MODEL = ProductionTotalsModel(
    model_name=(
        "ridge_epa_weather_qb_league_64_totals"
    ),
    model_version="0.1.0",
    feature_columns=(
        "offensive_epa_sum_last_4",
        "defensive_epa_allowed_sum_last_4",
        "is_indoor",
        "has_game_weather",
        "cold_degrees_below_50",
        "heat_degrees_above_80",
        "wind_mph_above_10",
        "listed_qb_rating_sum",
        "league_average_total_last_64",
    ),
    ridge_alpha=100.0,
    fallback_model_name=(
        "ridge_league_64_indoor_elo_totals"
    ),
    fallback_feature_columns=(
        "league_average_total_last_64",
        "is_indoor",
        "elo_rating_sum",
    ),
    fallback_ridge_alpha=1.0,
    target_column="target_total_points",
    scoring_environment_window=64,
    forward_test_season=2026,
)