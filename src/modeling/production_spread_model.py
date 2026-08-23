"""
NFL Analytics Platform
Production Spread Model Specification

Purpose:
    Define the spread model selected through chronological
    validation and one final 2025 holdout evaluation.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProductionSpreadModel:
    """Describe the frozen production spread model."""

    model_name: str
    model_version: str
    feature_columns: tuple[str, ...]
    ridge_alpha: float
    fallback_model_name: str
    fallback_feature_columns: tuple[str, ...]
    fallback_ridge_alpha: float
    target_column: str
    forward_test_season: int

    def __post_init__(self) -> None:
        """Validate the frozen specification."""

        if not self.model_name.strip():
            raise ValueError(
                "Spread model name must not be empty."
            )

        if not self.model_version.strip():
            raise ValueError(
                "Spread model version must not be empty."
            )

        if not self.feature_columns:
            raise ValueError(
                "Spread model features must not be empty."
            )

        if len(self.feature_columns) != len(
            set(self.feature_columns)
        ):
            raise ValueError(
                "Spread model features must be unique."
            )

        if self.ridge_alpha < 0.0:
            raise ValueError(
                "Spread Ridge alpha must not be negative."
            )

        if not self.fallback_feature_columns:
            raise ValueError(
                "Spread fallback features must not be empty."
            )

        if self.fallback_ridge_alpha < 0.0:
            raise ValueError(
                "Spread fallback Ridge alpha must not "
                "be negative."
            )

        if not self.target_column.strip():
            raise ValueError(
                "Spread target column must not be empty."
            )


PRODUCTION_SPREAD_MODEL = ProductionSpreadModel(
    model_name="external_nfelo_external_qb_spread",
    model_version="0.2.0",
    feature_columns=(
        "external_nfelo_rating_difference",
        "external_nfelo_qb_adjustment_difference",
    ),
    ridge_alpha=10.0,
    fallback_model_name="external_nfelo_external_qb_spread",
    fallback_feature_columns=(
        "external_nfelo_rating_difference",
        "external_nfelo_qb_adjustment_difference",
    ),
    fallback_ridge_alpha=10.0,
    target_column="target_point_differential",
    forward_test_season=2026,
)
