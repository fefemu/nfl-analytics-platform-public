"""
NFL Analytics Platform
Production Probability Model Specification

Purpose:
    Define the selected 2026 production probability
    routing, including the external nfelo primary blend
    and external Elo-QB logistic fallback.

Governance basis:
    Both routing layers were selected with chronological
    development backtests and confirmed once on the
    protected 2025 holdout.

Forward-test policy:
    The next untouched production evaluation season
    is 2026.

Author:
    Ferenc Kaizer

Version:
    0.2.0
"""

from dataclasses import dataclass

from src.modeling.run_logistic_injury_time_cv import (
    UNIT_BURDEN_FEATURES,
)


EXTERNAL_ELO_FEATURE = (
    "external_nfelo_rating_difference"
)

LISTED_QB_FEATURE = (
    "listed_qb_rating_difference"
)

EXTERNAL_QB_FEATURE = (
    "external_nfelo_qb_adjustment_difference"
)

EXTERNAL_PRIMARY_FEATURES = (
    EXTERNAL_ELO_FEATURE,
    LISTED_QB_FEATURE,
    EXTERNAL_QB_FEATURE,
    *UNIT_BURDEN_FEATURES,
)

EXTERNAL_FALLBACK_FEATURES = (
    EXTERNAL_ELO_FEATURE,
    EXTERNAL_QB_FEATURE,
)


@dataclass(frozen=True)
class ProductionProbabilityModel:
    """Describe the selected probability routing."""

    model_name: str
    model_version: str
    logistic_component_name: str
    logistic_feature_columns: tuple[str, ...]
    logistic_regularization_c: float
    logistic_weight: float
    elo_weight: float
    blend_probability_source: str
    classification_threshold: float
    requires_complete_injury_data: bool
    incomplete_injury_fallback_model: str
    fallback_feature_columns: tuple[str, ...]
    fallback_regularization_c: float
    forward_test_season: int

    def __post_init__(self) -> None:
        """Validate the frozen production specification."""

        if not self.model_name.strip():
            raise ValueError(
                "Production model name must not be empty."
            )

        if not self.model_version.strip():
            raise ValueError(
                "Production model version must not be empty."
            )

        if not self.logistic_component_name.strip():
            raise ValueError(
                "Production logistic component name "
                "must not be empty."
            )

        if not self.logistic_feature_columns:
            raise ValueError(
                "Production logistic features must not "
                "be empty."
            )

        if len(
            self.logistic_feature_columns
        ) != len(
            set(
                self.logistic_feature_columns
            )
        ):
            raise ValueError(
                "Production logistic features must be "
                "unique."
            )

        if self.logistic_regularization_c <= 0.0:
            raise ValueError(
                "Production logistic C must be positive."
            )

        if not 0.0 <= self.logistic_weight <= 1.0:
            raise ValueError(
                "Production logistic weight must be "
                "between zero and one."
            )

        if not 0.0 <= self.elo_weight <= 1.0:
            raise ValueError(
                "Production Elo weight must be between "
                "zero and one."
            )

        if abs(
            self.logistic_weight
            + self.elo_weight
            - 1.0
        ) > 0.000000001:
            raise ValueError(
                "Production blend weights must sum to one."
            )

        if not self.blend_probability_source.strip():
            raise ValueError(
                "Production blend probability source "
                "must not be empty."
            )

        if not 0.0 < self.classification_threshold < 1.0:
            raise ValueError(
                "Classification threshold must be "
                "between zero and one."
            )

        if not self.incomplete_injury_fallback_model.strip():
            raise ValueError(
                "Production fallback model must not "
                "be empty."
            )

        if not self.fallback_feature_columns:
            raise ValueError(
                "Production fallback features must not "
                "be empty."
            )

        if len(
            self.fallback_feature_columns
        ) != len(
            set(
                self.fallback_feature_columns
            )
        ):
            raise ValueError(
                "Production fallback features must be "
                "unique."
            )

        if self.fallback_regularization_c <= 0.0:
            raise ValueError(
                "Production fallback logistic C must "
                "be positive."
            )


PRODUCTION_MODEL_DEPLOYMENT_STATUS = (
    "selected_for_2026_forward_test"
)


PRODUCTION_PROBABILITY_MODEL = (
    ProductionProbabilityModel(
        model_name=(
            "external_nfelo_probability_routing"
        ),
        model_version="0.3.0",
        logistic_component_name=(
            "external_elo_qb_injury_logistic"
        ),
        logistic_feature_columns=(
            EXTERNAL_PRIMARY_FEATURES
        ),
        logistic_regularization_c=0.1,
        logistic_weight=0.70,
        elo_weight=0.30,
        blend_probability_source=(
            "published_nfelo_home_probability"
        ),
        classification_threshold=0.5,
        requires_complete_injury_data=True,
        incomplete_injury_fallback_model=(
            "external_elo_qb_logistic"
        ),
        fallback_feature_columns=(
            EXTERNAL_FALLBACK_FEATURES
        ),
        fallback_regularization_c=0.1,
        forward_test_season=2026,
    )
)