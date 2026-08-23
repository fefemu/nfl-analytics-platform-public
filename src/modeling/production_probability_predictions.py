"""
NFL Analytics Platform
Production Probability Predictions

Purpose:
    Route one game through the selected external-nfelo
    production probability architecture.

    Primary routing:
        70% external injury-logistic probability and
        30% published nfelo probability.

    Fallback routing:
        100% external Elo-QB logistic probability.

Author:
    Ferenc Kaizer

Version:
    0.2.0
"""

from dataclasses import dataclass
from math import isfinite

from src.modeling.production_probability_model import (
    PRODUCTION_PROBABILITY_MODEL,
    ProductionProbabilityModel,
)


EXTERNAL_NFELO_BLEND_PREDICTION_MODE = (
    "EXTERNAL_NFELO_BLEND"
)

EXTERNAL_ELO_QB_FALLBACK_PREDICTION_MODE = (
    "EXTERNAL_ELO_QB_FALLBACK"
)

COMPLETE_MODEL_FEATURES_REASON = (
    "complete_external_primary_features"
)

INCOMPLETE_MODEL_FEATURES_REASON = (
    "incomplete_external_primary_features"
)

BLEND_PREDICTION_MODE = (
    EXTERNAL_NFELO_BLEND_PREDICTION_MODE
)

ELO_FALLBACK_PREDICTION_MODE = (
    EXTERNAL_ELO_QB_FALLBACK_PREDICTION_MODE
)


@dataclass(frozen=True)
class ProductionProbabilityPrediction:
    """Store one routed production probability."""

    model_name: str
    model_version: str
    prediction_mode: str
    prediction_mode_reason: str
    home_win_probability: float
    away_win_probability: float
    published_nfelo_home_probability: float | None
    primary_logistic_home_win_probability: (
        float | None
    )
    fallback_logistic_home_win_probability: (
        float | None
    )
    applied_primary_logistic_weight: float
    applied_published_nfelo_weight: float

    @property
    def elo_home_win_probability(
        self,
    ) -> float | None:
        """
        Return the transitional legacy audit alias.

        The value now represents the published nfelo
        probability rather than the internal Elo model.
        """

        return self.published_nfelo_home_probability

    @property
    def logistic_home_win_probability(
        self,
    ) -> float | None:
        """Return the transitional primary-logistic alias."""

        return (
            self.primary_logistic_home_win_probability
        )

    @property
    def applied_logistic_weight(
        self,
    ) -> float:
        """Return the transitional logistic-weight alias."""

        return self.applied_primary_logistic_weight

    @property
    def applied_elo_weight(
        self,
    ) -> float:
        """Return the transitional published-weight alias."""

        return self.applied_published_nfelo_weight


def validate_probability(
    probability: float,
    probability_name: str,
) -> float:
    """Validate and return one finite probability."""

    numeric_probability = float(
        probability
    )

    if not isfinite(
        numeric_probability
    ):
        raise ValueError(
            f"{probability_name} must be finite."
        )

    if not (
        0.0
        <= numeric_probability
        <= 1.0
    ):
        raise ValueError(
            f"{probability_name} must be between "
            "zero and one."
        )

    return numeric_probability


def create_production_probability_prediction(
    published_nfelo_home_probability: float | None,
    primary_logistic_home_win_probability: (
        float | None
    ),
    fallback_logistic_home_win_probability: (
        float | None
    ),
    has_complete_primary_features: bool,
    production_model: (
        ProductionProbabilityModel
    ) = PRODUCTION_PROBABILITY_MODEL,
) -> ProductionProbabilityPrediction:
    """
    Route one game through the production model.

    Complete primary features activate the selected
    external logistic and published nfelo blend.

    Incomplete primary features activate the separately
    trained external Elo-QB logistic fallback.
    """

    if has_complete_primary_features:
        if published_nfelo_home_probability is None:
            raise ValueError(
                "Complete primary features require a "
                "published nfelo home-win probability."
            )

        published_probability = validate_probability(
            probability=(
                published_nfelo_home_probability
            ),
            probability_name=(
                "Published nfelo home-win probability"
            ),
        )

        if (
            primary_logistic_home_win_probability
            is None
        ):
            raise ValueError(
                "Complete primary features require a "
                "primary logistic home-win probability."
            )

        primary_probability = validate_probability(
            probability=(
                primary_logistic_home_win_probability
            ),
            probability_name=(
                "Primary logistic home-win probability"
            ),
        )

        home_probability = (
            production_model.logistic_weight
            * primary_probability
            + production_model.elo_weight
            * published_probability
        )

        fallback_probability = None

        prediction_mode = (
            EXTERNAL_NFELO_BLEND_PREDICTION_MODE
        )

        prediction_mode_reason = (
            COMPLETE_MODEL_FEATURES_REASON
        )

        applied_primary_logistic_weight = (
            production_model.logistic_weight
        )

        applied_published_nfelo_weight = (
            production_model.elo_weight
        )
    else:
        published_probability = (
            None
            if published_nfelo_home_probability is None
            else validate_probability(
                probability=(
                    published_nfelo_home_probability
                ),
                probability_name=(
                    "Published nfelo home-win probability"
                ),
            )
        )

        if (
            fallback_logistic_home_win_probability
            is None
        ):
            raise ValueError(
                "Incomplete primary features require an "
                "external fallback logistic home-win "
                "probability."
            )

        fallback_probability = validate_probability(
            probability=(
                fallback_logistic_home_win_probability
            ),
            probability_name=(
                "Fallback logistic home-win probability"
            ),
        )

        primary_probability = None
        home_probability = fallback_probability

        prediction_mode = (
            EXTERNAL_ELO_QB_FALLBACK_PREDICTION_MODE
        )

        prediction_mode_reason = (
            INCOMPLETE_MODEL_FEATURES_REASON
        )

        applied_primary_logistic_weight = 0.0
        applied_published_nfelo_weight = 0.0

    validated_home_probability = (
        validate_probability(
            probability=home_probability,
            probability_name=(
                "Production home-win probability"
            ),
        )
    )

    away_probability = (
        1.0
        - validated_home_probability
    )

    return ProductionProbabilityPrediction(
        model_name=production_model.model_name,
        model_version=production_model.model_version,
        prediction_mode=prediction_mode,
        prediction_mode_reason=(
            prediction_mode_reason
        ),
        home_win_probability=(
            validated_home_probability
        ),
        away_win_probability=away_probability,
        published_nfelo_home_probability=(
            published_probability
        ),
        primary_logistic_home_win_probability=(
            primary_probability
        ),
        fallback_logistic_home_win_probability=(
            fallback_probability
        ),
        applied_primary_logistic_weight=(
            applied_primary_logistic_weight
        ),
        applied_published_nfelo_weight=(
            applied_published_nfelo_weight
        ),
    )
