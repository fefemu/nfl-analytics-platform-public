"""
NFL Analytics Platform
Current Production Prediction Explanations

Purpose:
    Create auditable explanations for the selected
    external nfelo probability routing.

    Primary explanations show the external logistic and
    published nfelo blend.

    Fallback explanations show the independently trained
    external Elo-QB logistic output.

Author:
    Ferenc Kaizer

Version:
    0.2.0
"""

import numpy as np
import pandas as pd

from src.modeling.production_probability_predictions import (
    EXTERNAL_ELO_QB_FALLBACK_PREDICTION_MODE,
    EXTERNAL_NFELO_BLEND_PREDICTION_MODE,
)


REQUIRED_PREDICTION_COLUMNS = {
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "gametime",
    "home_team",
    "away_team",
    "is_neutral",
    "model_name",
    "model_version",
    "home_win_probability",
    "away_win_probability",
    "prediction_mode",
    "prediction_mode_reason",
    "published_nfelo_home_probability",
    "primary_logistic_home_win_probability",
    "fallback_logistic_home_win_probability",
    "applied_primary_logistic_weight",
    "applied_published_nfelo_weight",
    "has_complete_injury_data",
    "both_listed_qb_ratings_available",
    "has_complete_production_features",
    "has_complete_fallback_features",
    "external_nfelo_rating_difference",
    "listed_qb_rating_difference",
    "external_nfelo_qb_adjustment_difference",
    "offense_injury_burden_difference",
    "defense_injury_burden_difference",
    "special_teams_injury_burden_difference",
    "prediction_generated_at",
}

PRODUCTION_EXPLANATION_COLUMNS = (
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "gametime",
    "home_team",
    "away_team",
    "is_neutral",
    "model_name",
    "model_version",
    "prediction_mode",
    "prediction_mode_reason",
    "favorite",
    "underdog",
    "favorite_win_probability",
    "home_win_probability",
    "away_win_probability",
    "published_nfelo_home_probability",
    "published_nfelo_away_probability",
    "primary_logistic_home_win_probability",
    "primary_logistic_away_win_probability",
    "fallback_logistic_home_win_probability",
    "fallback_logistic_away_win_probability",
    "production_probability_adjustment_from_published_nfelo",
    "applied_primary_logistic_weight",
    "applied_published_nfelo_weight",
    "external_nfelo_rating_difference",
    "external_nfelo_qb_adjustment_difference",
    "listed_qb_rating_difference",
    "offense_injury_burden_difference",
    "defense_injury_burden_difference",
    "special_teams_injury_burden_difference",
    "has_complete_injury_data",
    "both_listed_qb_ratings_available",
    "has_complete_production_features",
    "has_complete_fallback_features",
    "matchup_label",
    "prediction_generated_at",
)


def classify_production_matchup(
    favorite_win_probability: float,
) -> str:
    """Classify matchup strength from final probability."""

    probability = float(
        favorite_win_probability
    )

    if not 0.5 <= probability <= 1.0:
        raise ValueError(
            "Favorite win probability must be "
            "between 0.5 and 1.0."
        )

    edge = probability - 0.5

    if edge < 0.025:
        return "toss_up"

    if edge < 0.075:
        return "slight_edge"

    if edge < 0.15:
        return "clear_edge"

    return "strong_edge"


def optional_probability(
    value: object,
) -> float | None:
    """Convert one optional probability."""

    if pd.isna(value):
        return None

    probability = float(value)

    if (
        not np.isfinite(probability)
        or not 0.0 <= probability <= 1.0
    ):
        raise ValueError(
            "Optional production probability must be "
            "finite and between zero and one."
        )

    return probability


def optional_float(
    value: object,
) -> float | None:
    """Convert one optional finite numeric value."""

    if pd.isna(value):
        return None

    numeric_value = float(value)

    if not np.isfinite(numeric_value):
        raise ValueError(
            "Optional production feature must be finite."
        )

    return numeric_value


def validate_probability_pair(
    home_probability: float,
    away_probability: float,
    probability_name: str,
) -> None:
    """Validate one complementary probability pair."""

    if (
        not np.isfinite(home_probability)
        or not np.isfinite(away_probability)
        or not 0.0 <= home_probability <= 1.0
        or not 0.0 <= away_probability <= 1.0
        or abs(
            home_probability
            + away_probability
            - 1.0
        ) > 0.000001
    ):
        raise ValueError(
            f"{probability_name} probabilities are "
            "invalid."
        )


def create_production_prediction_explanation_frame(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Create external primary and fallback explanations."""

    missing_columns = sorted(
        REQUIRED_PREDICTION_COLUMNS
        - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Current production predictions are "
            "missing explanation columns: "
            + ", ".join(missing_columns)
        )

    if predictions[
        "game_id"
    ].duplicated().any():
        raise ValueError(
            "Current production predictions contain "
            "duplicate game identifiers."
        )

    explanation_rows: list[
        dict[str, object]
    ] = []

    for prediction in predictions.itertuples(
        index=False
    ):
        home_probability = float(
            prediction.home_win_probability
        )

        away_probability = float(
            prediction.away_win_probability
        )

        validate_probability_pair(
            home_probability=home_probability,
            away_probability=away_probability,
            probability_name="Production",
        )

        published_home_probability = (
            optional_probability(
                prediction
                .published_nfelo_home_probability
            )
        )

        published_away_probability = (
            None
            if published_home_probability is None
            else 1.0 - published_home_probability
        )

        if published_home_probability is not None:
            validate_probability_pair(
                home_probability=(
                    published_home_probability
                ),
                away_probability=(
                    published_away_probability
                ),
                probability_name="Published nfelo",
            )

        primary_home_probability = (
            optional_probability(
                prediction
                .primary_logistic_home_win_probability
            )
        )

        primary_away_probability = (
            None
            if primary_home_probability is None
            else 1.0 - primary_home_probability
        )

        fallback_home_probability = (
            optional_probability(
                prediction
                .fallback_logistic_home_win_probability
            )
        )

        fallback_away_probability = (
            None
            if fallback_home_probability is None
            else 1.0 - fallback_home_probability
        )

        prediction_mode = str(
            prediction.prediction_mode
        )

        if (
            prediction_mode
            == EXTERNAL_NFELO_BLEND_PREDICTION_MODE
        ):
            if (
                published_home_probability is None
                or
                primary_home_probability is None
                or fallback_home_probability is not None
            ):
                raise ValueError(
                    "External primary explanation has "
                    "invalid component probabilities."
                )

        elif (
            prediction_mode
            == (
                EXTERNAL_ELO_QB_FALLBACK_PREDICTION_MODE
            )
        ):
            if (
                primary_home_probability is not None
                or fallback_home_probability is None
            ):
                raise ValueError(
                    "External fallback explanation has "
                    "invalid component probabilities."
                )

        else:
            raise ValueError(
                "Unknown production prediction mode: "
                f"{prediction_mode}"
            )

        if home_probability >= 0.5:
            favorite = str(
                prediction.home_team
            )

            underdog = str(
                prediction.away_team
            )

            favorite_probability = (
                home_probability
            )
        else:
            favorite = str(
                prediction.away_team
            )

            underdog = str(
                prediction.home_team
            )

            favorite_probability = (
                away_probability
            )

        explanation_rows.append(
            {
                "game_id": str(
                    prediction.game_id
                ),
                "season": int(
                    prediction.season
                ),
                "game_type": str(
                    prediction.game_type
                ),
                "week": int(
                    prediction.week
                ),
                "gameday": (
                    prediction.gameday
                ),
                "gametime": (
                    prediction.gametime
                ),
                "home_team": str(
                    prediction.home_team
                ),
                "away_team": str(
                    prediction.away_team
                ),
                "is_neutral": bool(
                    prediction.is_neutral
                ),
                "model_name": str(
                    prediction.model_name
                ),
                "model_version": str(
                    prediction.model_version
                ),
                "prediction_mode": (
                    prediction_mode
                ),
                "prediction_mode_reason": str(
                    prediction
                    .prediction_mode_reason
                ),
                "favorite": favorite,
                "underdog": underdog,
                "favorite_win_probability": (
                    favorite_probability
                ),
                "home_win_probability": (
                    home_probability
                ),
                "away_win_probability": (
                    away_probability
                ),
                "published_nfelo_home_probability": (
                    published_home_probability
                ),
                "published_nfelo_away_probability": (
                    published_away_probability
                ),
                "primary_logistic_home_win_probability": (
                    primary_home_probability
                ),
                "primary_logistic_away_win_probability": (
                    primary_away_probability
                ),
                "fallback_logistic_home_win_probability": (
                    fallback_home_probability
                ),
                "fallback_logistic_away_win_probability": (
                    fallback_away_probability
                ),
                "production_probability_adjustment_from_published_nfelo": (
                    None
                    if published_home_probability is None
                    else (
                        home_probability
                        - published_home_probability
                    )
                ),
                "applied_primary_logistic_weight": float(
                    prediction
                    .applied_primary_logistic_weight
                ),
                "applied_published_nfelo_weight": float(
                    prediction
                    .applied_published_nfelo_weight
                ),
                "external_nfelo_rating_difference": float(
                    prediction
                    .external_nfelo_rating_difference
                ),
                "external_nfelo_qb_adjustment_difference": float(
                    prediction
                    .external_nfelo_qb_adjustment_difference
                ),
                "listed_qb_rating_difference": (
                    optional_float(
                        prediction
                        .listed_qb_rating_difference
                    )
                ),
                "offense_injury_burden_difference": (
                    optional_float(
                        prediction
                        .offense_injury_burden_difference
                    )
                ),
                "defense_injury_burden_difference": (
                    optional_float(
                        prediction
                        .defense_injury_burden_difference
                    )
                ),
                "special_teams_injury_burden_difference": (
                    optional_float(
                        prediction
                        .special_teams_injury_burden_difference
                    )
                ),
                "has_complete_injury_data": bool(
                    prediction
                    .has_complete_injury_data
                ),
                "both_listed_qb_ratings_available": bool(
                    prediction
                    .both_listed_qb_ratings_available
                ),
                "has_complete_production_features": bool(
                    prediction
                    .has_complete_production_features
                ),
                "has_complete_fallback_features": bool(
                    prediction
                    .has_complete_fallback_features
                ),
                "matchup_label": (
                    classify_production_matchup(
                        favorite_probability
                    )
                ),
                "prediction_generated_at": (
                    prediction
                    .prediction_generated_at
                ),
            }
        )

    return pd.DataFrame(
        explanation_rows,
        columns=(
            PRODUCTION_EXPLANATION_COLUMNS
        ),
    )
