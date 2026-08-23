"""
NFL Analytics Platform
Current Production Features

Purpose:
    Convert upcoming game inputs into the exact external
    nfelo primary and fallback probability features.

    Transitional internal Elo fields remain available
    until every downstream production output has been
    migrated.

Author:
    Ferenc Kaizer

Version:
    0.2.0
"""

import numpy as np
import pandas as pd

from src.modeling.production_probability_model import (
    EXTERNAL_ELO_FEATURE,
    EXTERNAL_QB_FEATURE,
    LISTED_QB_FEATURE,
    PRODUCTION_PROBABILITY_MODEL,
)


GAME_ID_COLUMN = "game_id"

HOME_QB_RATING_COLUMN = (
    "home_listed_qb_rating"
)

AWAY_QB_RATING_COLUMN = (
    "away_listed_qb_rating"
)

INJURY_COVERAGE_COLUMN = (
    "has_complete_injury_data"
)

INTERNAL_ELO_FEATURE_COLUMN = (
    "elo_rating_difference"
)

PUBLISHED_PROBABILITY_COLUMN = (
    "published_nfelo_home_probability"
)

INJURY_FEATURE_COLUMNS = (
    "offense_injury_burden_difference",
    "defense_injury_burden_difference",
    "special_teams_injury_burden_difference",
)

REQUIRED_UPCOMING_COLUMNS = {
    GAME_ID_COLUMN,
    HOME_QB_RATING_COLUMN,
    AWAY_QB_RATING_COLUMN,
    INJURY_COVERAGE_COLUMN,
    EXTERNAL_ELO_FEATURE,
    EXTERNAL_QB_FEATURE,
    PUBLISHED_PROBABILITY_COLUMN,
    "external_game_available",
    *INJURY_FEATURE_COLUMNS,
}

REQUIRED_ELO_PREDICTION_COLUMNS = {
    GAME_ID_COLUMN,
    "home_rating_pregame",
    "away_rating_pregame",
    "home_win_probability",
}


def validate_feature_source_columns(
    data: pd.DataFrame,
    required_columns: set[str],
    data_name: str,
) -> None:
    """Validate one current feature source."""

    missing_columns = sorted(
        required_columns
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{data_name} is missing columns: "
            + ", ".join(missing_columns)
        )


def validate_unique_game_ids(
    data: pd.DataFrame,
    data_name: str,
) -> None:
    """Require one source row per game."""

    if data[
        GAME_ID_COLUMN
    ].duplicated().any():
        raise ValueError(
            f"{data_name} contains duplicate "
            "game identifiers."
        )


def validate_published_probabilities(
    features: pd.DataFrame,
) -> None:
    """Validate optional exact-game nfelo probabilities."""

    exact_game_available = features[
        "external_game_available"
    ].fillna(False).astype(bool)

    available_probabilities = features.loc[
        exact_game_available,
        PUBLISHED_PROBABILITY_COLUMN,
    ].to_numpy(dtype=float)

    if (
        not np.isfinite(
            available_probabilities
        ).all()
        or (
            available_probabilities <= 0.0
        ).any()
        or (
            available_probabilities >= 1.0
        ).any()
    ):
        raise ValueError(
            "Available published nfelo probabilities "
            "must be finite and strictly between zero "
            "and one."
        )

    if features.loc[
        ~exact_game_available,
        PUBLISHED_PROBABILITY_COLUMN,
    ].notna().any():
        raise ValueError(
            "Games without exact external data must "
            "not contain published nfelo probabilities."
        )


def create_current_production_feature_frame(
    upcoming_games: pd.DataFrame,
    elo_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create current external probability features.

    Every upcoming game is retained. Missing listed-QB or
    injury inputs make the primary model ineligible but
    do not affect the external Elo-QB fallback.

    The internal Elo prediction frame remains joined
    temporarily for backward-compatible metadata and
    output migration.
    """

    validate_feature_source_columns(
        data=upcoming_games,
        required_columns=(
            REQUIRED_UPCOMING_COLUMNS
        ),
        data_name="Upcoming game features",
    )

    validate_feature_source_columns(
        data=elo_predictions,
        required_columns=(
            REQUIRED_ELO_PREDICTION_COLUMNS
        ),
        data_name="Current Elo predictions",
    )

    validate_unique_game_ids(
        data=upcoming_games,
        data_name="Upcoming game features",
    )

    validate_unique_game_ids(
        data=elo_predictions,
        data_name="Current Elo predictions",
    )

    transitional_elo_source = (
        elo_predictions.loc[
            :,
            [
                GAME_ID_COLUMN,
                "home_rating_pregame",
                "away_rating_pregame",
                "home_win_probability",
            ],
        ].rename(
            columns={
                "home_win_probability": (
                    "internal_elo_home_win_probability"
                ),
            }
        )
    )

    features = upcoming_games.merge(
        transitional_elo_source,
        on=GAME_ID_COLUMN,
        how="left",
        validate="one_to_one",
        sort=False,
    )

    missing_internal_elo_mask = features.loc[
        :,
        [
            "home_rating_pregame",
            "away_rating_pregame",
            "internal_elo_home_win_probability",
        ],
    ].isna().any(axis=1)

    if missing_internal_elo_mask.any():
        missing_game_ids = ", ".join(
            features.loc[
                missing_internal_elo_mask,
                GAME_ID_COLUMN,
            ].astype(str)
        )

        raise RuntimeError(
            "Upcoming games are missing current "
            "internal Elo predictions: "
            f"{missing_game_ids}"
        )

    features[
        INTERNAL_ELO_FEATURE_COLUMN
    ] = (
        features["home_rating_pregame"]
        - features["away_rating_pregame"]
    )

    both_qb_ratings_available = (
        features[
            HOME_QB_RATING_COLUMN
        ].notna()
        & features[
            AWAY_QB_RATING_COLUMN
        ].notna()
    )

    features[
        "both_listed_qb_ratings_available"
    ] = both_qb_ratings_available

    features[
        LISTED_QB_FEATURE
    ] = (
        features[
            HOME_QB_RATING_COLUMN
        ]
        - features[
            AWAY_QB_RATING_COLUMN
        ]
    ).where(
        both_qb_ratings_available
    )

    features[
        INJURY_COVERAGE_COLUMN
    ] = (
        features[
            INJURY_COVERAGE_COLUMN
        ].fillna(False).astype(bool)
    )

    validate_published_probabilities(
        features
    )

    primary_feature_columns = list(
        PRODUCTION_PROBABILITY_MODEL
        .logistic_feature_columns
    )

    fallback_feature_columns = list(
        PRODUCTION_PROBABILITY_MODEL
        .fallback_feature_columns
    )

    features[
        "has_complete_production_features"
    ] = (
        features["external_game_available"]
        .fillna(False).astype(bool)
        & features[
            PUBLISHED_PROBABILITY_COLUMN
        ].notna()
        & features[
            INJURY_COVERAGE_COLUMN
        ]
        & features[
            primary_feature_columns
        ].notna().all(axis=1)
    )

    features[
        "has_complete_fallback_features"
    ] = features[
        fallback_feature_columns
    ].notna().all(axis=1)

    if not features[
        "has_complete_fallback_features"
    ].all():
        missing_game_ids = ", ".join(
            features.loc[
                ~features[
                    "has_complete_fallback_features"
                ],
                GAME_ID_COLUMN,
            ].astype(str)
        )

        raise RuntimeError(
            "Upcoming games are missing external "
            "probability fallback features: "
            f"{missing_game_ids}"
        )

    return features
