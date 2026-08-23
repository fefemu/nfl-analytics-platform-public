"""
NFL Analytics Platform
Production Logistic Component

Purpose:
    Train the selected injury-enhanced logistic component
    on all eligible historical games before the forward
    test season.

    Score only current games with a complete production
    feature set. Incomplete games remain available for
    the documented Elo fallback.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.modeling.production_probability_model import (
    PRODUCTION_PROBABILITY_MODEL,
    ProductionProbabilityModel,
)
from src.modeling.train_logistic_baseline import (
    TARGET_COLUMN,
    TRAIN_SPLIT,
    train_logistic_model,
)


GAME_ID_COLUMN = "game_id"
SEASON_COLUMN = "season"
SPLIT_COLUMN = "split_name"

INJURY_COVERAGE_COLUMN = (
    "has_complete_injury_data"
)

LOGISTIC_FEATURE_COVERAGE_COLUMN = (
    "has_complete_logistic_features"
)

LOGISTIC_PROBABILITY_COLUMN = (
    "logistic_home_win_probability"
)

STANDARDIZED_VALUE_SUFFIX = (
    "_standardized_value"
)

LOG_ODDS_CONTRIBUTION_SUFFIX = (
    "_log_odds_contribution"
)

LOGISTIC_INTERCEPT_COLUMN = (
    "logistic_intercept"
)

LOGISTIC_TOTAL_LOG_ODDS_COLUMN = (
    "logistic_total_log_odds"
)

LOGISTIC_RECONSTRUCTED_PROBABILITY_COLUMN = (
    "logistic_reconstructed_home_win_probability"
)

LOGISTIC_FEATURE_CONTRIBUTION_COLUMNS = (
    GAME_ID_COLUMN,
    "feature_name",
    "raw_feature_value",
    "standardized_feature_value",
    "coefficient",
    "log_odds_contribution",
    "absolute_log_odds_contribution",
    "contribution_rank",
    LOGISTIC_INTERCEPT_COLUMN,
    LOGISTIC_TOTAL_LOG_ODDS_COLUMN,
    LOGISTIC_RECONSTRUCTED_PROBABILITY_COLUMN,
)


def validate_required_columns(
    data: pd.DataFrame,
    required_columns: set[str],
    data_name: str,
) -> None:
    """Validate required DataFrame columns."""

    missing_columns = sorted(
        required_columns
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{data_name} is missing columns: "
            + ", ".join(missing_columns)
        )


def prepare_production_training_data(
    historical_data: pd.DataFrame,
    production_model: (
        ProductionProbabilityModel
    ) = PRODUCTION_PROBABILITY_MODEL,
) -> pd.DataFrame:
    """
    Prepare leakage-safe production training games.

    Only games before the configured forward-test season
    are eligible. Injury coverage, the binary target and
    every selected logistic feature must be complete.
    """

    required_columns = {
        GAME_ID_COLUMN,
        SEASON_COLUMN,
        TARGET_COLUMN,
        INJURY_COVERAGE_COLUMN,
        *production_model.logistic_feature_columns,
    }

    validate_required_columns(
        data=historical_data,
        required_columns=required_columns,
        data_name="Historical production data",
    )

    if historical_data[
        GAME_ID_COLUMN
    ].duplicated().any():
        raise ValueError(
            "Historical production data contains "
            "duplicate game identifiers."
        )

    feature_columns = list(
        production_model.logistic_feature_columns
    )

    eligible_mask = (
        historical_data[
            SEASON_COLUMN
        ].lt(
            production_model.forward_test_season
        )
        & historical_data[
            INJURY_COVERAGE_COLUMN
        ].fillna(False).astype(bool)
        & historical_data[
            TARGET_COLUMN
        ].notna()
        & historical_data[
            feature_columns
        ].notna().all(axis=1)
    )

    training_data = historical_data.loc[
        eligible_mask
    ].copy()

    if training_data.empty:
        raise RuntimeError(
            "No eligible historical games are "
            "available for production training."
        )

    training_data[
        SPLIT_COLUMN
    ] = TRAIN_SPLIT

    return training_data


def train_production_logistic_component(
    historical_data: pd.DataFrame,
    production_model: (
        ProductionProbabilityModel
    ) = PRODUCTION_PROBABILITY_MODEL,
) -> Pipeline:
    """Train the selected production logistic component."""

    training_data = (
        prepare_production_training_data(
            historical_data=historical_data,
            production_model=production_model,
        )
    )

    return train_logistic_model(
        development_data=training_data,
        feature_columns=(
            production_model
            .logistic_feature_columns
        ),
        regularization_c=(
            production_model
            .logistic_regularization_c
        ),
    )


def calculate_logistic_feature_contributions(
    current_features: pd.DataFrame,
    trained_model: Pipeline,
    production_model: (
        ProductionProbabilityModel
    ) = PRODUCTION_PROBABILITY_MODEL,
) -> pd.DataFrame:
    """
    Calculate exact standardized logistic contributions.

    One contribution equals the fitted standardized
    coefficient multiplied by the transformed feature
    value. Only games with a complete production feature
    set receive contribution rows.
    """

    required_columns = {
        GAME_ID_COLUMN,
        INJURY_COVERAGE_COLUMN,
        *production_model.logistic_feature_columns,
    }

    validate_required_columns(
        data=current_features,
        required_columns=required_columns,
        data_name="Current production features",
    )

    if current_features[
        GAME_ID_COLUMN
    ].duplicated().any():
        raise ValueError(
            "Current production features contain "
            "duplicate game identifiers."
        )

    feature_columns = list(
        production_model.logistic_feature_columns
    )

    contribution_columns = [
        GAME_ID_COLUMN,
        LOGISTIC_INTERCEPT_COLUMN,
        *[
            (
                feature_name
                + STANDARDIZED_VALUE_SUFFIX
            )
            for feature_name in feature_columns
        ],
        *[
            (
                feature_name
                + LOG_ODDS_CONTRIBUTION_SUFFIX
            )
            for feature_name in feature_columns
        ],
        LOGISTIC_TOTAL_LOG_ODDS_COLUMN,
        LOGISTIC_RECONSTRUCTED_PROBABILITY_COLUMN,
    ]

    complete_feature_mask = (
        current_features[
            INJURY_COVERAGE_COLUMN
        ].fillna(False).astype(bool)
        & current_features[
            feature_columns
        ].notna().all(axis=1)
    )

    complete_features = current_features.loc[
        complete_feature_mask,
        [
            GAME_ID_COLUMN,
            *feature_columns,
        ],
    ].copy()

    if complete_features.empty:
        return pd.DataFrame(
            columns=contribution_columns
        )

    preprocessor = trained_model.named_steps[
        "preprocessor"
    ]

    logistic_model = trained_model.named_steps[
        "model"
    ]

    standardized_values = np.asarray(
        preprocessor.transform(
            complete_features[
                feature_columns
            ]
        )
    )

    coefficients = np.asarray(
        logistic_model.coef_[0]
    )

    intercept = float(
        logistic_model.intercept_[0]
    )

    if standardized_values.shape[1] != len(
        feature_columns
    ):
        raise RuntimeError(
            "Transformed logistic feature count does "
            "not match the production specification."
        )

    if len(coefficients) != len(
        feature_columns
    ):
        raise RuntimeError(
            "Logistic coefficient count does not "
            "match the production specification."
        )

    contributions = (
        standardized_values
        * coefficients
    )

    total_log_odds = (
        intercept
        + contributions.sum(axis=1)
    )

    reconstructed_probabilities = (
        1.0
        / (
            1.0
            + np.exp(-total_log_odds)
        )
    )

    model_probabilities = (
        trained_model.predict_proba(
            complete_features[
                feature_columns
            ]
        )[:, 1]
    )

    if not np.allclose(
        reconstructed_probabilities,
        model_probabilities,
        atol=0.000000000001,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Reconstructed logistic probabilities do "
            "not match predict_proba."
        )

    result = pd.DataFrame(
        {
            GAME_ID_COLUMN: (
                complete_features[
                    GAME_ID_COLUMN
                ].to_numpy()
            ),
            LOGISTIC_INTERCEPT_COLUMN: intercept,
        }
    )

    for feature_index, feature_name in enumerate(
        feature_columns
    ):
        result[
            (
                feature_name
                + STANDARDIZED_VALUE_SUFFIX
            )
        ] = standardized_values[
            :,
            feature_index,
        ]

        result[
            (
                feature_name
                + LOG_ODDS_CONTRIBUTION_SUFFIX
            )
        ] = contributions[
            :,
            feature_index,
        ]

    result[
        LOGISTIC_TOTAL_LOG_ODDS_COLUMN
    ] = total_log_odds

    result[
        LOGISTIC_RECONSTRUCTED_PROBABILITY_COLUMN
    ] = reconstructed_probabilities

    return result.loc[
        :,
        contribution_columns,
    ]


def create_long_logistic_feature_contributions(
    current_features: pd.DataFrame,
    trained_model: Pipeline,
    production_model: (
        ProductionProbabilityModel
    ) = PRODUCTION_PROBABILITY_MODEL,
) -> pd.DataFrame:
    """
    Create one exact contribution row per game-feature.

    Only games with complete production features receive
    rows. Ranking uses absolute contribution magnitude,
    while the signed contribution preserves direction.
    """

    wide_contributions = (
        calculate_logistic_feature_contributions(
            current_features=current_features,
            trained_model=trained_model,
            production_model=production_model,
        )
    )

    if wide_contributions.empty:
        return pd.DataFrame(
            columns=(
                LOGISTIC_FEATURE_CONTRIBUTION_COLUMNS
            )
        )

    feature_columns = list(
        production_model.logistic_feature_columns
    )

    logistic_model = trained_model.named_steps[
        "model"
    ]

    coefficients = np.asarray(
        logistic_model.coef_[0]
    )

    if len(coefficients) != len(
        feature_columns
    ):
        raise RuntimeError(
            "Logistic coefficient count does not "
            "match the production specification."
        )

    raw_features = current_features.loc[
        :,
        [
            GAME_ID_COLUMN,
            *feature_columns,
        ],
    ]

    contribution_source = (
        wide_contributions.merge(
            raw_features,
            on=GAME_ID_COLUMN,
            how="left",
            validate="one_to_one",
            sort=False,
        )
    )

    contribution_rows: list[
        dict[str, object]
    ] = []

    for game in contribution_source.itertuples(
        index=False
    ):
        for feature_index, feature_name in enumerate(
            feature_columns
        ):
            contribution = float(
                getattr(
                    game,
                    (
                        feature_name
                        + LOG_ODDS_CONTRIBUTION_SUFFIX
                    ),
                )
            )

            contribution_rows.append(
                {
                    GAME_ID_COLUMN: str(
                        getattr(
                            game,
                            GAME_ID_COLUMN,
                        )
                    ),
                    "feature_name": feature_name,
                    "raw_feature_value": float(
                        getattr(
                            game,
                            feature_name,
                        )
                    ),
                    "standardized_feature_value": float(
                        getattr(
                            game,
                            (
                                feature_name
                                + STANDARDIZED_VALUE_SUFFIX
                            ),
                        )
                    ),
                    "coefficient": float(
                        coefficients[
                            feature_index
                        ]
                    ),
                    "log_odds_contribution": (
                        contribution
                    ),
                    "absolute_log_odds_contribution": (
                        abs(contribution)
                    ),
                    LOGISTIC_INTERCEPT_COLUMN: float(
                        getattr(
                            game,
                            LOGISTIC_INTERCEPT_COLUMN,
                        )
                    ),
                    LOGISTIC_TOTAL_LOG_ODDS_COLUMN: float(
                        getattr(
                            game,
                            LOGISTIC_TOTAL_LOG_ODDS_COLUMN,
                        )
                    ),
                    LOGISTIC_RECONSTRUCTED_PROBABILITY_COLUMN: float(
                        getattr(
                            game,
                            (
                                LOGISTIC_RECONSTRUCTED_PROBABILITY_COLUMN
                            ),
                        )
                    ),
                }
            )

    contribution_frame = pd.DataFrame(
        contribution_rows
    )

    contribution_frame[
        "contribution_rank"
    ] = (
        contribution_frame.groupby(
            GAME_ID_COLUMN,
            sort=False,
        )[
            "absolute_log_odds_contribution"
        ]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    contribution_frame = (
        contribution_frame.sort_values(
            by=[
                GAME_ID_COLUMN,
                "contribution_rank",
            ],
            kind="stable",
        ).reset_index(drop=True)
    )

    return contribution_frame.loc[
        :,
        LOGISTIC_FEATURE_CONTRIBUTION_COLUMNS,
    ]


def score_current_logistic_component(
    current_features: pd.DataFrame,
    trained_model: Pipeline,
    production_model: (
        ProductionProbabilityModel
    ) = PRODUCTION_PROBABILITY_MODEL,
) -> pd.DataFrame:
    """
    Score current games with complete logistic features.

    The returned frame preserves every input game.
    Incomplete games receive a null logistic probability
    and are marked for the Elo fallback.
    """

    required_columns = {
        GAME_ID_COLUMN,
        INJURY_COVERAGE_COLUMN,
        *production_model.logistic_feature_columns,
    }

    validate_required_columns(
        data=current_features,
        required_columns=required_columns,
        data_name="Current production features",
    )

    if current_features[
        GAME_ID_COLUMN
    ].duplicated().any():
        raise ValueError(
            "Current production features contain "
            "duplicate game identifiers."
        )

    feature_columns = list(
        production_model.logistic_feature_columns
    )

    scored_features = (
        current_features.copy()
    )

    complete_feature_mask = (
        scored_features[
            INJURY_COVERAGE_COLUMN
        ].fillna(False).astype(bool)
        & scored_features[
            feature_columns
        ].notna().all(axis=1)
    )

    scored_features[
        LOGISTIC_FEATURE_COVERAGE_COLUMN
    ] = complete_feature_mask

    scored_features[
        LOGISTIC_PROBABILITY_COLUMN
    ] = np.nan

    if complete_feature_mask.any():
        probabilities = (
            trained_model.predict_proba(
                scored_features.loc[
                    complete_feature_mask,
                    feature_columns,
                ]
            )[:, 1]
        )

        scored_features.loc[
            complete_feature_mask,
            LOGISTIC_PROBABILITY_COLUMN,
        ] = probabilities

    return scored_features