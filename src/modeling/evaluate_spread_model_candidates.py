"""
NFL Analytics Platform
Spread Model Candidate Evaluation

Purpose:
    Compare simple leakage-safe spread models on one
    identical train and validation sample.

Target:
    Home score minus away score. Positive predictions
    favor the home team; negative predictions favor the
    away team.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.modeling.train_logistic_baseline import (
    DATABASE_FILE,
    validate_database_file,
)


logger = logging.getLogger(__name__)

DATASET_FULL_NAME = (
    "analytics.game_modeling_dataset"
)

SPLIT_FULL_NAME = (
    "analytics.modeling_game_splits"
)

TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"

SPREAD_TARGET_COLUMN = (
    "target_point_differential"
)

ELO_SPREAD_FEATURES = (
    "elo_rating_difference",
)

ELO_QB_SPREAD_FEATURES = (
    *ELO_SPREAD_FEATURES,
    "listed_qb_rating_difference",
)

INJURY_SPREAD_FEATURES = (
    "offense_injury_burden_difference",
    "defense_injury_burden_difference",
    "special_teams_injury_burden_difference",
)

SPREAD_CORE_FEATURES = (
    *ELO_QB_SPREAD_FEATURES,
    *INJURY_SPREAD_FEATURES,
)

REQUIRED_DATA_COLUMNS = {
    "game_id",
    "season",
    "split_name",
    "has_complete_injury_data",
    SPREAD_TARGET_COLUMN,
    *SPREAD_CORE_FEATURES,
}


@dataclass(frozen=True)
class SpreadModelCandidate:
    """Describe one spread model candidate."""

    candidate_name: str
    feature_columns: tuple[str, ...]
    ridge_alpha: float


SPREAD_MODEL_CANDIDATES = (
    SpreadModelCandidate(
        candidate_name="ridge_elo",
        feature_columns=ELO_SPREAD_FEATURES,
        ridge_alpha=1.0,
    ),
    SpreadModelCandidate(
        candidate_name="ridge_elo_qb",
        feature_columns=ELO_QB_SPREAD_FEATURES,
        ridge_alpha=1.0,
    ),
    SpreadModelCandidate(
        candidate_name="ridge_elo_qb_injury",
        feature_columns=SPREAD_CORE_FEATURES,
        ridge_alpha=1.0,
    ),
)

RESULT_COLUMNS = (
    "candidate_name",
    "feature_count",
    "ridge_alpha",
    "train_game_count",
    "validation_game_count",
    "validation_mae",
    "validation_rmse",
    "validation_bias",
    "validation_r_squared",
)


def validate_required_columns(
    data: pd.DataFrame,
) -> None:
    """Validate the spread evaluation schema."""

    missing_columns = sorted(
        REQUIRED_DATA_COLUMNS
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Spread development data is missing columns: "
            + ", ".join(missing_columns)
        )


def load_spread_development_data(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Load train and validation spread inputs."""

    feature_select = ",\n            ".join(
        f"dataset.{feature_name}"
        for feature_name
        in SPREAD_CORE_FEATURES
    )

    data = connection.execute(
        f"""
        SELECT
            dataset.game_id,
            dataset.season,
            splits.split_name,
            dataset.has_complete_injury_data,
            dataset.{SPREAD_TARGET_COLUMN},
            {feature_select}

        FROM {DATASET_FULL_NAME}
            AS dataset

        INNER JOIN {SPLIT_FULL_NAME}
            AS splits
            ON dataset.game_id = splits.game_id

        WHERE splits.split_name IN (
            '{TRAIN_SPLIT}',
            '{VALIDATION_SPLIT}'
        )

        ORDER BY
            dataset.season,
            dataset.game_date,
            dataset.game_id
        """
    ).fetchdf()

    validate_required_columns(data)

    if data.empty:
        raise RuntimeError(
            "No spread development data is available."
        )

    return data


def prepare_common_spread_sample(
    development_data: pd.DataFrame,
) -> pd.DataFrame:
    """Create one identical complete candidate sample."""

    validate_required_columns(
        development_data
    )

    allowed_split_mask = (
        development_data["split_name"].isin(
            [
                TRAIN_SPLIT,
                VALIDATION_SPLIT,
            ]
        )
    )

    complete_mask = (
        allowed_split_mask
        & development_data[
            "has_complete_injury_data"
        ].fillna(False).astype(bool)
        & development_data[
            SPREAD_TARGET_COLUMN
        ].notna()
        & development_data[
            list(SPREAD_CORE_FEATURES)
        ].notna().all(axis=1)
    )

    sample = development_data.loc[
        complete_mask
    ].copy()

    if sample.empty:
        raise RuntimeError(
            "No complete spread candidate games "
            "are available."
        )

    if sample["game_id"].duplicated().any():
        raise ValueError(
            "Spread development data contains duplicate "
            "game identifiers."
        )

    available_splits = set(
        sample["split_name"]
    )

    if available_splits != {
        TRAIN_SPLIT,
        VALIDATION_SPLIT,
    }:
        raise RuntimeError(
            "Spread candidate sample must contain train "
            "and validation games."
        )

    return sample


def create_ridge_pipeline(
    ridge_alpha: float,
) -> Pipeline:
    """Create a standardized linear model pipeline."""

    if ridge_alpha < 0.0:
        raise ValueError(
            "Ridge alpha must not be negative."
        )

    regression_model = (
        LinearRegression()
        if ridge_alpha == 0.0
        else Ridge(
            alpha=ridge_alpha,
        )
    )

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                regression_model,
            ),
        ]
    )


def calculate_regression_metrics(
    actual_margin: pd.Series,
    predicted_margin: np.ndarray,
) -> dict[str, float]:
    """Calculate spread regression metrics."""

    actual = actual_margin.to_numpy(
        dtype=float
    )

    predicted = np.asarray(
        predicted_margin,
        dtype=float,
    )

    if actual.shape != predicted.shape:
        raise ValueError(
            "Actual and predicted margin shapes "
            "must match."
        )

    if actual.size == 0:
        raise ValueError(
            "Regression metrics require observations."
        )

    residual = predicted - actual

    return {
        "validation_mae": float(
            mean_absolute_error(
                actual,
                predicted,
            )
        ),
        "validation_rmse": float(
            np.sqrt(
                mean_squared_error(
                    actual,
                    predicted,
                )
            )
        ),
        "validation_bias": float(
            residual.mean()
        ),
        "validation_r_squared": float(
            r2_score(
                actual,
                predicted,
            )
        ),
    }


def evaluate_spread_model_candidates(
    development_data: pd.DataFrame,
    candidates: tuple[
        SpreadModelCandidate, ...
    ] = SPREAD_MODEL_CANDIDATES,
) -> pd.DataFrame:
    """Evaluate spread candidates on identical games."""

    if not candidates:
        raise ValueError(
            "At least one spread candidate is required."
        )

    sample = prepare_common_spread_sample(
        development_data
    )

    train_data = sample.loc[
        sample["split_name"]
        == TRAIN_SPLIT
    ]

    validation_data = sample.loc[
        sample["split_name"]
        == VALIDATION_SPLIT
    ]

    train_target = train_data[
        SPREAD_TARGET_COLUMN
    ]

    validation_target = validation_data[
        SPREAD_TARGET_COLUMN
    ]

    result_rows: list[
        dict[str, object]
    ] = []

    constant_prediction = np.full(
        shape=len(validation_data),
        fill_value=float(
            train_target.mean()
        ),
    )

    result_rows.append(
        {
            "candidate_name": (
                "constant_train_mean"
            ),
            "feature_count": 0,
            "ridge_alpha": None,
            "train_game_count": len(
                train_data
            ),
            "validation_game_count": len(
                validation_data
            ),
            **calculate_regression_metrics(
                actual_margin=validation_target,
                predicted_margin=(
                    constant_prediction
                ),
            ),
        }
    )

    candidate_names = [
        candidate.candidate_name
        for candidate in candidates
    ]

    if len(candidate_names) != len(
        set(candidate_names)
    ):
        raise ValueError(
            "Spread candidate names must be unique."
        )

    for candidate in candidates:
        model = create_ridge_pipeline(
            ridge_alpha=(
                candidate.ridge_alpha
            )
        )

        model.fit(
            train_data.loc[
                :,
                list(
                    candidate.feature_columns
                ),
            ],
            train_target,
        )

        predicted_margin = model.predict(
            validation_data.loc[
                :,
                list(
                    candidate.feature_columns
                ),
            ]
        )

        result_rows.append(
            {
                "candidate_name": (
                    candidate.candidate_name
                ),
                "feature_count": len(
                    candidate.feature_columns
                ),
                "ridge_alpha": (
                    candidate.ridge_alpha
                ),
                "train_game_count": len(
                    train_data
                ),
                "validation_game_count": len(
                    validation_data
                ),
                **calculate_regression_metrics(
                    actual_margin=(
                        validation_target
                    ),
                    predicted_margin=(
                        predicted_margin
                    ),
                ),
            }
        )

    return pd.DataFrame(
        result_rows,
        columns=RESULT_COLUMNS,
    ).sort_values(
        by=[
            "validation_mae",
            "validation_rmse",
            "candidate_name",
        ],
        kind="stable",
    ).reset_index(drop=True)


def run_spread_candidate_evaluation(
    database_file: Path = DATABASE_FILE,
) -> pd.DataFrame:
    """Run spread candidate evaluation from DuckDB."""

    validate_database_file(
        database_file
    )

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        development_data = (
            load_spread_development_data(
                connection
            )
        )

    results = (
        evaluate_spread_model_candidates(
            development_data
        )
    )

    logger.info(
        "Spread candidate evaluation completed on "
        "%s train and %s validation games.",
        int(
            results.iloc[0][
                "train_game_count"
            ]
        ),
        int(
            results.iloc[0][
                "validation_game_count"
            ]
        ),
    )

    return results


def main() -> None:
    """Run and print spread candidate evaluation."""

    results = (
        run_spread_candidate_evaluation()
    )

    print(
        results.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()