"""
NFL Analytics Platform
Totals Fallback Candidate Evaluation

Purpose:
    Compare totals fallback candidates that remain
    available without rolling, listed-QB or observed
    game-time weather inputs.

The 2025 holdout is never loaded or evaluated.

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

from src.modeling.evaluate_spread_model_candidates import (
    calculate_regression_metrics,
    create_ridge_pipeline,
)
from src.modeling.evaluate_totals_model_candidates import (
    DATASET_FULL_NAME,
    SPLIT_FULL_NAME,
    TOTALS_TARGET_COLUMN,
)
from src.modeling.train_logistic_baseline import (
    DATABASE_FILE,
    validate_database_file,
)


logger = logging.getLogger(__name__)

TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"

RAW_FALLBACK_FEATURE_COLUMNS = (
    "home_elo_rating",
    "away_elo_rating",
    "is_indoor",
    "league_average_total_last_64",
)

LEAGUE_64_FEATURES = (
    "league_average_total_last_64",
)

LEAGUE_64_INDOOR_FEATURES = (
    "league_average_total_last_64",
    "is_indoor",
)

LEAGUE_64_ELO_FEATURES = (
    "league_average_total_last_64",
    "elo_rating_sum",
)

LEAGUE_64_INDOOR_ELO_FEATURES = (
    "league_average_total_last_64",
    "is_indoor",
    "elo_rating_sum",
)

FALLBACK_CORE_FEATURES = (
    *LEAGUE_64_INDOOR_ELO_FEATURES,
)

REQUIRED_SOURCE_COLUMNS = {
    "game_id",
    "season",
    "split_name",
    TOTALS_TARGET_COLUMN,
    *RAW_FALLBACK_FEATURE_COLUMNS,
}


@dataclass(frozen=True)
class TotalsFallbackCandidate:
    """Describe one totals fallback candidate."""

    candidate_name: str
    feature_columns: tuple[str, ...]
    ridge_alpha: float


TOTALS_FALLBACK_CANDIDATES = (
    TotalsFallbackCandidate(
        candidate_name="ridge_league_64",
        feature_columns=LEAGUE_64_FEATURES,
        ridge_alpha=1.0,
    ),
    TotalsFallbackCandidate(
        candidate_name="ridge_league_64_indoor",
        feature_columns=(
            LEAGUE_64_INDOOR_FEATURES
        ),
        ridge_alpha=1.0,
    ),
    TotalsFallbackCandidate(
        candidate_name="ridge_league_64_elo",
        feature_columns=(
            LEAGUE_64_ELO_FEATURES
        ),
        ridge_alpha=1.0,
    ),
    TotalsFallbackCandidate(
        candidate_name=(
            "ridge_league_64_indoor_elo"
        ),
        feature_columns=(
            LEAGUE_64_INDOOR_ELO_FEATURES
        ),
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


def validate_source_columns(
    data: pd.DataFrame,
) -> None:
    """Validate the fallback source schema."""

    missing_columns = sorted(
        REQUIRED_SOURCE_COLUMNS
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Totals fallback data is missing columns: "
            + ", ".join(missing_columns)
        )


def create_totals_fallback_features(
    source_data: pd.DataFrame,
) -> pd.DataFrame:
    """Create team-order-invariant fallback features."""

    validate_source_columns(source_data)

    data = source_data.copy()

    data["elo_rating_sum"] = (
        data["home_elo_rating"]
        + data["away_elo_rating"]
    )

    return data


def load_totals_fallback_development_data(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Load train and validation fallback inputs."""

    feature_select = ",\n            ".join(
        f"dataset.{column_name}"
        for column_name
        in RAW_FALLBACK_FEATURE_COLUMNS
    )

    source_data = connection.execute(
        f"""
        SELECT
            dataset.game_id,
            dataset.season,
            splits.split_name,
            dataset.{TOTALS_TARGET_COLUMN},
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

    if source_data.empty:
        raise RuntimeError(
            "No totals fallback development data "
            "is available."
        )

    return create_totals_fallback_features(
        source_data
    )


def prepare_common_fallback_sample(
    development_data: pd.DataFrame,
) -> pd.DataFrame:
    """Create one identical complete fallback sample."""

    data = create_totals_fallback_features(
        development_data
    )

    allowed_split_mask = data[
        "split_name"
    ].isin(
        [
            TRAIN_SPLIT,
            VALIDATION_SPLIT,
        ]
    )

    complete_mask = (
        allowed_split_mask
        & data[TOTALS_TARGET_COLUMN].notna()
        & data[
            list(FALLBACK_CORE_FEATURES)
        ].notna().all(axis=1)
    )

    sample = data.loc[
        complete_mask
    ].copy()

    if sample.empty:
        raise RuntimeError(
            "No complete totals fallback games "
            "are available."
        )

    if sample["game_id"].duplicated().any():
        raise ValueError(
            "Totals fallback data contains duplicate "
            "game identifiers."
        )

    if set(sample["split_name"]) != {
        TRAIN_SPLIT,
        VALIDATION_SPLIT,
    }:
        raise RuntimeError(
            "Totals fallback sample must contain "
            "train and validation games."
        )

    return sample


def evaluate_totals_fallback_candidates(
    development_data: pd.DataFrame,
    candidates: tuple[
        TotalsFallbackCandidate, ...
    ] = TOTALS_FALLBACK_CANDIDATES,
) -> pd.DataFrame:
    """Evaluate fallback candidates on identical games."""

    if not candidates:
        raise ValueError(
            "At least one fallback candidate is required."
        )

    candidate_names = [
        candidate.candidate_name
        for candidate in candidates
    ]

    if len(candidate_names) != len(
        set(candidate_names)
    ):
        raise ValueError(
            "Fallback candidate names must be unique."
        )

    sample = prepare_common_fallback_sample(
        development_data
    )

    train_data = sample.loc[
        sample["split_name"] == TRAIN_SPLIT
    ]

    validation_data = sample.loc[
        sample["split_name"] == VALIDATION_SPLIT
    ]

    train_target = train_data[
        TOTALS_TARGET_COLUMN
    ]

    validation_target = validation_data[
        TOTALS_TARGET_COLUMN
    ]

    constant_prediction = np.full(
        shape=len(validation_data),
        fill_value=float(train_target.mean()),
    )

    result_rows: list[
        dict[str, object]
    ] = [
        {
            "candidate_name": (
                "constant_train_mean"
            ),
            "feature_count": 0,
            "ridge_alpha": None,
            "train_game_count": len(train_data),
            "validation_game_count": len(
                validation_data
            ),
            **calculate_regression_metrics(
                actual_margin=validation_target,
                predicted_margin=constant_prediction,
            ),
        }
    ]

    for candidate in candidates:
        feature_columns = list(
            candidate.feature_columns
        )

        model = create_ridge_pipeline(
            ridge_alpha=candidate.ridge_alpha
        )

        model.fit(
            train_data.loc[
                :,
                feature_columns,
            ],
            train_target,
        )

        predictions = model.predict(
            validation_data.loc[
                :,
                feature_columns,
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
                    actual_margin=validation_target,
                    predicted_margin=predictions,
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


def run_totals_fallback_evaluation(
    database_file: Path = DATABASE_FILE,
) -> pd.DataFrame:
    """Load DuckDB data and evaluate fallbacks."""

    validate_database_file(database_file)

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        development_data = (
            load_totals_fallback_development_data(
                connection
            )
        )

    results = evaluate_totals_fallback_candidates(
        development_data
    )

    logger.info(
        "Totals fallback evaluation completed on "
        "%s train and %s validation games.",
        int(results.iloc[0]["train_game_count"]),
        int(
            results.iloc[0][
                "validation_game_count"
            ]
        ),
    )

    return results


def main() -> None:
    """Run and print fallback evaluation."""

    results = run_totals_fallback_evaluation()

    print(
        results.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()