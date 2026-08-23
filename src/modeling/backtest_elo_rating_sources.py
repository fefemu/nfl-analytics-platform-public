"""
NFL Analytics Platform
Elo Rating Source Backtest

Purpose:
    Compare internal Elo, external nfelo and a simple
    50-50 blend on identical chronological validation
    games for win probability and scoring margin.

    The published nfelo probability is included only as
    an independent probability benchmark.

    The 2025 holdout is never loaded or evaluated.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.modeling.evaluate_spread_model_candidates import (
    calculate_regression_metrics,
    create_ridge_pipeline,
)
from src.modeling.train_logistic_baseline import (
    DATABASE_FILE,
    TARGET_COLUMN,
    create_logistic_pipeline,
    evaluate_probabilities,
    validate_database_file,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

DATASET_FULL_NAME = (
    "analytics.game_modeling_dataset"
)

SPLIT_FULL_NAME = (
    "analytics.modeling_game_splits"
)

EXTERNAL_RATINGS_FULL_NAME = (
    "processed.external_nfelo_game_ratings"
)

SPREAD_TARGET_COLUMN = (
    "target_point_differential"
)

BACKTEST_VALIDATION_SEASONS = (
    2021,
    2022,
    2023,
    2024,
)

LOGISTIC_REGULARIZATION_C = 1.0
SPREAD_RIDGE_ALPHA = 10.0

INTERNAL_FEATURE = (
    "internal_elo_rating_difference"
)

EXTERNAL_FEATURE = (
    "external_nfelo_rating_difference"
)

BLEND_25_INTERNAL_FEATURE = (
    "blend_25_internal_75_external"
)

BLEND_FEATURE = (
    "blended_elo_rating_difference"
)

BLEND_75_INTERNAL_FEATURE = (
    "blend_75_internal_25_external"
)

LISTED_QB_FEATURE = (
    "listed_qb_rating_difference"
)

EXTERNAL_QB_FEATURE = (
    "external_nfelo_qb_adjustment_difference"
)

TRAINED_CANDIDATES = {
    "internal_elo": (
        INTERNAL_FEATURE,
    ),
    "external_nfelo": (
        EXTERNAL_FEATURE,
    ),
    "blend_25_internal_75_external": (
        BLEND_25_INTERNAL_FEATURE,
    ),
    "internal_external_blend_50": (
        BLEND_FEATURE,
    ),
    "blend_75_internal_25_external": (
        BLEND_75_INTERNAL_FEATURE,
    ),
    "internal_elo_listed_qb": (
        INTERNAL_FEATURE,
        LISTED_QB_FEATURE,
    ),
    "external_nfelo_listed_qb": (
        EXTERNAL_FEATURE,
        LISTED_QB_FEATURE,
    ),
    "external_nfelo_external_qb": (
        EXTERNAL_FEATURE,
        EXTERNAL_QB_FEATURE,
    ),
    "external_nfelo_both_qb": (
        EXTERNAL_FEATURE,
        LISTED_QB_FEATURE,
        EXTERNAL_QB_FEATURE,
    ),
    "blend_25_internal_75_external_listed_qb": (
        BLEND_25_INTERNAL_FEATURE,
        LISTED_QB_FEATURE,
    ),
}

PUBLISHED_NFELO_CANDIDATE = (
    "published_nfelo_probability"
)

FOLD_RESULT_COLUMNS = (
    "candidate_name",
    "validation_season",
    "training_game_count",
    "validation_game_count",
    "accuracy",
    "brier_score",
    "log_loss",
    "spread_mae",
    "spread_rmse",
    "spread_bias",
    "spread_r_squared",
)

SUMMARY_COLUMNS = (
    "candidate_name",
    "fold_count",
    "validation_game_count",
    "accuracy",
    "brier_score",
    "log_loss",
    "spread_mae",
    "spread_rmse",
    "spread_bias",
    "spread_r_squared",
)


def validate_source_tables(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate rating-source backtest tables."""

    required_tables = {
        (
            "analytics",
            "game_modeling_dataset",
        ),
        (
            "analytics",
            "modeling_game_splits",
        ),
        (
            "processed",
            "external_nfelo_game_ratings",
        ),
    }

    existing_tables = {
        (row[0], row[1])
        for row in connection.execute(
            """
            SELECT
                table_schema,
                table_name
            FROM information_schema.tables
            """
        ).fetchall()
    }

    missing_tables = (
        required_tables - existing_tables
    )

    if missing_tables:
        missing_names = ", ".join(
            f"{schema}.{table}"
            for schema, table
            in sorted(missing_tables)
        )

        raise RuntimeError(
            "Missing Elo source backtest tables: "
            + missing_names
        )


def load_rating_source_backtest_data(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Load identical development games and ratings."""

    data = connection.execute(
        f"""
        SELECT
            dataset.game_id,
            dataset.season,
            dataset.game_date,
            splits.split_name,

            CAST(
                dataset.{TARGET_COLUMN}
                AS INTEGER
            ) AS {TARGET_COLUMN},

            dataset.{SPREAD_TARGET_COLUMN},

            dataset.elo_rating_difference
                AS {INTERNAL_FEATURE},

            dataset.listed_qb_rating_difference
                AS {LISTED_QB_FEATURE},

            external.starting_nfelo_home,
            external.starting_nfelo_away,

            external.starting_nfelo_home
                - external.starting_nfelo_away
                AS {EXTERNAL_FEATURE},

            0.25
                * dataset.elo_rating_difference
                + 0.75
                * (
                    external.starting_nfelo_home
                    - external.starting_nfelo_away
                )
                AS {BLEND_25_INTERNAL_FEATURE},

            0.5
                * dataset.elo_rating_difference
                + 0.5
                * (
                    external.starting_nfelo_home
                    - external.starting_nfelo_away
                )
                AS {BLEND_FEATURE},

            0.75
                * dataset.elo_rating_difference
                + 0.25
                * (
                    external.starting_nfelo_home
                    - external.starting_nfelo_away
                )
                AS {BLEND_75_INTERNAL_FEATURE},

            external.home_538_qb_adj
                - external.away_538_qb_adj
                AS {EXTERNAL_QB_FEATURE},

            external.nfelo_home_probability_open
                AS published_nfelo_home_probability

        FROM {DATASET_FULL_NAME}
            AS dataset

        INNER JOIN {SPLIT_FULL_NAME}
            AS splits
            ON dataset.game_id = splits.game_id

        INNER JOIN {EXTERNAL_RATINGS_FULL_NAME}
            AS external
            ON dataset.game_id
                = external.normalized_game_id

        WHERE splits.split_name IN (
            'train',
            'validation'
        )

        ORDER BY
            dataset.season,
            dataset.game_date,
            dataset.game_id
        """
    ).fetchdf()

    if data.empty:
        raise RuntimeError(
            "No Elo source backtest data is available."
        )

    return data


def prepare_common_backtest_sample(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Create one complete identical backtest sample."""

    required_columns = {
        "game_id",
        "season",
        "split_name",
        TARGET_COLUMN,
        SPREAD_TARGET_COLUMN,
        INTERNAL_FEATURE,
        EXTERNAL_FEATURE,
        BLEND_25_INTERNAL_FEATURE,
        BLEND_FEATURE,
        BLEND_75_INTERNAL_FEATURE,
        LISTED_QB_FEATURE,
        EXTERNAL_QB_FEATURE,
        "published_nfelo_home_probability",
    }

    missing_columns = sorted(
        required_columns - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Elo source backtest data is missing "
            "columns: "
            + ", ".join(missing_columns)
        )

    if data["game_id"].duplicated().any():
        raise ValueError(
            "Elo source backtest data contains "
            "duplicate game identifiers."
        )

    unexpected_splits = sorted(
        set(
            data["split_name"].dropna()
        )
        - {
            "train",
            "validation",
        }
    )

    if unexpected_splits:
        raise ValueError(
            "Elo source backtest must not contain "
            "holdout or unknown splits: "
            + ", ".join(unexpected_splits)
        )

    if int(data["season"].max()) >= 2025:
        raise ValueError(
            "Elo source backtest must end before "
            "the 2025 holdout season."
        )

    complete_columns = [
        TARGET_COLUMN,
        SPREAD_TARGET_COLUMN,
        INTERNAL_FEATURE,
        EXTERNAL_FEATURE,
        BLEND_25_INTERNAL_FEATURE,
        BLEND_FEATURE,
        BLEND_75_INTERNAL_FEATURE,
        LISTED_QB_FEATURE,
        EXTERNAL_QB_FEATURE,
        "published_nfelo_home_probability",
    ]

    sample = data.loc[
        data[
            complete_columns
        ].notna().all(axis=1)
    ].copy()

    if sample.empty:
        raise RuntimeError(
            "No complete Elo source backtest games "
            "are available."
        )

    target_values = set(
        sample[TARGET_COLUMN].unique()
    )

    if not target_values.issubset(
        {
            0,
            1,
        }
    ):
        raise ValueError(
            "Probability target must be binary."
        )

    invalid_probability_mask = (
        sample[
            "published_nfelo_home_probability"
        ].le(0.0)
        | sample[
            "published_nfelo_home_probability"
        ].ge(1.0)
    )

    if invalid_probability_mask.any():
        raise ValueError(
            "Published nfelo probabilities must "
            "be between zero and one."
        )

    return sample


def validate_backtest_seasons(
    sample: pd.DataFrame,
    validation_seasons: tuple[int, ...],
) -> None:
    """Validate chronological fold seasons."""

    if not validation_seasons:
        raise ValueError(
            "At least one validation season is required."
        )

    if tuple(
        sorted(validation_seasons)
    ) != validation_seasons:
        raise ValueError(
            "Validation seasons must be chronological."
        )

    if len(validation_seasons) != len(
        set(validation_seasons)
    ):
        raise ValueError(
            "Validation seasons must be unique."
        )

    missing_seasons = sorted(
        set(validation_seasons)
        - set(
            sample["season"].astype(int)
        )
    )

    if missing_seasons:
        raise ValueError(
            "Elo source validation seasons are missing: "
            + ", ".join(
                str(season)
                for season in missing_seasons
            )
        )

    if (
        validation_seasons[0]
        <= int(sample["season"].min())
    ):
        raise ValueError(
            "Every validation season requires "
            "earlier training data."
        )


def evaluate_rating_source_backtest(
    data: pd.DataFrame,
    validation_seasons: tuple[
        int, ...
    ] = BACKTEST_VALIDATION_SEASONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run expanding-window rating source backtests."""

    sample = prepare_common_backtest_sample(
        data
    )

    validate_backtest_seasons(
        sample=sample,
        validation_seasons=validation_seasons,
    )

    fold_rows: list[
        dict[str, object]
    ] = []

    prediction_rows: list[
        dict[str, object]
    ] = []

    for validation_season in validation_seasons:
        training_data = sample.loc[
            sample["season"]
            < validation_season
        ].copy()

        validation_data = sample.loc[
            sample["season"]
            == validation_season
        ].copy()

        if training_data.empty:
            raise RuntimeError(
                "No Elo source training games precede "
                f"{validation_season}."
            )

        if validation_data.empty:
            raise RuntimeError(
                "No Elo source validation games exist "
                f"for {validation_season}."
            )

        training_target = training_data[
            TARGET_COLUMN
        ]

        if training_target.nunique() != 2:
            raise RuntimeError(
                "Elo source training data must contain "
                "both probability target classes."
            )

        validation_target = validation_data[
            TARGET_COLUMN
        ]

        validation_margin = validation_data[
            SPREAD_TARGET_COLUMN
        ]

        for (
            candidate_name,
            feature_columns,
        ) in TRAINED_CANDIDATES.items():

            probability_model = (
                create_logistic_pipeline(
                    feature_columns=feature_columns,
                    regularization_c=(
                        LOGISTIC_REGULARIZATION_C
                    ),
                )
            )

            probability_model.fit(
                training_data.loc[
                    :,
                    feature_columns,
                ],
                training_target,
            )

            probabilities = (
                probability_model.predict_proba(
                    validation_data.loc[
                        :,
                        feature_columns,
                    ]
                )[:, 1]
            )

            probability_metrics = (
                evaluate_probabilities(
                    actual_values=validation_target,
                    probabilities=probabilities,
                )
            )

            spread_model = create_ridge_pipeline(
                ridge_alpha=SPREAD_RIDGE_ALPHA
            )

            spread_model.fit(
                training_data.loc[
                    :,
                    feature_columns,
                ],
                training_data[
                    SPREAD_TARGET_COLUMN
                ],
            )

            predicted_margin = spread_model.predict(
                validation_data.loc[
                    :,
                    feature_columns,
                ]
            )

            spread_metrics = (
                calculate_regression_metrics(
                    actual_margin=validation_margin,
                    predicted_margin=(
                        predicted_margin
                    ),
                )
            )

            fold_rows.append(
                {
                    "candidate_name": (
                        candidate_name
                    ),
                    "validation_season": (
                        validation_season
                    ),
                    "training_game_count": len(
                        training_data
                    ),
                    "validation_game_count": len(
                        validation_data
                    ),
                    "accuracy": (
                        probability_metrics.accuracy
                    ),
                    "brier_score": (
                        probability_metrics.brier_score
                    ),
                    "log_loss": (
                        probability_metrics.log_loss
                    ),
                    "spread_mae": (
                        spread_metrics[
                            "validation_mae"
                        ]
                    ),
                    "spread_rmse": (
                        spread_metrics[
                            "validation_rmse"
                        ]
                    ),
                    "spread_bias": (
                        spread_metrics[
                            "validation_bias"
                        ]
                    ),
                    "spread_r_squared": (
                        spread_metrics[
                            "validation_r_squared"
                        ]
                    ),
                }
            )

            for (
                game_id,
                actual_win,
                probability,
                actual_margin,
                margin_prediction,
            ) in zip(
                validation_data["game_id"],
                validation_target,
                probabilities,
                validation_margin,
                predicted_margin,
                strict=True,
            ):
                prediction_rows.append(
                    {
                        "candidate_name": (
                            candidate_name
                        ),
                        "validation_season": (
                            validation_season
                        ),
                        "game_id": game_id,
                        "actual_home_win": int(
                            actual_win
                        ),
                        "home_win_probability": float(
                            probability
                        ),
                        "actual_home_margin": float(
                            actual_margin
                        ),
                        "predicted_home_margin": float(
                            margin_prediction
                        ),
                    }
                )

        published_probabilities = validation_data[
            "published_nfelo_home_probability"
        ].to_numpy(dtype=float)

        published_metrics = evaluate_probabilities(
            actual_values=validation_target,
            probabilities=published_probabilities,
        )

        fold_rows.append(
            {
                "candidate_name": (
                    PUBLISHED_NFELO_CANDIDATE
                ),
                "validation_season": (
                    validation_season
                ),
                "training_game_count": len(
                    training_data
                ),
                "validation_game_count": len(
                    validation_data
                ),
                "accuracy": (
                    published_metrics.accuracy
                ),
                "brier_score": (
                    published_metrics.brier_score
                ),
                "log_loss": (
                    published_metrics.log_loss
                ),
                "spread_mae": None,
                "spread_rmse": None,
                "spread_bias": None,
                "spread_r_squared": None,
            }
        )

        for (
            game_id,
            actual_win,
            probability,
        ) in zip(
            validation_data["game_id"],
            validation_target,
            published_probabilities,
            strict=True,
        ):
            prediction_rows.append(
                {
                    "candidate_name": (
                        PUBLISHED_NFELO_CANDIDATE
                    ),
                    "validation_season": (
                        validation_season
                    ),
                    "game_id": game_id,
                    "actual_home_win": int(
                        actual_win
                    ),
                    "home_win_probability": float(
                        probability
                    ),
                    "actual_home_margin": None,
                    "predicted_home_margin": None,
                }
            )

    fold_results = pd.DataFrame(
        fold_rows,
        columns=FOLD_RESULT_COLUMNS,
    ).sort_values(
        by=[
            "validation_season",
            "brier_score",
            "candidate_name",
        ],
        kind="stable",
    ).reset_index(drop=True)

    predictions = pd.DataFrame(
        prediction_rows
    )

    summary_rows: list[
        dict[str, object]
    ] = []

    for candidate_name, group in predictions.groupby(
        "candidate_name",
        sort=False,
    ):
        probability_metrics = evaluate_probabilities(
            actual_values=group[
                "actual_home_win"
            ],
            probabilities=group[
                "home_win_probability"
            ],
        )

        spread_sample = group.loc[
            group[
                "actual_home_margin"
            ].notna()
            & group[
                "predicted_home_margin"
            ].notna()
        ]

        if spread_sample.empty:
            spread_metrics = {
                "validation_mae": None,
                "validation_rmse": None,
                "validation_bias": None,
                "validation_r_squared": None,
            }
        else:
            spread_metrics = (
                calculate_regression_metrics(
                    actual_margin=spread_sample[
                        "actual_home_margin"
                    ],
                    predicted_margin=(
                        spread_sample[
                            "predicted_home_margin"
                        ].to_numpy(dtype=float)
                    ),
                )
            )

        summary_rows.append(
            {
                "candidate_name": (
                    candidate_name
                ),
                "fold_count": int(
                    group[
                        "validation_season"
                    ].nunique()
                ),
                "validation_game_count": len(
                    group
                ),
                "accuracy": (
                    probability_metrics.accuracy
                ),
                "brier_score": (
                    probability_metrics.brier_score
                ),
                "log_loss": (
                    probability_metrics.log_loss
                ),
                "spread_mae": (
                    spread_metrics[
                        "validation_mae"
                    ]
                ),
                "spread_rmse": (
                    spread_metrics[
                        "validation_rmse"
                    ]
                ),
                "spread_bias": (
                    spread_metrics[
                        "validation_bias"
                    ]
                ),
                "spread_r_squared": (
                    spread_metrics[
                        "validation_r_squared"
                    ]
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows,
        columns=SUMMARY_COLUMNS,
    ).sort_values(
        by=[
            "brier_score",
            "log_loss",
            "candidate_name",
        ],
        kind="stable",
    ).reset_index(drop=True)

    return summary, fold_results


def run_rating_source_backtest(
    database_file: Path = DATABASE_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load DuckDB data and run source comparison."""

    validate_database_file(database_file)

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        validate_source_tables(connection)

        data = load_rating_source_backtest_data(
            connection
        )

    summary, fold_results = (
        evaluate_rating_source_backtest(
            data
        )
    )

    logger.info(
        "Elo rating source backtest completed: "
        "%s validation games across %s folds "
        "without opening holdout.",
        int(
            summary.iloc[0][
                "validation_game_count"
            ]
        ),
        len(BACKTEST_VALIDATION_SEASONS),
    )

    return summary, fold_results


def main() -> None:
    """Run and print rating-source diagnostics."""

    summary, fold_results = (
        run_rating_source_backtest()
    )

    print("\nELO RATING SOURCE BACKTEST SUMMARY\n")

    print(
        summary.to_string(
            index=False
        )
    )

    print("\nSEASON-LEVEL RESULTS\n")

    print(
        fold_results.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()