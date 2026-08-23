"""
NFL Analytics Platform
External Probability Fallback Backtest

Purpose:
    Compare current and external probability fallbacks
    on historical games that were not eligible for the
    complete injury-enhanced primary model.

    External fallback models are trained on all earlier
    complete games, but evaluated only on routed fallback
    validation games.

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

from src.modeling.backtest_external_elo_probability_champion import (
    EXTERNAL_ELO_FEATURE,
    EXTERNAL_LOGISTIC_FEATURES,
    EXTERNAL_QB_FEATURE,
    INTERNAL_LOGISTIC_FEATURES,
)
from src.modeling.backtest_elo_rating_sources import (
    BACKTEST_VALIDATION_SEASONS,
)
from src.modeling.production_probability_model import (
    PRODUCTION_PROBABILITY_MODEL,
)
from src.modeling.run_logistic_injury_time_cv import (
    INJURY_AVAILABILITY_COLUMN,
    UNIT_BURDEN_FEATURES,
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

PRIMARY_ELIGIBILITY_COLUMN = (
    "is_probability_primary_eligible"
)

CURRENT_FALLBACK_CANDIDATE = (
    "current_internal_elo_fallback"
)

PUBLISHED_NFELO_CANDIDATE = (
    "published_nfelo_fallback"
)

EXTERNAL_LOGISTIC_CANDIDATE = (
    "external_elo_qb_logistic_fallback"
)

EXTERNAL_BLEND_CANDIDATE = (
    "external_nfelo_logistic_blend_fallback"
)

CANDIDATE_NAMES = (
    CURRENT_FALLBACK_CANDIDATE,
    PUBLISHED_NFELO_CANDIDATE,
    EXTERNAL_LOGISTIC_CANDIDATE,
    EXTERNAL_BLEND_CANDIDATE,
)

COMPARISONS = {
    "published_nfelo_vs_current_fallback": (
        CURRENT_FALLBACK_CANDIDATE,
        PUBLISHED_NFELO_CANDIDATE,
    ),
    "external_logistic_vs_current_fallback": (
        CURRENT_FALLBACK_CANDIDATE,
        EXTERNAL_LOGISTIC_CANDIDATE,
    ),
    "external_blend_vs_current_fallback": (
        CURRENT_FALLBACK_CANDIDATE,
        EXTERNAL_BLEND_CANDIDATE,
    ),
}

DEFAULT_BOOTSTRAP_ITERATIONS = 10_000
DEFAULT_RANDOM_SEED = 42

SUMMARY_COLUMNS = (
    "candidate_name",
    "fold_count",
    "fallback_validation_game_count",
    "accuracy",
    "brier_score",
    "log_loss",
)

FOLD_RESULT_COLUMNS = (
    "candidate_name",
    "validation_season",
    "training_game_count",
    "fallback_validation_game_count",
    "accuracy",
    "brier_score",
    "log_loss",
)

PREDICTION_COLUMNS = (
    "candidate_name",
    "validation_season",
    "game_id",
    "actual_home_win",
    "home_win_probability",
    "brier_loss",
)

PAIRED_SUMMARY_COLUMNS = (
    "comparison_name",
    "base_candidate",
    "challenger_candidate",
    "fold_count",
    "fallback_validation_game_count",
    "base_brier_score",
    "challenger_brier_score",
    "brier_score_delta",
    "challenger_win_rate",
    "challenger_loss_rate",
    "equal_loss_rate",
    "bootstrap_mean_delta",
    "bootstrap_95_percent_lower",
    "bootstrap_95_percent_upper",
)


def load_probability_fallback_data(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Load development data for fallback routing."""

    injury_select = ",\n            ".join(
        f"dataset.{column_name}"
        for column_name in UNIT_BURDEN_FEATURES
    )

    data = connection.execute(
        f"""
        SELECT
            dataset.game_id,
            dataset.season,
            dataset.game_date,
            splits.split_name,
            splits.is_core_model_eligible,

            CAST(
                dataset.{TARGET_COLUMN}
                AS INTEGER
            ) AS {TARGET_COLUMN},

            dataset.{INJURY_AVAILABILITY_COLUMN},
            dataset.elo_rating_difference,
            dataset.listed_qb_rating_difference,
            dataset.elo_home_win_probability,

            {injury_select},

            external.starting_nfelo_home
                - external.starting_nfelo_away
                AS {EXTERNAL_ELO_FEATURE},

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
            dataset.game_date,
            dataset.game_id
        """
    ).fetchdf()

    if data.empty:
        raise RuntimeError(
            "No probability fallback development "
            "data is available."
        )

    return data


def prepare_probability_fallback_sample(
    source_data: pd.DataFrame,
) -> pd.DataFrame:
    """Create common data and primary routing status."""

    required_columns = {
        "game_id",
        "season",
        "split_name",
        "is_core_model_eligible",
        TARGET_COLUMN,
        INJURY_AVAILABILITY_COLUMN,
        "elo_home_win_probability",
        "published_nfelo_home_probability",
        *INTERNAL_LOGISTIC_FEATURES,
        EXTERNAL_ELO_FEATURE,
        EXTERNAL_QB_FEATURE,
    }

    missing_columns = sorted(
        required_columns
        - set(source_data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Probability fallback data is missing "
            "columns: "
            + ", ".join(missing_columns)
        )

    if source_data[
        "game_id"
    ].duplicated().any():
        raise ValueError(
            "Probability fallback data contains "
            "duplicate game identifiers."
        )

    unexpected_splits = sorted(
        set(
            source_data[
                "split_name"
            ].dropna()
        )
        - {
            "train",
            "validation",
        }
    )

    if unexpected_splits:
        raise ValueError(
            "Probability fallback backtest must not "
            "contain holdout or unknown splits: "
            + ", ".join(unexpected_splits)
        )

    if int(source_data["season"].max()) >= 2025:
        raise ValueError(
            "Probability fallback backtest must end "
            "before the 2025 holdout season."
        )

    common_columns = [
        TARGET_COLUMN,
        "elo_home_win_probability",
        "published_nfelo_home_probability",
        EXTERNAL_ELO_FEATURE,
        EXTERNAL_QB_FEATURE,
    ]

    sample = source_data.loc[
        source_data[
            common_columns
        ].notna().all(axis=1)
    ].copy()

    if sample.empty:
        raise RuntimeError(
            "No complete probability fallback games "
            "are available."
        )

    if not set(
        sample[TARGET_COLUMN].unique()
    ).issubset(
        {
            0,
            1,
        }
    ):
        raise ValueError(
            "Probability fallback target must be binary."
        )

    internal_primary_complete = sample[
        list(INTERNAL_LOGISTIC_FEATURES)
    ].notna().all(axis=1)

    external_primary_complete = sample[
        list(EXTERNAL_LOGISTIC_FEATURES)
    ].notna().all(axis=1)

    sample[
        PRIMARY_ELIGIBILITY_COLUMN
    ] = (
        sample[
            "is_core_model_eligible"
        ].fillna(False).astype(bool)
        & sample[
            INJURY_AVAILABILITY_COLUMN
        ].fillna(False).astype(bool)
        & internal_primary_complete
        & external_primary_complete
    )

    probability_values = sample[
        [
            "elo_home_win_probability",
            "published_nfelo_home_probability",
        ]
    ].to_numpy(dtype=float)

    if (
        not np.isfinite(
            probability_values
        ).all()
        or (
            probability_values <= 0.0
        ).any()
        or (
            probability_values >= 1.0
        ).any()
    ):
        raise ValueError(
            "Fallback source probabilities must be "
            "between zero and one."
        )

    return sample


def validate_validation_seasons(
    sample: pd.DataFrame,
    validation_seasons: tuple[int, ...],
) -> None:
    """Validate fallback chronological folds."""

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
            "Fallback validation seasons are missing: "
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


def bootstrap_paired_mean_delta(
    paired_deltas: np.ndarray,
    iteration_count: int = (
        DEFAULT_BOOTSTRAP_ITERATIONS
    ),
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, float]:
    """Bootstrap paired Brier-loss deltas."""

    deltas = np.asarray(
        paired_deltas,
        dtype=float,
    )

    if deltas.ndim != 1 or deltas.size == 0:
        raise ValueError(
            "Bootstrap requires a non-empty "
            "one-dimensional delta array."
        )

    if not np.isfinite(deltas).all():
        raise ValueError(
            "Bootstrap deltas must be finite."
        )

    if iteration_count <= 0:
        raise ValueError(
            "Bootstrap iteration count must be positive."
        )

    random_generator = np.random.default_rng(
        random_seed
    )

    bootstrap_means = np.empty(
        iteration_count,
        dtype=float,
    )

    for iteration_index in range(
        iteration_count
    ):
        bootstrap_sample = random_generator.choice(
            deltas,
            size=deltas.size,
            replace=True,
        )

        bootstrap_means[
            iteration_index
        ] = bootstrap_sample.mean()

    return {
        "bootstrap_mean_delta": float(
            bootstrap_means.mean()
        ),
        "bootstrap_95_percent_lower": float(
            np.percentile(
                bootstrap_means,
                2.5,
            )
        ),
        "bootstrap_95_percent_upper": float(
            np.percentile(
                bootstrap_means,
                97.5,
            )
        ),
    }


def create_fallback_oof_predictions(
    source_data: pd.DataFrame,
    validation_seasons: tuple[
        int, ...
    ] = BACKTEST_VALIDATION_SEASONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create predictions only for routed fallback games."""

    sample = prepare_probability_fallback_sample(
        source_data
    )

    validate_validation_seasons(
        sample=sample,
        validation_seasons=validation_seasons,
    )

    prediction_rows: list[
        dict[str, object]
    ] = []

    fold_rows: list[
        dict[str, object]
    ] = []

    regularization_c = (
        PRODUCTION_PROBABILITY_MODEL
        .logistic_regularization_c
    )

    logistic_weight = (
        PRODUCTION_PROBABILITY_MODEL
        .logistic_weight
    )

    elo_weight = (
        PRODUCTION_PROBABILITY_MODEL
        .elo_weight
    )

    fallback_features = (
        EXTERNAL_ELO_FEATURE,
        EXTERNAL_QB_FEATURE,
    )

    for validation_season in validation_seasons:
        training_data = sample.loc[
            sample["season"] < validation_season
        ].copy()

        validation_data = sample.loc[
            (
                sample["season"]
                == validation_season
            )
            & (
                ~sample[
                    PRIMARY_ELIGIBILITY_COLUMN
                ]
            )
        ].copy()

        if training_data.empty:
            raise RuntimeError(
                "No fallback training games precede "
                f"{validation_season}."
            )

        if validation_data.empty:
            raise RuntimeError(
                "No routed fallback validation games "
                f"exist for {validation_season}."
            )

        if (
            training_data[TARGET_COLUMN].nunique()
            != 2
        ):
            raise RuntimeError(
                "Fallback training data must contain "
                "both target classes."
            )

        external_model = (
            create_logistic_pipeline(
                feature_columns=(
                    fallback_features
                ),
                regularization_c=(
                    regularization_c
                ),
            )
        )

        external_model.fit(
            training_data.loc[
                :,
                fallback_features,
            ],
            training_data[TARGET_COLUMN],
        )

        external_logistic_probability = (
            external_model.predict_proba(
                validation_data.loc[
                    :,
                    fallback_features,
                ]
            )[:, 1]
        )

        current_probability = validation_data[
            "elo_home_win_probability"
        ].to_numpy(dtype=float)

        published_probability = validation_data[
            "published_nfelo_home_probability"
        ].to_numpy(dtype=float)

        external_blend_probability = (
            logistic_weight
            * external_logistic_probability
            + elo_weight
            * published_probability
        )

        probability_sets = {
            CURRENT_FALLBACK_CANDIDATE: (
                current_probability
            ),
            PUBLISHED_NFELO_CANDIDATE: (
                published_probability
            ),
            EXTERNAL_LOGISTIC_CANDIDATE: (
                external_logistic_probability
            ),
            EXTERNAL_BLEND_CANDIDATE: (
                external_blend_probability
            ),
        }

        actual_values = validation_data[
            TARGET_COLUMN
        ]

        actual_array = actual_values.to_numpy(
            dtype=int
        )

        for (
            candidate_name,
            probabilities,
        ) in probability_sets.items():
            evaluation = evaluate_probabilities(
                actual_values=actual_values,
                probabilities=probabilities,
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
                    "fallback_validation_game_count": (
                        len(validation_data)
                    ),
                    "accuracy": (
                        evaluation.accuracy
                    ),
                    "brier_score": (
                        evaluation.brier_score
                    ),
                    "log_loss": (
                        evaluation.log_loss
                    ),
                }
            )

            brier_losses = np.square(
                probabilities
                - actual_array
            )

            for (
                row_index,
                game_id,
            ) in enumerate(
                validation_data["game_id"]
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
                            actual_array[
                                row_index
                            ]
                        ),
                        "home_win_probability": float(
                            probabilities[
                                row_index
                            ]
                        ),
                        "brier_loss": float(
                            brier_losses[
                                row_index
                            ]
                        ),
                    }
                )

    predictions = pd.DataFrame(
        prediction_rows,
        columns=PREDICTION_COLUMNS,
    )

    fold_results = pd.DataFrame(
        fold_rows,
        columns=FOLD_RESULT_COLUMNS,
    )

    return predictions, fold_results


def create_candidate_summary(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Create pooled fallback metrics."""

    summary_rows: list[
        dict[str, object]
    ] = []

    for (
        candidate_name,
        candidate_predictions,
    ) in predictions.groupby(
        "candidate_name",
        sort=False,
    ):
        evaluation = evaluate_probabilities(
            actual_values=candidate_predictions[
                "actual_home_win"
            ],
            probabilities=candidate_predictions[
                "home_win_probability"
            ],
        )

        summary_rows.append(
            {
                "candidate_name": candidate_name,
                "fold_count": int(
                    candidate_predictions[
                        "validation_season"
                    ].nunique()
                ),
                "fallback_validation_game_count": (
                    len(candidate_predictions)
                ),
                "accuracy": (
                    evaluation.accuracy
                ),
                "brier_score": (
                    evaluation.brier_score
                ),
                "log_loss": (
                    evaluation.log_loss
                ),
            }
        )

    return pd.DataFrame(
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


def create_paired_summary(
    predictions: pd.DataFrame,
    bootstrap_iterations: int = (
        DEFAULT_BOOTSTRAP_ITERATIONS
    ),
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Create paired fallback comparisons."""

    summary_rows: list[
        dict[str, object]
    ] = []

    for comparison_index, (
        comparison_name,
        (
            base_candidate,
            challenger_candidate,
        ),
    ) in enumerate(COMPARISONS.items()):
        base = predictions.loc[
            predictions["candidate_name"]
            == base_candidate,
            [
                "game_id",
                "validation_season",
                "actual_home_win",
                "brier_loss",
            ],
        ].rename(
            columns={
                "brier_loss": (
                    "base_brier_loss"
                ),
            }
        )

        challenger = predictions.loc[
            predictions["candidate_name"]
            == challenger_candidate,
            [
                "game_id",
                "validation_season",
                "actual_home_win",
                "brier_loss",
            ],
        ].rename(
            columns={
                "brier_loss": (
                    "challenger_brier_loss"
                ),
            }
        )

        paired = base.merge(
            challenger,
            on=[
                "game_id",
                "validation_season",
                "actual_home_win",
            ],
            how="inner",
            validate="one_to_one",
        )

        if len(paired) != len(base):
            raise RuntimeError(
                "Fallback candidates do not cover "
                "identical validation games."
            )

        paired_deltas = (
            paired["challenger_brier_loss"]
            - paired["base_brier_loss"]
        ).to_numpy(dtype=float)

        bootstrap_results = (
            bootstrap_paired_mean_delta(
                paired_deltas=paired_deltas,
                iteration_count=(
                    bootstrap_iterations
                ),
                random_seed=(
                    random_seed
                    + comparison_index * 10
                ),
            )
        )

        game_count = len(paired)

        summary_rows.append(
            {
                "comparison_name": (
                    comparison_name
                ),
                "base_candidate": (
                    base_candidate
                ),
                "challenger_candidate": (
                    challenger_candidate
                ),
                "fold_count": int(
                    paired[
                        "validation_season"
                    ].nunique()
                ),
                "fallback_validation_game_count": (
                    game_count
                ),
                "base_brier_score": float(
                    paired[
                        "base_brier_loss"
                    ].mean()
                ),
                "challenger_brier_score": float(
                    paired[
                        "challenger_brier_loss"
                    ].mean()
                ),
                "brier_score_delta": float(
                    paired_deltas.mean()
                ),
                "challenger_win_rate": float(
                    np.mean(
                        paired_deltas < 0.0
                    )
                ),
                "challenger_loss_rate": float(
                    np.mean(
                        paired_deltas > 0.0
                    )
                ),
                "equal_loss_rate": float(
                    np.mean(
                        paired_deltas == 0.0
                    )
                ),
                **bootstrap_results,
            }
        )

    return pd.DataFrame(
        summary_rows,
        columns=PAIRED_SUMMARY_COLUMNS,
    )


def evaluate_probability_fallbacks(
    source_data: pd.DataFrame,
    validation_seasons: tuple[
        int, ...
    ] = BACKTEST_VALIDATION_SEASONS,
    bootstrap_iterations: int = (
        DEFAULT_BOOTSTRAP_ITERATIONS
    ),
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Run fallback candidate comparison."""

    (
        predictions,
        fold_results,
    ) = create_fallback_oof_predictions(
        source_data=source_data,
        validation_seasons=validation_seasons,
    )

    candidate_summary = (
        create_candidate_summary(
            predictions
        )
    )

    paired_summary = create_paired_summary(
        predictions=predictions,
        bootstrap_iterations=(
            bootstrap_iterations
        ),
        random_seed=random_seed,
    )

    return (
        candidate_summary,
        paired_summary,
        fold_results,
        predictions,
    )


def run_probability_fallback_backtest(
    database_file: Path = DATABASE_FILE,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load and evaluate fallback development data."""

    validate_database_file(database_file)

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        source_data = (
            load_probability_fallback_data(
                connection
            )
        )

    results = evaluate_probability_fallbacks(
        source_data=source_data
    )

    logger.info(
        "External probability fallback backtest "
        "completed without opening holdout."
    )

    return results


def main() -> None:
    """Run and print fallback diagnostics."""

    (
        candidate_summary,
        paired_summary,
        fold_results,
        _,
    ) = run_probability_fallback_backtest()

    print(
        "\nPROBABILITY FALLBACK BACKTEST SUMMARY\n"
    )

    print(
        candidate_summary.to_string(
            index=False
        )
    )

    print(
        "\nPAIRED PROBABILITY FALLBACK SUMMARY\n"
    )

    print(
        paired_summary.to_string(
            index=False
        )
    )

    print(
        "\nSEASON-LEVEL FALLBACK RESULTS\n"
    )

    print(
        fold_results.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()