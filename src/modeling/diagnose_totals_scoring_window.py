"""
NFL Analytics Platform
Totals Scoring Window Diagnostics

Purpose:
    Compare the selected 64-game scoring-environment
    model with the strongest 128-game alternative using
    paired game-level errors, bootstrap uncertainty and
    standardized coefficient stability.

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

from src.modeling.backtest_totals_model_candidates import (
    BACKTEST_VALIDATION_SEASONS,
)
from src.modeling.evaluate_totals_model_candidates import (
    LEAGUE_SCORING_64_TOTALS_FEATURES,
    LEAGUE_SCORING_128_TOTALS_FEATURES,
    TOTALS_SELECTED_BASE_FEATURES,
    TOTALS_TARGET_COLUMN,
    create_ridge_pipeline,
    load_totals_development_data,
    prepare_common_totals_sample,
)
from src.modeling.train_logistic_baseline import (
    DATABASE_FILE,
    validate_database_file,
)


logger = logging.getLogger(__name__)

DEFAULT_BASE_RIDGE_ALPHA = 100.0
DEFAULT_ALTERNATIVE_RIDGE_ALPHA = 300.0
DEFAULT_BOOTSTRAP_ITERATIONS = 10_000
DEFAULT_RANDOM_SEED = 42

BASE_MODEL_NAME = (
    "ridge_epa_weather_qb_league_64"
)

ALTERNATIVE_MODEL_NAME = (
    "ridge_epa_weather_qb_league_128"
)

BASE_FEATURES = (
    *TOTALS_SELECTED_BASE_FEATURES,
    *LEAGUE_SCORING_64_TOTALS_FEATURES,
)

ALTERNATIVE_FEATURES = (
    *TOTALS_SELECTED_BASE_FEATURES,
    *LEAGUE_SCORING_128_TOTALS_FEATURES,
)

PAIRED_RESULT_COLUMNS = (
    "game_id",
    "validation_season",
    "actual_total",
    "base_predicted_total",
    "alternative_predicted_total",
    "base_absolute_error",
    "alternative_absolute_error",
    "alternative_absolute_error_delta",
    "alternative_model_wins",
)

FOLD_RESULT_COLUMNS = (
    "validation_season",
    "train_game_count",
    "validation_game_count",
    "base_mae",
    "alternative_mae",
    "alternative_mae_delta",
    "alternative_model_win_rate",
)

COEFFICIENT_RESULT_COLUMNS = (
    "candidate_name",
    "validation_season",
    "feature_name",
    "standardized_coefficient",
)

SUMMARY_RESULT_COLUMNS = (
    "base_ridge_alpha",
    "alternative_ridge_alpha",
    "fold_count",
    "validation_game_count",
    "base_mae",
    "alternative_mae",
    "alternative_mae_delta",
    "alternative_model_win_rate",
    "alternative_model_loss_rate",
    "equal_error_rate",
    "bootstrap_mean_delta",
    "bootstrap_95_percent_lower",
    "bootstrap_95_percent_upper",
)


def extract_standardized_coefficients(
    model,
    candidate_name: str,
    validation_season: int,
    feature_columns: tuple[str, ...],
) -> list[dict[str, object]]:
    """Extract coefficients after feature standardization."""

    coefficients = np.asarray(
        model.named_steps["model"].coef_,
        dtype=float,
    )

    if coefficients.shape != (
        len(feature_columns),
    ):
        raise ValueError(
            "Coefficient count does not match "
            "the feature count."
        )

    return [
        {
            "candidate_name": candidate_name,
            "validation_season": validation_season,
            "feature_name": feature_name,
            "standardized_coefficient": float(
                coefficient
            ),
        }
        for feature_name, coefficient in zip(
            feature_columns,
            coefficients,
            strict=True,
        )
    ]


def bootstrap_paired_mean_delta(
    paired_deltas: np.ndarray,
    iteration_count: int = (
        DEFAULT_BOOTSTRAP_ITERATIONS
    ),
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, float]:
    """Bootstrap the paired mean absolute-error delta."""

    deltas = np.asarray(
        paired_deltas,
        dtype=float,
    )

    if deltas.ndim != 1 or deltas.size == 0:
        raise ValueError(
            "Bootstrap requires a non-empty "
            "one-dimensional delta array."
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
        sample = random_generator.choice(
            deltas,
            size=deltas.size,
            replace=True,
        )

        bootstrap_means[
            iteration_index
        ] = sample.mean()

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


def diagnose_totals_scoring_window(
    development_data: pd.DataFrame,
    validation_seasons: tuple[
        int, ...
    ] = BACKTEST_VALIDATION_SEASONS,
    base_ridge_alpha: float = (
        DEFAULT_BASE_RIDGE_ALPHA
    ),
    alternative_ridge_alpha: float = (
        DEFAULT_ALTERNATIVE_RIDGE_ALPHA
    ),
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
    """Compare base and alternative models on paired games."""

    if base_ridge_alpha < 0.0:
        raise ValueError(
            "Base Ridge alpha must not be negative."
        )

    if alternative_ridge_alpha < 0.0:
        raise ValueError(
            "Alternative Ridge alpha must not be negative."
        )

    if not validation_seasons:
        raise ValueError(
            "At least one validation season is required."
        )

    sample = prepare_common_totals_sample(
        development_data
    )

    paired_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    coefficient_rows: list[
        dict[str, object]
    ] = []

    for validation_season in validation_seasons:
        train_data = sample.loc[
            sample["season"] < validation_season
        ].copy()

        validation_data = sample.loc[
            sample["season"] == validation_season
        ].copy()

        if train_data.empty:
            raise RuntimeError(
                f"No training games precede "
                f"{validation_season}."
            )

        if validation_data.empty:
            raise RuntimeError(
                f"No validation games exist for "
                f"{validation_season}."
            )

        train_target = train_data[
            TOTALS_TARGET_COLUMN
        ]

        base_model = create_ridge_pipeline(
            ridge_alpha=base_ridge_alpha
        )

        alternative_model = create_ridge_pipeline(
            ridge_alpha=alternative_ridge_alpha
        )

        base_model.fit(
            train_data.loc[
                :,
                list(BASE_FEATURES),
            ],
            train_target,
        )

        alternative_model.fit(
            train_data.loc[
                :,
                list(ALTERNATIVE_FEATURES),
            ],
            train_target,
        )

        base_predictions = base_model.predict(
            validation_data.loc[
                :,
                list(BASE_FEATURES),
            ]
        )

        alternative_predictions = alternative_model.predict(
            validation_data.loc[
                :,
                list(ALTERNATIVE_FEATURES),
            ]
        )

        actual_total = validation_data[
            TOTALS_TARGET_COLUMN
        ].to_numpy(dtype=float)

        base_absolute_error = np.abs(
            base_predictions - actual_total
        )

        alternative_absolute_error = np.abs(
            alternative_predictions - actual_total
        )

        error_delta = (
            alternative_absolute_error
            - base_absolute_error
        )

        for row_index, game_id in enumerate(
            validation_data["game_id"]
        ):
            paired_rows.append(
                {
                    "game_id": game_id,
                    "validation_season": (
                        validation_season
                    ),
                    "actual_total": float(
                        actual_total[row_index]
                    ),
                    "base_predicted_total": float(
                        base_predictions[row_index]
                    ),
                    "alternative_predicted_total": float(
                        alternative_predictions[row_index]
                    ),
                    "base_absolute_error": float(
                        base_absolute_error[row_index]
                    ),
                    "alternative_absolute_error": float(
                        alternative_absolute_error[row_index]
                    ),
                    "alternative_absolute_error_delta": float(
                        error_delta[row_index]
                    ),
                    "alternative_model_wins": bool(
                        error_delta[row_index] < 0.0
                    ),
                }
            )

        fold_rows.append(
            {
                "validation_season": validation_season,
                "train_game_count": len(
                    train_data
                ),
                "validation_game_count": len(
                    validation_data
                ),
                "base_mae": float(
                    base_absolute_error.mean()
                ),
                "alternative_mae": float(
                    alternative_absolute_error.mean()
                ),
                "alternative_mae_delta": float(
                    error_delta.mean()
                ),
                "alternative_model_win_rate": float(
                    np.mean(error_delta < 0.0)
                ),
            }
        )

        coefficient_rows.extend(
            extract_standardized_coefficients(
                model=base_model,
                candidate_name=BASE_MODEL_NAME,
                validation_season=validation_season,
                feature_columns=(
                    BASE_FEATURES
                ),
            )
        )

        coefficient_rows.extend(
            extract_standardized_coefficients(
                model=alternative_model,
                candidate_name=ALTERNATIVE_MODEL_NAME,
                validation_season=validation_season,
                feature_columns=(
                    ALTERNATIVE_FEATURES
                ),
            )
        )

    paired_results = pd.DataFrame(
        paired_rows,
        columns=PAIRED_RESULT_COLUMNS,
    )

    fold_results = pd.DataFrame(
        fold_rows,
        columns=FOLD_RESULT_COLUMNS,
    )

    coefficient_results = pd.DataFrame(
        coefficient_rows,
        columns=COEFFICIENT_RESULT_COLUMNS,
    )

    paired_deltas = paired_results[
        "alternative_absolute_error_delta"
    ].to_numpy(dtype=float)

    bootstrap_results = (
        bootstrap_paired_mean_delta(
            paired_deltas=paired_deltas,
            iteration_count=bootstrap_iterations,
            random_seed=random_seed,
        )
    )

    wins = int(
        np.sum(paired_deltas < 0.0)
    )

    losses = int(
        np.sum(paired_deltas > 0.0)
    )

    equal_errors = int(
        np.sum(paired_deltas == 0.0)
    )

    game_count = len(paired_results)

    summary_results = pd.DataFrame(
        [
            {
                "base_ridge_alpha": (
                    base_ridge_alpha
                ),
                "alternative_ridge_alpha": (
                    alternative_ridge_alpha
                ),
                "fold_count": len(
                    validation_seasons
                ),
                "validation_game_count": game_count,
                "base_mae": float(
                    paired_results[
                        "base_absolute_error"
                    ].mean()
                ),
                "alternative_mae": float(
                    paired_results[
                        "alternative_absolute_error"
                    ].mean()
                ),
                "alternative_mae_delta": float(
                    paired_deltas.mean()
                ),
                "alternative_model_win_rate": (
                    wins / game_count
                ),
                "alternative_model_loss_rate": (
                    losses / game_count
                ),
                "equal_error_rate": (
                    equal_errors / game_count
                ),
                **bootstrap_results,
            }
        ],
        columns=SUMMARY_RESULT_COLUMNS,
    )

    return (
        summary_results,
        fold_results,
        coefficient_results,
        paired_results,
    )


def run_totals_scoring_window_diagnostics(
    database_file: Path = DATABASE_FILE,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load development data and run diagnostics."""

    validate_database_file(
        database_file
    )

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        development_data = (
            load_totals_development_data(
                connection
            )
        )

    results = diagnose_totals_scoring_window(
        development_data=development_data
    )

    logger.info(
        "Totals scoring-window diagnostics completed "
        "without opening holdout."
    )

    return results


def main() -> None:
    """Run and print alternative feature diagnostics."""

    (
        summary_results,
        fold_results,
        coefficient_results,
        _,
    ) = run_totals_scoring_window_diagnostics()

    print("\nPAIRED SCORING-WINDOW SUMMARY\n")

    print(
        summary_results.to_string(
            index=False
        )
    )

    print("\nSEASON-LEVEL COMPARISON\n")

    print(
        fold_results.to_string(
            index=False
        )
    )

    print("\nSTANDARDIZED COEFFICIENTS\n")

    print(
        coefficient_results.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()