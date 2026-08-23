"""
NFL Analytics Platform
Totals Model Expanding-Window Backtest

Purpose:
    Compare totals feature sets and Ridge alpha values
    across multiple chronological validation seasons.

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

from src.modeling.evaluate_totals_model_candidates import (
    LEAGUE_SCORING_64_TOTALS_FEATURES,
    LEAGUE_SCORING_128_TOTALS_FEATURES,
    TOTALS_SELECTED_BASE_FEATURES,
    TOTALS_TARGET_COLUMN,
    TotalsModelCandidate,
    create_ridge_pipeline,
    load_totals_development_data,
    prepare_common_totals_sample,
)
from src.modeling.evaluate_spread_model_candidates import (
    calculate_regression_metrics,
)
from src.modeling.train_logistic_baseline import (
    DATABASE_FILE,
    validate_database_file,
)


logger = logging.getLogger(__name__)

BACKTEST_VALIDATION_SEASONS = (
    2021,
    2022,
    2023,
    2024,
)

RIDGE_ALPHA_GRID = (
    0.0,
    0.1,
    1.0,
    10.0,
    100.0,
    300.0,
    1000.0,
)

BACKTEST_FEATURE_SETS = (
    (
        "ridge_epa_weather_qb_league_64",
        (
            *TOTALS_SELECTED_BASE_FEATURES,
            *LEAGUE_SCORING_64_TOTALS_FEATURES,
        ),
    ),
    (
        "ridge_epa_weather_qb_league_128",
        (
            *TOTALS_SELECTED_BASE_FEATURES,
            *LEAGUE_SCORING_128_TOTALS_FEATURES,
        ),
    ),
)

FOLD_RESULT_COLUMNS = (
    "candidate_name",
    "feature_count",
    "ridge_alpha",
    "validation_season",
    "train_first_season",
    "train_last_season",
    "train_game_count",
    "validation_game_count",
    "validation_mae",
    "validation_rmse",
    "validation_bias",
    "validation_r_squared",
)

SUMMARY_RESULT_COLUMNS = (
    "candidate_name",
    "feature_count",
    "ridge_alpha",
    "fold_count",
    "validation_game_count",
    "pooled_validation_mae",
    "mean_fold_mae",
    "standard_deviation_fold_mae",
    "worst_fold_mae",
    "pooled_validation_rmse",
    "pooled_validation_bias",
    "pooled_validation_r_squared",
)


def create_backtest_candidates(
    alpha_grid: tuple[float, ...] = RIDGE_ALPHA_GRID,
) -> tuple[TotalsModelCandidate, ...]:
    """Create every feature-set and alpha combination."""

    if not alpha_grid:
        raise ValueError(
            "Ridge alpha grid must not be empty."
        )

    if any(alpha < 0.0 for alpha in alpha_grid):
        raise ValueError(
            "Ridge alpha values must not be negative."
        )

    if len(alpha_grid) != len(set(alpha_grid)):
        raise ValueError(
            "Ridge alpha values must be unique."
        )

    return tuple(
        TotalsModelCandidate(
            candidate_name=candidate_name,
            feature_columns=feature_columns,
            ridge_alpha=float(alpha),
        )
        for candidate_name, feature_columns
        in BACKTEST_FEATURE_SETS
        for alpha in alpha_grid
    )


def validate_backtest_seasons(
    sample: pd.DataFrame,
    validation_seasons: tuple[int, ...],
) -> None:
    """Validate chronological backtest seasons."""

    if not validation_seasons:
        raise ValueError(
            "At least one validation season is required."
        )

    if len(validation_seasons) != len(
        set(validation_seasons)
    ):
        raise ValueError(
            "Validation seasons must be unique."
        )

    if tuple(sorted(validation_seasons)) != (
        validation_seasons
    ):
        raise ValueError(
            "Validation seasons must be chronological."
        )

    available_seasons = set(
        sample["season"].astype(int)
    )

    missing_seasons = sorted(
        set(validation_seasons)
        - available_seasons
    )

    if missing_seasons:
        raise ValueError(
            "Backtest validation seasons are missing: "
            + ", ".join(
                str(season)
                for season in missing_seasons
            )
        )

    earliest_season = int(
        sample["season"].min()
    )

    if validation_seasons[0] <= earliest_season:
        raise ValueError(
            "Every validation season requires earlier "
            "training data."
        )


def evaluate_totals_expanding_window(
    development_data: pd.DataFrame,
    validation_seasons: tuple[
        int, ...
    ] = BACKTEST_VALIDATION_SEASONS,
    alpha_grid: tuple[
        float, ...
    ] = RIDGE_ALPHA_GRID,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run chronological totals candidate backtests."""

    sample = prepare_common_totals_sample(
        development_data
    )

    validate_backtest_seasons(
        sample=sample,
        validation_seasons=validation_seasons,
    )

    candidates = create_backtest_candidates(
        alpha_grid=alpha_grid
    )

    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

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

        validation_target = validation_data[
            TOTALS_TARGET_COLUMN
        ]

        constant_prediction = np.full(
            shape=len(validation_data),
            fill_value=float(
                train_target.mean()
            ),
        )

        constant_metrics = (
            calculate_regression_metrics(
                actual_margin=validation_target,
                predicted_margin=constant_prediction,
            )
        )

        fold_rows.append(
            {
                "candidate_name": (
                    "constant_train_mean"
                ),
                "feature_count": 0,
                "ridge_alpha": None,
                "validation_season": (
                    validation_season
                ),
                "train_first_season": int(
                    train_data["season"].min()
                ),
                "train_last_season": int(
                    train_data["season"].max()
                ),
                "train_game_count": len(
                    train_data
                ),
                "validation_game_count": len(
                    validation_data
                ),
                **constant_metrics,
            }
        )

        for game_id, actual, predicted in zip(
            validation_data["game_id"],
            validation_target,
            constant_prediction,
            strict=True,
        ):
            prediction_rows.append(
                {
                    "candidate_name": (
                        "constant_train_mean"
                    ),
                    "ridge_alpha": None,
                    "validation_season": (
                        validation_season
                    ),
                    "game_id": game_id,
                    "actual_margin": float(actual),
                    "predicted_margin": float(
                        predicted
                    ),
                }
            )

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

            predicted_margin = model.predict(
                validation_data.loc[
                    :,
                    feature_columns,
                ]
            )

            metrics = calculate_regression_metrics(
                actual_margin=validation_target,
                predicted_margin=predicted_margin,
            )

            fold_rows.append(
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
                    "validation_season": (
                        validation_season
                    ),
                    "train_first_season": int(
                        train_data["season"].min()
                    ),
                    "train_last_season": int(
                        train_data["season"].max()
                    ),
                    "train_game_count": len(
                        train_data
                    ),
                    "validation_game_count": len(
                        validation_data
                    ),
                    **metrics,
                }
            )

            for game_id, actual, predicted in zip(
                validation_data["game_id"],
                validation_target,
                predicted_margin,
                strict=True,
            ):
                prediction_rows.append(
                    {
                        "candidate_name": (
                            candidate.candidate_name
                        ),
                        "ridge_alpha": (
                            candidate.ridge_alpha
                        ),
                        "validation_season": (
                            validation_season
                        ),
                        "game_id": game_id,
                        "actual_margin": float(
                            actual
                        ),
                        "predicted_margin": float(
                            predicted
                        ),
                    }
                )

    fold_results = pd.DataFrame(
        fold_rows,
        columns=FOLD_RESULT_COLUMNS,
    )

    predictions = pd.DataFrame(
        prediction_rows
    )

    summary_rows: list[dict[str, object]] = []

    grouping_columns = [
        "candidate_name",
        "ridge_alpha",
    ]

    grouped_predictions = predictions.groupby(
        grouping_columns,
        dropna=False,
        sort=False,
    )

    for (
        candidate_name,
        ridge_alpha,
    ), candidate_predictions in grouped_predictions:
        fold_mask = (
            fold_results["candidate_name"]
            == candidate_name
        )

        if pd.isna(ridge_alpha):
            fold_mask &= fold_results[
                "ridge_alpha"
            ].isna()
        else:
            fold_mask &= (
                fold_results["ridge_alpha"]
                == ridge_alpha
            )

        candidate_folds = fold_results.loc[
            fold_mask
        ]

        pooled_metrics = (
            calculate_regression_metrics(
                actual_margin=candidate_predictions[
                    "actual_margin"
                ],
                predicted_margin=(
                    candidate_predictions[
                        "predicted_margin"
                    ].to_numpy()
                ),
            )
        )

        summary_rows.append(
            {
                "candidate_name": candidate_name,
                "feature_count": int(
                    candidate_folds[
                        "feature_count"
                    ].iloc[0]
                ),
                "ridge_alpha": (
                    None
                    if pd.isna(ridge_alpha)
                    else float(ridge_alpha)
                ),
                "fold_count": len(
                    candidate_folds
                ),
                "validation_game_count": len(
                    candidate_predictions
                ),
                "pooled_validation_mae": (
                    pooled_metrics[
                        "validation_mae"
                    ]
                ),
                "mean_fold_mae": float(
                    candidate_folds[
                        "validation_mae"
                    ].mean()
                ),
                "standard_deviation_fold_mae": (
                    float(
                        candidate_folds[
                            "validation_mae"
                        ].std(ddof=0)
                    )
                ),
                "worst_fold_mae": float(
                    candidate_folds[
                        "validation_mae"
                    ].max()
                ),
                "pooled_validation_rmse": (
                    pooled_metrics[
                        "validation_rmse"
                    ]
                ),
                "pooled_validation_bias": (
                    pooled_metrics[
                        "validation_bias"
                    ]
                ),
                "pooled_validation_r_squared": (
                    pooled_metrics[
                        "validation_r_squared"
                    ]
                ),
            }
        )

    summary_results = pd.DataFrame(
        summary_rows,
        columns=SUMMARY_RESULT_COLUMNS,
    ).sort_values(
        by=[
            "pooled_validation_mae",
            "pooled_validation_rmse",
            "candidate_name",
            "ridge_alpha",
        ],
        kind="stable",
    ).reset_index(drop=True)

    fold_results = fold_results.sort_values(
        by=[
            "validation_season",
            "validation_mae",
            "candidate_name",
            "ridge_alpha",
        ],
        kind="stable",
    ).reset_index(drop=True)

    return summary_results, fold_results


def run_totals_expanding_window_backtest(
    database_file: Path = DATABASE_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load DuckDB data and run the totals backtest."""

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

    summary_results, fold_results = (
        evaluate_totals_expanding_window(
            development_data
        )
    )

    logger.info(
        "Totals expanding-window backtest completed "
        "with %s folds and %s candidate settings.",
        len(BACKTEST_VALIDATION_SEASONS),
        len(summary_results),
    )

    return summary_results, fold_results


def main() -> None:
    """Run and print the totals backtest."""

    summary_results, fold_results = (
        run_totals_expanding_window_backtest()
    )

    print("\nBACKTEST SUMMARY\n")

    print(
        summary_results.to_string(
            index=False
        )
    )

    best_candidate_settings = (
        summary_results.loc[
            summary_results[
                "candidate_name"
            ]
            != "constant_train_mean"
        ]
        .groupby(
            "candidate_name",
            sort=False,
        )
        .head(1)
    )

    best_fold_results = fold_results.merge(
        best_candidate_settings[
            [
                "candidate_name",
                "ridge_alpha",
            ]
        ],
        on=[
            "candidate_name",
            "ridge_alpha",
        ],
        how="inner",
        validate="many_to_one",
    )

    print("\nBEST SETTING BY FEATURE SET\n")

    print(
        best_candidate_settings.to_string(
            index=False
        )
    )

    print("\nSEASON RESULTS FOR BEST SETTINGS\n")

    print(
        best_fold_results.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()