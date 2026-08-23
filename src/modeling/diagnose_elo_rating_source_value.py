"""
NFL Analytics Platform
Elo Rating Source Paired Diagnostics

Purpose:
    Compare selected Elo and QB candidates on identical
    chronological validation games using paired Brier-loss
    and spread absolute-error deltas.

    Negative challenger deltas mean improvement.

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

from src.modeling.backtest_elo_rating_sources import (
    BACKTEST_VALIDATION_SEASONS,
    LOGISTIC_REGULARIZATION_C,
    PUBLISHED_NFELO_CANDIDATE,
    SPREAD_RIDGE_ALPHA,
    SPREAD_TARGET_COLUMN,
    TRAINED_CANDIDATES,
    load_rating_source_backtest_data,
    prepare_common_backtest_sample,
    validate_backtest_seasons,
    validate_source_tables,
)
from src.modeling.evaluate_spread_model_candidates import (
    create_ridge_pipeline,
)
from src.modeling.train_logistic_baseline import (
    DATABASE_FILE,
    TARGET_COLUMN,
    create_logistic_pipeline,
    validate_database_file,
)


logger = logging.getLogger(__name__)

DEFAULT_BOOTSTRAP_ITERATIONS = 10_000
DEFAULT_RANDOM_SEED = 42

COMPARISONS = {
    "external_nfelo_vs_internal_elo": (
        "internal_elo",
        "external_nfelo",
    ),
    "external_both_qb_vs_external_nfelo": (
        "external_nfelo",
        "external_nfelo_both_qb",
    ),
    "both_qb_vs_external_qb_only": (
        "external_nfelo_external_qb",
        "external_nfelo_both_qb",
    ),
    "published_probability_vs_external_both_qb": (
        "external_nfelo_both_qb",
        PUBLISHED_NFELO_CANDIDATE,
    ),
}

PREDICTION_COLUMNS = (
    "candidate_name",
    "validation_season",
    "game_id",
    "actual_home_win",
    "home_win_probability",
    "brier_loss",
    "actual_home_margin",
    "predicted_home_margin",
    "spread_absolute_error",
)

SUMMARY_COLUMNS = (
    "comparison_name",
    "base_candidate",
    "challenger_candidate",
    "fold_count",
    "validation_game_count",
    "base_brier_score",
    "challenger_brier_score",
    "brier_score_delta",
    "challenger_brier_win_rate",
    "brier_bootstrap_95_percent_lower",
    "brier_bootstrap_95_percent_upper",
    "base_spread_mae",
    "challenger_spread_mae",
    "spread_mae_delta",
    "challenger_spread_win_rate",
    "spread_bootstrap_95_percent_lower",
    "spread_bootstrap_95_percent_upper",
)

FOLD_RESULT_COLUMNS = (
    "comparison_name",
    "validation_season",
    "validation_game_count",
    "base_brier_score",
    "challenger_brier_score",
    "brier_score_delta",
    "challenger_brier_win_rate",
    "base_spread_mae",
    "challenger_spread_mae",
    "spread_mae_delta",
    "challenger_spread_win_rate",
)


def bootstrap_paired_mean_delta(
    paired_deltas: np.ndarray,
    iteration_count: int = (
        DEFAULT_BOOTSTRAP_ITERATIONS
    ),
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[float, float]:
    """Return a paired bootstrap 95 percent interval."""

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

    return (
        float(
            np.percentile(
                bootstrap_means,
                2.5,
            )
        ),
        float(
            np.percentile(
                bootstrap_means,
                97.5,
            )
        ),
    )


def create_oof_candidate_predictions(
    development_data: pd.DataFrame,
    validation_seasons: tuple[
        int, ...
    ] = BACKTEST_VALIDATION_SEASONS,
) -> pd.DataFrame:
    """Create expanding-window out-of-fold predictions."""

    sample = prepare_common_backtest_sample(
        development_data
    )

    validate_backtest_seasons(
        sample=sample,
        validation_seasons=validation_seasons,
    )

    prediction_rows: list[
        dict[str, object]
    ] = []

    for validation_season in validation_seasons:
        training_data = sample.loc[
            sample["season"] < validation_season
        ].copy()

        validation_data = sample.loc[
            sample["season"] == validation_season
        ].copy()

        if training_data.empty:
            raise RuntimeError(
                "No Elo diagnostic training games "
                f"precede {validation_season}."
            )

        if validation_data.empty:
            raise RuntimeError(
                "No Elo diagnostic validation games "
                f"exist for {validation_season}."
            )

        training_target = training_data[
            TARGET_COLUMN
        ]

        if training_target.nunique() != 2:
            raise RuntimeError(
                "Elo diagnostic training data must "
                "contain both target classes."
            )

        actual_home_win = validation_data[
            TARGET_COLUMN
        ].to_numpy(dtype=int)

        actual_home_margin = validation_data[
            SPREAD_TARGET_COLUMN
        ].to_numpy(dtype=float)

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

            home_win_probabilities = (
                probability_model.predict_proba(
                    validation_data.loc[
                        :,
                        feature_columns,
                    ]
                )[:, 1]
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

            predicted_home_margin = (
                spread_model.predict(
                    validation_data.loc[
                        :,
                        feature_columns,
                    ]
                )
            )

            brier_losses = np.square(
                home_win_probabilities
                - actual_home_win
            )

            spread_absolute_errors = np.abs(
                predicted_home_margin
                - actual_home_margin
            )

            for row_index, game_id in enumerate(
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
                            actual_home_win[
                                row_index
                            ]
                        ),
                        "home_win_probability": float(
                            home_win_probabilities[
                                row_index
                            ]
                        ),
                        "brier_loss": float(
                            brier_losses[
                                row_index
                            ]
                        ),
                        "actual_home_margin": float(
                            actual_home_margin[
                                row_index
                            ]
                        ),
                        "predicted_home_margin": float(
                            predicted_home_margin[
                                row_index
                            ]
                        ),
                        "spread_absolute_error": float(
                            spread_absolute_errors[
                                row_index
                            ]
                        ),
                    }
                )

        published_probabilities = validation_data[
            "published_nfelo_home_probability"
        ].to_numpy(dtype=float)

        published_brier_losses = np.square(
            published_probabilities
            - actual_home_win
        )

        for row_index, game_id in enumerate(
            validation_data["game_id"]
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
                        actual_home_win[
                            row_index
                        ]
                    ),
                    "home_win_probability": float(
                        published_probabilities[
                            row_index
                        ]
                    ),
                    "brier_loss": float(
                        published_brier_losses[
                            row_index
                        ]
                    ),
                    "actual_home_margin": float(
                        actual_home_margin[
                            row_index
                        ]
                    ),
                    "predicted_home_margin": None,
                    "spread_absolute_error": None,
                }
            )

    predictions = pd.DataFrame(
        prediction_rows,
        columns=PREDICTION_COLUMNS,
    )

    duplicate_mask = predictions.duplicated(
        subset=[
            "candidate_name",
            "game_id",
        ]
    )

    if duplicate_mask.any():
        raise RuntimeError(
            "Elo diagnostic predictions contain "
            "duplicate candidate-game rows."
        )

    return predictions


def create_paired_comparison(
    predictions: pd.DataFrame,
    base_candidate: str,
    challenger_candidate: str,
) -> pd.DataFrame:
    """Join two candidates on identical validation games."""

    available_candidates = set(
        predictions["candidate_name"]
    )

    missing_candidates = {
        base_candidate,
        challenger_candidate,
    } - available_candidates

    if missing_candidates:
        raise ValueError(
            "Elo diagnostic candidates are missing: "
            + ", ".join(
                sorted(missing_candidates)
            )
        )

    base = predictions.loc[
        predictions["candidate_name"]
        == base_candidate,
        [
            "game_id",
            "validation_season",
            "actual_home_win",
            "actual_home_margin",
            "brier_loss",
            "spread_absolute_error",
        ],
    ].rename(
        columns={
            "brier_loss": "base_brier_loss",
            "spread_absolute_error": (
                "base_spread_absolute_error"
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
            "actual_home_margin",
            "brier_loss",
            "spread_absolute_error",
        ],
    ].rename(
        columns={
            "brier_loss": (
                "challenger_brier_loss"
            ),
            "spread_absolute_error": (
                "challenger_spread_absolute_error"
            ),
        }
    )

    paired = base.merge(
        challenger,
        on=[
            "game_id",
            "validation_season",
            "actual_home_win",
            "actual_home_margin",
        ],
        how="inner",
        validate="one_to_one",
    )

    if len(paired) != len(base):
        raise RuntimeError(
            "Elo diagnostic candidates do not cover "
            "identical validation games."
        )

    paired["brier_loss_delta"] = (
        paired["challenger_brier_loss"]
        - paired["base_brier_loss"]
    )

    paired["spread_absolute_error_delta"] = (
        paired[
            "challenger_spread_absolute_error"
        ]
        - paired[
            "base_spread_absolute_error"
        ]
    )

    return paired


def summarize_paired_comparison(
    comparison_name: str,
    base_candidate: str,
    challenger_candidate: str,
    paired: pd.DataFrame,
    bootstrap_iterations: int,
    random_seed: int,
) -> dict[str, object]:
    """Summarize one paired candidate comparison."""

    brier_deltas = paired[
        "brier_loss_delta"
    ].to_numpy(dtype=float)

    (
        brier_lower,
        brier_upper,
    ) = bootstrap_paired_mean_delta(
        paired_deltas=brier_deltas,
        iteration_count=bootstrap_iterations,
        random_seed=random_seed,
    )

    complete_spread = paired.loc[
        paired[
            "base_spread_absolute_error"
        ].notna()
        & paired[
            "challenger_spread_absolute_error"
        ].notna()
    ]

    if complete_spread.empty:
        base_spread_mae = None
        challenger_spread_mae = None
        spread_mae_delta = None
        challenger_spread_win_rate = None
        spread_lower = None
        spread_upper = None
    else:
        spread_deltas = complete_spread[
            "spread_absolute_error_delta"
        ].to_numpy(dtype=float)

        (
            spread_lower,
            spread_upper,
        ) = bootstrap_paired_mean_delta(
            paired_deltas=spread_deltas,
            iteration_count=bootstrap_iterations,
            random_seed=random_seed + 1,
        )

        base_spread_mae = float(
            complete_spread[
                "base_spread_absolute_error"
            ].mean()
        )

        challenger_spread_mae = float(
            complete_spread[
                "challenger_spread_absolute_error"
            ].mean()
        )

        spread_mae_delta = float(
            spread_deltas.mean()
        )

        challenger_spread_win_rate = float(
            np.mean(spread_deltas < 0.0)
        )

    return {
        "comparison_name": comparison_name,
        "base_candidate": base_candidate,
        "challenger_candidate": (
            challenger_candidate
        ),
        "fold_count": int(
            paired["validation_season"].nunique()
        ),
        "validation_game_count": len(paired),
        "base_brier_score": float(
            paired["base_brier_loss"].mean()
        ),
        "challenger_brier_score": float(
            paired[
                "challenger_brier_loss"
            ].mean()
        ),
        "brier_score_delta": float(
            brier_deltas.mean()
        ),
        "challenger_brier_win_rate": float(
            np.mean(brier_deltas < 0.0)
        ),
        "brier_bootstrap_95_percent_lower": (
            brier_lower
        ),
        "brier_bootstrap_95_percent_upper": (
            brier_upper
        ),
        "base_spread_mae": base_spread_mae,
        "challenger_spread_mae": (
            challenger_spread_mae
        ),
        "spread_mae_delta": spread_mae_delta,
        "challenger_spread_win_rate": (
            challenger_spread_win_rate
        ),
        "spread_bootstrap_95_percent_lower": (
            spread_lower
        ),
        "spread_bootstrap_95_percent_upper": (
            spread_upper
        ),
    }


def create_fold_rows(
    comparison_name: str,
    paired: pd.DataFrame,
) -> list[dict[str, object]]:
    """Create season-level paired comparison rows."""

    fold_rows: list[dict[str, object]] = []

    for (
        validation_season,
        season_data,
    ) in paired.groupby(
        "validation_season",
        sort=True,
    ):
        brier_deltas = season_data[
            "brier_loss_delta"
        ].to_numpy(dtype=float)

        complete_spread = season_data.loc[
            season_data[
                "base_spread_absolute_error"
            ].notna()
            & season_data[
                "challenger_spread_absolute_error"
            ].notna()
        ]

        if complete_spread.empty:
            base_spread_mae = None
            challenger_spread_mae = None
            spread_mae_delta = None
            challenger_spread_win_rate = None
        else:
            spread_deltas = complete_spread[
                "spread_absolute_error_delta"
            ].to_numpy(dtype=float)

            base_spread_mae = float(
                complete_spread[
                    "base_spread_absolute_error"
                ].mean()
            )

            challenger_spread_mae = float(
                complete_spread[
                    "challenger_spread_absolute_error"
                ].mean()
            )

            spread_mae_delta = float(
                spread_deltas.mean()
            )

            challenger_spread_win_rate = float(
                np.mean(spread_deltas < 0.0)
            )

        fold_rows.append(
            {
                "comparison_name": (
                    comparison_name
                ),
                "validation_season": int(
                    validation_season
                ),
                "validation_game_count": len(
                    season_data
                ),
                "base_brier_score": float(
                    season_data[
                        "base_brier_loss"
                    ].mean()
                ),
                "challenger_brier_score": float(
                    season_data[
                        "challenger_brier_loss"
                    ].mean()
                ),
                "brier_score_delta": float(
                    brier_deltas.mean()
                ),
                "challenger_brier_win_rate": float(
                    np.mean(
                        brier_deltas < 0.0
                    )
                ),
                "base_spread_mae": (
                    base_spread_mae
                ),
                "challenger_spread_mae": (
                    challenger_spread_mae
                ),
                "spread_mae_delta": (
                    spread_mae_delta
                ),
                "challenger_spread_win_rate": (
                    challenger_spread_win_rate
                ),
            }
        )

    return fold_rows


def diagnose_elo_rating_source_value(
    development_data: pd.DataFrame,
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
]:
    """Run paired Elo and QB candidate diagnostics."""

    predictions = create_oof_candidate_predictions(
        development_data=development_data,
        validation_seasons=validation_seasons,
    )

    summary_rows: list[
        dict[str, object]
    ] = []

    fold_rows: list[
        dict[str, object]
    ] = []

    for comparison_index, (
        comparison_name,
        (
            base_candidate,
            challenger_candidate,
        ),
    ) in enumerate(COMPARISONS.items()):
        paired = create_paired_comparison(
            predictions=predictions,
            base_candidate=base_candidate,
            challenger_candidate=(
                challenger_candidate
            ),
        )

        summary_rows.append(
            summarize_paired_comparison(
                comparison_name=comparison_name,
                base_candidate=base_candidate,
                challenger_candidate=(
                    challenger_candidate
                ),
                paired=paired,
                bootstrap_iterations=(
                    bootstrap_iterations
                ),
                random_seed=(
                    random_seed
                    + comparison_index * 10
                ),
            )
        )

        fold_rows.extend(
            create_fold_rows(
                comparison_name=comparison_name,
                paired=paired,
            )
        )

    summary = pd.DataFrame(
        summary_rows,
        columns=SUMMARY_COLUMNS,
    )

    fold_results = pd.DataFrame(
        fold_rows,
        columns=FOLD_RESULT_COLUMNS,
    )

    return (
        summary,
        fold_results,
        predictions,
    )


def run_elo_rating_source_diagnostics(
    database_file: Path = DATABASE_FILE,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load development data and run diagnostics."""

    validate_database_file(database_file)

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        validate_source_tables(connection)

        development_data = (
            load_rating_source_backtest_data(
                connection
            )
        )

    results = diagnose_elo_rating_source_value(
        development_data=development_data
    )

    logger.info(
        "Paired Elo rating source diagnostics "
        "completed without opening holdout."
    )

    return results


def main() -> None:
    """Run and print paired Elo diagnostics."""

    (
        summary,
        fold_results,
        _,
    ) = run_elo_rating_source_diagnostics()

    print("\nPAIRED ELO SOURCE SUMMARY\n")

    print(
        summary.to_string(
            index=False
        )
    )

    print("\nSEASON-LEVEL PAIRED RESULTS\n")

    print(
        fold_results.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()