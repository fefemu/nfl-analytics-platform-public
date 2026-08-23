"""
NFL Analytics Platform
External Elo Totals Paired Diagnostics

Purpose:
    Measure the paired game-level value of external nfelo
    Elo and QB aggregates in the locked primary and
    fallback Totals routing layers.

    Negative challenger deltas mean lower absolute error.

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

from src.modeling.backtest_external_elo_totals_candidates import (
    CANDIDATES,
    ROUTING_FALLBACK,
    ROUTING_PRIMARY,
    ExternalEloTotalsCandidate,
    add_external_totals_features,
    load_external_totals_features,
    prepare_routing_sample,
    validate_external_source,
    validate_validation_seasons,
)
from src.modeling.backtest_totals_model_candidates import (
    BACKTEST_VALIDATION_SEASONS,
)
from src.modeling.evaluate_spread_model_candidates import (
    create_ridge_pipeline,
)
from src.modeling.evaluate_totals_fallback_candidates import (
    load_totals_fallback_development_data,
)
from src.modeling.evaluate_totals_model_candidates import (
    TOTALS_TARGET_COLUMN,
    load_totals_development_data,
)
from src.modeling.train_logistic_baseline import (
    DATABASE_FILE,
    validate_database_file,
)


logger = logging.getLogger(__name__)

DEFAULT_BOOTSTRAP_ITERATIONS = 10_000
DEFAULT_RANDOM_SEED = 42

COMPARISONS = {
    "primary_external_qb_vs_current": (
        ROUTING_PRIMARY,
        "primary_current_locked",
        "primary_external_qb_sum",
    ),
    "primary_external_elo_qb_vs_qb_only": (
        ROUTING_PRIMARY,
        "primary_external_qb_sum",
        "primary_external_elo_and_qb_sum",
    ),
    "fallback_external_elo_qb_vs_current": (
        ROUTING_FALLBACK,
        "fallback_current_locked",
        "fallback_external_elo_and_qb_sum",
    ),
    "fallback_external_vs_internal_elo_with_qb": (
        ROUTING_FALLBACK,
        "fallback_internal_elo_external_qb",
        "fallback_external_elo_and_qb_sum",
    ),
}

PREDICTION_COLUMNS = (
    "routing_layer",
    "candidate_name",
    "validation_season",
    "game_id",
    "actual_total",
    "predicted_total",
    "absolute_error",
)

SUMMARY_COLUMNS = (
    "comparison_name",
    "routing_layer",
    "base_candidate",
    "challenger_candidate",
    "fold_count",
    "validation_game_count",
    "base_mae",
    "challenger_mae",
    "mae_delta",
    "challenger_win_rate",
    "challenger_loss_rate",
    "equal_error_rate",
    "bootstrap_mean_delta",
    "bootstrap_95_percent_lower",
    "bootstrap_95_percent_upper",
)

FOLD_RESULT_COLUMNS = (
    "comparison_name",
    "routing_layer",
    "validation_season",
    "validation_game_count",
    "base_mae",
    "challenger_mae",
    "mae_delta",
    "challenger_win_rate",
)


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


def create_routing_oof_predictions(
    development_data: pd.DataFrame,
    candidates: tuple[
        ExternalEloTotalsCandidate, ...
    ],
    validation_seasons: tuple[
        int, ...
    ] = BACKTEST_VALIDATION_SEASONS,
) -> pd.DataFrame:
    """Create expanding-window Totals predictions."""

    sample = prepare_routing_sample(
        development_data=development_data,
        candidates=candidates,
    )

    validate_validation_seasons(
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
                "No Totals diagnostic training games "
                f"precede {validation_season}."
            )

        if validation_data.empty:
            raise RuntimeError(
                "No Totals diagnostic validation games "
                f"exist for {validation_season}."
            )

        actual_totals = validation_data[
            TOTALS_TARGET_COLUMN
        ].to_numpy(dtype=float)

        for candidate in candidates:
            model = create_ridge_pipeline(
                ridge_alpha=candidate.ridge_alpha
            )

            model.fit(
                training_data.loc[
                    :,
                    candidate.feature_columns,
                ],
                training_data[
                    TOTALS_TARGET_COLUMN
                ],
            )

            predicted_totals = model.predict(
                validation_data.loc[
                    :,
                    candidate.feature_columns,
                ]
            )

            absolute_errors = np.abs(
                predicted_totals
                - actual_totals
            )

            for row_index, game_id in enumerate(
                validation_data["game_id"]
            ):
                prediction_rows.append(
                    {
                        "routing_layer": (
                            candidate.routing_layer
                        ),
                        "candidate_name": (
                            candidate.candidate_name
                        ),
                        "validation_season": (
                            validation_season
                        ),
                        "game_id": game_id,
                        "actual_total": float(
                            actual_totals[
                                row_index
                            ]
                        ),
                        "predicted_total": float(
                            predicted_totals[
                                row_index
                            ]
                        ),
                        "absolute_error": float(
                            absolute_errors[
                                row_index
                            ]
                        ),
                    }
                )

    predictions = pd.DataFrame(
        prediction_rows,
        columns=PREDICTION_COLUMNS,
    )

    duplicate_mask = predictions.duplicated(
        subset=[
            "routing_layer",
            "candidate_name",
            "game_id",
        ]
    )

    if duplicate_mask.any():
        raise RuntimeError(
            "Totals diagnostic predictions contain "
            "duplicate candidate-game rows."
        )

    return predictions


def create_paired_comparison(
    predictions: pd.DataFrame,
    routing_layer: str,
    base_candidate: str,
    challenger_candidate: str,
) -> pd.DataFrame:
    """Join two candidates on identical games."""

    layer_predictions = predictions.loc[
        predictions["routing_layer"]
        == routing_layer
    ]

    available_candidates = set(
        layer_predictions["candidate_name"]
    )

    missing_candidates = {
        base_candidate,
        challenger_candidate,
    } - available_candidates

    if missing_candidates:
        raise ValueError(
            "Totals diagnostic candidates are missing: "
            + ", ".join(
                sorted(missing_candidates)
            )
        )

    base = layer_predictions.loc[
        layer_predictions["candidate_name"]
        == base_candidate,
        [
            "game_id",
            "validation_season",
            "actual_total",
            "predicted_total",
            "absolute_error",
        ],
    ].rename(
        columns={
            "predicted_total": (
                "base_predicted_total"
            ),
            "absolute_error": (
                "base_absolute_error"
            ),
        }
    )

    challenger = layer_predictions.loc[
        layer_predictions["candidate_name"]
        == challenger_candidate,
        [
            "game_id",
            "validation_season",
            "actual_total",
            "predicted_total",
            "absolute_error",
        ],
    ].rename(
        columns={
            "predicted_total": (
                "challenger_predicted_total"
            ),
            "absolute_error": (
                "challenger_absolute_error"
            ),
        }
    )

    paired = base.merge(
        challenger,
        on=[
            "game_id",
            "validation_season",
            "actual_total",
        ],
        how="inner",
        validate="one_to_one",
    )

    if len(paired) != len(base):
        raise RuntimeError(
            "Totals diagnostic candidates do not "
            "cover identical validation games."
        )

    paired["absolute_error_delta"] = (
        paired["challenger_absolute_error"]
        - paired["base_absolute_error"]
    )

    return paired


def create_summary_row(
    comparison_name: str,
    routing_layer: str,
    base_candidate: str,
    challenger_candidate: str,
    paired: pd.DataFrame,
    bootstrap_iterations: int,
    random_seed: int,
) -> dict[str, object]:
    """Summarize one paired Totals comparison."""

    paired_deltas = paired[
        "absolute_error_delta"
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

    game_count = len(paired)

    return {
        "comparison_name": comparison_name,
        "routing_layer": routing_layer,
        "base_candidate": base_candidate,
        "challenger_candidate": (
            challenger_candidate
        ),
        "fold_count": int(
            paired["validation_season"].nunique()
        ),
        "validation_game_count": game_count,
        "base_mae": float(
            paired["base_absolute_error"].mean()
        ),
        "challenger_mae": float(
            paired[
                "challenger_absolute_error"
            ].mean()
        ),
        "mae_delta": float(
            paired_deltas.mean()
        ),
        "challenger_win_rate": (
            wins / game_count
        ),
        "challenger_loss_rate": (
            losses / game_count
        ),
        "equal_error_rate": (
            equal_errors / game_count
        ),
        **bootstrap_results,
    }


def create_fold_rows(
    comparison_name: str,
    routing_layer: str,
    paired: pd.DataFrame,
) -> list[dict[str, object]]:
    """Create season-level paired Totals rows."""

    fold_rows: list[
        dict[str, object]
    ] = []

    for (
        validation_season,
        season_data,
    ) in paired.groupby(
        "validation_season",
        sort=True,
    ):
        paired_deltas = season_data[
            "absolute_error_delta"
        ].to_numpy(dtype=float)

        fold_rows.append(
            {
                "comparison_name": (
                    comparison_name
                ),
                "routing_layer": routing_layer,
                "validation_season": int(
                    validation_season
                ),
                "validation_game_count": len(
                    season_data
                ),
                "base_mae": float(
                    season_data[
                        "base_absolute_error"
                    ].mean()
                ),
                "challenger_mae": float(
                    season_data[
                        "challenger_absolute_error"
                    ].mean()
                ),
                "mae_delta": float(
                    paired_deltas.mean()
                ),
                "challenger_win_rate": float(
                    np.mean(
                        paired_deltas < 0.0
                    )
                ),
            }
        )

    return fold_rows


def diagnose_external_elo_totals_value(
    primary_data: pd.DataFrame,
    fallback_data: pd.DataFrame,
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
    """Run paired primary and fallback diagnostics."""

    primary_candidates = tuple(
        candidate
        for candidate in CANDIDATES
        if candidate.routing_layer
        == ROUTING_PRIMARY
    )

    fallback_candidates = tuple(
        candidate
        for candidate in CANDIDATES
        if candidate.routing_layer
        == ROUTING_FALLBACK
    )

    primary_predictions = (
        create_routing_oof_predictions(
            development_data=primary_data,
            candidates=primary_candidates,
            validation_seasons=validation_seasons,
        )
    )

    fallback_predictions = (
        create_routing_oof_predictions(
            development_data=fallback_data,
            candidates=fallback_candidates,
            validation_seasons=validation_seasons,
        )
    )

    predictions = pd.concat(
        [
            primary_predictions,
            fallback_predictions,
        ],
        ignore_index=True,
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
            routing_layer,
            base_candidate,
            challenger_candidate,
        ),
    ) in enumerate(COMPARISONS.items()):
        paired = create_paired_comparison(
            predictions=predictions,
            routing_layer=routing_layer,
            base_candidate=base_candidate,
            challenger_candidate=(
                challenger_candidate
            ),
        )

        summary_rows.append(
            create_summary_row(
                comparison_name=comparison_name,
                routing_layer=routing_layer,
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
                routing_layer=routing_layer,
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


def run_external_elo_totals_diagnostics(
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
        validate_external_source(connection)

        primary_data = (
            load_totals_development_data(
                connection
            )
        )

        fallback_data = (
            load_totals_fallback_development_data(
                connection
            )
        )

        external_features = (
            load_external_totals_features(
                connection
            )
        )

    primary_data = add_external_totals_features(
        development_data=primary_data,
        external_features=external_features,
    )

    fallback_data = add_external_totals_features(
        development_data=fallback_data,
        external_features=external_features,
    )

    results = diagnose_external_elo_totals_value(
        primary_data=primary_data,
        fallback_data=fallback_data,
    )

    logger.info(
        "External Elo Totals paired diagnostics "
        "completed without opening holdout."
    )

    return results


def main() -> None:
    """Run and print paired Totals diagnostics."""

    (
        summary,
        fold_results,
        _,
    ) = run_external_elo_totals_diagnostics()

    print(
        "\nPAIRED EXTERNAL ELO TOTALS SUMMARY\n"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nSEASON-LEVEL PAIRED TOTALS RESULTS\n"
    )

    print(
        fold_results.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()