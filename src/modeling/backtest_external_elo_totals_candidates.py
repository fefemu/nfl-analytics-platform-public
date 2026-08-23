"""
NFL Analytics Platform
External Elo Totals Candidate Backtest

Purpose:
    Test external nfelo rating and QB aggregates in the
    locked primary and fallback Totals model structures.

    Primary and fallback candidates are evaluated on
    separate identical chronological samples.

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

from src.modeling.backtest_totals_model_candidates import (
    BACKTEST_VALIDATION_SEASONS,
)
from src.modeling.evaluate_spread_model_candidates import (
    calculate_regression_metrics,
    create_ridge_pipeline,
)
from src.modeling.evaluate_totals_fallback_candidates import (
    load_totals_fallback_development_data,
)
from src.modeling.evaluate_totals_model_candidates import (
    TOTALS_TARGET_COLUMN,
    load_totals_development_data,
)
from src.modeling.production_totals_model import (
    PRODUCTION_TOTALS_MODEL,
)
from src.modeling.train_logistic_baseline import (
    DATABASE_FILE,
    validate_database_file,
)


logger = logging.getLogger(__name__)

EXTERNAL_RATINGS_FULL_NAME = (
    "processed.external_nfelo_game_ratings"
)

EXTERNAL_ELO_SUM_FEATURE = (
    "external_nfelo_rating_sum"
)

EXTERNAL_QB_SUM_FEATURE = (
    "external_nfelo_qb_adjustment_sum"
)

ROUTING_PRIMARY = "PRIMARY"
ROUTING_FALLBACK = "FALLBACK"

FOLD_RESULT_COLUMNS = (
    "routing_layer",
    "candidate_name",
    "feature_count",
    "ridge_alpha",
    "validation_season",
    "training_game_count",
    "validation_game_count",
    "validation_mae",
    "validation_rmse",
    "validation_bias",
    "validation_r_squared",
)

SUMMARY_COLUMNS = (
    "routing_layer",
    "candidate_name",
    "feature_count",
    "ridge_alpha",
    "fold_count",
    "validation_game_count",
    "validation_mae",
    "validation_rmse",
    "validation_bias",
    "validation_r_squared",
)


@dataclass(frozen=True)
class ExternalEloTotalsCandidate:
    """Describe one locked-alpha Totals candidate."""

    routing_layer: str
    candidate_name: str
    feature_columns: tuple[str, ...]
    ridge_alpha: float


PRIMARY_BASE_FEATURES = tuple(
    PRODUCTION_TOTALS_MODEL.feature_columns
)

FALLBACK_BASE_FEATURES = tuple(
    PRODUCTION_TOTALS_MODEL.fallback_feature_columns
)

FALLBACK_WITH_EXTERNAL_ELO_FEATURES = (
    "league_average_total_last_64",
    "is_indoor",
    EXTERNAL_ELO_SUM_FEATURE,
)

CANDIDATES = (
    ExternalEloTotalsCandidate(
        routing_layer=ROUTING_PRIMARY,
        candidate_name="primary_current_locked",
        feature_columns=PRIMARY_BASE_FEATURES,
        ridge_alpha=(
            PRODUCTION_TOTALS_MODEL.ridge_alpha
        ),
    ),
    ExternalEloTotalsCandidate(
        routing_layer=ROUTING_PRIMARY,
        candidate_name="primary_external_elo_sum",
        feature_columns=(
            *PRIMARY_BASE_FEATURES,
            EXTERNAL_ELO_SUM_FEATURE,
        ),
        ridge_alpha=(
            PRODUCTION_TOTALS_MODEL.ridge_alpha
        ),
    ),
    ExternalEloTotalsCandidate(
        routing_layer=ROUTING_PRIMARY,
        candidate_name="primary_external_qb_sum",
        feature_columns=(
            *PRIMARY_BASE_FEATURES,
            EXTERNAL_QB_SUM_FEATURE,
        ),
        ridge_alpha=(
            PRODUCTION_TOTALS_MODEL.ridge_alpha
        ),
    ),
    ExternalEloTotalsCandidate(
        routing_layer=ROUTING_PRIMARY,
        candidate_name=(
            "primary_external_elo_and_qb_sum"
        ),
        feature_columns=(
            *PRIMARY_BASE_FEATURES,
            EXTERNAL_ELO_SUM_FEATURE,
            EXTERNAL_QB_SUM_FEATURE,
        ),
        ridge_alpha=(
            PRODUCTION_TOTALS_MODEL.ridge_alpha
        ),
    ),
    ExternalEloTotalsCandidate(
        routing_layer=ROUTING_FALLBACK,
        candidate_name="fallback_current_locked",
        feature_columns=FALLBACK_BASE_FEATURES,
        ridge_alpha=(
            PRODUCTION_TOTALS_MODEL
            .fallback_ridge_alpha
        ),
    ),
    ExternalEloTotalsCandidate(
        routing_layer=ROUTING_FALLBACK,
        candidate_name=(
            "fallback_internal_elo_external_qb"
        ),
        feature_columns=(
            *FALLBACK_BASE_FEATURES,
            EXTERNAL_QB_SUM_FEATURE,
        ),
        ridge_alpha=(
            PRODUCTION_TOTALS_MODEL
            .fallback_ridge_alpha
        ),
    ),
    ExternalEloTotalsCandidate(
        routing_layer=ROUTING_FALLBACK,
        candidate_name="fallback_external_qb_no_elo",
        feature_columns=(
            "league_average_total_last_64",
            "is_indoor",
            EXTERNAL_QB_SUM_FEATURE,
        ),
        ridge_alpha=(
            PRODUCTION_TOTALS_MODEL
            .fallback_ridge_alpha
        ),
    ),
    ExternalEloTotalsCandidate(
        routing_layer=ROUTING_FALLBACK,
        candidate_name="fallback_external_elo_sum",
        feature_columns=(
            FALLBACK_WITH_EXTERNAL_ELO_FEATURES
        ),
        ridge_alpha=(
            PRODUCTION_TOTALS_MODEL
            .fallback_ridge_alpha
        ),
    ),
    ExternalEloTotalsCandidate(
        routing_layer=ROUTING_FALLBACK,
        candidate_name=(
            "fallback_external_elo_and_qb_sum"
        ),
        feature_columns=(
            *FALLBACK_WITH_EXTERNAL_ELO_FEATURES,
            EXTERNAL_QB_SUM_FEATURE,
        ),
        ridge_alpha=(
            PRODUCTION_TOTALS_MODEL
            .fallback_ridge_alpha
        ),
    ),
)


def validate_external_source(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate the external nfelo source table."""

    table_exists = connection.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = 'processed'
          AND table_name
              = 'external_nfelo_game_ratings'
        """
    ).fetchone()[0]

    if table_exists != 1:
        raise RuntimeError(
            "Missing external nfelo source table: "
            + EXTERNAL_RATINGS_FULL_NAME
        )


def load_external_totals_features(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Load team-order-invariant external Totals features."""

    external_features = connection.execute(
        f"""
        SELECT
            normalized_game_id AS game_id,

            starting_nfelo_home
                + starting_nfelo_away
                AS {EXTERNAL_ELO_SUM_FEATURE},

            home_538_qb_adj
                + away_538_qb_adj
                AS {EXTERNAL_QB_SUM_FEATURE}

        FROM {EXTERNAL_RATINGS_FULL_NAME}

        ORDER BY normalized_game_id
        """
    ).fetchdf()

    if external_features.empty:
        raise RuntimeError(
            "No external nfelo Totals features "
            "are available."
        )

    if external_features[
        "game_id"
    ].duplicated().any():
        raise RuntimeError(
            "External nfelo Totals features contain "
            "duplicate game identifiers."
        )

    return external_features


def add_external_totals_features(
    development_data: pd.DataFrame,
    external_features: pd.DataFrame,
) -> pd.DataFrame:
    """Join external features to development games."""

    required_development_columns = {
        "game_id",
        "season",
        "split_name",
        TOTALS_TARGET_COLUMN,
    }

    missing_development_columns = sorted(
        required_development_columns
        - set(development_data.columns)
    )

    if missing_development_columns:
        raise ValueError(
            "Totals development data is missing "
            "columns: "
            + ", ".join(
                missing_development_columns
            )
        )

    required_external_columns = {
        "game_id",
        EXTERNAL_ELO_SUM_FEATURE,
        EXTERNAL_QB_SUM_FEATURE,
    }

    missing_external_columns = sorted(
        required_external_columns
        - set(external_features.columns)
    )

    if missing_external_columns:
        raise ValueError(
            "External Totals data is missing columns: "
            + ", ".join(
                missing_external_columns
            )
        )

    if development_data[
        "game_id"
    ].duplicated().any():
        raise ValueError(
            "Totals development data contains "
            "duplicate game identifiers."
        )

    if external_features[
        "game_id"
    ].duplicated().any():
        raise ValueError(
            "External Totals data contains duplicate "
            "game identifiers."
        )

    merged = development_data.merge(
        external_features,
        on="game_id",
        how="left",
        validate="one_to_one",
    )

    missing_external_count = int(
        merged[
            EXTERNAL_ELO_SUM_FEATURE
        ].isna().sum()
    )

    if missing_external_count:
        raise RuntimeError(
            "External nfelo Totals coverage is "
            f"missing for {missing_external_count} "
            "development games."
        )

    return merged


def prepare_routing_sample(
    development_data: pd.DataFrame,
    candidates: tuple[
        ExternalEloTotalsCandidate, ...
    ],
) -> pd.DataFrame:
    """Create one identical complete routing sample."""

    if not candidates:
        raise ValueError(
            "At least one Totals candidate is required."
        )

    routing_layers = {
        candidate.routing_layer
        for candidate in candidates
    }

    if len(routing_layers) != 1:
        raise ValueError(
            "A routing sample must contain candidates "
            "from exactly one routing layer."
        )

    candidate_names = [
        candidate.candidate_name
        for candidate in candidates
    ]

    if len(candidate_names) != len(
        set(candidate_names)
    ):
        raise ValueError(
            "Totals candidate names must be unique."
        )

    required_columns = {
        "game_id",
        "season",
        "split_name",
        TOTALS_TARGET_COLUMN,
    }

    for candidate in candidates:
        required_columns.update(
            candidate.feature_columns
        )

    missing_columns = sorted(
        required_columns
        - set(development_data.columns)
    )

    if missing_columns:
        raise ValueError(
            "External Elo Totals data is missing "
            "columns: "
            + ", ".join(missing_columns)
        )

    if development_data[
        "game_id"
    ].duplicated().any():
        raise ValueError(
            "External Elo Totals data contains "
            "duplicate game identifiers."
        )

    unexpected_splits = sorted(
        set(
            development_data[
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
            "External Elo Totals backtest must not "
            "contain holdout or unknown splits: "
            + ", ".join(unexpected_splits)
        )

    if int(
        development_data["season"].max()
    ) >= 2025:
        raise ValueError(
            "External Elo Totals backtest must end "
            "before the 2025 holdout season."
        )

    complete_columns = sorted(
        required_columns
        - {
            "game_id",
            "split_name",
        }
    )

    sample = development_data.loc[
        development_data[
            complete_columns
        ].notna().all(axis=1)
    ].copy()

    if sample.empty:
        raise RuntimeError(
            "No complete external Elo Totals games "
            "are available."
        )

    return sample


def validate_validation_seasons(
    sample: pd.DataFrame,
    validation_seasons: tuple[int, ...],
) -> None:
    """Validate chronological validation folds."""

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
            "Totals validation seasons are missing: "
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


def evaluate_routing_candidates(
    development_data: pd.DataFrame,
    candidates: tuple[
        ExternalEloTotalsCandidate, ...
    ],
    validation_seasons: tuple[
        int, ...
    ] = BACKTEST_VALIDATION_SEASONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one routing layer chronologically."""

    sample = prepare_routing_sample(
        development_data=development_data,
        candidates=candidates,
    )

    validate_validation_seasons(
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
            sample["season"] < validation_season
        ].copy()

        validation_data = sample.loc[
            sample["season"] == validation_season
        ].copy()

        if training_data.empty:
            raise RuntimeError(
                "No Totals training games precede "
                f"{validation_season}."
            )

        if validation_data.empty:
            raise RuntimeError(
                "No Totals validation games exist "
                f"for {validation_season}."
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

            predictions = model.predict(
                validation_data.loc[
                    :,
                    candidate.feature_columns,
                ]
            )

            metrics = calculate_regression_metrics(
                actual_margin=validation_data[
                    TOTALS_TARGET_COLUMN
                ],
                predicted_margin=predictions,
            )

            fold_rows.append(
                {
                    "routing_layer": (
                        candidate.routing_layer
                    ),
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
                    "training_game_count": len(
                        training_data
                    ),
                    "validation_game_count": len(
                        validation_data
                    ),
                    "validation_mae": metrics[
                        "validation_mae"
                    ],
                    "validation_rmse": metrics[
                        "validation_rmse"
                    ],
                    "validation_bias": metrics[
                        "validation_bias"
                    ],
                    "validation_r_squared": metrics[
                        "validation_r_squared"
                    ],
                }
            )

            for (
                game_id,
                actual_total,
                predicted_total,
            ) in zip(
                validation_data["game_id"],
                actual_totals,
                predictions,
                strict=True,
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
                            actual_total
                        ),
                        "predicted_total": float(
                            predicted_total
                        ),
                    }
                )

    fold_results = pd.DataFrame(
        fold_rows,
        columns=FOLD_RESULT_COLUMNS,
    )

    prediction_results = pd.DataFrame(
        prediction_rows
    )

    summary_rows: list[
        dict[str, object]
    ] = []

    grouped_predictions = (
        prediction_results.groupby(
            [
                "routing_layer",
                "candidate_name",
            ],
            sort=False,
        )
    )

    for (
        (
            routing_layer,
            candidate_name,
        ),
        candidate_predictions,
    ) in grouped_predictions:
        candidate = next(
            candidate
            for candidate in candidates
            if candidate.candidate_name
            == candidate_name
        )

        metrics = calculate_regression_metrics(
            actual_margin=(
                candidate_predictions[
                    "actual_total"
                ]
            ),
            predicted_margin=(
                candidate_predictions[
                    "predicted_total"
                ].to_numpy(dtype=float)
            ),
        )

        summary_rows.append(
            {
                "routing_layer": routing_layer,
                "candidate_name": candidate_name,
                "feature_count": len(
                    candidate.feature_columns
                ),
                "ridge_alpha": (
                    candidate.ridge_alpha
                ),
                "fold_count": int(
                    candidate_predictions[
                        "validation_season"
                    ].nunique()
                ),
                "validation_game_count": len(
                    candidate_predictions
                ),
                "validation_mae": metrics[
                    "validation_mae"
                ],
                "validation_rmse": metrics[
                    "validation_rmse"
                ],
                "validation_bias": metrics[
                    "validation_bias"
                ],
                "validation_r_squared": metrics[
                    "validation_r_squared"
                ],
            }
        )

    summary = pd.DataFrame(
        summary_rows,
        columns=SUMMARY_COLUMNS,
    ).sort_values(
        by=[
            "validation_mae",
            "validation_rmse",
            "candidate_name",
        ],
        kind="stable",
    ).reset_index(drop=True)

    return summary, fold_results


def evaluate_external_elo_totals_candidates(
    primary_data: pd.DataFrame,
    fallback_data: pd.DataFrame,
    validation_seasons: tuple[
        int, ...
    ] = BACKTEST_VALIDATION_SEASONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate primary and fallback external Elo panels."""

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

    (
        primary_summary,
        primary_folds,
    ) = evaluate_routing_candidates(
        development_data=primary_data,
        candidates=primary_candidates,
        validation_seasons=validation_seasons,
    )

    (
        fallback_summary,
        fallback_folds,
    ) = evaluate_routing_candidates(
        development_data=fallback_data,
        candidates=fallback_candidates,
        validation_seasons=validation_seasons,
    )

    summary = pd.concat(
        [
            primary_summary,
            fallback_summary,
        ],
        ignore_index=True,
    ).sort_values(
        by=[
            "routing_layer",
            "validation_mae",
            "candidate_name",
        ],
        kind="stable",
    ).reset_index(drop=True)

    fold_results = pd.concat(
        [
            primary_folds,
            fallback_folds,
        ],
        ignore_index=True,
    ).sort_values(
        by=[
            "routing_layer",
            "validation_season",
            "validation_mae",
            "candidate_name",
        ],
        kind="stable",
    ).reset_index(drop=True)

    return summary, fold_results


def run_external_elo_totals_backtest(
    database_file: Path = DATABASE_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load development data and run Totals backtest."""

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

    summary, fold_results = (
        evaluate_external_elo_totals_candidates(
            primary_data=primary_data,
            fallback_data=fallback_data,
        )
    )

    logger.info(
        "External Elo Totals backtest completed "
        "without opening holdout."
    )

    return summary, fold_results


def main() -> None:
    """Run and print external Elo Totals results."""

    summary, fold_results = (
        run_external_elo_totals_backtest()
    )

    print(
        "\nEXTERNAL ELO TOTALS BACKTEST SUMMARY\n"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nSEASON-LEVEL TOTALS RESULTS\n"
    )

    print(
        fold_results.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()