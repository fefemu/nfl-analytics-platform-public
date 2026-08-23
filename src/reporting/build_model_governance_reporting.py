"""
NFL Analytics Platform
Model Governance Reporting Builder

Purpose:
    Persist validated governance, season and blend
    scorecards for the Streamlit Data Science Lab.

The reporting tables contain model evaluation outputs,
not training data or postgame prediction features.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging
from pathlib import Path

import duckdb
import pandas as pd

from src.modeling.production_probability_model import (
    PRODUCTION_MODEL_DEPLOYMENT_STATUS,
    PRODUCTION_PROBABILITY_MODEL,
)
from src.modeling.run_elo_injury_blend_scorecard import (
    AUDIT_SEASON,
    AUDIT_SELECTION_SEASONS,
    create_elo_injury_oof_predictions,
    evaluate_blend_weights,
    evaluate_probability_models,
    select_best_blend_weight,
)
from src.modeling.run_model_governance_scorecard import (
    GOVERNANCE_VALIDATION_SEASONS,
    aggregate_governance_results,
    evaluate_governance_models,
    load_governance_data,
    validate_governance_dataset_columns,
)
from src.modeling.train_logistic_baseline import (
    DATABASE_FILE,
    validate_database_file,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


TARGET_SCHEMA = "analytics"

GOVERNANCE_SCORECARD_TABLE = (
    "model_governance_scorecard"
)

GOVERNANCE_SEASON_TABLE = (
    "model_governance_season_results"
)

BLEND_WEIGHT_TABLE = (
    "model_blend_weight_grid"
)

BLEND_SCORECARD_TABLE = (
    "model_blend_scorecard"
)

PRODUCTION_REGISTRY_TABLE = (
    "production_model_registry"
)


def build_reporting_frames(
    governance_data: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build all dashboard-ready reporting frames."""

    governance_season_results = (
        evaluate_governance_models(
            governance_data
        )
    )

    governance_scorecard = (
        aggregate_governance_results(
            governance_season_results
        )
    )

    predictions = (
        create_elo_injury_oof_predictions(
            governance_data
        )
    )

    audit_weight_results = (
        evaluate_blend_weights(
            predictions=predictions,
            selection_seasons=(
                AUDIT_SELECTION_SEASONS
            ),
        )
    ).assign(
        selection_scope="audit_2020_2024"
    )

    production_weight_results = (
        evaluate_blend_weights(
            predictions=predictions,
            selection_seasons=(
                GOVERNANCE_VALIDATION_SEASONS
            ),
        )
    ).assign(
        selection_scope="production_2020_2025"
    )

    audit_weight = select_best_blend_weight(
        audit_weight_results
    )

    production_weight = (
        PRODUCTION_PROBABILITY_MODEL
        .logistic_weight
    )

    selected_grid_weight = (
        select_best_blend_weight(
            production_weight_results
        )
    )

    if abs(
        production_weight
        - selected_grid_weight
    ) > 0.000000001:
        raise RuntimeError(
            "Production model weight does not match "
            "the governance blend grid."
        )

    blend_scorecard = pd.concat(
        [
            evaluate_probability_models(
                predictions=predictions,
                seasons=(
                    AUDIT_SELECTION_SEASONS
                ),
                injury_weight=audit_weight,
                evaluation_period=(
                    "selection_2020_2024"
                ),
            ),
            evaluate_probability_models(
                predictions=predictions,
                seasons=(
                    AUDIT_SEASON,
                ),
                injury_weight=audit_weight,
                evaluation_period=(
                    "historical_audit_2025"
                ),
            ),
            evaluate_probability_models(
                predictions=predictions,
                seasons=(
                    GOVERNANCE_VALIDATION_SEASONS
                ),
                injury_weight=production_weight,
                evaluation_period=(
                    "production_selection_2020_2025"
                ),
            ),
        ],
        ignore_index=True,
    )

    blend_weight_grid = pd.concat(
        [
            audit_weight_results,
            production_weight_results,
        ],
        ignore_index=True,
    )

    production_registry = pd.DataFrame(
        [
            {
                "model_name": (
                    PRODUCTION_PROBABILITY_MODEL
                    .model_name
                ),
                "model_version": (
                    PRODUCTION_PROBABILITY_MODEL
                    .model_version
                ),
                "deployment_status": (
                    PRODUCTION_MODEL_DEPLOYMENT_STATUS
                ),
                "logistic_component_name": (
                    PRODUCTION_PROBABILITY_MODEL
                    .logistic_component_name
                ),
                "logistic_feature_columns": (
                    ", ".join(
                        PRODUCTION_PROBABILITY_MODEL
                        .logistic_feature_columns
                    )
                ),
                "logistic_regularization_c": (
                    PRODUCTION_PROBABILITY_MODEL
                    .logistic_regularization_c
                ),
                "logistic_weight": (
                    PRODUCTION_PROBABILITY_MODEL
                    .logistic_weight
                ),
                "elo_weight": (
                    PRODUCTION_PROBABILITY_MODEL
                    .elo_weight
                ),
                "classification_threshold": (
                    PRODUCTION_PROBABILITY_MODEL
                    .classification_threshold
                ),
                "requires_complete_injury_data": (
                    PRODUCTION_PROBABILITY_MODEL
                    .requires_complete_injury_data
                ),
                "incomplete_injury_fallback_model": (
                    PRODUCTION_PROBABILITY_MODEL
                    .incomplete_injury_fallback_model
                ),
                "forward_test_season": (
                    PRODUCTION_PROBABILITY_MODEL
                    .forward_test_season
                ),
            }
        ]
    )

    return {
        GOVERNANCE_SCORECARD_TABLE: (
            governance_scorecard
        ),
        GOVERNANCE_SEASON_TABLE: (
            governance_season_results
        ),
        BLEND_WEIGHT_TABLE: (
            blend_weight_grid
        ),
        BLEND_SCORECARD_TABLE: (
            blend_scorecard
        ),
        PRODUCTION_REGISTRY_TABLE: (
            production_registry
        ),
    }


def persist_reporting_frames(
    connection: duckdb.DuckDBPyConnection,
    reporting_frames: dict[
        str,
        pd.DataFrame,
    ],
) -> None:
    """Persist reporting frames transactionally."""

    connection.execute(
        f"CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA}"
    )

    for (
        table_name,
        frame,
    ) in reporting_frames.items():
        if frame.empty:
            raise RuntimeError(
                f"Reporting frame is empty: {table_name}"
            )

        temporary_view_name = (
            f"_reporting_{table_name}"
        )

        connection.register(
            temporary_view_name,
            frame,
        )

        try:
            connection.execute(
                f"""
                CREATE OR REPLACE TABLE
                    {TARGET_SCHEMA}.{table_name}
                AS
                SELECT *
                FROM {temporary_view_name}
                """
            )

        finally:
            connection.unregister(
                temporary_view_name
            )


def validate_reporting_tables(
    connection: duckdb.DuckDBPyConnection,
) -> dict[str, int]:
    """Validate persisted reporting table counts."""

    expected_counts = {
        GOVERNANCE_SCORECARD_TABLE: 5,
        GOVERNANCE_SEASON_TABLE: 30,
        BLEND_WEIGHT_TABLE: 42,
        BLEND_SCORECARD_TABLE: 9,
        PRODUCTION_REGISTRY_TABLE: 1,
    }

    actual_counts: dict[
        str,
        int,
    ] = {}

    for (
        table_name,
        expected_count,
    ) in expected_counts.items():
        row_count = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM {TARGET_SCHEMA}.{table_name}
            """
        ).fetchone()[0]

        if row_count != expected_count:
            raise RuntimeError(
                f"Reporting table {table_name} has "
                f"{row_count} rows; expected "
                f"{expected_count}."
            )

        actual_counts[
            table_name
        ] = row_count

    invalid_registry_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {
            TARGET_SCHEMA
        }.{PRODUCTION_REGISTRY_TABLE}
        WHERE model_name IS NULL
           OR model_version IS NULL
           OR deployment_status IS NULL
           OR logistic_weight NOT BETWEEN 0.0 AND 1.0
           OR elo_weight NOT BETWEEN 0.0 AND 1.0
           OR ABS(
                logistic_weight
                + elo_weight
                - 1.0
           ) > 0.000000001
           OR forward_test_season != 2026
        """
    ).fetchone()[0]

    if invalid_registry_count > 0:
        raise RuntimeError(
            "Production model registry is inconsistent."
        )

    return actual_counts


def build_model_governance_reporting(
    database_file: Path = DATABASE_FILE,
) -> None:
    """Build all model-governance reporting tables."""

    validate_database_file(
        database_file
    )

    logger.info(
        "Starting model governance reporting build..."
    )

    with duckdb.connect(
        str(database_file)
    ) as connection:
        validate_governance_dataset_columns(
            connection
        )

        governance_data = load_governance_data(
            connection
        )

        reporting_frames = build_reporting_frames(
            governance_data
        )

        connection.execute(
            "BEGIN TRANSACTION"
        )

        try:
            persist_reporting_frames(
                connection=connection,
                reporting_frames=reporting_frames,
            )

            table_counts = (
                validate_reporting_tables(
                    connection
                )
            )

            connection.execute(
                "COMMIT"
            )

        except Exception:
            connection.execute(
                "ROLLBACK"
            )
            raise

    for (
        table_name,
        row_count,
    ) in table_counts.items():
        logger.info(
            "Reporting table validated: "
            "analytics.%s | rows=%s",
            table_name,
            row_count,
        )

    logger.info(
        "Model governance reporting "
        "completed successfully."
    )


def main() -> None:
    """Run the governance reporting builder."""

    try:
        build_model_governance_reporting()

    except Exception:
        logger.exception(
            "Model governance reporting failed."
        )
        raise


if __name__ == "__main__":
    main()