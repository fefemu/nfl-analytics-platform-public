"""
NFL Analytics Platform
Current Spread Prediction Builder

Purpose:
    Train the frozen production spread models, score
    upcoming games and persist validated predictions in
    DuckDB.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging
from pathlib import Path

import duckdb
import pandas as pd

from src.modeling.build_current_game_predictions import (
    create_current_prediction_frame,
)
from src.modeling.current_production_data import (
    MODELING_DATASET_FULL_NAME,
    load_current_production_inputs,
    validate_current_production_sources,
)
from src.modeling.current_spread_predictions import (
    CURRENT_SPREAD_PREDICTION_COLUMNS,
    create_current_spread_prediction_frame,
)
from src.modeling.production_spread_model import (
    PRODUCTION_SPREAD_MODEL,
)
from src.modeling.train_logistic_baseline import (
    DATABASE_FILE,
    validate_database_file,
)
from src.processing.build_external_nfelo_game_ratings import (
    TARGET_FULL_NAME as EXTERNAL_NFELO_FULL_NAME,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

TARGET_SCHEMA = "analytics"
TARGET_TABLE = "current_game_spread_predictions"
TARGET_FULL_NAME = (
    f"{TARGET_SCHEMA}.{TARGET_TABLE}"
)


def load_production_spread_training_data(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Load historical spread model-training data."""

    historical_data = connection.execute(
        f"""
        SELECT
            dataset.game_id,
            dataset.season,
            {
                PRODUCTION_SPREAD_MODEL
                .target_column
            },
            external.starting_nfelo_home
                - external.starting_nfelo_away
                AS external_nfelo_rating_difference,
            external.home_538_qb_adj
                - external.away_538_qb_adj
                AS external_nfelo_qb_adjustment_difference

        FROM {MODELING_DATASET_FULL_NAME}
            AS dataset

        INNER JOIN {EXTERNAL_NFELO_FULL_NAME}
            AS external
            ON dataset.game_id
                = external.normalized_game_id

        WHERE dataset.season
            < {
                PRODUCTION_SPREAD_MODEL
                .forward_test_season
            }

        ORDER BY
            dataset.season,
            dataset.game_date,
            dataset.game_id
        """
    ).fetchdf()

    if historical_data.empty:
        raise RuntimeError(
            "Production spread training source "
            "is empty."
        )

    logger.info(
        "Production spread training data loaded: "
        "%s historical games.",
        len(historical_data),
    )

    return historical_data


def create_current_spread_predictions_table(
    connection: duckdb.DuckDBPyConnection,
    predictions: pd.DataFrame,
) -> None:
    """Create the persisted spread prediction table."""

    missing_columns = sorted(
        set(CURRENT_SPREAD_PREDICTION_COLUMNS)
        - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Current spread predictions are missing "
            "columns: "
            + ", ".join(missing_columns)
        )

    connection.execute(
        f"CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA}"
    )

    connection.execute(
        f"""
        CREATE OR REPLACE TABLE {TARGET_FULL_NAME} (
            game_id VARCHAR,
            season INTEGER,
            game_type VARCHAR,
            week INTEGER,
            gameday DATE,
            gametime VARCHAR,
            home_team VARCHAR,
            away_team VARCHAR,
            is_neutral BOOLEAN,
            model_name VARCHAR,
            model_version VARCHAR,
            prediction_mode VARCHAR,
            prediction_mode_reason VARCHAR,
            ridge_alpha DOUBLE,
            primary_training_game_count INTEGER,
            fallback_training_game_count INTEGER,
            external_nfelo_rating_difference DOUBLE,
            listed_qb_rating_difference DOUBLE,
            external_nfelo_qb_adjustment_difference DOUBLE,
            both_listed_qb_ratings_available BOOLEAN,
            predicted_home_margin DOUBLE,
            predicted_away_margin DOUBLE,
            predicted_winner VARCHAR,
            prediction_generated_at TIMESTAMP
        )
        """
    )

    if predictions.empty:
        return

    rows = [
        tuple(
            (
                None
                if pd.isna(row[column_name])
                else row[column_name]
            )
            for column_name
            in CURRENT_SPREAD_PREDICTION_COLUMNS
        )
        for row in predictions.to_dict(
            orient="records"
        )
    ]

    placeholders = ", ".join(
        "?"
        for _ in CURRENT_SPREAD_PREDICTION_COLUMNS
    )

    connection.executemany(
        f"""
        INSERT INTO {TARGET_FULL_NAME}
        VALUES ({placeholders})
        """,
        rows,
    )


def validate_current_spread_predictions_table(
    connection: duckdb.DuckDBPyConnection,
    expected_row_count: int,
) -> None:
    """Validate persisted production spread predictions."""

    actual_row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    if actual_row_count != expected_row_count:
        raise RuntimeError(
            "Current spread prediction row count does "
            f"not match: expected {expected_row_count}, "
            f"found {actual_row_count}."
        )

    duplicate_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT game_id
            FROM {TARGET_FULL_NAME}
            GROUP BY game_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_count > 0:
        raise RuntimeError(
            "Duplicate current spread predictions found."
        )

    invalid_margin_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE predicted_home_margin IS NULL
           OR predicted_away_margin IS NULL
           OR NOT isfinite(predicted_home_margin)
           OR NOT isfinite(predicted_away_margin)
           OR ABS(
                predicted_home_margin
                + predicted_away_margin
              ) > 0.000001
        """
    ).fetchone()[0]

    if invalid_margin_count > 0:
        raise RuntimeError(
            "Invalid current spread prediction "
            "margins found."
        )

    invalid_winner_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE predicted_winner NOT IN (
                home_team,
                away_team
              )
           OR (
                predicted_home_margin >= 0.0
                AND predicted_winner <> home_team
              )
           OR (
                predicted_home_margin < 0.0
                AND predicted_winner <> away_team
              )
        """
    ).fetchone()[0]

    if invalid_winner_count > 0:
        raise RuntimeError(
            "Invalid current spread predicted "
            "winners found."
        )

    invalid_routing_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE (
                prediction_mode = 'EXTERNAL_NFELO_QB_RIDGE'
                AND (
                    prediction_mode_reason
                        <> 'complete_external_nfelo_qb_features'
                    OR model_name
                        <> 'external_nfelo_external_qb_spread'
                    OR ridge_alpha <> 10.0
                )
              )
           OR prediction_mode NOT IN (
                'EXTERNAL_NFELO_QB_RIDGE'
              )
        """
    ).fetchone()[0]

    if invalid_routing_count > 0:
        raise RuntimeError(
            "Invalid current spread prediction "
            "routing found."
        )

    invalid_metadata_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE game_id IS NULL
           OR season IS NULL
           OR home_team IS NULL
           OR away_team IS NULL
           OR home_team = away_team
           OR model_version <> '0.2.0'
           OR external_nfelo_rating_difference IS NULL
           OR external_nfelo_qb_adjustment_difference
                IS NULL
           OR primary_training_game_count <= 0
           OR fallback_training_game_count <= 0
           OR prediction_generated_at IS NULL
        """
    ).fetchone()[0]

    if invalid_metadata_count > 0:
        raise RuntimeError(
            "Invalid current spread prediction "
            "metadata found."
        )

    logger.info(
        "Current spread prediction table validated: "
        "%s rows in %s.",
        actual_row_count,
        TARGET_FULL_NAME,
    )


def build_current_spread_predictions(
    database_file: Path = DATABASE_FILE,
) -> pd.DataFrame:
    """Build and persist current spread predictions."""

    validate_database_file(
        database_file
    )

    logger.info(
        "Starting current spread prediction build..."
    )

    with duckdb.connect(
        str(database_file)
    ) as connection:
        validate_current_production_sources(
            connection
        )

        upcoming_games = (
            load_current_production_inputs(
                connection
            )
        )

        elo_predictions = (
            create_current_prediction_frame(
                upcoming_games
            )
        )

        historical_data = (
            load_production_spread_training_data(
                connection
            )
        )

        predictions = (
            create_current_spread_prediction_frame(
                upcoming_games=upcoming_games,
                elo_predictions=elo_predictions,
                historical_data=historical_data,
            )
        )

        connection.execute(
            "BEGIN TRANSACTION"
        )

        try:
            create_current_spread_predictions_table(
                connection=connection,
                predictions=predictions,
            )

            validate_current_spread_predictions_table(
                connection=connection,
                expected_row_count=len(
                    predictions
                ),
            )

            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise

    primary_count = int(
        predictions["prediction_mode"]
        .eq("EXTERNAL_NFELO_QB_RIDGE")
        .sum()
    )

    fallback_count = int(
        predictions["prediction_mode"]
        .eq("__NO_FALLBACK__")
        .sum()
    )

    logger.info(
        "Current spread prediction build completed: "
        "%s primary and %s fallback predictions.",
        primary_count,
        fallback_count,
    )

    return predictions


def main() -> None:
    """Run the current spread prediction builder."""

    try:
        build_current_spread_predictions()
    except Exception:
        logger.exception(
            "Current spread prediction build failed."
        )
        raise


if __name__ == "__main__":
    main()
