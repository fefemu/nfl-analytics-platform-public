"""
NFL Analytics Platform
Current Totals Prediction Builder

Purpose:
    Load production totals inputs, train the frozen
    primary and fallback models, score upcoming games and
    persist validated predictions in DuckDB.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging
from pathlib import Path

import duckdb
import pandas as pd

from src.modeling.current_totals_predictions import (
    CURRENT_TOTALS_PREDICTION_COLUMNS,
    create_current_totals_prediction_frame,
)
from src.modeling.evaluate_totals_fallback_candidates import (
    create_totals_fallback_features,
)
from src.modeling.evaluate_totals_model_candidates import (
    RAW_TOTALS_FEATURE_COLUMNS,
    create_totals_aggregate_features,
)
from src.modeling.production_totals_model import (
    PRODUCTION_TOTALS_MODEL,
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
TARGET_TABLE = "current_game_total_predictions"
TARGET_FULL_NAME = (
    f"{TARGET_SCHEMA}.{TARGET_TABLE}"
)

SCHEDULE_FULL_NAME = "processed.schedule"
TEAM_EFFICIENCY_FULL_NAME = (
    "processed.team_game_efficiency"
)
CURRENT_ELO_FULL_NAME = (
    "analytics.current_elo_ratings"
)
CURRENT_QB_FULL_NAME = (
    "analytics.current_qb_ratings"
)
WEATHER_FULL_NAME = (
    "analytics.game_weather_features"
)
SCORING_ENVIRONMENT_FULL_NAME = (
    "analytics.game_scoring_environment_features"
)
MODELING_DATASET_FULL_NAME = (
    "analytics.game_modeling_dataset"
)


def load_current_totals_inputs(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Load upcoming games and production-safe inputs."""

    upcoming_games = connection.execute(
        f"""
        SELECT
            schedule.game_id,
            schedule.season,
            schedule.game_type,
            schedule.week,
            schedule.gameday,
            schedule.gametime,
            schedule.home_team,
            schedule.away_team,

            COALESCE(
                UPPER(TRIM(schedule.location))
                    = 'NEUTRAL',
                FALSE
            ) AS is_neutral,

            home_history.prior_game_count
                AS home_prior_season_games,

            away_history.prior_game_count
                AS away_prior_season_games,

            home_history.offensive_epa_last_4
                AS home_offensive_epa_per_play_last_4,

            away_history.offensive_epa_last_4
                AS away_offensive_epa_per_play_last_4,

            home_history.defensive_epa_last_4
                AS
                home_defensive_epa_allowed_per_play_last_4,

            away_history.defensive_epa_last_4
                AS
                away_defensive_epa_allowed_per_play_last_4,

            home_qb.qb_rating
                AS home_listed_qb_rating,

            away_qb.qb_rating
                AS away_listed_qb_rating,

            home_elo.elo_rating
                AS home_elo_rating,

            away_elo.elo_rating
                AS away_elo_rating,

            weather.is_indoor,
            weather.has_game_weather,
            weather.cold_degrees_below_50,
            weather.heat_degrees_above_80,
            weather.wind_mph_above_10,

            scoring.league_average_total_last_64

        FROM {SCHEDULE_FULL_NAME} AS schedule

        LEFT JOIN {CURRENT_ELO_FULL_NAME}
            AS home_elo
            ON schedule.home_team = home_elo.team

        LEFT JOIN {CURRENT_ELO_FULL_NAME}
            AS away_elo
            ON schedule.away_team = away_elo.team

        LEFT JOIN {CURRENT_QB_FULL_NAME}
            AS home_qb
            ON schedule.home_qb_id = home_qb.qb_id

        LEFT JOIN {CURRENT_QB_FULL_NAME}
            AS away_qb
            ON schedule.away_qb_id = away_qb.qb_id

        LEFT JOIN {WEATHER_FULL_NAME}
            AS weather
            ON schedule.game_id = weather.game_id

        LEFT JOIN {SCORING_ENVIRONMENT_FULL_NAME}
            AS scoring
            ON schedule.game_id = scoring.game_id

        LEFT JOIN LATERAL (
            SELECT
                COUNT(*) AS prior_game_count,
                AVG(
                    recent.offensive_epa_per_play
                ) AS offensive_epa_last_4,
                AVG(
                    recent
                        .defensive_epa_allowed_per_play
                ) AS defensive_epa_last_4
            FROM (
                SELECT
                    history.offensive_epa_per_play,
                    history
                        .defensive_epa_allowed_per_play
                FROM {TEAM_EFFICIENCY_FULL_NAME}
                    AS history
                WHERE history.team
                    = schedule.home_team
                  AND history.season
                    = schedule.season
                  AND history.game_date
                    < schedule.gameday
                ORDER BY
                    history.game_date DESC,
                    history.game_id DESC
                LIMIT 4
            ) AS recent
        ) AS home_history
            ON TRUE

        LEFT JOIN LATERAL (
            SELECT
                COUNT(*) AS prior_game_count,
                AVG(
                    recent.offensive_epa_per_play
                ) AS offensive_epa_last_4,
                AVG(
                    recent
                        .defensive_epa_allowed_per_play
                ) AS defensive_epa_last_4
            FROM (
                SELECT
                    history.offensive_epa_per_play,
                    history
                        .defensive_epa_allowed_per_play
                FROM {TEAM_EFFICIENCY_FULL_NAME}
                    AS history
                WHERE history.team
                    = schedule.away_team
                  AND history.season
                    = schedule.season
                  AND history.game_date
                    < schedule.gameday
                ORDER BY
                    history.game_date DESC,
                    history.game_id DESC
                LIMIT 4
            ) AS recent
        ) AS away_history
            ON TRUE

        WHERE schedule.is_completed = FALSE
          AND schedule.game_type IN (
                'REG',
                'POST'
          )

        ORDER BY
            schedule.gameday,
            schedule.gametime,
            schedule.game_id
        """
    ).fetchdf()

    if upcoming_games.empty:
        raise RuntimeError(
            "No upcoming totals games are available."
        )

    logger.info(
        "Current totals inputs loaded: %s games.",
        len(upcoming_games),
    )

    return upcoming_games


def load_production_totals_training_data(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Load and aggregate historical totals inputs."""

    raw_feature_select = ",\n            ".join(
        RAW_TOTALS_FEATURE_COLUMNS
    )

    historical_data = connection.execute(
        f"""
        SELECT
            game_id,
            season,
            'production' AS split_name,
            both_short_windows_complete,
            {
                PRODUCTION_TOTALS_MODEL
                .target_column
            },
            home_elo_rating,
            away_elo_rating,
            {raw_feature_select}

        FROM {MODELING_DATASET_FULL_NAME}

        WHERE season
            < {
                PRODUCTION_TOTALS_MODEL
                .forward_test_season
            }

        ORDER BY
            season,
            game_date,
            game_id
        """
    ).fetchdf()

    if historical_data.empty:
        raise RuntimeError(
            "Production totals training source is empty."
        )

    historical_data = (
        create_totals_aggregate_features(
            historical_data
        )
    )

    historical_data = (
        create_totals_fallback_features(
            historical_data
        )
    )

    logger.info(
        "Production totals training data loaded: "
        "%s historical games.",
        len(historical_data),
    )

    return historical_data


def create_current_totals_predictions_table(
    connection: duckdb.DuckDBPyConnection,
    predictions: pd.DataFrame,
) -> None:
    """Persist the current totals prediction table."""

    missing_columns = sorted(
        set(CURRENT_TOTALS_PREDICTION_COLUMNS)
        - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Current totals predictions are missing "
            "columns: "
            + ", ".join(missing_columns)
        )

    connection.execute(
        f"CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA}"
    )

    connection.register(
        "_current_totals_predictions",
        predictions.loc[
            :,
            CURRENT_TOTALS_PREDICTION_COLUMNS,
        ],
    )

    try:
        connection.execute(
            f"""
            CREATE OR REPLACE TABLE {TARGET_FULL_NAME} AS
            SELECT *
            FROM _current_totals_predictions
            """
        )
    finally:
        connection.unregister(
            "_current_totals_predictions"
        )


def validate_current_totals_predictions_table(
    connection: duckdb.DuckDBPyConnection,
    expected_row_count: int,
) -> None:
    """Validate persisted production totals predictions."""

    actual_row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    if actual_row_count != expected_row_count:
        raise RuntimeError(
            "Current totals prediction row count does "
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
            "Duplicate current totals predictions found."
        )

    invalid_prediction_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE predicted_total_points IS NULL
           OR NOT isfinite(predicted_total_points)
        """
    ).fetchone()[0]

    if invalid_prediction_count > 0:
        raise RuntimeError(
            "Invalid current totals predictions found."
        )

    invalid_routing_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE (
                prediction_mode
                    = 'RIDGE_TOTALS_PRIMARY'
                AND (
                    prediction_mode_reason
                        <> 'complete_locked_totals_features'
                    OR model_name
                        <> 'ridge_epa_weather_qb_league_64_totals'
                    OR ridge_alpha <> 100.0
                    OR NOT has_complete_primary_features
                    OR NOT both_short_windows_complete
                    OR NOT
                        both_listed_qb_ratings_available
                )
              )
           OR (
                prediction_mode
                    = 'RIDGE_TOTALS_FALLBACK'
                AND (
                    prediction_mode_reason
                        <> 'missing_primary_rolling_or_qb_features'
                    OR model_name
                        <> 'ridge_league_64_indoor_elo_totals'
                    OR ridge_alpha <> 1.0
                    OR has_complete_primary_features
                )
              )
           OR prediction_mode NOT IN (
                'RIDGE_TOTALS_PRIMARY',
                'RIDGE_TOTALS_FALLBACK'
              )
        """
    ).fetchone()[0]

    if invalid_routing_count > 0:
        raise RuntimeError(
            "Invalid current totals routing found."
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
           OR model_version <> '0.1.0'
           OR elo_rating_sum IS NULL
           OR is_indoor IS NULL
           OR league_average_total_last_64 IS NULL
           OR primary_training_game_count <= 0
           OR fallback_training_game_count <= 0
           OR prediction_generated_at IS NULL
        """
    ).fetchone()[0]

    if invalid_metadata_count > 0:
        raise RuntimeError(
            "Invalid current totals metadata found."
        )

    logger.info(
        "Current totals prediction table validated: "
        "%s rows in %s.",
        actual_row_count,
        TARGET_FULL_NAME,
    )


def build_current_totals_predictions(
    database_file: Path = DATABASE_FILE,
) -> pd.DataFrame:
    """Build and persist current totals predictions."""

    validate_database_file(database_file)

    logger.info(
        "Starting current totals prediction build..."
    )

    with duckdb.connect(
        str(database_file)
    ) as connection:
        upcoming_games = (
            load_current_totals_inputs(
                connection
            )
        )

        historical_data = (
            load_production_totals_training_data(
                connection
            )
        )

        predictions = (
            create_current_totals_prediction_frame(
                upcoming_games=upcoming_games,
                historical_data=historical_data,
            )
        )

        connection.execute("BEGIN TRANSACTION")

        try:
            create_current_totals_predictions_table(
                connection=connection,
                predictions=predictions,
            )

            validate_current_totals_predictions_table(
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
        .eq("RIDGE_TOTALS_PRIMARY")
        .sum()
    )

    fallback_count = int(
        predictions["prediction_mode"]
        .eq("RIDGE_TOTALS_FALLBACK")
        .sum()
    )

    logger.info(
        "Current totals prediction build completed: "
        "%s primary and %s fallback predictions.",
        primary_count,
        fallback_count,
    )

    return predictions


def main() -> None:
    """Run the current totals prediction builder."""

    try:
        build_current_totals_predictions()
    except Exception:
        logger.exception(
            "Current totals prediction build failed."
        )
        raise


if __name__ == "__main__":
    main()