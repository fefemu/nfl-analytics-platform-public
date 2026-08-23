"""
NFL Analytics Platform
Current Game Prediction Builder

Purpose:
    Load and validate upcoming schedule games together with
    current Elo ratings for production prediction workflows.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging
from pathlib import Path

import duckdb
import pandas as pd

from src.analytics.build_elo_ratings import (
    CURRENT_FULL_NAME as ELO_RATINGS_FULL_NAME,
    DATABASE_FILE,
    validate_database_file,
)
from src.processing.build_processed_schedule import (
    TARGET_FULL_NAME as SCHEDULE_FULL_NAME,
)
from src.modeling.current_game_predictions import (
    PREDICTION_COLUMNS,
    create_current_prediction_frame,
)
from src.modeling.current_game_prediction_explanations import (
    EXPLANATION_COLUMNS,
    create_prediction_explanation_frame,
)
from src.modeling.current_production_data import (
    load_current_production_inputs,
    load_production_training_data,
    validate_current_production_sources,
)
from src.modeling.current_production_predictions import (
    PRODUCTION_AUDIT_COLUMNS,
    create_current_production_predictions,
)
from src.modeling.build_current_logistic_feature_contributions import (
    create_current_logistic_feature_contributions_table,
    validate_current_logistic_feature_contributions_table,
)
from src.modeling.build_current_prediction_data_science_view import (
    create_current_prediction_data_science_view,
    validate_current_prediction_data_science_view,
)
from src.modeling.production_probability_model import (
    PRODUCTION_PROBABILITY_MODEL,
)
from src.modeling.current_game_prediction_narratives import (
    create_current_game_prediction_narratives,
)
from src.modeling.build_current_game_prediction_narratives import (
    create_current_game_prediction_narratives_table,
    validate_current_game_prediction_narratives_table,
)
from src.modeling.current_production_prediction_explanations import (
    PRODUCTION_EXPLANATION_COLUMNS,
    create_production_prediction_explanation_frame,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


SCHEDULE_SCHEMA = "processed"
SCHEDULE_TABLE = "schedule"

ELO_RATINGS_SCHEMA = "analytics"
ELO_RATINGS_TABLE = "current_elo_ratings"

TARGET_SCHEMA = "analytics"
TARGET_TABLE = "current_game_predictions"
TARGET_FULL_NAME = (
    f"{TARGET_SCHEMA}.{TARGET_TABLE}"
)

EXPLANATION_TABLE = (
    "current_game_prediction_explanations"
)
EXPLANATION_FULL_NAME = (
    f"{TARGET_SCHEMA}.{EXPLANATION_TABLE}"
)

REQUIRED_SCHEDULE_COLUMNS = {
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "gametime",
    "home_team",
    "away_team",
    "location",
    "is_completed",
}

REQUIRED_ELO_COLUMNS = {
    "team",
    "elo_rating",
    "as_of_gameday",
    "last_completed_season",
}


CURRENT_PRODUCTION_PREDICTION_COLUMNS = (
    *PREDICTION_COLUMNS,
    *(
        column_name
        for column_name
        in PRODUCTION_AUDIT_COLUMNS
        if column_name
        not in PREDICTION_COLUMNS
    ),
)


def get_table_columns(
    connection: duckdb.DuckDBPyConnection,
    schema_name: str,
    table_name: str,
) -> set[str]:
    """Return the available columns for one table."""

    return {
        row[0]
        for row in connection.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = ?
              AND table_name = ?
            """,
            [
                schema_name,
                table_name,
            ],
        ).fetchall()
    }


def validate_prediction_sources(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate schedule and current Elo source tables."""

    source_definitions = (
        (
            SCHEDULE_SCHEMA,
            SCHEDULE_TABLE,
            SCHEDULE_FULL_NAME,
            REQUIRED_SCHEDULE_COLUMNS,
        ),
        (
            ELO_RATINGS_SCHEMA,
            ELO_RATINGS_TABLE,
            ELO_RATINGS_FULL_NAME,
            REQUIRED_ELO_COLUMNS,
        ),
    )

    for (
        schema_name,
        table_name,
        full_name,
        required_columns,
    ) in source_definitions:
        available_columns = get_table_columns(
            connection=connection,
            schema_name=schema_name,
            table_name=table_name,
        )

        if not available_columns:
            raise RuntimeError(
                f"Prediction source table does not "
                f"exist: {full_name}"
            )

        missing_columns = sorted(
            required_columns - available_columns
        )

        if missing_columns:
            raise RuntimeError(
                f"Prediction source {full_name} is "
                f"missing columns: "
                + ", ".join(missing_columns)
            )

    logger.info(
        "Current prediction sources validated: %s "
        "and %s.",
        SCHEDULE_FULL_NAME,
        ELO_RATINGS_FULL_NAME,
    )


def load_upcoming_game_inputs(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Load upcoming games with both current Elo ratings."""

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
            schedule.location,

            home_ratings.elo_rating
                AS home_elo_rating,

            away_ratings.elo_rating
                AS away_elo_rating,

            home_ratings.last_completed_season
                AS home_rating_season,

            away_ratings.last_completed_season
                AS away_rating_season,

            home_ratings.as_of_gameday
                AS home_rating_as_of,

            away_ratings.as_of_gameday
                AS away_rating_as_of

        FROM {SCHEDULE_FULL_NAME} AS schedule

        LEFT JOIN {ELO_RATINGS_FULL_NAME}
            AS home_ratings
            ON schedule.home_team
                = home_ratings.team

        LEFT JOIN {ELO_RATINGS_FULL_NAME}
            AS away_ratings
            ON schedule.away_team
                = away_ratings.team

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

    rating_columns = (
        "home_elo_rating",
        "away_elo_rating",
        "home_rating_season",
        "away_rating_season",
        "home_rating_as_of",
        "away_rating_as_of",
    )

    if not upcoming_games.empty:
        missing_rating_mask = upcoming_games.loc[
            :,
            rating_columns,
        ].isna().any(axis=1)

        if missing_rating_mask.any():
            missing_game_ids = ", ".join(
                upcoming_games.loc[
                    missing_rating_mask,
                    "game_id",
                ].astype(str)
            )

            raise RuntimeError(
                "Upcoming games are missing current "
                f"Elo ratings: {missing_game_ids}"
            )

    logger.info(
        "Upcoming prediction inputs loaded: %s games.",
        len(upcoming_games),
    )

    return upcoming_games


def create_current_predictions_table(
    connection: duckdb.DuckDBPyConnection,
    predictions: pd.DataFrame,
) -> None:
    """Create the current production prediction table."""

    missing_columns = sorted(
        set(PREDICTION_COLUMNS)
        - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Current predictions are missing columns: "
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
            home_rating_current DOUBLE,
            away_rating_current DOUBLE,
            home_rating_pregame DOUBLE,
            away_rating_pregame DOUBLE,
            applied_home_advantage DOUBLE,
            home_win_probability DOUBLE,
            away_win_probability DOUBLE,
            predicted_winner VARCHAR,
            home_rating_as_of DATE,
            away_rating_as_of DATE,
            prediction_generated_at TIMESTAMP
        )
        """
    )

    if predictions.empty:
        return

    rows = [
        tuple(
            row[column_name]
            for column_name in PREDICTION_COLUMNS
        )
        for row in predictions.to_dict(
            orient="records"
        )
    ]

    placeholders = ", ".join(
        "?"
        for _ in PREDICTION_COLUMNS
    )

    connection.executemany(
        f"""
        INSERT INTO {TARGET_FULL_NAME}
        VALUES ({placeholders})
        """,
        rows,
    )


def create_current_production_predictions_table(
    connection: duckdb.DuckDBPyConnection,
    predictions: pd.DataFrame,
) -> None:
    """Create the auditable production prediction table."""

    missing_columns = sorted(
        set(
            CURRENT_PRODUCTION_PREDICTION_COLUMNS
        )
        - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Current production predictions are "
            "missing columns: "
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
            home_rating_current DOUBLE,
            away_rating_current DOUBLE,
            home_rating_pregame DOUBLE,
            away_rating_pregame DOUBLE,
            applied_home_advantage DOUBLE,
            home_win_probability DOUBLE,
            away_win_probability DOUBLE,
            predicted_winner VARCHAR,
            home_rating_as_of DATE,
            away_rating_as_of DATE,
            prediction_generated_at TIMESTAMP,
            prediction_mode VARCHAR,
            prediction_mode_reason VARCHAR,
            published_nfelo_home_probability DOUBLE,
            primary_logistic_home_win_probability DOUBLE,
            fallback_logistic_home_win_probability DOUBLE,
            applied_primary_logistic_weight DOUBLE,
            applied_published_nfelo_weight DOUBLE,
            elo_home_win_probability DOUBLE,
            logistic_home_win_probability DOUBLE,
            applied_logistic_weight DOUBLE,
            applied_elo_weight DOUBLE,
            has_complete_injury_data BOOLEAN,
            both_listed_qb_ratings_available BOOLEAN,
            has_complete_production_features BOOLEAN,
            has_complete_fallback_features BOOLEAN,
            external_nfelo_rating_difference DOUBLE,
            listed_qb_rating_difference DOUBLE,
            external_nfelo_qb_adjustment_difference DOUBLE,
            offense_injury_burden_difference DOUBLE,
            defense_injury_burden_difference DOUBLE,
            special_teams_injury_burden_difference DOUBLE
        )
        """
    )

    if predictions.empty:
        return

    rows = [
        tuple(
            (
                None
                if pd.isna(
                    row[column_name]
                )
                else row[column_name]
            )
            for column_name
            in CURRENT_PRODUCTION_PREDICTION_COLUMNS
        )
        for row in predictions.to_dict(
            orient="records"
        )
    ]

    placeholders = ", ".join(
        "?"
        for _ in (
            CURRENT_PRODUCTION_PREDICTION_COLUMNS
        )
    )

    connection.executemany(
        f"""
        INSERT INTO {TARGET_FULL_NAME}
        VALUES ({placeholders})
        """,
        rows,
    )


def create_current_explanations_table(
    connection: duckdb.DuckDBPyConnection,
    explanations: pd.DataFrame,
) -> None:
    """Create current prediction explanations."""

    missing_columns = sorted(
        set(EXPLANATION_COLUMNS)
        - set(explanations.columns)
    )

    if missing_columns:
        raise ValueError(
            "Current explanations are missing columns: "
            + ", ".join(missing_columns)
        )

    connection.execute(
        f"""
        CREATE OR REPLACE TABLE
            {EXPLANATION_FULL_NAME} (
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
            favorite VARCHAR,
            underdog VARCHAR,
            favorite_win_probability DOUBLE,
            home_win_probability DOUBLE,
            away_win_probability DOUBLE,
            neutral_home_win_probability DOUBLE,
            home_field_probability_lift DOUBLE,
            home_rating DOUBLE,
            away_rating DOUBLE,
            raw_home_rating_edge DOUBLE,
            applied_home_advantage DOUBLE,
            adjusted_home_rating_edge DOUBLE,
            team_strength_log_odds_contribution DOUBLE,
            home_field_log_odds_contribution DOUBLE,
            total_home_log_odds DOUBLE,
            matchup_label VARCHAR,
            prediction_generated_at TIMESTAMP
        )
        """
    )

    if explanations.empty:
        return

    rows = [
        tuple(
            row[column_name]
            for column_name in EXPLANATION_COLUMNS
        )
        for row in explanations.to_dict(
            orient="records"
        )
    ]

    placeholders = ", ".join(
        "?"
        for _ in EXPLANATION_COLUMNS
    )

    connection.executemany(
        f"""
        INSERT INTO {EXPLANATION_FULL_NAME}
        VALUES ({placeholders})
        """,
        rows,
    )

def create_current_production_explanations_table(
    connection: duckdb.DuckDBPyConnection,
    explanations: pd.DataFrame,
) -> None:
    """Create external production explanations."""

    missing_columns = sorted(
        set(
            PRODUCTION_EXPLANATION_COLUMNS
        )
        - set(explanations.columns)
    )

    if missing_columns:
        raise ValueError(
            "Current production explanations are "
            "missing columns: "
            + ", ".join(missing_columns)
        )

    connection.execute(
        f"""
        CREATE OR REPLACE TABLE
            {EXPLANATION_FULL_NAME} (
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
            favorite VARCHAR,
            underdog VARCHAR,
            favorite_win_probability DOUBLE,
            home_win_probability DOUBLE,
            away_win_probability DOUBLE,
            published_nfelo_home_probability DOUBLE,
            published_nfelo_away_probability DOUBLE,
            primary_logistic_home_win_probability DOUBLE,
            primary_logistic_away_win_probability DOUBLE,
            fallback_logistic_home_win_probability DOUBLE,
            fallback_logistic_away_win_probability DOUBLE,
            production_probability_adjustment_from_published_nfelo DOUBLE,
            applied_primary_logistic_weight DOUBLE,
            applied_published_nfelo_weight DOUBLE,
            external_nfelo_rating_difference DOUBLE,
            external_nfelo_qb_adjustment_difference DOUBLE,
            listed_qb_rating_difference DOUBLE,
            offense_injury_burden_difference DOUBLE,
            defense_injury_burden_difference DOUBLE,
            special_teams_injury_burden_difference DOUBLE,
            has_complete_injury_data BOOLEAN,
            both_listed_qb_ratings_available BOOLEAN,
            has_complete_production_features BOOLEAN,
            has_complete_fallback_features BOOLEAN,
            matchup_label VARCHAR,
            prediction_generated_at TIMESTAMP
        )
        """
    )

    if explanations.empty:
        return

    rows = [
        tuple(
            (
                None
                if pd.isna(
                    row[column_name]
                )
                else row[column_name]
            )
            for column_name
            in PRODUCTION_EXPLANATION_COLUMNS
        )
        for row in explanations.to_dict(
            orient="records"
        )
    ]

    placeholders = ", ".join(
        "?"
        for _ in (
            PRODUCTION_EXPLANATION_COLUMNS
        )
    )

    connection.executemany(
        f"""
        INSERT INTO {EXPLANATION_FULL_NAME}
        VALUES ({placeholders})
        """,
        rows,
    )


def validate_current_explanations_table(
    connection: duckdb.DuckDBPyConnection,
    expected_row_count: int,
) -> None:
    """Validate persisted prediction explanations."""

    actual_row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {EXPLANATION_FULL_NAME}
        """
    ).fetchone()[0]

    if actual_row_count != expected_row_count:
        raise RuntimeError(
            "Current explanation row count does not "
            f"match: expected {expected_row_count}, "
            f"found {actual_row_count}."
        )

    invalid_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {EXPLANATION_FULL_NAME}
        WHERE favorite NOT IN (
                home_team,
                away_team
              )
           OR underdog NOT IN (
                home_team,
                away_team
              )
           OR favorite = underdog
           OR favorite_win_probability
                NOT BETWEEN 0.5 AND 1.0
           OR home_win_probability
                NOT BETWEEN 0.0 AND 1.0
           OR away_win_probability
                NOT BETWEEN 0.0 AND 1.0
           OR ABS(
                home_win_probability
                + away_win_probability
                - 1.0
              ) > 0.000001
           OR ABS(
                raw_home_rating_edge
                + applied_home_advantage
                - adjusted_home_rating_edge
              ) > 0.000001
           OR ABS(
                team_strength_log_odds_contribution
                + home_field_log_odds_contribution
                - total_home_log_odds
              ) > 0.000001
           OR matchup_label NOT IN (
                'toss_up',
                'slight_edge',
                'clear_edge',
                'strong_edge'
              )
           OR prediction_generated_at IS NULL
        """
    ).fetchone()[0]

    if invalid_count > 0:
        raise RuntimeError(
            "Invalid current prediction explanations "
            "found."
        )

    duplicate_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT game_id
            FROM {EXPLANATION_FULL_NAME}
            GROUP BY game_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_count > 0:
        raise RuntimeError(
            "Duplicate current prediction "
            "explanations found."
        )

    logger.info(
        "Current explanation table validated: %s "
        "rows in %s.",
        actual_row_count,
        EXPLANATION_FULL_NAME,
    )


def validate_current_production_explanations_table(
    connection: duckdb.DuckDBPyConnection,
    expected_row_count: int,
) -> None:
    """Validate external production explanations."""

    actual_row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {EXPLANATION_FULL_NAME}
        """
    ).fetchone()[0]

    if actual_row_count != expected_row_count:
        raise RuntimeError(
            "Current production explanation row count "
            f"does not match: expected "
            f"{expected_row_count}, found "
            f"{actual_row_count}."
        )

    duplicate_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT game_id
            FROM {EXPLANATION_FULL_NAME}
            GROUP BY game_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_count > 0:
        raise RuntimeError(
            "Duplicate current production "
            "explanations found."
        )

    invalid_probability_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {EXPLANATION_FULL_NAME}
        WHERE home_win_probability
                NOT BETWEEN 0.0 AND 1.0
           OR away_win_probability
                NOT BETWEEN 0.0 AND 1.0
           OR published_nfelo_home_probability
                NOT BETWEEN 0.0 AND 1.0
           OR published_nfelo_away_probability
                NOT BETWEEN 0.0 AND 1.0
           OR ABS(
                home_win_probability
                + away_win_probability
                - 1.0
              ) > 0.000001
           OR ABS(
                published_nfelo_home_probability
                + published_nfelo_away_probability
                - 1.0
              ) > 0.000001
           OR ABS(
                production_probability_adjustment_from_published_nfelo
                - (
                    home_win_probability
                    - published_nfelo_home_probability
                )
              ) > 0.000001
           OR favorite_win_probability
                NOT BETWEEN 0.5 AND 1.0
           OR ABS(
                favorite_win_probability
                - GREATEST(
                    home_win_probability,
                    away_win_probability
                )
              ) > 0.000001
        """
    ).fetchone()[0]

    if invalid_probability_count > 0:
        raise RuntimeError(
            "Invalid current production explanation "
            "probabilities found."
        )

    invalid_primary_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {EXPLANATION_FULL_NAME}
        WHERE prediction_mode
                = 'EXTERNAL_NFELO_BLEND'
          AND (
                prediction_mode_reason
                    <> 'complete_external_primary_features'
                OR primary_logistic_home_win_probability IS NULL
                OR primary_logistic_home_win_probability
                    NOT BETWEEN 0.0 AND 1.0
                OR primary_logistic_away_win_probability
                    NOT BETWEEN 0.0 AND 1.0
                OR ABS(
                    primary_logistic_home_win_probability
                    + primary_logistic_away_win_probability
                    - 1.0
                  ) > 0.000001
                OR fallback_logistic_home_win_probability
                    IS NOT NULL
                OR fallback_logistic_away_win_probability
                    IS NOT NULL
                OR applied_primary_logistic_weight
                    <= 0.0
                OR applied_published_nfelo_weight
                    <= 0.0
                OR ABS(
                    applied_primary_logistic_weight
                    + applied_published_nfelo_weight
                    - 1.0
                  ) > 0.000001
                OR NOT has_complete_injury_data
                OR NOT
                    both_listed_qb_ratings_available
                OR NOT
                    has_complete_production_features
                OR NOT
                    has_complete_fallback_features
                OR ABS(
                    home_win_probability
                    - (
                        applied_primary_logistic_weight
                        * primary_logistic_home_win_probability
                        + applied_published_nfelo_weight
                        * published_nfelo_home_probability
                    )
                  ) > 0.000001
              )
        """
    ).fetchone()[0]

    if invalid_primary_count > 0:
        raise RuntimeError(
            "Invalid current production primary "
            "explanation routing found."
        )

    invalid_fallback_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {EXPLANATION_FULL_NAME}
        WHERE prediction_mode
                = 'EXTERNAL_ELO_QB_FALLBACK'
          AND (
                prediction_mode_reason
                    <> 'incomplete_external_primary_features'
                OR primary_logistic_home_win_probability
                    IS NOT NULL
                OR primary_logistic_away_win_probability
                    IS NOT NULL
                OR fallback_logistic_home_win_probability IS NULL
                OR fallback_logistic_home_win_probability
                    NOT BETWEEN 0.0 AND 1.0
                OR fallback_logistic_away_win_probability
                    NOT BETWEEN 0.0 AND 1.0
                OR ABS(
                    fallback_logistic_home_win_probability
                    + fallback_logistic_away_win_probability
                    - 1.0
                  ) > 0.000001
                OR applied_primary_logistic_weight
                    <> 0.0
                OR applied_published_nfelo_weight
                    <> 0.0
                OR has_complete_production_features
                OR NOT
                    has_complete_fallback_features
                OR ABS(
                    home_win_probability
                    - fallback_logistic_home_win_probability
                  ) > 0.000001
              )
        """
    ).fetchone()[0]

    if invalid_fallback_count > 0:
        raise RuntimeError(
            "Invalid current production fallback "
            "explanation routing found."
        )

    unknown_routing_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {EXPLANATION_FULL_NAME}
        WHERE prediction_mode NOT IN (
            'EXTERNAL_NFELO_BLEND',
            'EXTERNAL_ELO_QB_FALLBACK'
        )
        """
    ).fetchone()[0]

    if unknown_routing_count > 0:
        raise RuntimeError(
            "Unknown current production explanation "
            "routing found."
        )

    invalid_feature_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {EXPLANATION_FULL_NAME}
        WHERE external_nfelo_rating_difference IS NULL
           OR external_nfelo_qb_adjustment_difference
                IS NULL
           OR NOT has_complete_fallback_features
           OR (
                both_listed_qb_ratings_available
                AND listed_qb_rating_difference IS NULL
              )
           OR (
                has_complete_production_features
                AND (
                    listed_qb_rating_difference IS NULL
                    OR offense_injury_burden_difference
                        IS NULL
                    OR defense_injury_burden_difference
                        IS NULL
                    OR special_teams_injury_burden_difference
                        IS NULL
                )
              )
        """
    ).fetchone()[0]

    if invalid_feature_count > 0:
        raise RuntimeError(
            "Invalid current production explanation "
            "features found."
        )

    invalid_metadata_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {EXPLANATION_FULL_NAME}
        WHERE favorite NOT IN (
                home_team,
                away_team
              )
           OR underdog NOT IN (
                home_team,
                away_team
              )
           OR favorite = underdog
           OR model_name
                <> 'external_nfelo_probability_routing'
           OR model_version <> '0.3.0'
           OR prediction_mode_reason IS NULL
           OR matchup_label NOT IN (
                'toss_up',
                'slight_edge',
                'clear_edge',
                'strong_edge'
              )
           OR prediction_generated_at IS NULL
        """
    ).fetchone()[0]

    if invalid_metadata_count > 0:
        raise RuntimeError(
            "Invalid current production explanation "
            "metadata found."
        )

    logger.info(
        "Current production explanation table "
        "validated: %s rows in %s.",
        actual_row_count,
        EXPLANATION_FULL_NAME,
    )


def validate_current_predictions_table(
    connection: duckdb.DuckDBPyConnection,
    expected_row_count: int,
) -> None:
    """Validate current production predictions."""

    actual_row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    if actual_row_count != expected_row_count:
        raise RuntimeError(
            "Current prediction row count does not "
            f"match: expected {expected_row_count}, "
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
            "Duplicate current game predictions found."
        )

    invalid_probability_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE home_win_probability IS NULL
           OR away_win_probability IS NULL
           OR home_win_probability NOT BETWEEN 0.0 AND 1.0
           OR away_win_probability NOT BETWEEN 0.0 AND 1.0
           OR ABS(
                home_win_probability
                + away_win_probability
                - 1.0
           ) > 0.000001
        """
    ).fetchone()[0]

    if invalid_probability_count > 0:
        raise RuntimeError(
            "Invalid current prediction probabilities "
            "found."
        )

    invalid_metadata_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE model_name IS NULL
           OR model_version IS NULL
           OR prediction_generated_at IS NULL
           OR predicted_winner NOT IN (
                home_team,
                away_team
           )
        """
    ).fetchone()[0]

    if invalid_metadata_count > 0:
        raise RuntimeError(
            "Invalid current prediction metadata found."
        )

    logger.info(
        "Current prediction table validated: %s rows "
        "in %s.",
        actual_row_count,
        TARGET_FULL_NAME,
    )


def validate_current_production_predictions_table(
    connection: duckdb.DuckDBPyConnection,
    expected_row_count: int,
) -> None:
    """Validate routed external production predictions."""

    actual_row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    if actual_row_count != expected_row_count:
        raise RuntimeError(
            "Current production prediction row count "
            f"does not match: expected "
            f"{expected_row_count}, found "
            f"{actual_row_count}."
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
            "Duplicate current production "
            "predictions found."
        )

    invalid_probability_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE home_win_probability
                NOT BETWEEN 0.0 AND 1.0
           OR away_win_probability
                NOT BETWEEN 0.0 AND 1.0
           OR published_nfelo_home_probability
                NOT BETWEEN 0.0 AND 1.0
           OR ABS(
                home_win_probability
                + away_win_probability
                - 1.0
              ) > 0.000001
           OR applied_primary_logistic_weight
                NOT BETWEEN 0.0 AND 1.0
           OR applied_published_nfelo_weight
                NOT BETWEEN 0.0 AND 1.0
        """
    ).fetchone()[0]

    if invalid_probability_count > 0:
        raise RuntimeError(
            "Invalid current production "
            "probabilities found."
        )

    invalid_primary_routing_count = (
        connection.execute(
            f"""
            SELECT COUNT(*)
            FROM {TARGET_FULL_NAME}
            WHERE prediction_mode
                    = 'EXTERNAL_NFELO_BLEND'
              AND (
                    prediction_mode_reason
                        <> 'complete_external_primary_features'
                    OR primary_logistic_home_win_probability IS NULL
                    OR primary_logistic_home_win_probability
                        NOT BETWEEN 0.0 AND 1.0
                    OR fallback_logistic_home_win_probability
                        IS NOT NULL
                    OR applied_primary_logistic_weight
                        <= 0.0
                    OR applied_published_nfelo_weight
                        <= 0.0
                    OR NOT has_complete_injury_data
                    OR NOT
                        both_listed_qb_ratings_available
                    OR NOT
                        has_complete_production_features
                    OR NOT
                        has_complete_fallback_features
                    OR ABS(
                        home_win_probability
                        - (
                            applied_primary_logistic_weight
                            * primary_logistic_home_win_probability
                            + applied_published_nfelo_weight
                            * published_nfelo_home_probability
                        )
                    ) > 0.000001
                  )
            """
        ).fetchone()[0]
    )

    if invalid_primary_routing_count > 0:
        raise RuntimeError(
            "Invalid current production primary "
            "routing records found."
        )

    invalid_fallback_routing_count = (
        connection.execute(
            f"""
            SELECT COUNT(*)
            FROM {TARGET_FULL_NAME}
            WHERE prediction_mode
                    = 'EXTERNAL_ELO_QB_FALLBACK'
              AND (
                    prediction_mode_reason
                        <> 'incomplete_external_primary_features'
                    OR primary_logistic_home_win_probability
                        IS NOT NULL
                    OR fallback_logistic_home_win_probability IS NULL
                    OR fallback_logistic_home_win_probability
                        NOT BETWEEN 0.0 AND 1.0
                    OR applied_primary_logistic_weight
                        <> 0.0
                    OR applied_published_nfelo_weight
                        <> 0.0
                    OR has_complete_production_features
                    OR NOT
                        has_complete_fallback_features
                    OR ABS(
                        home_win_probability
                        - fallback_logistic_home_win_probability
                    ) > 0.000001
                  )
            """
        ).fetchone()[0]
    )

    if invalid_fallback_routing_count > 0:
        raise RuntimeError(
            "Invalid current production fallback "
            "routing records found."
        )

    unknown_routing_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE prediction_mode NOT IN (
            'EXTERNAL_NFELO_BLEND',
            'EXTERNAL_ELO_QB_FALLBACK'
        )
        """
    ).fetchone()[0]

    if unknown_routing_count > 0:
        raise RuntimeError(
            "Unknown current production routing "
            "records found."
        )

    invalid_feature_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE external_nfelo_rating_difference IS NULL
           OR external_nfelo_qb_adjustment_difference
                IS NULL
           OR NOT has_complete_fallback_features
           OR (
                both_listed_qb_ratings_available
                AND listed_qb_rating_difference IS NULL
              )
           OR (
                has_complete_production_features
                AND (
                    listed_qb_rating_difference IS NULL
                    OR offense_injury_burden_difference
                        IS NULL
                    OR defense_injury_burden_difference
                        IS NULL
                    OR special_teams_injury_burden_difference
                        IS NULL
                )
              )
        """
    ).fetchone()[0]

    if invalid_feature_count > 0:
        raise RuntimeError(
            "Invalid current production features found."
        )

    invalid_legacy_alias_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE ABS(
                elo_home_win_probability
                - published_nfelo_home_probability
              ) > 0.000001
           OR (
                prediction_mode
                    = 'EXTERNAL_NFELO_BLEND'
                AND (
                    ABS(
                        logistic_home_win_probability
                        - primary_logistic_home_win_probability
                    ) > 0.000001
                    OR ABS(
                        applied_logistic_weight
                        - applied_primary_logistic_weight
                    ) > 0.000001
                    OR ABS(
                        applied_elo_weight
                        - applied_published_nfelo_weight
                    ) > 0.000001
                )
              )
           OR (
                prediction_mode
                    = 'EXTERNAL_ELO_QB_FALLBACK'
                AND (
                    logistic_home_win_probability
                        IS NOT NULL
                    OR applied_logistic_weight <> 0.0
                    OR applied_elo_weight <> 0.0
                )
              )
        """
    ).fetchone()[0]

    if invalid_legacy_alias_count > 0:
        raise RuntimeError(
            "Invalid transitional production audit "
            "aliases found."
        )

    invalid_metadata_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE model_name
                <> 'external_nfelo_probability_routing'
           OR model_version <> '0.3.0'
           OR prediction_mode IS NULL
           OR prediction_mode_reason IS NULL
           OR prediction_generated_at IS NULL
           OR predicted_winner NOT IN (
                home_team,
                away_team
              )
        """
    ).fetchone()[0]

    if invalid_metadata_count > 0:
        raise RuntimeError(
            "Invalid current production metadata found."
        )

    logger.info(
        "Current production prediction table "
        "validated: %s rows in %s.",
        actual_row_count,
        TARGET_FULL_NAME,
    )


def build_current_game_predictions(
    database_file: Path = DATABASE_FILE,
) -> pd.DataFrame:
    """Build current production blend predictions."""

    validate_database_file(database_file)

    logger.info(
        "Starting current production prediction build..."
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
            load_production_training_data(
                connection
            )
        )

        (
            predictions,
            feature_contributions,
        ) = (
            create_current_production_predictions(
                upcoming_games=upcoming_games,
                elo_predictions=elo_predictions,
                historical_data=historical_data,
                return_feature_contributions=True,
            )
        )

        blend_count = int(
            predictions[
                "prediction_mode"
            ].eq("EXTERNAL_NFELO_BLEND").sum()
        )

        fallback_count = int(
            predictions[
                "prediction_mode"
            ].eq("EXTERNAL_ELO_QB_FALLBACK").sum()
        )

        logger.info(
            "Current production routing completed: "
            "%s external nfelo blend predictions and "
            "%s external Elo-QB fallback predictions.",
            blend_count,
            fallback_count,
        )

        explanations = (
            create_production_prediction_explanation_frame(
                predictions
            )
        )

        narratives = (
            create_current_game_prediction_narratives(
                explanations=explanations,
                feature_contributions=(
                    feature_contributions
                ),
            )
        )

        connection.execute(
            "BEGIN TRANSACTION"
        )

        try:
            create_current_production_predictions_table(
                connection=connection,
                predictions=predictions,
            )

            create_current_production_explanations_table(
                connection=connection,
                explanations=explanations,
            )

            create_current_logistic_feature_contributions_table(
                connection=connection,
                contributions=(
                    feature_contributions
                ),
            )

            create_current_game_prediction_narratives_table(
                connection=connection,
                narratives=narratives,
            )

            create_current_prediction_data_science_view(
                connection=connection,
            )

            validate_current_production_predictions_table(
                connection=connection,
                expected_row_count=len(
                    predictions
                ),
            )

            validate_current_production_explanations_table(
                connection=connection,
                expected_row_count=len(
                    explanations
                ),
            )

            validate_current_logistic_feature_contributions_table(
                connection=connection,
                expected_row_count=len(
                    feature_contributions
                ),
                expected_feature_count=len(
                    PRODUCTION_PROBABILITY_MODEL
                    .logistic_feature_columns
                ),
            )

            validate_current_game_prediction_narratives_table(
                connection=connection,
                expected_row_count=len(
                    narratives
                ),
            )

            validate_current_prediction_data_science_view(
                connection=connection,
                expected_prediction_count=len(
                    predictions
                ),
                expected_feature_count=len(
                    PRODUCTION_PROBABILITY_MODEL
                    .logistic_feature_columns
                ),
            )

            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise

    logger.info(
        "Current production prediction build "
        "completed: %s predictions.",
        len(predictions),
    )

    return predictions


def main() -> None:
    """Run the current prediction builder."""

    try:
        build_current_game_predictions()
    except Exception:
        logger.exception(
            "Current game prediction build failed."
        )
        raise


if __name__ == "__main__":
    main()
