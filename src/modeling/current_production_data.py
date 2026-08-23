"""
NFL Analytics Platform
Current Production Data Access

Purpose:
    Load and validate the DuckDB sources required by the
    selected external-nfelo production probability model.

    Internal Elo inputs remain available temporarily for
    downstream components that have not yet been migrated.

    This module performs read-only data access. It does
    not refresh external sources or write generated
    tables.

Author:
    Ferenc Kaizer

Version:
    0.2.0
"""

import logging

import duckdb
import pandas as pd

from src.analytics.build_elo_ratings import (
    CURRENT_FULL_NAME as ELO_RATINGS_FULL_NAME,
)
from src.analytics.build_game_injury_features import (
    TARGET_FULL_NAME as INJURY_FEATURES_FULL_NAME,
)
from src.analytics.build_qb_ratings import (
    CURRENT_FULL_NAME as QB_RATINGS_FULL_NAME,
)
from src.modeling.build_game_modeling_dataset import (
    TARGET_FULL_NAME as MODELING_DATASET_FULL_NAME,
)
from src.modeling.production_probability_model import (
    EXTERNAL_ELO_FEATURE,
    EXTERNAL_QB_FEATURE,
    LISTED_QB_FEATURE,
    PRODUCTION_PROBABILITY_MODEL,
)
from src.modeling.train_logistic_baseline import (
    TARGET_COLUMN,
)
from src.processing.build_external_nfelo_game_ratings import (
    TARGET_FULL_NAME as EXTERNAL_NFELO_FULL_NAME,
)
from src.processing.build_processed_schedule import (
    TARGET_FULL_NAME as SCHEDULE_FULL_NAME,
)


logger = logging.getLogger(__name__)


SCHEDULE_SCHEMA = "processed"
SCHEDULE_TABLE = "schedule"

ELO_RATINGS_SCHEMA = "analytics"
ELO_RATINGS_TABLE = "current_elo_ratings"

QB_RATINGS_SCHEMA = "analytics"
QB_RATINGS_TABLE = "current_qb_ratings"

INJURY_FEATURES_SCHEMA = "analytics"
INJURY_FEATURES_TABLE = "game_injury_features"

MODELING_DATASET_SCHEMA = "analytics"
MODELING_DATASET_TABLE = "game_modeling_dataset"

EXTERNAL_NFELO_SCHEMA = "processed"
EXTERNAL_NFELO_TABLE = (
    "external_nfelo_game_ratings"
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
    "home_qb_id",
    "home_qb_name",
    "away_qb_id",
    "away_qb_name",
    "is_completed",
}

REQUIRED_ELO_COLUMNS = {
    "team",
    "elo_rating",
    "as_of_gameday",
    "last_completed_season",
}

REQUIRED_QB_COLUMNS = {
    "qb_id",
    "qb_name",
    "current_team",
    "qb_rating",
    "as_of_date",
    "rating_standard_error",
}

REQUIRED_INJURY_COLUMNS = {
    "game_id",
    "has_complete_injury_data",
    "offense_injury_burden_difference",
    "defense_injury_burden_difference",
    "special_teams_injury_burden_difference",
}

REQUIRED_MODELING_COLUMNS = {
    "game_id",
    "season",
    "game_date",
    TARGET_COLUMN,
    "has_complete_injury_data",
    LISTED_QB_FEATURE,
    "offense_injury_burden_difference",
    "defense_injury_burden_difference",
    "special_teams_injury_burden_difference",
}

REQUIRED_EXTERNAL_NFELO_COLUMNS = {
    "normalized_game_id",
    "source_season",
    "source_week",
    "home_team",
    "away_team",
    "starting_nfelo_home",
    "starting_nfelo_away",
    "home_538_qb_adj",
    "away_538_qb_adj",
    "nfelo_home_probability_open",
}


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


def validate_current_production_sources(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate all current production source tables."""

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
        (
            QB_RATINGS_SCHEMA,
            QB_RATINGS_TABLE,
            QB_RATINGS_FULL_NAME,
            REQUIRED_QB_COLUMNS,
        ),
        (
            INJURY_FEATURES_SCHEMA,
            INJURY_FEATURES_TABLE,
            INJURY_FEATURES_FULL_NAME,
            REQUIRED_INJURY_COLUMNS,
        ),
        (
            MODELING_DATASET_SCHEMA,
            MODELING_DATASET_TABLE,
            MODELING_DATASET_FULL_NAME,
            REQUIRED_MODELING_COLUMNS,
        ),
        (
            EXTERNAL_NFELO_SCHEMA,
            EXTERNAL_NFELO_TABLE,
            EXTERNAL_NFELO_FULL_NAME,
            REQUIRED_EXTERNAL_NFELO_COLUMNS,
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
                "Current production source table "
                f"does not exist: {full_name}"
            )

        missing_columns = sorted(
            required_columns
            - available_columns
        )

        if missing_columns:
            raise RuntimeError(
                f"Current production source {full_name} "
                "is missing columns: "
                + ", ".join(missing_columns)
            )

    logger.info(
        "Current production sources validated: "
        "%s, %s, %s, %s, %s and %s.",
        SCHEDULE_FULL_NAME,
        ELO_RATINGS_FULL_NAME,
        QB_RATINGS_FULL_NAME,
        INJURY_FEATURES_FULL_NAME,
        MODELING_DATASET_FULL_NAME,
        EXTERNAL_NFELO_FULL_NAME,
    )


def load_current_production_inputs(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """
    Load upcoming games with all production inputs.

    Exact external game rows provide published nfelo
    probabilities when they are currently available.

    The latest external team-level rating and QB snapshot
    supplies fallback features for every future game.

    QB and injury joins remain optional so incomplete
    games can use the external Elo-QB logistic fallback.
    """

    upcoming_games = connection.execute(
        f"""
        WITH external_team_rows AS (
            SELECT
                source_season,
                source_week,
                normalized_game_id,
                home_team AS team,
                starting_nfelo_home
                    AS external_nfelo_rating,
                home_538_qb_adj
                    AS external_nfelo_qb_adjustment

            FROM {EXTERNAL_NFELO_FULL_NAME}

            UNION ALL

            SELECT
                source_season,
                source_week,
                normalized_game_id,
                away_team AS team,
                starting_nfelo_away
                    AS external_nfelo_rating,
                away_538_qb_adj
                    AS external_nfelo_qb_adjustment

            FROM {EXTERNAL_NFELO_FULL_NAME}
        ),

        ranked_external_team_rows AS (
            SELECT
                team,
                source_season,
                source_week,
                external_nfelo_rating,
                external_nfelo_qb_adjustment,

                ROW_NUMBER() OVER (
                    PARTITION BY team
                    ORDER BY
                        source_season DESC,
                        source_week DESC,
                        normalized_game_id DESC
                ) AS recency_rank

            FROM external_team_rows

            WHERE external_nfelo_rating IS NOT NULL
              AND external_nfelo_qb_adjustment IS NOT NULL
        ),

        current_external_team_ratings AS (
            SELECT
                team,
                source_season,
                source_week,
                external_nfelo_rating,
                external_nfelo_qb_adjustment

            FROM ranked_external_team_rows

            WHERE recency_rank = 1
        )

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

            schedule.home_qb_id
                AS home_listed_qb_id,

            schedule.home_qb_name
                AS home_listed_qb_name,

            schedule.away_qb_id
                AS away_listed_qb_id,

            schedule.away_qb_name
                AS away_listed_qb_name,

            home_elo.elo_rating
                AS home_elo_rating,

            away_elo.elo_rating
                AS away_elo_rating,

            home_elo.last_completed_season
                AS home_rating_season,

            away_elo.last_completed_season
                AS away_rating_season,

            home_elo.as_of_gameday
                AS home_rating_as_of,

            away_elo.as_of_gameday
                AS away_rating_as_of,

            home_external.external_nfelo_rating
                AS home_external_nfelo_rating,

            away_external.external_nfelo_rating
                AS away_external_nfelo_rating,

            home_external.external_nfelo_rating
                - away_external.external_nfelo_rating
                AS {EXTERNAL_ELO_FEATURE},

            home_external
                .external_nfelo_qb_adjustment
                AS home_external_nfelo_qb_adjustment,

            away_external
                .external_nfelo_qb_adjustment
                AS away_external_nfelo_qb_adjustment,

            home_external
                .external_nfelo_qb_adjustment
                - away_external
                    .external_nfelo_qb_adjustment
                AS {EXTERNAL_QB_FEATURE},

            home_external.source_season
                AS home_external_rating_season,

            away_external.source_season
                AS away_external_rating_season,

            home_external.source_week
                AS home_external_rating_week,

            away_external.source_week
                AS away_external_rating_week,

            exact_external.nfelo_home_probability_open
                AS published_nfelo_home_probability,

            exact_external.normalized_game_id
                IS NOT NULL
                AS external_game_available,

            home_qb.qb_rating
                AS home_listed_qb_rating,

            away_qb.qb_rating
                AS away_listed_qb_rating,

            home_qb.rating_standard_error
                AS home_listed_qb_rating_standard_error,

            away_qb.rating_standard_error
                AS away_listed_qb_rating_standard_error,

            home_qb.as_of_date
                AS home_qb_rating_as_of,

            away_qb.as_of_date
                AS away_qb_rating_as_of,

            injury.has_complete_injury_data,

            injury
                .offense_injury_burden_difference,

            injury
                .defense_injury_burden_difference,

            injury
                .special_teams_injury_burden_difference

        FROM {SCHEDULE_FULL_NAME}
            AS schedule

        LEFT JOIN {ELO_RATINGS_FULL_NAME}
            AS home_elo
            ON schedule.home_team
                = home_elo.team

        LEFT JOIN {ELO_RATINGS_FULL_NAME}
            AS away_elo
            ON schedule.away_team
                = away_elo.team

        LEFT JOIN {QB_RATINGS_FULL_NAME}
            AS home_qb
            ON schedule.home_qb_id
                = home_qb.qb_id

        LEFT JOIN {QB_RATINGS_FULL_NAME}
            AS away_qb
            ON schedule.away_qb_id
                = away_qb.qb_id

        LEFT JOIN {INJURY_FEATURES_FULL_NAME}
            AS injury
            ON schedule.game_id
                = injury.game_id

        LEFT JOIN current_external_team_ratings
            AS home_external
            ON schedule.home_team
                = home_external.team

        LEFT JOIN current_external_team_ratings
            AS away_external
            ON schedule.away_team
                = away_external.team

        LEFT JOIN {EXTERNAL_NFELO_FULL_NAME}
            AS exact_external
            ON schedule.game_id
                = exact_external.normalized_game_id

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

    internal_elo_columns = (
        "home_elo_rating",
        "away_elo_rating",
        "home_rating_season",
        "away_rating_season",
        "home_rating_as_of",
        "away_rating_as_of",
    )

    external_snapshot_columns = (
        "home_external_nfelo_rating",
        "away_external_nfelo_rating",
        EXTERNAL_ELO_FEATURE,
        "home_external_nfelo_qb_adjustment",
        "away_external_nfelo_qb_adjustment",
        EXTERNAL_QB_FEATURE,
        "home_external_rating_season",
        "away_external_rating_season",
        "home_external_rating_week",
        "away_external_rating_week",
    )

    if not upcoming_games.empty:
        missing_internal_mask = upcoming_games.loc[
            :,
            internal_elo_columns,
        ].isna().any(axis=1)

        if missing_internal_mask.any():
            missing_game_ids = ", ".join(
                upcoming_games.loc[
                    missing_internal_mask,
                    "game_id",
                ].astype(str)
            )

            raise RuntimeError(
                "Upcoming production games are missing "
                "current internal Elo ratings: "
                f"{missing_game_ids}"
            )

        missing_external_mask = upcoming_games.loc[
            :,
            external_snapshot_columns,
        ].isna().any(axis=1)

        if missing_external_mask.any():
            missing_game_ids = ", ".join(
                upcoming_games.loc[
                    missing_external_mask,
                    "game_id",
                ].astype(str)
            )

            raise RuntimeError(
                "Upcoming production games are missing "
                "external nfelo team snapshots: "
                f"{missing_game_ids}"
            )

        available_probability = upcoming_games.loc[
            upcoming_games[
                "external_game_available"
            ].fillna(False).astype(bool),
            "published_nfelo_home_probability",
        ]

        invalid_probability_mask = (
            available_probability.isna()
            | available_probability.le(0.0)
            | available_probability.ge(1.0)
        )

        if invalid_probability_mask.any():
            invalid_game_ids = ", ".join(
                upcoming_games.loc[
                    invalid_probability_mask.index[
                        invalid_probability_mask
                    ],
                    "game_id",
                ].astype(str)
            )

            raise RuntimeError(
                "Available external games contain "
                "invalid published nfelo probabilities: "
                f"{invalid_game_ids}"
            )

        unavailable_probability_mask = (
            ~upcoming_games[
                "external_game_available"
            ].fillna(False).astype(bool)
            & upcoming_games[
                "published_nfelo_home_probability"
            ].notna()
        )

        if unavailable_probability_mask.any():
            invalid_game_ids = ", ".join(
                upcoming_games.loc[
                    unavailable_probability_mask,
                    "game_id",
                ].astype(str)
            )

            raise RuntimeError(
                "Unavailable external games unexpectedly "
                "contain published probabilities: "
                f"{invalid_game_ids}"
            )

    exact_game_count = int(
        upcoming_games[
            "external_game_available"
        ].fillna(False).sum()
    )

    logger.info(
        "Current production inputs loaded: %s games, "
        "%s with exact external nfelo game data.",
        len(upcoming_games),
        exact_game_count,
    )

    return upcoming_games


def load_production_training_data(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """
    Load historical rows for both production models.

    The same external source provides the primary
    external Elo and QB features and the fallback feature
    pair. Historical coverage is required before fitting.
    """

    historical_data = connection.execute(
        f"""
        SELECT
            dataset.game_id,
            dataset.season,
            dataset.game_date,
            dataset.{TARGET_COLUMN},
            dataset.has_complete_injury_data,
            dataset.{LISTED_QB_FEATURE},

            dataset
                .offense_injury_burden_difference,

            dataset
                .defense_injury_burden_difference,

            dataset
                .special_teams_injury_burden_difference,

            external.starting_nfelo_home
                - external.starting_nfelo_away
                AS {EXTERNAL_ELO_FEATURE},

            external.home_538_qb_adj
                - external.away_538_qb_adj
                AS {EXTERNAL_QB_FEATURE},

            external.nfelo_home_probability_open
                AS published_nfelo_home_probability

        FROM {MODELING_DATASET_FULL_NAME}
            AS dataset

        LEFT JOIN {EXTERNAL_NFELO_FULL_NAME}
            AS external
            ON dataset.game_id
                = external.normalized_game_id

        WHERE dataset.season
            < {
                PRODUCTION_PROBABILITY_MODEL
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
            "Production training source is empty."
        )

    required_external_columns = (
        EXTERNAL_ELO_FEATURE,
        EXTERNAL_QB_FEATURE,
        "published_nfelo_home_probability",
    )

    missing_external_mask = historical_data.loc[
        :,
        required_external_columns,
    ].isna().any(axis=1)

    if missing_external_mask.any():
        missing_game_ids = ", ".join(
            historical_data.loc[
                missing_external_mask,
                "game_id",
            ].astype(str)
        )

        raise RuntimeError(
            "Production training games are missing "
            "external nfelo inputs: "
            f"{missing_game_ids}"
        )

    published_probability = historical_data[
        "published_nfelo_home_probability"
    ]

    invalid_probability_mask = (
        published_probability.le(0.0)
        | published_probability.ge(1.0)
    )

    if invalid_probability_mask.any():
        invalid_game_ids = ", ".join(
            historical_data.loc[
                invalid_probability_mask,
                "game_id",
            ].astype(str)
        )

        raise RuntimeError(
            "Production training games contain invalid "
            "published nfelo probabilities: "
            f"{invalid_game_ids}"
        )

    logger.info(
        "Production training data loaded: %s "
        "historical games.",
        len(historical_data),
    )

    return historical_data