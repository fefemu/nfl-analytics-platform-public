"""
NFL Analytics Platform
Rolling Team Features Builder

Purpose:
    Build leakage-safe pregame rolling team features
    from team-game efficiency data.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging
from pathlib import Path

import duckdb


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_FILE = PROJECT_ROOT / "data" / "nfl_analytics.duckdb"

SOURCE_SCHEMA = "processed"
SOURCE_TABLE = "team_game_efficiency"

TARGET_SCHEMA = "analytics"
TARGET_TABLE = "rolling_team_features"

SOURCE_FULL_NAME = f"{SOURCE_SCHEMA}.{SOURCE_TABLE}"
TARGET_FULL_NAME = f"{TARGET_SCHEMA}.{TARGET_TABLE}"

SHORT_WINDOW = 4
LONG_WINDOW = 8

REQUIRED_SOURCE_COLUMNS = {
    "game_id",
    "season",
    "season_type",
    "week",
    "game_date",
    "team",
    "opponent",
    "is_home",
    "offensive_plays",
    "points_scored",
    "points_allowed",
    "offensive_epa_per_play",
    "competitive_epa_per_play",
    "dropback_epa_per_play",
    "designed_rush_epa_per_play",
    "early_down_epa_per_play",
    "success_rate",
    "dropback_success_rate",
    "designed_rush_success_rate",
    "explosive_play_rate",
    "sack_rate",
    "turnover_rate",
    "defensive_epa_allowed_per_play",
    "competitive_defensive_epa_allowed_per_play",
    "defensive_success_rate_allowed",
    "explosive_play_rate_allowed",
    "sack_rate_generated",
    "turnover_rate_generated",
}


ROLLING_METRICS = (
    "offensive_plays",
    "points_scored",
    "points_allowed",
    "offensive_epa_per_play",
    "competitive_epa_per_play",
    "dropback_epa_per_play",
    "designed_rush_epa_per_play",
    "early_down_epa_per_play",
    "success_rate",
    "explosive_play_rate",
    "sack_rate",
    "turnover_rate",
    "defensive_epa_allowed_per_play",
    "competitive_defensive_epa_allowed_per_play",
    "defensive_success_rate_allowed",
    "explosive_play_rate_allowed",
    "sack_rate_generated",
    "turnover_rate_generated",
)


def build_rolling_average_expressions(
    window_size: int,
) -> str:
    """Build leakage-safe SQL rolling-average expressions."""

    if window_size <= 0:
        raise ValueError(
            "Rolling window size must be positive."
        )

    expressions = []

    for metric in ROLLING_METRICS:
        expressions.append(
            f"""
            AVG({metric}) OVER (
                PARTITION BY team, season
                ORDER BY
                    game_date,
                    game_id
                ROWS BETWEEN {window_size} PRECEDING
                         AND 1 PRECEDING
            ) AS pregame_{metric}_last_{window_size}
            """.strip()
        )

    return ",\n            ".join(expressions)


def validate_database_file(
    database_file: Path = DATABASE_FILE,
) -> None:
    """Validate that the DuckDB database file exists."""

    if not database_file.exists():
        raise FileNotFoundError(
            f"Database file does not exist: {database_file}"
        )

    if not database_file.is_file():
        raise RuntimeError(
            f"Database path is not a file: {database_file}"
        )

    logger.info(
        "Database file validated: %s",
        database_file,
    )


def validate_source_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate the team-game efficiency source table."""

    table_exists = connection.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = ?
          AND table_name = ?
        """,
        [SOURCE_SCHEMA, SOURCE_TABLE],
    ).fetchone()[0]

    if table_exists == 0:
        raise RuntimeError(
            f"Source table does not exist: {SOURCE_FULL_NAME}"
        )

    available_columns = {
        row[0]
        for row in connection.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = ?
              AND table_name = ?
            """,
            [SOURCE_SCHEMA, SOURCE_TABLE],
        ).fetchall()
    }

    missing_columns = sorted(
        REQUIRED_SOURCE_COLUMNS - available_columns
    )

    if missing_columns:
        missing_names = ", ".join(missing_columns)
        raise RuntimeError(
            "Missing required team-game efficiency columns: "
            f"{missing_names}"
        )

    source_row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {SOURCE_FULL_NAME}
        """
    ).fetchone()[0]

    if source_row_count == 0:
        raise RuntimeError(
            f"Source table is empty: {SOURCE_FULL_NAME}"
        )

    logger.info(
        "Rolling feature source validated: %s rows in %s.",
        source_row_count,
        SOURCE_FULL_NAME,
    )


def create_rolling_team_features_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create leakage-safe rolling team features."""

    short_expressions = build_rolling_average_expressions(
        SHORT_WINDOW
    )
    long_expressions = build_rolling_average_expressions(
        LONG_WINDOW
    )

    connection.execute(
        f"""
        CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA};

        CREATE OR REPLACE TABLE {TARGET_FULL_NAME} AS
        SELECT
            game_id,
            season,
            season_type,
            week,
            game_date,
            team,
            opponent,
            is_home,

            ROW_NUMBER() OVER (
                PARTITION BY team, season
                ORDER BY
                    game_date,
                    game_id
            ) - 1 AS season_games_played_before,

            COUNT(*) OVER (
                PARTITION BY team, season
                ORDER BY
                    game_date,
                    game_id
                ROWS BETWEEN {SHORT_WINDOW} PRECEDING
                         AND 1 PRECEDING
            ) AS short_window_games,

            COUNT(*) OVER (
                PARTITION BY team, season
                ORDER BY
                    game_date,
                    game_id
                ROWS BETWEEN {LONG_WINDOW} PRECEDING
                         AND 1 PRECEDING
            ) AS long_window_games,

            {short_expressions},

            {long_expressions}

        FROM {SOURCE_FULL_NAME}
        """
    )

    row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    logger.info(
        "Rolling team features created: %s rows in %s.",
        row_count,
        TARGET_FULL_NAME,
    )


def validate_rolling_team_features(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate rolling team feature structure and window sizes."""

    target_row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    source_row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {SOURCE_FULL_NAME}
        """
    ).fetchone()[0]

    if target_row_count == 0:
        raise RuntimeError(
            f"Target table is empty: {TARGET_FULL_NAME}"
        )

    if target_row_count != source_row_count:
        raise RuntimeError(
            "Rolling feature row count does not match source: "
            f"target={target_row_count}, "
            f"source={source_row_count}"
        )

    duplicate_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT
                game_id,
                team
            FROM {TARGET_FULL_NAME}
            GROUP BY
                game_id,
                team
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_count > 0:
        raise RuntimeError(
            "Duplicate rolling feature business keys found: "
            f"{duplicate_count}"
        )

    invalid_window_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE season_games_played_before < 0
           OR short_window_games
              <> LEAST(
                    season_games_played_before,
                    {SHORT_WINDOW}
                 )
           OR long_window_games
              <> LEAST(
                    season_games_played_before,
                    {LONG_WINDOW}
                 )
        """
    ).fetchone()[0]

    if invalid_window_count > 0:
        raise RuntimeError(
            "Invalid rolling window sizes found: "
            f"{invalid_window_count}"
        )

    invalid_first_game_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE season_games_played_before = 0
          AND (
                short_window_games <> 0
                OR long_window_games <> 0
                OR pregame_offensive_plays_last_4
                IS NOT NULL
                OR pregame_offensive_plays_last_8
                IS NOT NULL
                OR pregame_points_scored_last_4
                IS NOT NULL
                OR pregame_points_scored_last_8
                IS NOT NULL
                OR pregame_points_allowed_last_4
                IS NOT NULL
                OR pregame_points_allowed_last_8
                IS NOT NULL
                OR pregame_offensive_epa_per_play_last_4
                   IS NOT NULL
                OR pregame_offensive_epa_per_play_last_8
                   IS NOT NULL
          )
        """
    ).fetchone()[0]

    if invalid_first_game_count > 0:
        raise RuntimeError(
            "First games contain unexpected historical features: "
            f"{invalid_first_game_count}"
        )

    forbidden_columns = {
        "points_scored",
        "points_allowed",
        "point_differential",
        "team_win",
        "is_tie",
    }

    available_columns = {
        row[0]
        for row in connection.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = ?
              AND table_name = ?
            """,
            [TARGET_SCHEMA, TARGET_TABLE],
        ).fetchall()
    }

    leaked_columns = sorted(
        forbidden_columns & available_columns
    )

    if leaked_columns:
        leaked_names = ", ".join(leaked_columns)
        raise RuntimeError(
            "Postgame result columns leaked into rolling features: "
            f"{leaked_names}"
        )

    logger.info(
        "Rolling team features validated successfully: %s rows.",
        target_row_count,
    )


def build_rolling_team_features(
    database_file: Path = DATABASE_FILE,
) -> None:
    """Build and validate leakage-safe rolling team features."""

    validate_database_file(database_file)

    logger.info(
        "Starting rolling team features build..."
    )

    with duckdb.connect(str(database_file)) as connection:
        validate_source_table(connection)

        connection.execute("BEGIN TRANSACTION")

        try:
            create_rolling_team_features_table(connection)
            validate_rolling_team_features(connection)

            connection.execute("COMMIT")

            logger.info(
                "Rolling team features transaction committed."
            )

        except Exception:
            connection.execute("ROLLBACK")

            logger.exception(
                "Rolling team features build failed; "
                "transaction rolled back."
            )
            raise

    logger.info(
        "Rolling team features build completed: %s.",
        TARGET_FULL_NAME,
    )


def main() -> None:
    """Run the rolling team features builder."""

    try:
        build_rolling_team_features()
    except Exception:
        logger.exception(
            "Rolling team features builder failed."
        )
        raise


if __name__ == "__main__":
    main()


