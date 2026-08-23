"""
NFL Analytics Platform
Legacy Player-Game Depth-Chart Builder

Purpose:
    Build one normalized weekly legacy depth-chart role
    per player, team and scheduled game for 2018-2024.

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

DEPTH_SOURCE_SCHEMA = "raw"
DEPTH_SOURCE_TABLE = "depth_charts_legacy"
DEPTH_SOURCE_FULL_NAME = (
    f"{DEPTH_SOURCE_SCHEMA}.{DEPTH_SOURCE_TABLE}"
)

SCHEDULE_SCHEMA = "processed"
SCHEDULE_TABLE = "schedule"
SCHEDULE_FULL_NAME = (
    f"{SCHEDULE_SCHEMA}.{SCHEDULE_TABLE}"
)

TARGET_SCHEMA = "processed"
TARGET_TABLE = "player_game_depth_chart_legacy"
TARGET_FULL_NAME = (
    f"{TARGET_SCHEMA}.{TARGET_TABLE}"
)

FIRST_LEGACY_SEASON = 2018
LAST_LEGACY_SEASON = 2024

REQUIRED_DEPTH_COLUMNS = {
    "season",
    "club_code",
    "week",
    "game_type",
    "depth_team",
    "formation",
    "gsis_id",
    "position",
    "depth_position",
    "full_name",
}

REQUIRED_SCHEDULE_COLUMNS = {
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "home_team",
    "away_team",
}


def validate_database_file(
    database_file: Path = DATABASE_FILE,
) -> None:
    """Validate the DuckDB database file."""

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


def get_table_columns(
    connection: duckdb.DuckDBPyConnection,
    schema_name: str,
    table_name: str,
) -> set[str]:
    """Return available columns for one table."""

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


def validate_source_table(
    connection: duckdb.DuckDBPyConnection,
    schema_name: str,
    table_name: str,
    required_columns: set[str],
) -> None:
    """Validate one source table and its columns."""

    full_name = f"{schema_name}.{table_name}"

    table_exists = connection.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = ?
          AND table_name = ?
        """,
        [
            schema_name,
            table_name,
        ],
    ).fetchone()[0]

    if table_exists == 0:
        raise RuntimeError(
            f"Source table does not exist: {full_name}"
        )

    available_columns = get_table_columns(
        connection=connection,
        schema_name=schema_name,
        table_name=table_name,
    )

    missing_columns = sorted(
        required_columns - available_columns
    )

    if missing_columns:
        raise RuntimeError(
            f"Source table {full_name} is missing columns: "
            + ", ".join(missing_columns)
        )


def validate_source_tables(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate legacy depth-chart and schedule sources."""

    validate_source_table(
        connection=connection,
        schema_name=DEPTH_SOURCE_SCHEMA,
        table_name=DEPTH_SOURCE_TABLE,
        required_columns=REQUIRED_DEPTH_COLUMNS,
    )
    validate_source_table(
        connection=connection,
        schema_name=SCHEDULE_SCHEMA,
        table_name=SCHEDULE_TABLE,
        required_columns=REQUIRED_SCHEDULE_COLUMNS,
    )

    logger.info(
        "Legacy depth-chart sources validated: %s and %s.",
        DEPTH_SOURCE_FULL_NAME,
        SCHEDULE_FULL_NAME,
    )


def create_legacy_player_game_depth_chart(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create the normalized legacy player-game table."""

    connection.execute(
        f"""
        CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA}
        """
    )

    connection.execute(
        f"""
        CREATE OR REPLACE TABLE {TARGET_FULL_NAME} AS
        WITH scheduled_team_games AS (
            SELECT
                game_id,
                season,
                game_type,
                week,
                CAST(gameday AS DATE) AS gameday,
                home_team AS team,
                away_team AS opponent,
                TRUE AS is_home
            FROM {SCHEDULE_FULL_NAME}
            WHERE season BETWEEN
                {FIRST_LEGACY_SEASON}
                AND {LAST_LEGACY_SEASON}

            UNION ALL

            SELECT
                game_id,
                season,
                game_type,
                week,
                CAST(gameday AS DATE) AS gameday,
                away_team AS team,
                home_team AS opponent,
                FALSE AS is_home
            FROM {SCHEDULE_FULL_NAME}
            WHERE season BETWEEN
                {FIRST_LEGACY_SEASON}
                AND {LAST_LEGACY_SEASON}
        ),
        matched_source_rows AS (
            SELECT
                schedule.game_id,
                schedule.season,
                schedule.game_type,
                schedule.week,
                schedule.gameday,
                schedule.team,
                schedule.opponent,
                schedule.is_home,
                depth.gsis_id,
                NULLIF(
                    TRIM(
                        CAST(depth.full_name AS VARCHAR)
                    ),
                    ''
                ) AS player_name,
                NULLIF(
                    TRIM(
                        CAST(depth.position AS VARCHAR)
                    ),
                    ''
                ) AS player_position,
                NULLIF(
                    TRIM(
                        CAST(depth.formation AS VARCHAR)
                    ),
                    ''
                ) AS formation,
                NULLIF(
                    TRIM(
                        CAST(
                            depth.depth_position
                            AS VARCHAR
                        )
                    ),
                    ''
                ) AS depth_position,
                CAST(
                    depth.depth_team AS INTEGER
                ) AS source_depth_rank
            FROM {DEPTH_SOURCE_FULL_NAME} AS depth
            INNER JOIN scheduled_team_games AS schedule
                ON depth.season = schedule.season
               AND depth.game_type = schedule.game_type
               AND depth.week = schedule.week
               AND depth.club_code = schedule.team
        )
        SELECT
            game_id,
            season,
            game_type,
            week,
            gameday,
            team,
            opponent,
            is_home,
            gsis_id AS player_key,
            gsis_id,
            CAST(NULL AS VARCHAR) AS espn_id,
            ANY_VALUE(player_name) AS player_name,
            ANY_VALUE(player_position)
                AS player_position,
            formation,
            depth_position,
            MIN(source_depth_rank)
                AS depth_rank,
            MIN(source_depth_rank) = 1
                AS is_starter,
            MIN(source_depth_rank) = 2
                AS is_primary_backup,
            MIN(source_depth_rank) = 3
                AS is_reserve,
            COUNT(*) AS source_record_count,
            COUNT(
                DISTINCT source_depth_rank
            ) AS source_rank_count,
            COUNT(
                DISTINCT source_depth_rank
            ) > 1 AS has_conflicting_ranks,
            'legacy_nfl'::VARCHAR
                AS source_generation,
            CAST(NULL AS TIMESTAMPTZ)
                AS source_snapshot_at
        FROM matched_source_rows
        GROUP BY
            game_id,
            season,
            game_type,
            week,
            gameday,
            team,
            opponent,
            is_home,
            gsis_id,
            formation,
            depth_position
        """
    )


def count_scheduled_team_games(
    connection: duckdb.DuckDBPyConnection,
) -> int:
    """Count legacy-period scheduled team-game sides."""

    return connection.execute(
        f"""
        SELECT COUNT(*) * 2
        FROM {SCHEDULE_FULL_NAME}
        WHERE season BETWEEN
            {FIRST_LEGACY_SEASON}
            AND {LAST_LEGACY_SEASON}
        """
    ).fetchone()[0]


def validate_target_table(
    connection: duckdb.DuckDBPyConnection,
) -> tuple[int, int, int]:
    """Validate the legacy player-game depth-chart table."""

    row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    if row_count == 0:
        raise RuntimeError(
            "Legacy player-game depth-chart table is empty."
        )

    duplicate_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT
                game_id,
                team,
                player_key,
                formation,
                depth_position
            FROM {TARGET_FULL_NAME}
            GROUP BY
                game_id,
                team,
                player_key,
                formation,
                depth_position
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_count > 0:
        raise RuntimeError(
            "Legacy player-game depth-chart table "
            "contains duplicate role records."
        )

    null_key_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE game_id IS NULL
           OR team IS NULL
           OR player_key IS NULL
           OR formation IS NULL
           OR depth_position IS NULL
        """
    ).fetchone()[0]

    if null_key_count > 0:
        raise RuntimeError(
            "Legacy player-game depth-chart table "
            "contains null business keys."
        )

    invalid_rank_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE depth_rank NOT BETWEEN 1 AND 3
           OR source_rank_count < 1
           OR source_record_count < 1
        """
    ).fetchone()[0]

    if invalid_rank_count > 0:
        raise RuntimeError(
            "Legacy player-game depth-chart table "
            "contains invalid depth ranks."
        )

    scheduled_team_game_count = (
        count_scheduled_team_games(
            connection
        )
    )

    covered_team_game_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT DISTINCT
                game_id,
                team
            FROM {TARGET_FULL_NAME}
        )
        """
    ).fetchone()[0]

    if (
        covered_team_game_count
        != scheduled_team_game_count
    ):
        raise RuntimeError(
            "Legacy depth-chart coverage does not match "
            "scheduled team-games: "
            f"{covered_team_game_count} covered != "
            f"{scheduled_team_game_count} scheduled."
        )

    conflicting_role_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE has_conflicting_ranks
        """
    ).fetchone()[0]

    return (
        row_count,
        covered_team_game_count,
        conflicting_role_count,
    )


def build_legacy_player_game_depth_chart(
    database_file: Path = DATABASE_FILE,
) -> None:
    """Build the legacy processed depth-chart table."""

    validate_database_file(
        database_file
    )

    logger.info(
        "Starting legacy player-game depth-chart build..."
    )

    try:
        with duckdb.connect(
            str(database_file)
        ) as connection:
            validate_source_tables(
                connection
            )

            create_legacy_player_game_depth_chart(
                connection
            )

            (
                row_count,
                covered_team_game_count,
                conflicting_role_count,
            ) = validate_target_table(
                connection
            )

    except Exception:
        logger.exception(
            "Legacy player-game depth-chart build failed."
        )
        raise

    logger.info(
        "Legacy player-game depth chart validated: "
        "%s role rows across %s team-games.",
        row_count,
        covered_team_game_count,
    )
    logger.info(
        "Legacy roles consolidated from conflicting ranks: %s.",
        conflicting_role_count,
    )
    logger.info(
        "Legacy player-game depth-chart build "
        "completed successfully."
    )


def main() -> None:
    """Run the legacy player-game depth-chart builder."""

    build_legacy_player_game_depth_chart()


if __name__ == "__main__":
    main()