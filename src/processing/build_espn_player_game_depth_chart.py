"""
NFL Analytics Platform
ESPN Player-Game Depth-Chart Builder

Purpose:
    Build one normalized ESPN depth-chart role per player,
    team and scheduled game using the latest timestamped
    snapshot available no later than the game date.

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
DEPTH_SOURCE_TABLE = "depth_charts_espn"
DEPTH_SOURCE_FULL_NAME = (
    f"{DEPTH_SOURCE_SCHEMA}.{DEPTH_SOURCE_TABLE}"
)

SCHEDULE_SCHEMA = "processed"
SCHEDULE_TABLE = "schedule"
SCHEDULE_FULL_NAME = (
    f"{SCHEDULE_SCHEMA}.{SCHEDULE_TABLE}"
)

TARGET_SCHEMA = "processed"
TARGET_TABLE = "player_game_depth_chart_espn"
TARGET_FULL_NAME = (
    f"{TARGET_SCHEMA}.{TARGET_TABLE}"
)

FIRST_ESPN_SEASON = 2025
CURRENT_ESPN_SEASON = 2026

REQUIRED_DEPTH_COLUMNS = {
    "source_season",
    "dt",
    "team",
    "player_name",
    "espn_id",
    "gsis_id",
    "pos_grp",
    "pos_name",
    "pos_abb",
    "pos_slot",
    "pos_rank",
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
    """Return available columns for one DuckDB table."""

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
    """Validate one required source table."""

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
    """Validate ESPN depth-chart and schedule sources."""

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
        "ESPN depth-chart sources validated: %s and %s.",
        DEPTH_SOURCE_FULL_NAME,
        SCHEDULE_FULL_NAME,
    )


def create_espn_player_game_depth_chart(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create the timestamped ESPN player-game table."""

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
                {FIRST_ESPN_SEASON}
                AND {CURRENT_ESPN_SEASON}

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
                {FIRST_ESPN_SEASON}
                AND {CURRENT_ESPN_SEASON}
        ),
        parsed_depth_chart AS (
            SELECT
                source_season,
                CAST(dt AS TIMESTAMPTZ)
                    AS snapshot_at,
                team,
                player_name,
                espn_id,
                gsis_id,
                pos_grp,
                pos_name,
                pos_abb,
                pos_slot,
                pos_rank
            FROM {DEPTH_SOURCE_FULL_NAME}
        ),
        latest_team_snapshots AS (
            SELECT
                schedule.game_id,
                schedule.season,
                schedule.game_type,
                schedule.week,
                schedule.gameday,
                schedule.team,
                schedule.opponent,
                schedule.is_home,
                MAX(depth.snapshot_at)
                    AS source_snapshot_at
            FROM scheduled_team_games AS schedule
            LEFT JOIN parsed_depth_chart AS depth
                ON schedule.season
                    = depth.source_season
               AND schedule.team = depth.team
               AND CAST(
                    depth.snapshot_at AS DATE
                   ) <= schedule.gameday
            GROUP BY
                schedule.game_id,
                schedule.season,
                schedule.game_type,
                schedule.week,
                schedule.gameday,
                schedule.team,
                schedule.opponent,
                schedule.is_home
        ),
        matched_source_rows AS (
            SELECT
                snapshot.game_id,
                snapshot.season,
                snapshot.game_type,
                snapshot.week,
                snapshot.gameday,
                snapshot.team,
                snapshot.opponent,
                snapshot.is_home,
                COALESCE(
                    depth.gsis_id,
                    'ESPN:' || depth.espn_id
                ) AS player_key,
                depth.gsis_id,
                depth.espn_id,
                NULLIF(
                    TRIM(
                        CAST(depth.player_name AS VARCHAR)
                    ),
                    ''
                ) AS player_name,
                NULLIF(
                    TRIM(
                        CAST(depth.pos_abb AS VARCHAR)
                    ),
                    ''
                ) AS player_position,
                CASE
                    WHEN depth.pos_grp
                        = 'Special Teams'
                        THEN 'Special Teams'
                    WHEN depth.pos_grp
                        LIKE 'Base % D'
                        THEN 'Defense'
                    ELSE 'Offense'
                END AS formation,
                NULLIF(
                    TRIM(
                        CAST(depth.pos_name AS VARCHAR)
                    ),
                    ''
                ) AS depth_position,
                depth.pos_slot,
                depth.pos_rank
                    AS source_depth_rank,
                snapshot.source_snapshot_at
            FROM latest_team_snapshots AS snapshot
            INNER JOIN parsed_depth_chart AS depth
                ON snapshot.season
                    = depth.source_season
               AND snapshot.team = depth.team
               AND snapshot.source_snapshot_at
                    = depth.snapshot_at
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
            player_key,
            ANY_VALUE(gsis_id) AS gsis_id,
            ANY_VALUE(espn_id) AS espn_id,
            ANY_VALUE(player_name)
                AS player_name,
            ANY_VALUE(player_position)
                AS player_position,
            formation,
            depth_position,
            pos_slot,
            MIN(source_depth_rank)
                AS depth_rank,
            MIN(source_depth_rank) = 1
                AS is_starter,
            MIN(source_depth_rank) = 2
                AS is_primary_backup,
            MIN(source_depth_rank) >= 3
                AS is_reserve,
            COUNT(*) AS source_record_count,
            COUNT(
                DISTINCT source_depth_rank
            ) AS source_rank_count,
            COUNT(
                DISTINCT source_depth_rank
            ) > 1 AS has_conflicting_ranks,
            'espn'::VARCHAR
                AS source_generation,
            source_snapshot_at
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
            player_key,
            formation,
            depth_position,
            pos_slot,
            source_snapshot_at
        """
    )


def count_scheduled_team_games(
    connection: duckdb.DuckDBPyConnection,
) -> int:
    """Count ESPN-period scheduled team-game sides."""

    return connection.execute(
        f"""
        SELECT COUNT(*) * 2
        FROM {SCHEDULE_FULL_NAME}
        WHERE season BETWEEN
            {FIRST_ESPN_SEASON}
            AND {CURRENT_ESPN_SEASON}
        """
    ).fetchone()[0]


def validate_target_table(
    connection: duckdb.DuckDBPyConnection,
) -> tuple[int, int, int, int]:
    """Validate the ESPN player-game depth-chart table."""

    row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    if row_count == 0:
        raise RuntimeError(
            "ESPN player-game depth-chart table is empty."
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
                depth_position,
                pos_slot
            FROM {TARGET_FULL_NAME}
            GROUP BY
                game_id,
                team,
                player_key,
                formation,
                depth_position,
                pos_slot
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_count > 0:
        raise RuntimeError(
            "ESPN player-game depth-chart table "
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
           OR pos_slot IS NULL
           OR source_snapshot_at IS NULL
        """
    ).fetchone()[0]

    if null_key_count > 0:
        raise RuntimeError(
            "ESPN player-game depth-chart table "
            "contains null business keys."
        )

    invalid_rank_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE depth_rank < 1
           OR source_rank_count < 1
           OR source_record_count < 1
        """
    ).fetchone()[0]

    if invalid_rank_count > 0:
        raise RuntimeError(
            "ESPN player-game depth-chart table "
            "contains invalid depth ranks."
        )

    future_snapshot_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE CAST(
                source_snapshot_at AS DATE
              ) > gameday
        """
    ).fetchone()[0]

    if future_snapshot_count > 0:
        raise RuntimeError(
            "ESPN player-game depth-chart table "
            "contains post-game-date snapshots."
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
            "ESPN depth-chart coverage does not match "
            "scheduled team-games: "
            f"{covered_team_game_count} covered != "
            f"{scheduled_team_game_count} scheduled."
        )

    missing_gsis_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE gsis_id IS NULL
        """
    ).fetchone()[0]

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
        missing_gsis_count,
        conflicting_role_count,
    )


def build_espn_player_game_depth_chart(
    database_file: Path = DATABASE_FILE,
) -> None:
    """Build the ESPN processed depth-chart table."""

    validate_database_file(
        database_file
    )

    logger.info(
        "Starting ESPN player-game depth-chart build..."
    )

    try:
        with duckdb.connect(
            str(database_file)
        ) as connection:
            validate_source_tables(
                connection
            )

            create_espn_player_game_depth_chart(
                connection
            )

            (
                row_count,
                covered_team_game_count,
                missing_gsis_count,
                conflicting_role_count,
            ) = validate_target_table(
                connection
            )

    except Exception:
        logger.exception(
            "ESPN player-game depth-chart build failed."
        )
        raise

    logger.info(
        "ESPN player-game depth chart validated: "
        "%s role rows across %s team-games.",
        row_count,
        covered_team_game_count,
    )
    logger.info(
        "ESPN role rows without GSIS ID: %s.",
        missing_gsis_count,
    )
    logger.info(
        "ESPN roles consolidated from conflicting ranks: %s.",
        conflicting_role_count,
    )
    logger.info(
        "ESPN player-game depth-chart build "
        "completed successfully."
    )


def main() -> None:
    """Run the ESPN player-game depth-chart builder."""

    build_espn_player_game_depth_chart()


if __name__ == "__main__":
    main()