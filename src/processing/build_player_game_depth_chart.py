"""
NFL Analytics Platform
Unified Player-Game Depth-Chart Builder

Purpose:
    Combine legacy NFL and timestamped ESPN player-game
    depth charts into one source-independent business table.

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

LEGACY_SCHEMA = "processed"
LEGACY_TABLE = "player_game_depth_chart_legacy"
LEGACY_FULL_NAME = (
    f"{LEGACY_SCHEMA}.{LEGACY_TABLE}"
)

ESPN_SCHEMA = "processed"
ESPN_TABLE = "player_game_depth_chart_espn"
ESPN_FULL_NAME = (
    f"{ESPN_SCHEMA}.{ESPN_TABLE}"
)

TARGET_SCHEMA = "processed"
TARGET_TABLE = "player_game_depth_chart"
TARGET_FULL_NAME = (
    f"{TARGET_SCHEMA}.{TARGET_TABLE}"
)

COMMON_REQUIRED_COLUMNS = {
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "team",
    "opponent",
    "is_home",
    "player_key",
    "gsis_id",
    "espn_id",
    "player_name",
    "player_position",
    "formation",
    "depth_position",
    "depth_rank",
    "is_starter",
    "is_primary_backup",
    "is_reserve",
    "source_record_count",
    "source_rank_count",
    "has_conflicting_ranks",
    "source_generation",
    "source_snapshot_at",
}

ESPN_ONLY_REQUIRED_COLUMNS = {
    "pos_slot",
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
    """Return available columns for one source table."""

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
    """Validate one processed depth-chart source."""

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
    """Validate legacy and ESPN processed sources."""

    validate_source_table(
        connection=connection,
        schema_name=LEGACY_SCHEMA,
        table_name=LEGACY_TABLE,
        required_columns=COMMON_REQUIRED_COLUMNS,
    )
    validate_source_table(
        connection=connection,
        schema_name=ESPN_SCHEMA,
        table_name=ESPN_TABLE,
        required_columns=(
            COMMON_REQUIRED_COLUMNS
            | ESPN_ONLY_REQUIRED_COLUMNS
        ),
    )

    logger.info(
        "Unified depth-chart sources validated: %s and %s.",
        LEGACY_FULL_NAME,
        ESPN_FULL_NAME,
    )


def create_player_game_depth_chart(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create the unified player-game depth-chart table."""

    connection.execute(
        f"""
        CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA}
        """
    )

    connection.execute(
        f"""
        CREATE OR REPLACE TABLE {TARGET_FULL_NAME} AS
        WITH combined_depth_charts AS (
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
                gsis_id,
                espn_id,
                player_name,
                player_position,
                formation,
                depth_position,
                CAST(NULL AS INTEGER)
                    AS position_slot,
                depth_rank,
                is_starter,
                is_primary_backup,
                is_reserve,
                source_record_count,
                source_rank_count,
                has_conflicting_ranks,
                source_generation,
                source_snapshot_at
            FROM {LEGACY_FULL_NAME}

            UNION ALL

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
                gsis_id,
                espn_id,
                player_name,
                player_position,
                formation,
                depth_position,
                pos_slot AS position_slot,
                depth_rank,
                is_starter,
                is_primary_backup,
                is_reserve,
                source_record_count,
                source_rank_count,
                has_conflicting_ranks,
                source_generation,
                source_snapshot_at
            FROM {ESPN_FULL_NAME}
        )
        SELECT
            *,
            gsis_id IS NOT NULL
                AS has_gsis_id,
            CASE
                WHEN gsis_id IS NOT NULL
                    THEN 'GSIS'
                ELSE 'ESPN'
            END AS player_identifier_source,
            CASE
                WHEN is_starter
                    THEN 'STARTER'
                WHEN is_primary_backup
                    THEN 'PRIMARY_BACKUP'
                ELSE 'RESERVE'
            END AS depth_tier,
            formation = 'Offense'
                AS is_offense_role,
            formation = 'Defense'
                AS is_defense_role,
            formation = 'Special Teams'
                AS is_special_teams_role,
            source_snapshot_at IS NOT NULL
                AS has_timestamped_snapshot
        FROM combined_depth_charts
        """
    )


def count_source_rows(
    connection: duckdb.DuckDBPyConnection,
) -> int:
    """Count rows across both processed sources."""

    return connection.execute(
        f"""
        SELECT
            (
                SELECT COUNT(*)
                FROM {LEGACY_FULL_NAME}
            )
            +
            (
                SELECT COUNT(*)
                FROM {ESPN_FULL_NAME}
            )
        """
    ).fetchone()[0]


def validate_target_table(
    connection: duckdb.DuckDBPyConnection,
    expected_row_count: int,
) -> tuple[int, int, int, int]:
    """Validate the unified player-game depth chart."""

    row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    if row_count != expected_row_count:
        raise RuntimeError(
            "Unified depth-chart row count does not "
            "match its processed sources: "
            f"{row_count} != {expected_row_count}."
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
                COALESCE(
                    position_slot,
                    -1
                ) AS position_slot_key
            FROM {TARGET_FULL_NAME}
            GROUP BY
                game_id,
                team,
                player_key,
                formation,
                depth_position,
                position_slot_key
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_count > 0:
        raise RuntimeError(
            "Unified player-game depth chart "
            "contains duplicate role records."
        )

    null_key_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE game_id IS NULL
           OR season IS NULL
           OR team IS NULL
           OR player_key IS NULL
           OR formation IS NULL
           OR depth_position IS NULL
           OR depth_rank IS NULL
           OR depth_tier IS NULL
        """
    ).fetchone()[0]

    if null_key_count > 0:
        raise RuntimeError(
            "Unified player-game depth chart "
            "contains null business keys."
        )

    invalid_flag_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE has_gsis_id
                IS DISTINCT FROM
                    (
                        gsis_id IS NOT NULL
                    )
           OR has_timestamped_snapshot
                IS DISTINCT FROM
                    (
                        source_snapshot_at IS NOT NULL
                    )
           OR is_offense_role
                IS DISTINCT FROM
                    (
                        formation = 'Offense'
                    )
           OR is_defense_role
                IS DISTINCT FROM
                    (
                        formation = 'Defense'
                    )
           OR is_special_teams_role
                IS DISTINCT FROM
                    (
                        formation = 'Special Teams'
                    )
        """
    ).fetchone()[0]

    if invalid_flag_count > 0:
        raise RuntimeError(
            "Unified player-game depth chart "
            "contains inconsistent derived flags."
        )

    team_game_count = connection.execute(
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

    missing_gsis_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE NOT has_gsis_id
        """
    ).fetchone()[0]

    starter_role_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE is_starter
        """
    ).fetchone()[0]

    return (
        row_count,
        team_game_count,
        missing_gsis_count,
        starter_role_count,
    )


def build_player_game_depth_chart(
    database_file: Path = DATABASE_FILE,
) -> None:
    """Build the unified processed depth-chart table."""

    validate_database_file(
        database_file
    )

    logger.info(
        "Starting unified player-game depth-chart build..."
    )

    try:
        with duckdb.connect(
            str(database_file)
        ) as connection:
            validate_source_tables(
                connection
            )

            expected_row_count = count_source_rows(
                connection
            )

            create_player_game_depth_chart(
                connection
            )

            (
                row_count,
                team_game_count,
                missing_gsis_count,
                starter_role_count,
            ) = validate_target_table(
                connection=connection,
                expected_row_count=expected_row_count,
            )

    except Exception:
        logger.exception(
            "Unified player-game depth-chart build failed."
        )
        raise

    logger.info(
        "Unified player-game depth chart validated: "
        "%s role rows across %s team-games.",
        row_count,
        team_game_count,
    )
    logger.info(
        "Unified role rows without GSIS ID: %s.",
        missing_gsis_count,
    )
    logger.info(
        "Unified starter-role rows: %s.",
        starter_role_count,
    )
    logger.info(
        "Unified player-game depth-chart build "
        "completed successfully."
    )


def main() -> None:
    """Run the unified player-game depth-chart builder."""

    build_player_game_depth_chart()


if __name__ == "__main__":
    main()