"""
NFL Analytics Platform
Player-Game Injury Context Builder

Purpose:
    Combine final player-game injury status, pregame
    depth-chart importance and strictly prior snap-share
    history into one leakage-safe player context table.

Grain:
    One row per injury-report player, team and game.

Leakage boundary:
    Snap history is eligible only when its availability date
    is strictly earlier than the injury game's scheduled date.

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
DATABASE_FILE = (
    PROJECT_ROOT
    / "data"
    / "nfl_analytics.duckdb"
)

INJURY_SCHEMA = "processed"
INJURY_TABLE = "player_game_injury_status"
INJURY_FULL_NAME = (
    f"{INJURY_SCHEMA}.{INJURY_TABLE}"
)

DEPTH_SCHEMA = "processed"
DEPTH_TABLE = "player_game_depth_chart"
DEPTH_FULL_NAME = (
    f"{DEPTH_SCHEMA}.{DEPTH_TABLE}"
)

SNAP_SCHEMA = "analytics"
SNAP_TABLE = "player_snap_share_history"
SNAP_FULL_NAME = (
    f"{SNAP_SCHEMA}.{SNAP_TABLE}"
)

TARGET_SCHEMA = "analytics"
TARGET_TABLE = "player_game_injury_context"
TARGET_FULL_NAME = (
    f"{TARGET_SCHEMA}.{TARGET_TABLE}"
)

INJURY_REQUIRED_COLUMNS = {
    "game_id",
    "season",
    "season_type",
    "game_type",
    "week",
    "gameday",
    "team",
    "opponent",
    "is_home",
    "gsis_id",
    "position",
    "full_name",
    "report_primary_injury",
    "report_secondary_injury",
    "report_status",
    "practice_primary_injury",
    "practice_secondary_injury",
    "practice_status",
    "is_out",
    "is_doubtful",
    "is_questionable",
    "did_not_practice",
    "limited_practice",
    "full_practice",
    "source_date_modified",
    "has_source_timestamp",
    "source_snapshot_count",
}

DEPTH_REQUIRED_COLUMNS = {
    "game_id",
    "team",
    "gsis_id",
    "formation",
    "depth_position",
    "depth_rank",
    "is_starter",
    "is_primary_backup",
    "is_reserve",
    "source_generation",
}

SNAP_REQUIRED_COLUMNS = {
    "game_id",
    "gameday",
    "available_after_gameday",
    "team",
    "player_key",
    "career_appearance_count",
    "team_appearance_count",
    "career_games_last_4",
    "career_games_last_8",
    "team_games_last_4",
    "team_games_last_8",
    "career_offense_snap_share_last_4",
    "career_defense_snap_share_last_4",
    "career_special_teams_snap_share_last_4",
    "career_offense_snap_share_last_8",
    "career_defense_snap_share_last_8",
    "career_special_teams_snap_share_last_8",
    "team_offense_snap_share_last_4",
    "team_defense_snap_share_last_4",
    "team_special_teams_snap_share_last_4",
    "team_offense_snap_share_last_8",
    "team_defense_snap_share_last_8",
    "team_special_teams_snap_share_last_8",
}


def validate_database_file(
    database_file: Path = DATABASE_FILE,
) -> None:
    """Validate that the DuckDB database exists."""

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
        required_columns
        - available_columns
    )

    if missing_columns:
        raise RuntimeError(
            f"Source table {full_name} is missing columns: "
            + ", ".join(
                missing_columns
            )
        )


def validate_source_tables(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate injury, depth and snap-history sources."""

    validate_source_table(
        connection=connection,
        schema_name=INJURY_SCHEMA,
        table_name=INJURY_TABLE,
        required_columns=INJURY_REQUIRED_COLUMNS,
    )
    validate_source_table(
        connection=connection,
        schema_name=DEPTH_SCHEMA,
        table_name=DEPTH_TABLE,
        required_columns=DEPTH_REQUIRED_COLUMNS,
    )
    validate_source_table(
        connection=connection,
        schema_name=SNAP_SCHEMA,
        table_name=SNAP_TABLE,
        required_columns=SNAP_REQUIRED_COLUMNS,
    )

    logger.info(
        "Player injury-context sources validated: "
        "%s, %s and %s.",
        INJURY_FULL_NAME,
        DEPTH_FULL_NAME,
        SNAP_FULL_NAME,
    )


def count_source_rows(
    connection: duckdb.DuckDBPyConnection,
) -> int:
    """Count processed player-game injury records."""

    return connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {INJURY_FULL_NAME}
        """
    ).fetchone()[0]


def create_player_game_injury_context(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create analytics.player_game_injury_context."""

    connection.execute(
        f"""
        CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA}
        """
    )

    connection.execute(
        f"""
        CREATE OR REPLACE TABLE {TARGET_FULL_NAME} AS
        WITH depth_roles AS (
            SELECT
                game_id,
                team,
                gsis_id,

                COUNT(*) AS depth_role_count,
                COUNT(
                    DISTINCT formation
                ) AS depth_formation_count,

                MIN(depth_rank)
                    AS best_depth_rank,

                BOOL_OR(is_starter)
                    AS has_starter_role,

                BOOL_OR(is_primary_backup)
                    AS has_primary_backup_role,

                BOOL_OR(is_reserve)
                    AS has_reserve_role,

                BOOL_OR(
                    formation = 'Offense'
                ) AS has_offense_role,

                BOOL_OR(
                    formation = 'Defense'
                ) AS has_defense_role,

                BOOL_OR(
                    formation = 'Special Teams'
                ) AS has_special_teams_role,

                ANY_VALUE(source_generation)
                    AS depth_source_generation

            FROM {DEPTH_FULL_NAME}

            WHERE gsis_id IS NOT NULL

            GROUP BY
                game_id,
                team,
                gsis_id
        )

        SELECT
            injury.game_id,
            injury.season,
            injury.season_type,
            injury.game_type,
            injury.week,
            injury.gameday,
            injury.team,
            injury.opponent,
            injury.is_home,

            injury.gsis_id AS player_key,
            injury.gsis_id,
            injury.position,
            injury.full_name,

            injury.report_primary_injury,
            injury.report_secondary_injury,
            injury.report_status,
            injury.practice_primary_injury,
            injury.practice_secondary_injury,
            injury.practice_status,

            injury.is_out,
            injury.is_doubtful,
            injury.is_questionable,
            injury.did_not_practice,
            injury.limited_practice,
            injury.full_practice,

            injury.source_date_modified,
            injury.has_source_timestamp,
            injury.source_snapshot_count,

            depth.gsis_id IS NOT NULL
                AS has_depth_chart_match,

            COALESCE(
                depth.depth_role_count,
                0
            ) AS depth_role_count,

            COALESCE(
                depth.depth_formation_count,
                0
            ) AS depth_formation_count,

            depth.best_depth_rank,

            COALESCE(
                depth.has_starter_role,
                FALSE
            ) AS has_starter_role,

            COALESCE(
                depth.has_primary_backup_role,
                FALSE
            ) AS has_primary_backup_role,

            COALESCE(
                depth.has_reserve_role,
                FALSE
            ) AS has_reserve_role,

            COALESCE(
                depth.has_offense_role,
                FALSE
            ) AS has_offense_role,

            COALESCE(
                depth.has_defense_role,
                FALSE
            ) AS has_defense_role,

            COALESCE(
                depth.has_special_teams_role,
                FALSE
            ) AS has_special_teams_role,

            depth.depth_source_generation,

            CASE
                WHEN depth.has_starter_role
                    THEN 'STARTER'
                WHEN depth.has_primary_backup_role
                    THEN 'PRIMARY_BACKUP'
                WHEN depth.gsis_id IS NOT NULL
                    THEN 'RESERVE'
                ELSE 'UNKNOWN'
            END AS depth_tier,

            snap_history.game_id
                AS prior_snap_history_game_id,

            snap_history.gameday
                AS prior_snap_history_gameday,

            snap_history.team
                AS prior_snap_history_team,

            snap_history.game_id IS NOT NULL
                AS has_prior_snap_history,

            COALESCE(
                snap_history.team = injury.team,
                FALSE
            ) AS prior_snap_history_same_team,

            CASE
                WHEN snap_history.game_id IS NULL
                    THEN 'NONE'
                WHEN snap_history.team = injury.team
                    THEN 'TEAM'
                ELSE 'CAREER'
            END AS snap_history_source,

            CASE
                WHEN snap_history.gameday IS NULL
                    THEN NULL
                ELSE DATE_DIFF(
                    'day',
                    snap_history.gameday,
                    injury.gameday
                )
            END AS days_since_prior_snap_history,

            snap_history.career_appearance_count
                AS prior_career_appearance_count,

            CASE
                WHEN snap_history.team = injury.team
                    THEN snap_history.team_appearance_count
                ELSE snap_history.career_appearance_count
            END AS prior_selected_appearance_count,

            CASE
                WHEN snap_history.team = injury.team
                    THEN snap_history.team_games_last_4
                ELSE snap_history.career_games_last_4
            END AS prior_snap_games_last_4,

            CASE
                WHEN snap_history.team = injury.team
                    THEN snap_history.team_games_last_8
                ELSE snap_history.career_games_last_8
            END AS prior_snap_games_last_8,

            CASE
                WHEN snap_history.team = injury.team
                    THEN
                        snap_history
                            .team_offense_snap_share_last_4
                ELSE
                    snap_history
                        .career_offense_snap_share_last_4
            END AS prior_offense_snap_share_last_4,

            CASE
                WHEN snap_history.team = injury.team
                    THEN
                        snap_history
                            .team_defense_snap_share_last_4
                ELSE
                    snap_history
                        .career_defense_snap_share_last_4
            END AS prior_defense_snap_share_last_4,

            CASE
                WHEN snap_history.team = injury.team
                    THEN
                        snap_history
                            .team_special_teams_snap_share_last_4
                ELSE
                    snap_history
                        .career_special_teams_snap_share_last_4
            END AS prior_special_teams_snap_share_last_4,

            CASE
                WHEN snap_history.team = injury.team
                    THEN
                        snap_history
                            .team_offense_snap_share_last_8
                ELSE
                    snap_history
                        .career_offense_snap_share_last_8
            END AS prior_offense_snap_share_last_8,

            CASE
                WHEN snap_history.team = injury.team
                    THEN
                        snap_history
                            .team_defense_snap_share_last_8
                ELSE
                    snap_history
                        .career_defense_snap_share_last_8
            END AS prior_defense_snap_share_last_8,

            CASE
                WHEN snap_history.team = injury.team
                    THEN
                        snap_history
                            .team_special_teams_snap_share_last_8
                ELSE
                    snap_history
                        .career_special_teams_snap_share_last_8
            END AS prior_special_teams_snap_share_last_8

        FROM {INJURY_FULL_NAME}
            AS injury

        LEFT JOIN depth_roles
            AS depth
            ON injury.game_id = depth.game_id
           AND injury.team = depth.team
           AND injury.gsis_id = depth.gsis_id

        LEFT JOIN LATERAL (
            SELECT
                history.game_id,
                history.gameday,
                history.team,

                history.career_appearance_count,
                history.team_appearance_count,

                history.career_games_last_4,
                history.career_games_last_8,
                history.team_games_last_4,
                history.team_games_last_8,

                history.career_offense_snap_share_last_4,
                history.career_defense_snap_share_last_4,
                history
                    .career_special_teams_snap_share_last_4,

                history.career_offense_snap_share_last_8,
                history.career_defense_snap_share_last_8,
                history
                    .career_special_teams_snap_share_last_8,

                history.team_offense_snap_share_last_4,
                history.team_defense_snap_share_last_4,
                history
                    .team_special_teams_snap_share_last_4,

                history.team_offense_snap_share_last_8,
                history.team_defense_snap_share_last_8,
                history
                    .team_special_teams_snap_share_last_8

            FROM {SNAP_FULL_NAME}
                AS history

            WHERE history.player_key = injury.gsis_id
              AND history.available_after_gameday
                    < injury.gameday

            ORDER BY
                history.available_after_gameday DESC,
                history.game_id DESC

            LIMIT 1
        ) AS snap_history
            ON TRUE
        """
    )


def validate_target_table(
    connection: duckdb.DuckDBPyConnection,
    expected_row_count: int,
) -> tuple[int, int, int, int, int]:
    """Validate the player-game injury context."""

    row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    if row_count != expected_row_count:
        raise RuntimeError(
            "Player injury-context row count does not "
            "match its injury source: "
            f"{row_count} != {expected_row_count}."
        )

    duplicate_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT
                game_id,
                team,
                gsis_id
            FROM {TARGET_FULL_NAME}
            GROUP BY
                game_id,
                team,
                gsis_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_count > 0:
        raise RuntimeError(
            "Player injury context contains "
            f"{duplicate_count} duplicate business keys."
        )

    null_key_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE game_id IS NULL
           OR season IS NULL
           OR gameday IS NULL
           OR team IS NULL
           OR opponent IS NULL
           OR player_key IS NULL
           OR gsis_id IS NULL
           OR depth_tier IS NULL
           OR snap_history_source IS NULL
        """
    ).fetchone()[0]

    if null_key_count > 0:
        raise RuntimeError(
            "Player injury context contains "
            f"{null_key_count} rows with null business keys."
        )

    future_history_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE prior_snap_history_gameday
                >= gameday
        """
    ).fetchone()[0]

    if future_history_count > 0:
        raise RuntimeError(
            "Player injury context contains "
            f"{future_history_count} non-pregame "
            "snap-history matches."
        )

    invalid_snap_share_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE prior_offense_snap_share_last_4
                NOT BETWEEN 0 AND 1
           OR prior_defense_snap_share_last_4
                NOT BETWEEN 0 AND 1
           OR prior_special_teams_snap_share_last_4
                NOT BETWEEN 0 AND 1
           OR prior_offense_snap_share_last_8
                NOT BETWEEN 0 AND 1
           OR prior_defense_snap_share_last_8
                NOT BETWEEN 0 AND 1
           OR prior_special_teams_snap_share_last_8
                NOT BETWEEN 0 AND 1
        """
    ).fetchone()[0]

    if invalid_snap_share_count > 0:
        raise RuntimeError(
            "Player injury context contains "
            f"{invalid_snap_share_count} invalid "
            "prior snap shares."
        )

    inconsistent_depth_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE has_depth_chart_match
                IS DISTINCT FROM (
                    best_depth_rank IS NOT NULL
                )
           OR (
                NOT has_depth_chart_match
                AND (
                    depth_role_count != 0
                    OR depth_formation_count != 0
                    OR has_starter_role
                    OR has_primary_backup_role
                    OR has_reserve_role
                )
              )
           OR depth_tier
                IS DISTINCT FROM
                    CASE
                        WHEN has_starter_role
                            THEN 'STARTER'
                        WHEN has_primary_backup_role
                            THEN 'PRIMARY_BACKUP'
                        WHEN has_depth_chart_match
                            THEN 'RESERVE'
                        ELSE 'UNKNOWN'
                    END
        """
    ).fetchone()[0]

    if inconsistent_depth_count > 0:
        raise RuntimeError(
            "Player injury context contains "
            f"{inconsistent_depth_count} inconsistent "
            "depth-chart fields."
        )

    inconsistent_snap_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE has_prior_snap_history
                IS DISTINCT FROM (
                    prior_snap_history_game_id IS NOT NULL
                )
           OR prior_snap_history_same_team
                IS DISTINCT FROM (
                    has_prior_snap_history
                    AND prior_snap_history_team = team
                )
           OR snap_history_source
                IS DISTINCT FROM
                    CASE
                        WHEN NOT has_prior_snap_history
                            THEN 'NONE'
                        WHEN prior_snap_history_team = team
                            THEN 'TEAM'
                        ELSE 'CAREER'
                    END
           OR (
                NOT has_prior_snap_history
                AND (
                    prior_snap_history_gameday IS NOT NULL
                    OR prior_snap_history_team IS NOT NULL
                    OR days_since_prior_snap_history IS NOT NULL
                    OR prior_snap_games_last_4 IS NOT NULL
                    OR prior_snap_games_last_8 IS NOT NULL
                )
              )
        """
    ).fetchone()[0]

    if inconsistent_snap_count > 0:
        raise RuntimeError(
            "Player injury context contains "
            f"{inconsistent_snap_count} inconsistent "
            "snap-history fields."
        )

    depth_match_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE has_depth_chart_match
        """
    ).fetchone()[0]

    prior_snap_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE has_prior_snap_history
        """
    ).fetchone()[0]

    team_snap_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE snap_history_source = 'TEAM'
        """
    ).fetchone()[0]

    career_fallback_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE snap_history_source = 'CAREER'
        """
    ).fetchone()[0]

    return (
        row_count,
        depth_match_count,
        prior_snap_count,
        team_snap_count,
        career_fallback_count,
    )


def build_player_game_injury_context(
    database_file: Path = DATABASE_FILE,
) -> None:
    """Build analytics.player_game_injury_context."""

    validate_database_file(
        database_file
    )

    logger.info(
        "Starting player-game injury context build..."
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

            connection.execute(
                "BEGIN TRANSACTION"
            )

            try:
                create_player_game_injury_context(
                    connection
                )

                (
                    row_count,
                    depth_match_count,
                    prior_snap_count,
                    team_snap_count,
                    career_fallback_count,
                ) = validate_target_table(
                    connection=connection,
                    expected_row_count=expected_row_count,
                )

                connection.execute(
                    "COMMIT"
                )

            except Exception:
                connection.execute(
                    "ROLLBACK"
                )
                raise

    except Exception:
        logger.exception(
            "Player-game injury context build failed."
        )
        raise

    logger.info(
        "Player-game injury context validated: %s rows.",
        row_count,
    )
    logger.info(
        "Depth-chart matches: %s.",
        depth_match_count,
    )
    logger.info(
        "Prior snap-history matches: %s.",
        prior_snap_count,
    )
    logger.info(
        "Same-team snap histories: %s.",
        team_snap_count,
    )
    logger.info(
        "Career snap-history fallbacks: %s.",
        career_fallback_count,
    )
    logger.info(
        "Player-game injury context build "
        "completed successfully."
    )


def main() -> None:
    """Run the player-game injury-context builder."""

    build_player_game_injury_context()


if __name__ == "__main__":
    main()