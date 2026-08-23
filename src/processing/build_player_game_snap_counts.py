"""
NFL Analytics Platform
Player-Game Snap-Count Builder

Purpose:
    Connect raw player snap counts to scheduled games and
    the player identity directory, producing one canonical
    player-team-game participation table.

Leakage note:
    This table contains actual participation from completed
    games. Its current-game values must not enter pregame
    predictions for the same game. Pregame feature builders
    must use shifted prior-game history only.

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

SNAP_SCHEMA = "raw"
SNAP_TABLE = "player_snap_counts"
SNAP_FULL_NAME = (
    f"{SNAP_SCHEMA}.{SNAP_TABLE}"
)

PLAYER_SCHEMA = "raw"
PLAYER_TABLE = "player_directory"
PLAYER_FULL_NAME = (
    f"{PLAYER_SCHEMA}.{PLAYER_TABLE}"
)

SCHEDULE_SCHEMA = "processed"
SCHEDULE_TABLE = "schedule"
SCHEDULE_FULL_NAME = (
    f"{SCHEDULE_SCHEMA}.{SCHEDULE_TABLE}"
)

TARGET_SCHEMA = "processed"
TARGET_TABLE = "player_game_snap_counts"
TARGET_FULL_NAME = (
    f"{TARGET_SCHEMA}.{TARGET_TABLE}"
)

SNAP_REQUIRED_COLUMNS = {
    "game_id",
    "pfr_game_id",
    "season",
    "game_type",
    "week",
    "player",
    "pfr_player_id",
    "position",
    "team",
    "opponent",
    "offense_snaps",
    "offense_pct",
    "defense_snaps",
    "defense_pct",
    "st_snaps",
    "st_pct",
    "source_file",
}

PLAYER_REQUIRED_COLUMNS = {
    "gsis_id",
    "pfr_id",
    "espn_id",
    "display_name",
    "position_group",
    "position",
}

SCHEDULE_REQUIRED_COLUMNS = {
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
    """Validate snap, player and schedule sources."""

    validate_source_table(
        connection=connection,
        schema_name=SNAP_SCHEMA,
        table_name=SNAP_TABLE,
        required_columns=SNAP_REQUIRED_COLUMNS,
    )
    validate_source_table(
        connection=connection,
        schema_name=PLAYER_SCHEMA,
        table_name=PLAYER_TABLE,
        required_columns=PLAYER_REQUIRED_COLUMNS,
    )
    validate_source_table(
        connection=connection,
        schema_name=SCHEDULE_SCHEMA,
        table_name=SCHEDULE_TABLE,
        required_columns=SCHEDULE_REQUIRED_COLUMNS,
    )

    logger.info(
        "Player-game snap sources validated: "
        "%s, %s and %s.",
        SNAP_FULL_NAME,
        PLAYER_FULL_NAME,
        SCHEDULE_FULL_NAME,
    )


def count_source_rows(
    connection: duckdb.DuckDBPyConnection,
) -> int:
    """Count raw player snap-count records."""

    return connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {SNAP_FULL_NAME}
        """
    ).fetchone()[0]


def validate_schedule_coverage(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Require every snap record to match its scheduled game."""

    missing_schedule_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {SNAP_FULL_NAME}
            AS snap

        LEFT JOIN {SCHEDULE_FULL_NAME}
            AS schedule
            ON snap.game_id = schedule.game_id

        WHERE schedule.game_id IS NULL
        """
    ).fetchone()[0]

    if missing_schedule_count > 0:
        raise RuntimeError(
            "Snap-count schedule coverage is incomplete: "
            f"{missing_schedule_count} rows have no game."
        )

    invalid_team_context_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {SNAP_FULL_NAME}
            AS snap

        INNER JOIN {SCHEDULE_FULL_NAME}
            AS schedule
            ON snap.game_id = schedule.game_id

        WHERE NOT (
                snap.team = schedule.home_team
                AND snap.opponent = schedule.away_team
              )
          AND NOT (
                snap.team = schedule.away_team
                AND snap.opponent = schedule.home_team
              )
        """
    ).fetchone()[0]

    if invalid_team_context_count > 0:
        raise RuntimeError(
            "Snap-count schedule coverage contains "
            f"{invalid_team_context_count} invalid "
            "team-opponent assignments."
        )

    logger.info(
        "Snap-count schedule coverage validated."
    )


def create_player_game_snap_counts(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create processed.player_game_snap_counts."""

    connection.execute(
        f"""
        CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA}
        """
    )

    connection.execute(
        f"""
        CREATE OR REPLACE TABLE {TARGET_FULL_NAME} AS
        SELECT
            snap.game_id,
            snap.pfr_game_id,
            schedule.season,
            schedule.game_type,
            schedule.week,
            schedule.gameday,

            snap.team,
            snap.opponent,
            snap.team = schedule.home_team
                AS is_home,

            COALESCE(
                player.gsis_id,
                'PFR:' || snap.pfr_player_id
            ) AS player_key,

            player.gsis_id,
            snap.pfr_player_id,
            player.espn_id,

            COALESCE(
                player.display_name,
                snap.player
            ) AS player_name,

            snap.player
                AS source_player_name,
            snap.position
                AS source_position,
            player.position
                AS directory_position,
            player.position_group
                AS directory_position_group,

            snap.offense_snaps,
            snap.offense_pct
                AS source_offense_snap_share,

            CASE
                WHEN snap.offense_pct IS NULL
                    THEN NULL
                ELSE LEAST(
                    snap.offense_pct,
                    1.0
                )
            END AS offense_snap_share,

            snap.defense_snaps,
            snap.defense_pct
                AS source_defense_snap_share,

            CASE
                WHEN snap.defense_pct IS NULL
                    THEN NULL
                ELSE LEAST(
                    snap.defense_pct,
                    1.0
                )
            END AS defense_snap_share,

            snap.st_snaps,
            snap.st_pct
                AS source_special_teams_snap_share,

            CASE
                WHEN snap.st_pct IS NULL
                    THEN NULL
                ELSE LEAST(
                    snap.st_pct,
                    1.0
                )
            END AS special_teams_snap_share,

            COALESCE(
                snap.offense_snaps,
                0
            )
            +
            COALESCE(
                snap.defense_snaps,
                0
            )
            +
            COALESCE(
                snap.st_snaps,
                0
            ) AS total_snaps,

            COALESCE(
                snap.offense_snaps,
                0
            ) > 0 AS played_offense,

            COALESCE(
                snap.defense_snaps,
                0
            ) > 0 AS played_defense,

            COALESCE(
                snap.st_snaps,
                0
            ) > 0 AS played_special_teams,

            (
                COALESCE(
                    snap.offense_pct,
                    0
                ) > 1
                OR COALESCE(
                    snap.defense_pct,
                    0
                ) > 1
                OR COALESCE(
                    snap.st_pct,
                    0
                ) > 1
            ) AS has_source_rounding_adjustment,

            player.gsis_id IS NOT NULL
                AS has_player_directory_match,

            CASE
                WHEN player.gsis_id IS NOT NULL
                    THEN 'GSIS'
                ELSE 'PFR'
            END AS player_identifier_source,

            snap.source_file

        FROM {SNAP_FULL_NAME}
            AS snap

        INNER JOIN {SCHEDULE_FULL_NAME}
            AS schedule
            ON snap.game_id = schedule.game_id

        LEFT JOIN {PLAYER_FULL_NAME}
            AS player
            ON snap.pfr_player_id = player.pfr_id
        """
    )


def validate_target_table(
    connection: duckdb.DuckDBPyConnection,
    expected_row_count: int,
) -> tuple[int, int, int, int]:
    """Validate processed player-game snap counts."""

    row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    if row_count != expected_row_count:
        raise RuntimeError(
            "Processed player-game snap row count "
            "does not match the raw source: "
            f"{row_count} != {expected_row_count}."
        )

    duplicate_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT
                game_id,
                team,
                pfr_player_id
            FROM {TARGET_FULL_NAME}
            GROUP BY
                game_id,
                team,
                pfr_player_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_count > 0:
        raise RuntimeError(
            "Processed player-game snap table contains "
            f"{duplicate_count} duplicate business keys."
        )

    null_key_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE game_id IS NULL
           OR season IS NULL
           OR week IS NULL
           OR gameday IS NULL
           OR team IS NULL
           OR opponent IS NULL
           OR player_key IS NULL
           OR pfr_player_id IS NULL
           OR player_identifier_source IS NULL
        """
    ).fetchone()[0]

    if null_key_count > 0:
        raise RuntimeError(
            "Processed player-game snap table contains "
            f"{null_key_count} rows with null business keys."
        )

    invalid_share_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE offense_snap_share < 0
           OR offense_snap_share > 1
           OR defense_snap_share < 0
           OR defense_snap_share > 1
           OR special_teams_snap_share < 0
           OR special_teams_snap_share > 1
        """
    ).fetchone()[0]

    if invalid_share_count > 0:
        raise RuntimeError(
            "Processed player-game snap table contains "
            f"{invalid_share_count} normalized shares "
            "outside the range 0 to 1."
        )

    inconsistent_flag_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE played_offense
                IS DISTINCT FROM (
                    COALESCE(
                        offense_snaps,
                        0
                    ) > 0
                )
           OR played_defense
                IS DISTINCT FROM (
                    COALESCE(
                        defense_snaps,
                        0
                    ) > 0
                )
           OR played_special_teams
                IS DISTINCT FROM (
                    COALESCE(
                        st_snaps,
                        0
                    ) > 0
                )
           OR has_player_directory_match
                IS DISTINCT FROM (
                    gsis_id IS NOT NULL
                )
           OR player_identifier_source
                IS DISTINCT FROM
                    CASE
                        WHEN gsis_id IS NOT NULL
                            THEN 'GSIS'
                        ELSE 'PFR'
                    END
        """
    ).fetchone()[0]

    if inconsistent_flag_count > 0:
        raise RuntimeError(
            "Processed player-game snap table contains "
            f"{inconsistent_flag_count} inconsistent flags."
        )

    schedule_mismatch_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
            AS snap

        INNER JOIN {SCHEDULE_FULL_NAME}
            AS schedule
            ON snap.game_id = schedule.game_id

        WHERE snap.season
                IS DISTINCT FROM schedule.season
           OR snap.game_type
                IS DISTINCT FROM schedule.game_type
           OR snap.week
                IS DISTINCT FROM schedule.week
           OR snap.gameday
                IS DISTINCT FROM schedule.gameday
           OR (
                snap.is_home
                AND snap.team
                    IS DISTINCT FROM schedule.home_team
              )
           OR (
                snap.is_home
                AND snap.opponent
                    IS DISTINCT FROM schedule.away_team
              )
           OR (
                NOT snap.is_home
                AND snap.team
                    IS DISTINCT FROM schedule.away_team
              )
           OR (
                NOT snap.is_home
                AND snap.opponent
                    IS DISTINCT FROM schedule.home_team
              )
        """
    ).fetchone()[0]

    if schedule_mismatch_count > 0:
        raise RuntimeError(
            "Processed player-game snap table contains "
            f"{schedule_mismatch_count} schedule mismatches."
        )

    unmatched_player_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE NOT has_player_directory_match
        """
    ).fetchone()[0]

    unmatched_unique_player_count = connection.execute(
        f"""
        SELECT COUNT(
            DISTINCT pfr_player_id
        )
        FROM {TARGET_FULL_NAME}
        WHERE NOT has_player_directory_match
        """
    ).fetchone()[0]

    rounding_adjustment_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE has_source_rounding_adjustment
        """
    ).fetchone()[0]

    return (
        row_count,
        unmatched_player_count,
        unmatched_unique_player_count,
        rounding_adjustment_count,
    )


def build_player_game_snap_counts(
    database_file: Path = DATABASE_FILE,
) -> None:
    """Build processed.player_game_snap_counts."""

    validate_database_file(
        database_file
    )

    logger.info(
        "Starting player-game snap-count build..."
    )

    try:
        with duckdb.connect(
            str(database_file)
        ) as connection:
            validate_source_tables(
                connection
            )

            validate_schedule_coverage(
                connection
            )

            expected_row_count = count_source_rows(
                connection
            )

            connection.execute(
                "BEGIN TRANSACTION"
            )

            try:
                create_player_game_snap_counts(
                    connection
                )

                (
                    row_count,
                    unmatched_player_count,
                    unmatched_unique_player_count,
                    rounding_adjustment_count,
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
            "Player-game snap-count build failed."
        )
        raise

    logger.info(
        "Player-game snap-count table validated: %s rows.",
        row_count,
    )
    logger.info(
        "Snap records without player-directory match: "
        "%s rows across %s players.",
        unmatched_player_count,
        unmatched_unique_player_count,
    )
    logger.info(
        "Source snap shares normalized from above 1.0: %s.",
        rounding_adjustment_count,
    )
    logger.info(
        "Player-game snap-count build completed successfully."
    )


def main() -> None:
    """Run the player-game snap-count builder."""

    build_player_game_snap_counts()


if __name__ == "__main__":
    main()