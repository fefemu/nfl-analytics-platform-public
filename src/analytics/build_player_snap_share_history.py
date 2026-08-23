"""
NFL Analytics Platform
Player Snap-Share History Builder

Purpose:
    Build time-indexed player usage history from completed
    player-game snap counts.

Leakage boundary:
    Rolling values in this table include the row's completed
    game and become available only after that game. A pregame
    feature for a target game may use only history rows whose
    gameday is strictly earlier than the target gameday.

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

SOURCE_SCHEMA = "processed"
SOURCE_TABLE = "player_game_snap_counts"
SOURCE_FULL_NAME = (
    f"{SOURCE_SCHEMA}.{SOURCE_TABLE}"
)

TARGET_SCHEMA = "analytics"
TARGET_TABLE = "player_snap_share_history"
TARGET_FULL_NAME = (
    f"{TARGET_SCHEMA}.{TARGET_TABLE}"
)

SHORT_WINDOW = 4
LONG_WINDOW = 8

SOURCE_REQUIRED_COLUMNS = {
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
    "pfr_player_id",
    "espn_id",
    "player_name",
    "source_position",
    "directory_position",
    "directory_position_group",
    "offense_snap_share",
    "defense_snap_share",
    "special_teams_snap_share",
    "total_snaps",
    "has_player_directory_match",
    "player_identifier_source",
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
) -> None:
    """Validate the processed snap-count source."""

    table_exists = connection.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = ?
          AND table_name = ?
        """,
        [
            SOURCE_SCHEMA,
            SOURCE_TABLE,
        ],
    ).fetchone()[0]

    if table_exists == 0:
        raise RuntimeError(
            f"Source table does not exist: {SOURCE_FULL_NAME}"
        )

    available_columns = get_table_columns(
        connection=connection,
        schema_name=SOURCE_SCHEMA,
        table_name=SOURCE_TABLE,
    )

    missing_columns = sorted(
        SOURCE_REQUIRED_COLUMNS
        - available_columns
    )

    if missing_columns:
        raise RuntimeError(
            f"Source table {SOURCE_FULL_NAME} "
            "is missing columns: "
            + ", ".join(
                missing_columns
            )
        )

    logger.info(
        "Player snap-share history source validated: %s.",
        SOURCE_FULL_NAME,
    )


def count_source_rows(
    connection: duckdb.DuckDBPyConnection,
) -> int:
    """Count processed player-game snap rows."""

    return connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {SOURCE_FULL_NAME}
        """
    ).fetchone()[0]


def create_player_snap_share_history(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create analytics.player_snap_share_history."""

    connection.execute(
        f"""
        CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA}
        """
    )

    connection.execute(
        f"""
        CREATE OR REPLACE TABLE {TARGET_FULL_NAME} AS
        WITH rolling_history AS (
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
                pfr_player_id,
                espn_id,
                player_name,
                source_position,
                directory_position,
                directory_position_group,
                has_player_directory_match,
                player_identifier_source,

                offense_snap_share,
                defense_snap_share,
                special_teams_snap_share,
                total_snaps,

                COUNT(*) OVER (
                    PARTITION BY player_key
                    ORDER BY
                        gameday,
                        game_id,
                        team
                    ROWS BETWEEN UNBOUNDED PRECEDING
                        AND CURRENT ROW
                ) AS career_appearance_count,

                COUNT(*) OVER (
                    PARTITION BY
                        player_key,
                        team
                    ORDER BY
                        gameday,
                        game_id
                    ROWS BETWEEN UNBOUNDED PRECEDING
                        AND CURRENT ROW
                ) AS team_appearance_count,

                LAG(game_id) OVER (
                    PARTITION BY player_key
                    ORDER BY
                        gameday,
                        game_id,
                        team
                ) AS previous_career_game_id,

                LAG(gameday) OVER (
                    PARTITION BY player_key
                    ORDER BY
                        gameday,
                        game_id,
                        team
                ) AS previous_career_gameday,

                LAG(game_id) OVER (
                    PARTITION BY
                        player_key,
                        team
                    ORDER BY
                        gameday,
                        game_id
                ) AS previous_team_game_id,

                LAG(gameday) OVER (
                    PARTITION BY
                        player_key,
                        team
                    ORDER BY
                        gameday,
                        game_id
                ) AS previous_team_gameday,

                COUNT(*) OVER (
                    PARTITION BY player_key
                    ORDER BY
                        gameday,
                        game_id,
                        team
                    ROWS BETWEEN
                        {SHORT_WINDOW - 1} PRECEDING
                        AND CURRENT ROW
                ) AS career_games_last_4,

                COUNT(*) OVER (
                    PARTITION BY player_key
                    ORDER BY
                        gameday,
                        game_id,
                        team
                    ROWS BETWEEN
                        {LONG_WINDOW - 1} PRECEDING
                        AND CURRENT ROW
                ) AS career_games_last_8,

                COUNT(*) OVER (
                    PARTITION BY
                        player_key,
                        team
                    ORDER BY
                        gameday,
                        game_id
                    ROWS BETWEEN
                        {SHORT_WINDOW - 1} PRECEDING
                        AND CURRENT ROW
                ) AS team_games_last_4,

                COUNT(*) OVER (
                    PARTITION BY
                        player_key,
                        team
                    ORDER BY
                        gameday,
                        game_id
                    ROWS BETWEEN
                        {LONG_WINDOW - 1} PRECEDING
                        AND CURRENT ROW
                ) AS team_games_last_8,

                AVG(offense_snap_share) OVER (
                    PARTITION BY player_key
                    ORDER BY
                        gameday,
                        game_id,
                        team
                    ROWS BETWEEN
                        {SHORT_WINDOW - 1} PRECEDING
                        AND CURRENT ROW
                ) AS career_offense_snap_share_last_4,

                AVG(defense_snap_share) OVER (
                    PARTITION BY player_key
                    ORDER BY
                        gameday,
                        game_id,
                        team
                    ROWS BETWEEN
                        {SHORT_WINDOW - 1} PRECEDING
                        AND CURRENT ROW
                ) AS career_defense_snap_share_last_4,

                AVG(special_teams_snap_share) OVER (
                    PARTITION BY player_key
                    ORDER BY
                        gameday,
                        game_id,
                        team
                    ROWS BETWEEN
                        {SHORT_WINDOW - 1} PRECEDING
                        AND CURRENT ROW
                ) AS career_special_teams_snap_share_last_4,

                AVG(offense_snap_share) OVER (
                    PARTITION BY player_key
                    ORDER BY
                        gameday,
                        game_id,
                        team
                    ROWS BETWEEN
                        {LONG_WINDOW - 1} PRECEDING
                        AND CURRENT ROW
                ) AS career_offense_snap_share_last_8,

                AVG(defense_snap_share) OVER (
                    PARTITION BY player_key
                    ORDER BY
                        gameday,
                        game_id,
                        team
                    ROWS BETWEEN
                        {LONG_WINDOW - 1} PRECEDING
                        AND CURRENT ROW
                ) AS career_defense_snap_share_last_8,

                AVG(special_teams_snap_share) OVER (
                    PARTITION BY player_key
                    ORDER BY
                        gameday,
                        game_id,
                        team
                    ROWS BETWEEN
                        {LONG_WINDOW - 1} PRECEDING
                        AND CURRENT ROW
                ) AS career_special_teams_snap_share_last_8,

                AVG(offense_snap_share) OVER (
                    PARTITION BY
                        player_key,
                        team
                    ORDER BY
                        gameday,
                        game_id
                    ROWS BETWEEN
                        {SHORT_WINDOW - 1} PRECEDING
                        AND CURRENT ROW
                ) AS team_offense_snap_share_last_4,

                AVG(defense_snap_share) OVER (
                    PARTITION BY
                        player_key,
                        team
                    ORDER BY
                        gameday,
                        game_id
                    ROWS BETWEEN
                        {SHORT_WINDOW - 1} PRECEDING
                        AND CURRENT ROW
                ) AS team_defense_snap_share_last_4,

                AVG(special_teams_snap_share) OVER (
                    PARTITION BY
                        player_key,
                        team
                    ORDER BY
                        gameday,
                        game_id
                    ROWS BETWEEN
                        {SHORT_WINDOW - 1} PRECEDING
                        AND CURRENT ROW
                ) AS team_special_teams_snap_share_last_4,

                AVG(offense_snap_share) OVER (
                    PARTITION BY
                        player_key,
                        team
                    ORDER BY
                        gameday,
                        game_id
                    ROWS BETWEEN
                        {LONG_WINDOW - 1} PRECEDING
                        AND CURRENT ROW
                ) AS team_offense_snap_share_last_8,

                AVG(defense_snap_share) OVER (
                    PARTITION BY
                        player_key,
                        team
                    ORDER BY
                        gameday,
                        game_id
                    ROWS BETWEEN
                        {LONG_WINDOW - 1} PRECEDING
                        AND CURRENT ROW
                ) AS team_defense_snap_share_last_8,

                AVG(special_teams_snap_share) OVER (
                    PARTITION BY
                        player_key,
                        team
                    ORDER BY
                        gameday,
                        game_id
                    ROWS BETWEEN
                        {LONG_WINDOW - 1} PRECEDING
                        AND CURRENT ROW
                ) AS team_special_teams_snap_share_last_8

            FROM {SOURCE_FULL_NAME}
        )

        SELECT
            *,

            CASE
                WHEN previous_career_gameday IS NULL
                    THEN NULL
                ELSE DATE_DIFF(
                    'day',
                    previous_career_gameday,
                    gameday
                )
            END AS days_since_previous_career_appearance,

            CASE
                WHEN previous_team_gameday IS NULL
                    THEN NULL
                ELSE DATE_DIFF(
                    'day',
                    previous_team_gameday,
                    gameday
                )
            END AS days_since_previous_team_appearance,

            gameday AS available_after_gameday

        FROM rolling_history
        """
    )


def validate_target_table(
    connection: duckdb.DuckDBPyConnection,
    expected_row_count: int,
) -> tuple[int, int, int, int]:
    """Validate the player snap-share history table."""

    row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    if row_count != expected_row_count:
        raise RuntimeError(
            "Player snap-share history row count "
            "does not match its source: "
            f"{row_count} != {expected_row_count}."
        )

    duplicate_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT
                game_id,
                team,
                player_key
            FROM {TARGET_FULL_NAME}
            GROUP BY
                game_id,
                team,
                player_key
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_count > 0:
        raise RuntimeError(
            "Player snap-share history contains "
            f"{duplicate_count} duplicate business keys."
        )

    null_key_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE game_id IS NULL
           OR gameday IS NULL
           OR team IS NULL
           OR player_key IS NULL
           OR career_appearance_count IS NULL
           OR team_appearance_count IS NULL
           OR available_after_gameday IS NULL
        """
    ).fetchone()[0]

    if null_key_count > 0:
        raise RuntimeError(
            "Player snap-share history contains "
            f"{null_key_count} rows with null business keys."
        )

    invalid_window_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE career_games_last_4 NOT BETWEEN 1 AND 4
           OR career_games_last_8 NOT BETWEEN 1 AND 8
           OR team_games_last_4 NOT BETWEEN 1 AND 4
           OR team_games_last_8 NOT BETWEEN 1 AND 8
           OR career_games_last_4
                > career_appearance_count
           OR career_games_last_8
                > career_appearance_count
           OR team_games_last_4
                > team_appearance_count
           OR team_games_last_8
                > team_appearance_count
        """
    ).fetchone()[0]

    if invalid_window_count > 0:
        raise RuntimeError(
            "Player snap-share history contains "
            f"{invalid_window_count} invalid rolling windows."
        )

    invalid_share_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE career_offense_snap_share_last_4
                NOT BETWEEN 0 AND 1
           OR career_defense_snap_share_last_4
                NOT BETWEEN 0 AND 1
           OR career_special_teams_snap_share_last_4
                NOT BETWEEN 0 AND 1
           OR career_offense_snap_share_last_8
                NOT BETWEEN 0 AND 1
           OR career_defense_snap_share_last_8
                NOT BETWEEN 0 AND 1
           OR career_special_teams_snap_share_last_8
                NOT BETWEEN 0 AND 1
           OR team_offense_snap_share_last_4
                NOT BETWEEN 0 AND 1
           OR team_defense_snap_share_last_4
                NOT BETWEEN 0 AND 1
           OR team_special_teams_snap_share_last_4
                NOT BETWEEN 0 AND 1
           OR team_offense_snap_share_last_8
                NOT BETWEEN 0 AND 1
           OR team_defense_snap_share_last_8
                NOT BETWEEN 0 AND 1
           OR team_special_teams_snap_share_last_8
                NOT BETWEEN 0 AND 1
        """
    ).fetchone()[0]

    if invalid_share_count > 0:
        raise RuntimeError(
            "Player snap-share history contains "
            f"{invalid_share_count} rolling shares "
            "outside the range 0 to 1."
        )

    invalid_sequence_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE career_appearance_count < 1
           OR team_appearance_count < 1
           OR days_since_previous_career_appearance < 0
           OR days_since_previous_team_appearance < 0
           OR available_after_gameday
                IS DISTINCT FROM gameday
        """
    ).fetchone()[0]

    if invalid_sequence_count > 0:
        raise RuntimeError(
            "Player snap-share history contains "
            f"{invalid_sequence_count} invalid sequences."
        )

    first_career_row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE career_appearance_count = 1
        """
    ).fetchone()[0]

    first_team_row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE team_appearance_count = 1
        """
    ).fetchone()[0]

    multi_team_player_season_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT
                season,
                player_key
            FROM {TARGET_FULL_NAME}
            GROUP BY
                season,
                player_key
            HAVING COUNT(
                DISTINCT team
            ) > 1
        )
        """
    ).fetchone()[0]

    return (
        row_count,
        first_career_row_count,
        first_team_row_count,
        multi_team_player_season_count,
    )


def build_player_snap_share_history(
    database_file: Path = DATABASE_FILE,
) -> None:
    """Build analytics.player_snap_share_history."""

    validate_database_file(
        database_file
    )

    logger.info(
        "Starting player snap-share history build..."
    )

    try:
        with duckdb.connect(
            str(database_file)
        ) as connection:
            validate_source_table(
                connection
            )

            expected_row_count = count_source_rows(
                connection
            )

            connection.execute(
                "BEGIN TRANSACTION"
            )

            try:
                create_player_snap_share_history(
                    connection
                )

                (
                    row_count,
                    first_career_row_count,
                    first_team_row_count,
                    multi_team_player_season_count,
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
            "Player snap-share history build failed."
        )
        raise

    logger.info(
        "Player snap-share history validated: %s rows.",
        row_count,
    )
    logger.info(
        "First career history rows: %s.",
        first_career_row_count,
    )
    logger.info(
        "First team history rows: %s.",
        first_team_row_count,
    )
    logger.info(
        "Multi-team player-seasons: %s.",
        multi_team_player_season_count,
    )
    logger.info(
        "Player snap-share history build "
        "completed successfully."
    )


def main() -> None:
    """Run the player snap-share history builder."""

    build_player_snap_share_history()


if __name__ == "__main__":
    main()