"""
NFL Analytics Platform
Game-Level Injury Feature Builder

Purpose:
    Convert schedule-complete team-game injury burdens
    into one row per game with home, away and
    home-minus-away features.

Missing-data policy:
    Zero burden and missing injury-report data remain
    distinguishable through explicit availability flags.

QB policy:
    QB burden is retained for audit and explanation.
    Generic non-QB burden remains separately available
    to avoid double counting the QB-rating layer.

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

SOURCE_SCHEMA = "analytics"
SOURCE_TABLE = "team_game_injury_burden"
SOURCE_FULL_NAME = (
    f"{SOURCE_SCHEMA}.{SOURCE_TABLE}"
)

TARGET_SCHEMA = "analytics"
TARGET_TABLE = "game_injury_features"
TARGET_FULL_NAME = (
    f"{TARGET_SCHEMA}.{TARGET_TABLE}"
)

SOURCE_REQUIRED_COLUMNS = {
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "team",
    "opponent",
    "is_home",
    "has_injury_report_data",
    "injury_report_player_count",
    "out_player_count",
    "doubtful_player_count",
    "questionable_player_count",
    "starter_out_count",
    "qb_out_count",
    "total_injury_burden",
    "qb_injury_burden",
    "non_qb_injury_burden",
    "offense_injury_burden",
    "defense_injury_burden",
    "special_teams_injury_burden",
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


def get_source_columns(
    connection: duckdb.DuckDBPyConnection,
) -> set[str]:
    """Return source-table columns."""

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
                SOURCE_SCHEMA,
                SOURCE_TABLE,
            ],
        ).fetchall()
    }


def validate_source_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate schedule-complete team-game burdens."""

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

    missing_columns = sorted(
        SOURCE_REQUIRED_COLUMNS
        - get_source_columns(
            connection
        )
    )

    if missing_columns:
        raise RuntimeError(
            f"Source table {SOURCE_FULL_NAME} "
            "is missing columns: "
            + ", ".join(
                missing_columns
            )
        )

    duplicate_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT
                game_id,
                team
            FROM {SOURCE_FULL_NAME}
            GROUP BY
                game_id,
                team
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_count > 0:
        raise RuntimeError(
            "Team-game injury source contains "
            f"{duplicate_count} duplicate business keys."
        )

    invalid_game_grain_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT
                game_id,
                COUNT(*) AS team_count,
                COUNT(*) FILTER (
                    WHERE is_home
                ) AS home_count,
                COUNT(*) FILTER (
                    WHERE NOT is_home
                ) AS away_count
            FROM {SOURCE_FULL_NAME}
            GROUP BY game_id
            HAVING team_count != 2
                OR home_count != 1
                OR away_count != 1
        )
        """
    ).fetchone()[0]

    if invalid_game_grain_count > 0:
        raise RuntimeError(
            "Team-game injury source contains "
            f"{invalid_game_grain_count} games without "
            "exactly one home and one away row."
        )

    logger.info(
        "Game injury-feature source validated: %s.",
        SOURCE_FULL_NAME,
    )


def count_source_games(
    connection: duckdb.DuckDBPyConnection,
) -> int:
    """Count distinct games in the source."""

    return connection.execute(
        f"""
        SELECT COUNT(DISTINCT game_id)
        FROM {SOURCE_FULL_NAME}
        """
    ).fetchone()[0]


def create_game_injury_features(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create analytics.game_injury_features."""

    connection.execute(
        f"""
        CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA}
        """
    )

    connection.execute(
        f"""
        CREATE OR REPLACE TABLE {TARGET_FULL_NAME} AS

        SELECT
            home.game_id,
            home.season,
            home.game_type,
            home.week,
            home.gameday,
            home.team AS home_team,
            away.team AS away_team,

            home.has_injury_report_data
                AS home_has_injury_report_data,

            away.has_injury_report_data
                AS away_has_injury_report_data,

            (
                home.has_injury_report_data
                AND away.has_injury_report_data
            ) AS has_complete_injury_data,

            home.injury_report_player_count
                AS home_injury_report_player_count,

            away.injury_report_player_count
                AS away_injury_report_player_count,

            (
                home.injury_report_player_count
                - away.injury_report_player_count
            ) AS injury_report_player_count_difference,

            home.out_player_count
                AS home_out_player_count,

            away.out_player_count
                AS away_out_player_count,

            (
                home.out_player_count
                - away.out_player_count
            ) AS out_player_count_difference,

            home.doubtful_player_count
                AS home_doubtful_player_count,

            away.doubtful_player_count
                AS away_doubtful_player_count,

            (
                home.doubtful_player_count
                - away.doubtful_player_count
            ) AS doubtful_player_count_difference,

            home.questionable_player_count
                AS home_questionable_player_count,

            away.questionable_player_count
                AS away_questionable_player_count,

            (
                home.questionable_player_count
                - away.questionable_player_count
            ) AS questionable_player_count_difference,

            home.starter_out_count
                AS home_starter_out_count,

            away.starter_out_count
                AS away_starter_out_count,

            (
                home.starter_out_count
                - away.starter_out_count
            ) AS starter_out_count_difference,

            home.qb_out_count
                AS home_qb_out_count,

            away.qb_out_count
                AS away_qb_out_count,

            (
                home.qb_out_count
                - away.qb_out_count
            ) AS qb_out_count_difference,

            home.total_injury_burden
                AS home_total_injury_burden,

            away.total_injury_burden
                AS away_total_injury_burden,

            (
                home.total_injury_burden
                - away.total_injury_burden
            ) AS total_injury_burden_difference,

            home.qb_injury_burden
                AS home_qb_injury_burden,

            away.qb_injury_burden
                AS away_qb_injury_burden,

            (
                home.qb_injury_burden
                - away.qb_injury_burden
            ) AS qb_injury_burden_difference,

            home.non_qb_injury_burden
                AS home_non_qb_injury_burden,

            away.non_qb_injury_burden
                AS away_non_qb_injury_burden,

            (
                home.non_qb_injury_burden
                - away.non_qb_injury_burden
            ) AS non_qb_injury_burden_difference,

            home.offense_injury_burden
                AS home_offense_injury_burden,

            away.offense_injury_burden
                AS away_offense_injury_burden,

            (
                home.offense_injury_burden
                - away.offense_injury_burden
            ) AS offense_injury_burden_difference,

            home.defense_injury_burden
                AS home_defense_injury_burden,

            away.defense_injury_burden
                AS away_defense_injury_burden,

            (
                home.defense_injury_burden
                - away.defense_injury_burden
            ) AS defense_injury_burden_difference,

            home.special_teams_injury_burden
                AS home_special_teams_injury_burden,

            away.special_teams_injury_burden
                AS away_special_teams_injury_burden,

            (
                home.special_teams_injury_burden
                - away.special_teams_injury_burden
            ) AS special_teams_injury_burden_difference

        FROM {SOURCE_FULL_NAME} AS home

        INNER JOIN {SOURCE_FULL_NAME} AS away
            ON home.game_id = away.game_id
           AND home.team = away.opponent
           AND home.opponent = away.team
           AND home.is_home
           AND NOT away.is_home
        """
    )


def validate_target_table(
    connection: duckdb.DuckDBPyConnection,
    expected_game_count: int,
) -> tuple[int, int]:
    """Validate game-level injury features."""

    row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    if row_count != expected_game_count:
        raise RuntimeError(
            "Game injury-feature row count does not "
            "match source games: "
            f"{row_count} != {expected_game_count}."
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
            "Game injury features contain "
            f"{duplicate_count} duplicate games."
        )

    null_key_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE game_id IS NULL
           OR season IS NULL
           OR game_type IS NULL
           OR week IS NULL
           OR gameday IS NULL
           OR home_team IS NULL
           OR away_team IS NULL
           OR home_has_injury_report_data IS NULL
           OR away_has_injury_report_data IS NULL
           OR has_complete_injury_data IS NULL
           OR home_total_injury_burden IS NULL
           OR away_total_injury_burden IS NULL
           OR total_injury_burden_difference IS NULL
           OR non_qb_injury_burden_difference IS NULL
        """
    ).fetchone()[0]

    if null_key_count > 0:
        raise RuntimeError(
            "Game injury features contain "
            f"{null_key_count} rows with null required values."
        )

    invalid_coverage_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE has_complete_injury_data
            != (
                home_has_injury_report_data
                AND away_has_injury_report_data
            )
        """
    ).fetchone()[0]

    if invalid_coverage_count > 0:
        raise RuntimeError(
            "Game injury features contain "
            f"{invalid_coverage_count} invalid coverage flags."
        )

    invalid_score_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE home_total_injury_burden < 0
           OR away_total_injury_burden < 0
           OR home_qb_injury_burden < 0
           OR away_qb_injury_burden < 0
           OR home_non_qb_injury_burden < 0
           OR away_non_qb_injury_burden < 0
           OR home_offense_injury_burden < 0
           OR away_offense_injury_burden < 0
           OR home_defense_injury_burden < 0
           OR away_defense_injury_burden < 0
           OR home_special_teams_injury_burden < 0
           OR away_special_teams_injury_burden < 0
        """
    ).fetchone()[0]

    if invalid_score_count > 0:
        raise RuntimeError(
            "Game injury features contain "
            f"{invalid_score_count} invalid burden scores."
        )

    invalid_difference_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE ABS(
                total_injury_burden_difference
                - (
                    home_total_injury_burden
                    - away_total_injury_burden
                )
              ) > 0.000000001
           OR ABS(
                qb_injury_burden_difference
                - (
                    home_qb_injury_burden
                    - away_qb_injury_burden
                )
              ) > 0.000000001
           OR ABS(
                non_qb_injury_burden_difference
                - (
                    home_non_qb_injury_burden
                    - away_non_qb_injury_burden
                )
              ) > 0.000000001
           OR ABS(
                offense_injury_burden_difference
                - (
                    home_offense_injury_burden
                    - away_offense_injury_burden
                )
              ) > 0.000000001
           OR ABS(
                defense_injury_burden_difference
                - (
                    home_defense_injury_burden
                    - away_defense_injury_burden
                )
              ) > 0.000000001
           OR ABS(
                special_teams_injury_burden_difference
                - (
                    home_special_teams_injury_burden
                    - away_special_teams_injury_burden
                )
              ) > 0.000000001
           OR out_player_count_difference
                != (
                    home_out_player_count
                    - away_out_player_count
                )
           OR starter_out_count_difference
                != (
                    home_starter_out_count
                    - away_starter_out_count
                )
        """
    ).fetchone()[0]

    if invalid_difference_count > 0:
        raise RuntimeError(
            "Game injury features contain "
            f"{invalid_difference_count} invalid differences."
        )

    incomplete_game_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE NOT has_complete_injury_data
        """
    ).fetchone()[0]

    return (
        row_count,
        incomplete_game_count,
    )


def build_game_injury_features(
    database_file: Path = DATABASE_FILE,
) -> None:
    """Build analytics.game_injury_features."""

    validate_database_file(
        database_file
    )

    logger.info(
        "Starting game injury-feature build..."
    )

    try:
        with duckdb.connect(
            str(database_file)
        ) as connection:
            validate_source_table(
                connection
            )

            expected_game_count = count_source_games(
                connection
            )

            connection.execute(
                "BEGIN TRANSACTION"
            )

            try:
                create_game_injury_features(
                    connection
                )

                (
                    row_count,
                    incomplete_game_count,
                ) = validate_target_table(
                    connection=connection,
                    expected_game_count=(
                        expected_game_count
                    ),
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
            "Game injury-feature build failed."
        )
        raise

    logger.info(
        "Game injury-feature table validated: %s rows.",
        row_count,
    )
    logger.info(
        "Games without complete injury data: %s.",
        incomplete_game_count,
    )
    logger.info(
        "Game injury-feature build completed successfully."
    )


def main() -> None:
    """Run the game injury-feature builder."""

    build_game_injury_features()


if __name__ == "__main__":
    main()