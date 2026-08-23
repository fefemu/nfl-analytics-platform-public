"""
NFL Analytics Platform
Team-Game Injury Burden Builder

Purpose:
    Aggregate player injury-impact scores to one row per
    team and scheduled game for modeling and explanation.

QB policy:
    QB injury impact remains separately available and is
    excluded from generic non-QB and offensive burden to
    avoid double counting the project QB-rating layer.

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
SOURCE_TABLE = "player_injury_impact"
SOURCE_FULL_NAME = (
    f"{SOURCE_SCHEMA}.{SOURCE_TABLE}"
)

SCHEDULE_SCHEMA = "processed"
SCHEDULE_TABLE = "schedule"
SCHEDULE_FULL_NAME = (
    f"{SCHEDULE_SCHEMA}.{SCHEDULE_TABLE}"
)

SCHEDULE_REQUIRED_COLUMNS = {
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "home_team",
    "away_team",
}

TARGET_SCHEMA = "analytics"
TARGET_TABLE = "team_game_injury_burden"
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
    "gsis_id",
    "position",
    "full_name",
    "report_status",
    "practice_status",
    "is_out",
    "is_doubtful",
    "is_questionable",
    "did_not_practice",
    "limited_practice",
    "full_practice",
    "has_depth_chart_match",
    "depth_tier",
    "has_starter_role",
    "has_prior_snap_history",
    "snap_history_source",
    "is_qb",
    "availability_severity_score",
    "player_importance_score",
    "injury_impact_score",
    "non_qb_injury_impact_score",
    "offense_injury_impact_score",
    "defense_injury_impact_score",
    "special_teams_injury_impact_score",
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
    table_schema: str = SOURCE_SCHEMA,
    table_name: str = SOURCE_TABLE,
) -> set[str]:
    """Return columns for a DuckDB table."""

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
                table_schema,
                table_name,
            ],
        ).fetchall()
    }


def validate_source_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate the player injury-impact source."""

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
        - get_table_columns(
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

    logger.info(
        "Team injury-burden source validated: %s.",
        SOURCE_FULL_NAME,
    )


def validate_schedule_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate the schedule source used as target grain."""

    table_exists = connection.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = ?
          AND table_name = ?
        """,
        [
            SCHEDULE_SCHEMA,
            SCHEDULE_TABLE,
        ],
    ).fetchone()[0]

    if table_exists == 0:
        raise RuntimeError(
            f"Schedule table does not exist: {SCHEDULE_FULL_NAME}"
        )

    schedule_columns = get_table_columns(
        connection=connection,
        table_schema=SCHEDULE_SCHEMA,
        table_name=SCHEDULE_TABLE,
    )

    missing_columns = sorted(
        SCHEDULE_REQUIRED_COLUMNS
        - schedule_columns
    )

    if missing_columns:
        raise RuntimeError(
            f"Schedule table {SCHEDULE_FULL_NAME} "
            "is missing columns: "
            + ", ".join(
                missing_columns
            )
        )

    logger.info(
        "Team injury-burden schedule source validated: %s.",
        SCHEDULE_FULL_NAME,
    )


def count_schedule_team_games(
    connection: duckdb.DuckDBPyConnection,
) -> int:
    """Count scheduled team-games across source seasons."""

    return connection.execute(
        f"""
        WITH source_season_bounds AS (
            SELECT
                MIN(season) AS minimum_season,
                MAX(season) AS maximum_season
            FROM {SOURCE_FULL_NAME}
        )
        SELECT
            COUNT(*) * 2
        FROM {SCHEDULE_FULL_NAME}
        CROSS JOIN source_season_bounds
        WHERE season BETWEEN
            minimum_season
            AND maximum_season
        """
    ).fetchone()[0]


def create_team_game_injury_burden(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create a schedule-complete team-game injury table."""

    connection.execute(
        f"""
        CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA}
        """
    )

    connection.execute(
        f"""
        CREATE OR REPLACE TABLE {TARGET_FULL_NAME} AS

        WITH source_season_bounds AS (
            SELECT
                MIN(season) AS minimum_season,
                MAX(season) AS maximum_season
            FROM {SOURCE_FULL_NAME}
        ),

        schedule_team_games AS (
            SELECT
                schedule.game_id,
                schedule.season,
                schedule.game_type,
                schedule.week,
                schedule.gameday,
                schedule.home_team AS team,
                schedule.away_team AS opponent,
                TRUE AS is_home
            FROM {SCHEDULE_FULL_NAME} AS schedule
            CROSS JOIN source_season_bounds
            WHERE schedule.season BETWEEN
                minimum_season
                AND maximum_season

            UNION ALL

            SELECT
                schedule.game_id,
                schedule.season,
                schedule.game_type,
                schedule.week,
                schedule.gameday,
                schedule.away_team AS team,
                schedule.home_team AS opponent,
                FALSE AS is_home
            FROM {SCHEDULE_FULL_NAME} AS schedule
            CROSS JOIN source_season_bounds
            WHERE schedule.season BETWEEN
                minimum_season
                AND maximum_season
        ),

        injury_burden AS (
            SELECT
                game_id,
                team,

                COUNT(*) AS injury_report_player_count,

                COUNT(*) FILTER (
                    WHERE report_status IS NOT NULL
                ) AS game_status_player_count,

                COUNT(*) FILTER (
                    WHERE is_out
                ) AS out_player_count,

                COUNT(*) FILTER (
                    WHERE is_doubtful
                ) AS doubtful_player_count,

                COUNT(*) FILTER (
                    WHERE is_questionable
                ) AS questionable_player_count,

                COUNT(*) FILTER (
                    WHERE did_not_practice
                ) AS did_not_practice_player_count,

                COUNT(*) FILTER (
                    WHERE limited_practice
                ) AS limited_practice_player_count,

                COUNT(*) FILTER (
                    WHERE full_practice
                ) AS full_practice_player_count,

                COUNT(*) FILTER (
                    WHERE has_starter_role
                      AND report_status IS NOT NULL
                ) AS starter_game_status_count,

                COUNT(*) FILTER (
                    WHERE has_starter_role
                      AND is_out
                ) AS starter_out_count,

                COUNT(*) FILTER (
                    WHERE is_qb
                      AND report_status IS NOT NULL
                ) AS qb_game_status_count,

                COUNT(*) FILTER (
                    WHERE is_qb
                      AND is_out
                ) AS qb_out_count,

                COUNT(*) FILTER (
                    WHERE NOT is_qb
                      AND report_status IS NOT NULL
                ) AS non_qb_game_status_count,

                COUNT(*) FILTER (
                    WHERE NOT has_depth_chart_match
                ) AS missing_depth_chart_count,

                COUNT(*) FILTER (
                    WHERE NOT has_prior_snap_history
                ) AS missing_snap_history_count,

                COUNT(*) FILTER (
                    WHERE snap_history_source = 'CAREER'
                ) AS career_snap_fallback_count,

                AVG(
                    CAST(
                        has_depth_chart_match
                        AS DOUBLE
                    )
                ) AS depth_chart_match_rate,

                AVG(
                    CAST(
                        has_prior_snap_history
                        AS DOUBLE
                    )
                ) AS snap_history_match_rate,

                SUM(
                    availability_severity_score
                ) AS availability_severity_sum,

                SUM(
                    injury_impact_score
                ) AS total_injury_burden,

                SUM(
                    CASE
                        WHEN is_qb
                            THEN injury_impact_score
                        ELSE 0.0
                    END
                ) AS qb_injury_burden,

                SUM(
                    non_qb_injury_impact_score
                ) AS non_qb_injury_burden,

                SUM(
                    offense_injury_impact_score
                ) AS offense_injury_burden,

                SUM(
                    defense_injury_impact_score
                ) AS defense_injury_burden,

                SUM(
                    special_teams_injury_impact_score
                ) AS special_teams_injury_burden,

                MAX(
                    injury_impact_score
                ) AS maximum_player_injury_impact,

                AVG(
                    injury_impact_score
                ) FILTER (
                    WHERE report_status IS NOT NULL
                ) AS average_game_status_injury_impact,

                ARG_MAX(
                    full_name,
                    injury_impact_score
                ) FILTER (
                    WHERE injury_impact_score > 0
                ) AS top_impact_player_name,

                ARG_MAX(
                    gsis_id,
                    injury_impact_score
                ) FILTER (
                    WHERE injury_impact_score > 0
                ) AS top_impact_player_gsis_id,

                ARG_MAX(
                    position,
                    injury_impact_score
                ) FILTER (
                    WHERE injury_impact_score > 0
                ) AS top_impact_player_position,

                ARG_MAX(
                    report_status,
                    injury_impact_score
                ) FILTER (
                    WHERE injury_impact_score > 0
                ) AS top_impact_player_status,

                ARG_MAX(
                    injury_impact_score,
                    injury_impact_score
                ) FILTER (
                    WHERE injury_impact_score > 0
                ) AS top_impact_player_score

            FROM {SOURCE_FULL_NAME}

            GROUP BY
                game_id,
                team
        )

        SELECT
            schedule.game_id,
            schedule.season,
            schedule.game_type,
            schedule.week,
            schedule.gameday,
            schedule.team,
            schedule.opponent,
            schedule.is_home,

            burden.game_id IS NOT NULL
                AS has_injury_report_data,

            COALESCE(
                burden.injury_report_player_count,
                0
            ) AS injury_report_player_count,

            COALESCE(
                burden.game_status_player_count,
                0
            ) AS game_status_player_count,

            COALESCE(
                burden.out_player_count,
                0
            ) AS out_player_count,

            COALESCE(
                burden.doubtful_player_count,
                0
            ) AS doubtful_player_count,

            COALESCE(
                burden.questionable_player_count,
                0
            ) AS questionable_player_count,

            COALESCE(
                burden.did_not_practice_player_count,
                0
            ) AS did_not_practice_player_count,

            COALESCE(
                burden.limited_practice_player_count,
                0
            ) AS limited_practice_player_count,

            COALESCE(
                burden.full_practice_player_count,
                0
            ) AS full_practice_player_count,

            COALESCE(
                burden.starter_game_status_count,
                0
            ) AS starter_game_status_count,

            COALESCE(
                burden.starter_out_count,
                0
            ) AS starter_out_count,

            COALESCE(
                burden.qb_game_status_count,
                0
            ) AS qb_game_status_count,

            COALESCE(
                burden.qb_out_count,
                0
            ) AS qb_out_count,

            COALESCE(
                burden.non_qb_game_status_count,
                0
            ) AS non_qb_game_status_count,

            COALESCE(
                burden.missing_depth_chart_count,
                0
            ) AS missing_depth_chart_count,

            COALESCE(
                burden.missing_snap_history_count,
                0
            ) AS missing_snap_history_count,

            COALESCE(
                burden.career_snap_fallback_count,
                0
            ) AS career_snap_fallback_count,

            burden.depth_chart_match_rate,
            burden.snap_history_match_rate,

            COALESCE(
                burden.availability_severity_sum,
                0.0
            ) AS availability_severity_sum,

            COALESCE(
                burden.total_injury_burden,
                0.0
            ) AS total_injury_burden,

            COALESCE(
                burden.qb_injury_burden,
                0.0
            ) AS qb_injury_burden,

            COALESCE(
                burden.non_qb_injury_burden,
                0.0
            ) AS non_qb_injury_burden,

            COALESCE(
                burden.offense_injury_burden,
                0.0
            ) AS offense_injury_burden,

            COALESCE(
                burden.defense_injury_burden,
                0.0
            ) AS defense_injury_burden,

            COALESCE(
                burden.special_teams_injury_burden,
                0.0
            ) AS special_teams_injury_burden,

            burden.maximum_player_injury_impact,
            burden.average_game_status_injury_impact,
            burden.top_impact_player_name,
            burden.top_impact_player_gsis_id,
            burden.top_impact_player_position,
            burden.top_impact_player_status,
            burden.top_impact_player_score

        FROM schedule_team_games AS schedule

        LEFT JOIN injury_burden AS burden
            ON schedule.game_id = burden.game_id
           AND schedule.team = burden.team
        """
    )


def validate_target_table(
    connection: duckdb.DuckDBPyConnection,
    expected_team_game_count: int,
) -> tuple[int, int, int, int, int]:
    """Validate team-game injury burden."""

    row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    if row_count != expected_team_game_count:
        raise RuntimeError(
            "Team injury-burden row count does not "
            "match scheduled team-games: "
            f"{row_count} != {expected_team_game_count}."
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
            "Team injury burden contains "
            f"{duplicate_count} duplicate team-games."
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
           OR is_home IS NULL
           OR has_injury_report_data IS NULL
           OR injury_report_player_count IS NULL
        """
    ).fetchone()[0]

    if null_key_count > 0:
        raise RuntimeError(
            "Team injury burden contains "
            f"{null_key_count} rows with null business keys."
        )

    inconsistent_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE game_status_player_count
                != (
                    out_player_count
                    + doubtful_player_count
                    + questionable_player_count
                )
           OR starter_out_count
                > starter_game_status_count
           OR qb_out_count
                > qb_game_status_count
           OR qb_game_status_count
                + non_qb_game_status_count
                != game_status_player_count
           OR missing_depth_chart_count
                > injury_report_player_count
           OR missing_snap_history_count
                > injury_report_player_count
           OR career_snap_fallback_count
                > injury_report_player_count
           OR (
                has_injury_report_data
                AND injury_report_player_count = 0
           )
           OR (
                NOT has_injury_report_data
                AND (
                    injury_report_player_count != 0
                    OR game_status_player_count != 0
                    OR out_player_count != 0
                    OR doubtful_player_count != 0
                    OR questionable_player_count != 0
                    OR did_not_practice_player_count != 0
                    OR limited_practice_player_count != 0
                    OR full_practice_player_count != 0
                    OR total_injury_burden != 0
                    OR qb_injury_burden != 0
                    OR non_qb_injury_burden != 0
                    OR offense_injury_burden != 0
                    OR defense_injury_burden != 0
                    OR special_teams_injury_burden != 0
                    OR depth_chart_match_rate IS NOT NULL
                    OR snap_history_match_rate IS NOT NULL
                    OR maximum_player_injury_impact IS NOT NULL
                    OR average_game_status_injury_impact IS NOT NULL
                    OR top_impact_player_name IS NOT NULL
                    OR top_impact_player_gsis_id IS NOT NULL
                    OR top_impact_player_position IS NOT NULL
                    OR top_impact_player_status IS NOT NULL
                    OR top_impact_player_score IS NOT NULL
                )
           )
        """
    ).fetchone()[0]

    if inconsistent_count > 0:
        raise RuntimeError(
            "Team injury burden contains "
            f"{inconsistent_count} inconsistent counts."
        )

    invalid_score_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE depth_chart_match_rate NOT BETWEEN 0 AND 1
           OR snap_history_match_rate NOT BETWEEN 0 AND 1
           OR availability_severity_sum < 0
           OR total_injury_burden < 0
           OR qb_injury_burden < 0
           OR non_qb_injury_burden < 0
           OR offense_injury_burden < 0
           OR defense_injury_burden < 0
           OR special_teams_injury_burden < 0
           OR maximum_player_injury_impact NOT BETWEEN 0 AND 1
           OR average_game_status_injury_impact
                NOT BETWEEN 0 AND 1
           OR top_impact_player_score
                NOT BETWEEN 0 AND 1
        """
    ).fetchone()[0]

    if invalid_score_count > 0:
        raise RuntimeError(
            "Team injury burden contains "
            f"{invalid_score_count} invalid scores."
        )

    inconsistent_top_player_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE (
                total_injury_burden = 0
                AND (
                    top_impact_player_name IS NOT NULL
                    OR top_impact_player_score IS NOT NULL
                )
              )
           OR (
                total_injury_burden > 0
                AND (
                    top_impact_player_name IS NULL
                    OR top_impact_player_score IS NULL
                )
              )
            OR (
                    total_injury_burden > 0
                    AND top_impact_player_score
                        IS DISTINCT FROM
                            maximum_player_injury_impact
            )
        """
    ).fetchone()[0]

    if inconsistent_top_player_count > 0:
        raise RuntimeError(
            "Team injury burden contains "
            f"{inconsistent_top_player_count} inconsistent "
            "top-player fields."
        )

    team_game_with_burden_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE total_injury_burden > 0
        """
    ).fetchone()[0]

    team_game_with_qb_burden_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE qb_injury_burden > 0
        """
    ).fetchone()[0]

    team_game_with_non_qb_burden_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE non_qb_injury_burden > 0
        """
    ).fetchone()[0]

    missing_injury_report_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE NOT has_injury_report_data
        """
    ).fetchone()[0]

    return (
        row_count,
        team_game_with_burden_count,
        team_game_with_qb_burden_count,
        team_game_with_non_qb_burden_count,
        missing_injury_report_count,
    )


def build_team_game_injury_burden(
    database_file: Path = DATABASE_FILE,
) -> None:
    """Build analytics.team_game_injury_burden."""

    validate_database_file(
        database_file
    )

    logger.info(
        "Starting team-game injury-burden build..."
    )

    try:
        with duckdb.connect(
            str(database_file)
        ) as connection:
            validate_source_table(
                connection
            )

            validate_schedule_table(
                connection
            )

            expected_team_game_count = (
                count_schedule_team_games(
                    connection
                )
            )

            connection.execute(
                "BEGIN TRANSACTION"
            )

            try:
                create_team_game_injury_burden(
                    connection
                )

                (
                    row_count,
                    team_game_with_burden_count,
                    team_game_with_qb_burden_count,
                    team_game_with_non_qb_burden_count,
                    missing_injury_report_count,
                ) = validate_target_table(
                    connection=connection,
                    expected_team_game_count=(
                        expected_team_game_count
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
            "Team-game injury-burden build failed."
        )
        raise

    logger.info(
        "Team-game injury burden validated: %s rows.",
        row_count,
    )
    logger.info(
        "Team-games with positive burden: %s.",
        team_game_with_burden_count,
    )
    logger.info(
        "Team-games with QB burden: %s.",
        team_game_with_qb_burden_count,
    )
    logger.info(
        "Team-games with non-QB burden: %s.",
        team_game_with_non_qb_burden_count,
    )
    logger.info(
        "Team-games without injury-report data: %s.",
        missing_injury_report_count,
    )
    logger.info(
        "Team-game injury-burden build "
        "completed successfully."
    )


def main() -> None:
    """Run the team-game injury-burden builder."""

    build_team_game_injury_burden()


if __name__ == "__main__":
    main()