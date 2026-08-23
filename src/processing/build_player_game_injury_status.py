"""
NFL Analytics Platform
Player-Game Injury Status Builder

Purpose:
    Build one cleaned pregame injury-status record per
    player and NFL game from raw weekly injury reports.

Notes:
    When multiple timestamped source snapshots exist for
    one player-team-week, the latest available snapshot is
    selected.

    The source does not provide a complete daily snapshot
    history for every season. The resulting table represents
    the final available weekly game-day report, not an
    arbitrary earlier prediction-time snapshot.

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

INJURY_SOURCE_SCHEMA = "raw"
INJURY_SOURCE_TABLE = "injury_reports"
INJURY_SOURCE_FULL_NAME = (
    f"{INJURY_SOURCE_SCHEMA}.{INJURY_SOURCE_TABLE}"
)

SCHEDULE_SOURCE_SCHEMA = "processed"
SCHEDULE_SOURCE_TABLE = "schedule"
SCHEDULE_SOURCE_FULL_NAME = (
    f"{SCHEDULE_SOURCE_SCHEMA}.{SCHEDULE_SOURCE_TABLE}"
)

TARGET_SCHEMA = "processed"
TARGET_TABLE = "player_game_injury_status"
TARGET_FULL_NAME = f"{TARGET_SCHEMA}.{TARGET_TABLE}"

KNOWN_UNPLAYED_TEAM_WEEKS = {
    (
        2022,
        "REG",
        "BUF",
        17,
    ),
    (
        2022,
        "REG",
        "CIN",
        17,
    ),
}

REQUIRED_INJURY_COLUMNS = {
    "season",
    "season_type",
    "game_type",
    "team",
    "week",
    "gsis_id",
    "position",
    "full_name",
    "report_primary_injury",
    "report_secondary_injury",
    "report_status",
    "practice_primary_injury",
    "practice_secondary_injury",
    "practice_status",
    "date_modified",
}

REQUIRED_SCHEDULE_COLUMNS = {
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "away_team",
    "home_team",
}

VALID_REPORT_STATUSES = {
    "Out",
    "Doubtful",
    "Questionable",
}

VALID_PRACTICE_STATUSES = {
    "Did Not Participate In Practice",
    "Limited Participation in Practice",
    "Full Participation in Practice",
}


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


def get_table_columns(
    connection: duckdb.DuckDBPyConnection,
    schema_name: str,
    table_name: str,
) -> set[str]:
    """Return the available columns for one DuckDB table."""

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
    """Validate one required DuckDB source table."""

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
    """Validate injury and schedule source tables."""

    validate_source_table(
        connection=connection,
        schema_name=INJURY_SOURCE_SCHEMA,
        table_name=INJURY_SOURCE_TABLE,
        required_columns=REQUIRED_INJURY_COLUMNS,
    )

    validate_source_table(
        connection=connection,
        schema_name=SCHEDULE_SOURCE_SCHEMA,
        table_name=SCHEDULE_SOURCE_TABLE,
        required_columns=REQUIRED_SCHEDULE_COLUMNS,
    )

    logger.info(
        "Player-game injury sources validated: %s and %s.",
        INJURY_SOURCE_FULL_NAME,
        SCHEDULE_SOURCE_FULL_NAME,
    )


def count_source_player_week_keys(
    connection: duckdb.DuckDBPyConnection,
) -> int:
    """Count distinct injury player-team-week source keys."""

    return connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT DISTINCT
                season,
                game_type,
                team,
                week,
                gsis_id
            FROM {INJURY_SOURCE_FULL_NAME}
        )
        """
    ).fetchone()[0]


def validate_schedule_coverage(
    connection: duckdb.DuckDBPyConnection,
    source_key_count: int,
) -> tuple[int, int]:
    """Validate unmatched injury keys against known unplayed games."""

    unmatched_rows = connection.execute(
        f"""
        WITH injury_keys AS (
            SELECT DISTINCT
                season,
                game_type,
                team,
                week,
                gsis_id
            FROM {INJURY_SOURCE_FULL_NAME}
        )
        SELECT
            injury.season,
            injury.game_type,
            injury.team,
            injury.week,
            injury.gsis_id
        FROM injury_keys AS injury
        LEFT JOIN {SCHEDULE_SOURCE_FULL_NAME} AS schedule
            ON injury.season = schedule.season
           AND injury.game_type = schedule.game_type
           AND injury.week = schedule.week
           AND (
                injury.team = schedule.home_team
                OR injury.team = schedule.away_team
           )
        WHERE schedule.game_id IS NULL
        """
    ).fetchall()

    unmatched_team_weeks = {
        (
            row[0],
            row[1],
            row[2],
            row[3],
        )
        for row in unmatched_rows
    }

    unexpected_team_weeks = sorted(
        unmatched_team_weeks
        - KNOWN_UNPLAYED_TEAM_WEEKS
    )

    if unexpected_team_weeks:
        formatted_keys = ", ".join(
            str(team_week)
            for team_week in unexpected_team_weeks
        )
        raise RuntimeError(
            "Injury source contains unexpected unmatched "
            "team-week keys: "
            f"{formatted_keys}"
        )

    unmatched_key_count = len(
        unmatched_rows
    )
    expected_target_count = (
        source_key_count
        - unmatched_key_count
    )

    logger.info(
        "Injury schedule coverage validated: "
        "%s matched keys and %s known unplayed-game keys.",
        expected_target_count,
        unmatched_key_count,
    )

    return (
        expected_target_count,
        unmatched_key_count,
    )


def create_player_game_injury_status(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create the processed player-game injury table."""

    connection.execute(
        f"""
        CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA}
        """
    )

    connection.execute(
        f"""
        CREATE OR REPLACE TABLE {TARGET_FULL_NAME} AS
        WITH matched_snapshots AS (
            SELECT
                schedule.game_id,
                injury.season,
                injury.season_type,
                injury.game_type,
                injury.week,
                schedule.gameday,
                injury.team,
                CASE
                    WHEN injury.team = schedule.home_team
                        THEN schedule.away_team
                    ELSE schedule.home_team
                END AS opponent,
                injury.team = schedule.home_team AS is_home,
                injury.gsis_id,
                injury.position,
                injury.full_name,
                injury.report_primary_injury,
                injury.report_secondary_injury,
                injury.report_status,
                injury.practice_primary_injury,
                injury.practice_secondary_injury,
                injury.practice_status,
                injury.date_modified,
                COUNT(*) OVER (
                    PARTITION BY
                        injury.season,
                        injury.game_type,
                        injury.team,
                        injury.week,
                        injury.gsis_id
                ) AS source_snapshot_count,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        injury.season,
                        injury.game_type,
                        injury.team,
                        injury.week,
                        injury.gsis_id
                    ORDER BY
                        injury.date_modified DESC NULLS LAST,
                        injury.report_status DESC NULLS LAST,
                        injury.practice_status DESC NULLS LAST
                ) AS snapshot_rank
            FROM {INJURY_SOURCE_FULL_NAME} AS injury
            INNER JOIN {SCHEDULE_SOURCE_FULL_NAME} AS schedule
                ON injury.season = schedule.season
               AND injury.game_type = schedule.game_type
               AND injury.week = schedule.week
               AND (
                    injury.team = schedule.home_team
                    OR injury.team = schedule.away_team
               )
        ),
        latest_snapshots AS (
            SELECT *
            FROM matched_snapshots
            WHERE snapshot_rank = 1
        ),
        cleaned_snapshots AS (
            SELECT
                game_id,
                season,
                season_type,
                game_type,
                week,
                gameday,
                team,
                opponent,
                is_home,
                gsis_id,
                NULLIF(
                    TRIM(
                        CAST(position AS VARCHAR)
                    ),
                    ''
                ) AS position,
                NULLIF(
                    TRIM(
                        CAST(full_name AS VARCHAR)
                    ),
                    ''
                ) AS full_name,
                NULLIF(
                    TRIM(
                        CAST(
                            report_primary_injury
                            AS VARCHAR
                        )
                    ),
                    ''
                ) AS report_primary_injury,
                NULLIF(
                    TRIM(
                        CAST(
                            report_secondary_injury
                            AS VARCHAR
                        )
                    ),
                    ''
                ) AS report_secondary_injury,
                CASE
                    WHEN TRIM(
                        CAST(
                            report_status AS VARCHAR
                        )
                    ) IN (
                        'Out',
                        'Doubtful',
                        'Questionable'
                    )
                        THEN TRIM(
                            CAST(
                                report_status AS VARCHAR
                            )
                        )
                    ELSE NULL
                END AS report_status,
                NULLIF(
                    TRIM(
                        CAST(
                            practice_primary_injury
                            AS VARCHAR
                        )
                    ),
                    ''
                ) AS practice_primary_injury,
                NULLIF(
                    TRIM(
                        CAST(
                            practice_secondary_injury
                            AS VARCHAR
                        )
                    ),
                    ''
                ) AS practice_secondary_injury,
                CASE
                    WHEN TRIM(
                        CAST(
                            practice_status AS VARCHAR
                        )
                    ) IN (
                        'Did Not Participate In Practice',
                        'Limited Participation in Practice',
                        'Full Participation in Practice'
                    )
                        THEN TRIM(
                            CAST(
                                practice_status AS VARCHAR
                            )
                        )
                    ELSE NULL
                END AS practice_status,
                date_modified AS source_date_modified,
                date_modified IS NOT NULL
                    AS has_source_timestamp,
                source_snapshot_count
            FROM latest_snapshots
        )
        SELECT
            game_id,
            season,
            season_type,
            game_type,
            week,
            gameday,
            team,
            opponent,
            is_home,
            gsis_id,
            position,
            full_name,
            report_primary_injury,
            report_secondary_injury,
            report_status,
            practice_primary_injury,
            practice_secondary_injury,
            practice_status,
            COALESCE(
                report_status = 'Out',
                FALSE
            ) AS is_out,
            COALESCE(
                report_status = 'Doubtful',
                FALSE
            ) AS is_doubtful,
            COALESCE(
                report_status = 'Questionable',
                FALSE
            ) AS is_questionable,
            COALESCE(
                practice_status =
                    'Did Not Participate In Practice',
                FALSE
            ) AS did_not_practice,
            COALESCE(
                practice_status =
                    'Limited Participation in Practice',
                FALSE
            ) AS limited_practice,
            COALESCE(
                practice_status =
                    'Full Participation in Practice',
                FALSE
            ) AS full_practice,
            source_date_modified,
            has_source_timestamp,
            source_snapshot_count
        FROM cleaned_snapshots
        """
    )


def validate_target_table(
    connection: duckdb.DuckDBPyConnection,
    source_key_count: int,
) -> tuple[int, int]:
    """Validate the processed player-game injury table."""

    row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    if row_count == 0:
        raise RuntimeError(
            "Processed player-game injury table is empty."
        )

    if row_count != source_key_count:
        raise RuntimeError(
            "Not every distinct injury player-team-week key "
            "matched exactly one scheduled game: "
            f"{row_count} processed rows != "
            f"{source_key_count} source keys."
        )

    duplicate_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT
                game_id,
                team,
                gsis_id,
                COUNT(*) AS record_count
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
            "Processed player-game injury table contains "
            "duplicate player-game records."
        )

    null_key_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE game_id IS NULL
           OR season IS NULL
           OR team IS NULL
           OR gsis_id IS NULL
        """
    ).fetchone()[0]

    if null_key_count > 0:
        raise RuntimeError(
            "Processed player-game injury table contains "
            "null business keys."
        )

    invalid_report_status_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE report_status IS NOT NULL
          AND report_status NOT IN (
              'Out',
              'Doubtful',
              'Questionable'
          )
        """
    ).fetchone()[0]

    if invalid_report_status_count > 0:
        raise RuntimeError(
            "Processed player-game injury table contains "
            "invalid report statuses."
        )

    invalid_practice_status_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE practice_status IS NOT NULL
          AND practice_status NOT IN (
              'Did Not Participate In Practice',
              'Limited Participation in Practice',
              'Full Participation in Practice'
          )
        """
    ).fetchone()[0]

    if invalid_practice_status_count > 0:
        raise RuntimeError(
            "Processed player-game injury table contains "
            "invalid practice statuses."
        )

    multiple_snapshot_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE source_snapshot_count > 1
        """
    ).fetchone()[0]

    return (
        row_count,
        multiple_snapshot_count,
    )


def build_player_game_injury_status(
    database_file: Path = DATABASE_FILE,
) -> None:
    """Build processed.player_game_injury_status."""

    validate_database_file(
        database_file
    )

    logger.info(
        "Starting player-game injury status build..."
    )

    try:
        with duckdb.connect(
            str(database_file)
        ) as connection:
            validate_source_tables(
                connection
            )

            source_key_count = (
                count_source_player_week_keys(
                    connection
                )
            )

            (
                expected_target_count,
                unmatched_unplayed_count,
            ) = validate_schedule_coverage(
                connection=connection,
                source_key_count=source_key_count,
            )

            create_player_game_injury_status(
                connection
            )

            (
                row_count,
                multiple_snapshot_count,
            ) = validate_target_table(
                connection=connection,
                source_key_count=expected_target_count,
            )

    except Exception:
        logger.exception(
            "Player-game injury status build failed."
        )
        raise

    logger.info(
        "Player-game injury status table validated: "
        "%s rows.",
        row_count,
    )
    logger.info(
        "Player-game records selected from multiple "
        "source snapshots: %s.",
        multiple_snapshot_count,
    )
    logger.info(
        "Known unplayed-game injury keys excluded: %s.",
        unmatched_unplayed_count,
    )
    logger.info(
        "Player-game injury status build completed "
        "successfully."
    )


def main() -> None:
    """Run the player-game injury builder."""

    build_player_game_injury_status()


if __name__ == "__main__":
    main()