"""
NFL Analytics Platform
Quarterback Game Performance Builder

Purpose:
    Build one quarterback-level performance record per NFL game
    from nflverse play-by-play data.

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
PBP_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "pbp"

SCHEDULE_SCHEMA = "processed"
SCHEDULE_TABLE = "schedule"
SCHEDULE_FULL_NAME = (
    f"{SCHEDULE_SCHEMA}.{SCHEDULE_TABLE}"
)

TARGET_SCHEMA = "processed"
TARGET_TABLE = "qb_game_performance"
TARGET_FULL_NAME = f"{TARGET_SCHEMA}.{TARGET_TABLE}"

REQUIRED_PBP_COLUMNS = {
    "game_id",
    "season",
    "season_type",
    "week",
    "game_date",
    "posteam",
    "defteam",
    "home_team",
    "away_team",
    "wp",
    "qb_kneel",
    "qb_spike",
    "aborted_play",
    "qb_dropback",
    "qb_scramble",
    "passer_player_id",
    "passer_player_name",
    "rusher_player_id",
    "rusher_player_name",
    "pass_attempt",
    "complete_pass",
    "incomplete_pass",
    "passing_yards",
    "air_yards",
    "sack",
    "qb_hit",
    "interception",
    "fumble_lost",
    "two_point_attempt",
    "special_teams_play",
    "epa",
    "success",
    "cpoe",
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


def get_pbp_files(
    pbp_directory: Path = PBP_DIRECTORY,
) -> list[Path]:
    """Return available season-level PBP Parquet files."""

    if not pbp_directory.exists():
        raise FileNotFoundError(
            f"PBP directory does not exist: {pbp_directory}"
        )

    pbp_files = sorted(
        pbp_directory.glob("pbp_*.parquet")
    )

    if not pbp_files:
        raise FileNotFoundError(
            f"No PBP Parquet files found in: {pbp_directory}"
        )

    logger.info(
        "PBP files discovered: %s file(s) in %s.",
        len(pbp_files),
        pbp_directory,
    )

    return pbp_files


def build_parquet_source(
    pbp_files: list[Path],
) -> str:
    """Build a DuckDB-compatible Parquet file-list expression."""

    escaped_paths = [
        str(path.resolve()).replace("'", "''")
        for path in pbp_files
    ]

    quoted_paths = ", ".join(
        f"'{path}'"
        for path in escaped_paths
    )

    return f"[{quoted_paths}]"


def validate_pbp_columns(
    connection: duckdb.DuckDBPyConnection,
    parquet_source: str,
) -> None:
    """Validate columns required for QB performance metrics."""

    description = connection.execute(
        f"""
        SELECT *
        FROM read_parquet(
            {parquet_source},
            union_by_name = true
        )
        LIMIT 0
        """
    ).description

    available_columns = {
        column[0]
        for column in description
    }

    missing_columns = sorted(
        REQUIRED_PBP_COLUMNS - available_columns
    )

    if missing_columns:
        missing_names = ", ".join(missing_columns)
        raise RuntimeError(
            f"Missing required QB PBP columns: {missing_names}"
        )

    logger.info(
        "Required QB PBP columns validated successfully: "
        "%s columns.",
        len(REQUIRED_PBP_COLUMNS),
    )


def validate_schedule_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate schedule fields used for starter identification."""

    table_exists = connection.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = ?
          AND table_name = ?
        """,
        [SCHEDULE_SCHEMA, SCHEDULE_TABLE],
    ).fetchone()[0]

    if table_exists == 0:
        raise RuntimeError(
            f"Schedule table does not exist: {SCHEDULE_FULL_NAME}"
        )

    required_columns = {
        "game_id",
        "home_team",
        "away_team",
        "home_qb_id",
        "home_qb_name",
        "away_qb_id",
        "away_qb_name",
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
            [SCHEDULE_SCHEMA, SCHEDULE_TABLE],
        ).fetchall()
    }

    missing_columns = sorted(
        required_columns - available_columns
    )

    if missing_columns:
        missing_names = ", ".join(missing_columns)
        raise RuntimeError(
            "Missing required schedule QB columns: "
            f"{missing_names}"
        )

    logger.info(
        "Schedule QB fields validated: %s.",
        SCHEDULE_FULL_NAME,
    )


def create_valid_qb_dropbacks(
    connection: duckdb.DuckDBPyConnection,
    parquet_source: str,
) -> None:
    """Create valid quarterback dropbacks with unified QB identity."""

    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE valid_qb_dropbacks AS
        SELECT
            game_id,
            season,
            season_type,
            week,
            CAST(game_date AS DATE) AS game_date,
            posteam AS team,
            defteam AS opponent,
            home_team,
            away_team,

            CASE
                WHEN qb_scramble = 1
                THEN COALESCE(
                    rusher_player_id,
                    passer_player_id
                )
                ELSE passer_player_id
            END AS qb_id,

            CASE
                WHEN qb_scramble = 1
                THEN COALESCE(
                    rusher_player_name,
                    passer_player_name
                )
                ELSE passer_player_name
            END AS qb_name,

            CASE
                WHEN posteam = home_team THEN TRUE
                ELSE FALSE
            END AS is_home,

            CASE
                WHEN wp BETWEEN 0.10 AND 0.90 THEN TRUE
                ELSE FALSE
            END AS is_competitive_dropback,

            CASE
                WHEN COALESCE(pass_attempt, 0) = 1
                 AND COALESCE(sack, 0) = 0
                 AND COALESCE(qb_scramble, 0) = 0
                THEN TRUE
                ELSE FALSE
            END AS is_throw_attempt,

            COALESCE(complete_pass, 0)
                AS complete_pass,

            COALESCE(incomplete_pass, 0)
                AS incomplete_pass,

            COALESCE(passing_yards, 0)
                AS passing_yards,

            air_yards,
            COALESCE(sack, 0) AS sack,
            COALESCE(qb_hit, 0) AS qb_hit,
            COALESCE(qb_scramble, 0) AS qb_scramble,
            COALESCE(interception, 0) AS interception,
            COALESCE(fumble_lost, 0) AS fumble_lost,
            epa,
            success,
            cpoe

        FROM read_parquet(
            {parquet_source},
            union_by_name = true
        )

        WHERE game_id IS NOT NULL
          AND posteam IS NOT NULL
          AND defteam IS NOT NULL
          AND qb_dropback = 1
          AND epa IS NOT NULL
          AND COALESCE(qb_kneel, 0) = 0
          AND COALESCE(qb_spike, 0) = 0
          AND COALESCE(aborted_play, 0) = 0
          AND COALESCE(two_point_attempt, 0) = 0
          AND COALESCE(special_teams_play, 0) = 0
          AND (
                CASE
                    WHEN qb_scramble = 1
                    THEN COALESCE(
                        rusher_player_id,
                        passer_player_id
                    )
                    ELSE passer_player_id
                END
              ) IS NOT NULL
        """
    )

    dropback_count = connection.execute(
        """
        SELECT COUNT(*)
        FROM valid_qb_dropbacks
        """
    ).fetchone()[0]

    qb_game_count = connection.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT DISTINCT
                game_id,
                qb_id
            FROM valid_qb_dropbacks
        )
        """
    ).fetchone()[0]

    if dropback_count == 0:
        raise RuntimeError(
            "No valid quarterback dropbacks were created."
        )

    logger.info(
        "Valid QB dropbacks created: %s rows across "
        "%s QB-game records.",
        dropback_count,
        qb_game_count,
    )


def create_qb_game_aggregates(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Aggregate valid dropbacks to one row per QB and game."""

    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE qb_game_aggregates AS
        SELECT
            game_id,
            season,
            season_type,
            week,
            game_date,
            team,
            opponent,
            home_team,
            away_team,
            is_home,
            qb_id,
            MAX(qb_name) AS qb_name,

            COUNT(*) AS dropbacks,

            COUNT(*) FILTER (
                WHERE is_competitive_dropback
            ) AS competitive_dropbacks,

            COUNT(*) FILTER (
                WHERE is_throw_attempt
            ) AS throw_attempts,

            SUM(complete_pass) FILTER (
                WHERE is_throw_attempt
            ) AS completions,

            SUM(incomplete_pass) FILTER (
                WHERE is_throw_attempt
            ) AS incompletions,

            SUM(passing_yards) FILTER (
                WHERE is_throw_attempt
            ) AS passing_yards,

            AVG(epa) AS epa_per_dropback,

            AVG(epa) FILTER (
                WHERE is_competitive_dropback
            ) AS competitive_epa_per_dropback,

            AVG(success) AS success_rate,

            SUM(complete_pass) FILTER (
                WHERE is_throw_attempt
            ) / NULLIF(
                COUNT(*) FILTER (
                    WHERE is_throw_attempt
                ),
                0
            )::DOUBLE AS completion_rate,

            AVG(cpoe) FILTER (
                WHERE is_throw_attempt
                  AND cpoe IS NOT NULL
            ) AS cpoe,

            AVG(air_yards) FILTER (
                WHERE is_throw_attempt
                  AND air_yards IS NOT NULL
            ) AS air_yards_per_attempt,

            SUM(sack) AS sacks,

            SUM(sack)
                / NULLIF(COUNT(*), 0)::DOUBLE
                AS sack_rate,

            SUM(qb_hit) AS qb_hits,

            SUM(qb_hit)
                / NULLIF(COUNT(*), 0)::DOUBLE
                AS qb_hit_rate,

            SUM(qb_scramble) AS scrambles,

            SUM(qb_scramble)
                / NULLIF(COUNT(*), 0)::DOUBLE
                AS scramble_rate,

            SUM(interception) AS interceptions,

            SUM(interception)
                / NULLIF(
                    COUNT(*) FILTER (
                        WHERE is_throw_attempt
                    ),
                    0
                )::DOUBLE AS interception_rate,

            SUM(fumble_lost) AS fumbles_lost,

            SUM(
                interception + fumble_lost
            ) AS turnovers,

            SUM(
                interception + fumble_lost
            ) / NULLIF(COUNT(*), 0)::DOUBLE
                AS turnover_rate

        FROM valid_qb_dropbacks

        GROUP BY
            game_id,
            season,
            season_type,
            week,
            game_date,
            team,
            opponent,
            home_team,
            away_team,
            is_home,
            qb_id
        """
    )

    row_count = connection.execute(
        """
        SELECT COUNT(*)
        FROM qb_game_aggregates
        """
    ).fetchone()[0]

    if row_count == 0:
        raise RuntimeError(
            "No QB-game aggregate records were created."
        )

    logger.info(
        "QB-game aggregates created: %s rows.",
        row_count,
    )


def create_qb_game_performance_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create the processed quarterback game performance table."""

    connection.execute(
        f"""
        CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA};

        CREATE OR REPLACE TABLE {TARGET_FULL_NAME} AS
        WITH qb_context AS (
            SELECT
                qb.*,

                CASE
                    WHEN qb.is_home
                    THEN schedule.home_qb_id
                    ELSE schedule.away_qb_id
                END AS listed_starter_qb_id,

                CASE
                    WHEN qb.is_home
                    THEN schedule.home_qb_name
                    ELSE schedule.away_qb_name
                END AS listed_starter_qb_name,

                qb.dropbacks
                    / NULLIF(
                        SUM(qb.dropbacks) OVER (
                            PARTITION BY
                                qb.game_id,
                                qb.team
                        ),
                        0
                    )::DOUBLE AS team_dropback_share,

                ROW_NUMBER() OVER (
                    PARTITION BY
                        qb.game_id,
                        qb.team
                    ORDER BY
                        qb.dropbacks DESC,
                        qb.qb_id
                ) AS qb_rank_by_dropbacks

            FROM qb_game_aggregates AS qb

            INNER JOIN {SCHEDULE_FULL_NAME} AS schedule
                ON qb.game_id = schedule.game_id
        )

        SELECT
            game_id,
            season,
            season_type,
            week,
            game_date,
            team,
            opponent,
            home_team,
            away_team,
            is_home,
            qb_id,
            qb_name,

            listed_starter_qb_id,
            listed_starter_qb_name,

            CASE
                WHEN qb_id = listed_starter_qb_id
                THEN TRUE
                ELSE FALSE
            END AS is_listed_starter,

            CASE
                WHEN qb_rank_by_dropbacks = 1
                THEN TRUE
                ELSE FALSE
            END AS is_primary_qb,

            qb_rank_by_dropbacks,
            team_dropback_share,

            dropbacks,
            competitive_dropbacks,
            throw_attempts,
            completions,
            incompletions,
            passing_yards,

            epa_per_dropback,
            competitive_epa_per_dropback,
            success_rate,
            completion_rate,
            cpoe,
            air_yards_per_attempt,

            sacks,
            sack_rate,
            qb_hits,
            qb_hit_rate,
            scrambles,
            scramble_rate,

            interceptions,
            interception_rate,
            fumbles_lost,
            turnovers,
            turnover_rate

        FROM qb_context
        """
    )

    row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    logger.info(
        "QB game performance table created: %s rows in %s.",
        row_count,
        TARGET_FULL_NAME,
    )


def validate_qb_game_performance(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate processed quarterback game performance data."""

    target_row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    source_row_count = connection.execute(
        """
        SELECT COUNT(*)
        FROM qb_game_aggregates
        """
    ).fetchone()[0]

    if target_row_count == 0:
        raise RuntimeError(
            f"Target table is empty: {TARGET_FULL_NAME}"
        )

    if target_row_count != source_row_count:
        raise RuntimeError(
            "QB-game row count does not match aggregation: "
            f"target={target_row_count}, "
            f"source={source_row_count}"
        )

    duplicate_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT
                game_id,
                team,
                qb_id
            FROM {TARGET_FULL_NAME}
            GROUP BY
                game_id,
                team,
                qb_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_count > 0:
        raise RuntimeError(
            "Duplicate QB-game business keys found: "
            f"{duplicate_count}"
        )

    invalid_primary_count = connection.execute(
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
            HAVING SUM(
                CASE
                    WHEN is_primary_qb THEN 1
                    ELSE 0
                END
            ) <> 1
        )
        """
    ).fetchone()[0]

    if invalid_primary_count > 0:
        raise RuntimeError(
            "Team-games without exactly one primary QB found: "
            f"{invalid_primary_count}"
        )

    invalid_share_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT
                game_id,
                team,
                SUM(team_dropback_share) AS total_share
            FROM {TARGET_FULL_NAME}
            GROUP BY
                game_id,
                team
            HAVING ABS(total_share - 1.0) > 0.000000001
        )
        """
    ).fetchone()[0]

    if invalid_share_count > 0:
        raise RuntimeError(
            "Invalid team QB dropback shares found: "
            f"{invalid_share_count}"
        )

    invalid_count_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE dropbacks <= 0
           OR competitive_dropbacks < 0
           OR competitive_dropbacks > dropbacks
           OR throw_attempts < 0
           OR throw_attempts > dropbacks
           OR completions < 0
           OR completions > throw_attempts
           OR sacks < 0
           OR sacks > dropbacks
           OR scrambles < 0
           OR scrambles > dropbacks
           OR interceptions < 0
           OR interceptions > throw_attempts
           OR turnovers < 0
        """
    ).fetchone()[0]

    if invalid_count_count > 0:
        raise RuntimeError(
            "Invalid QB-game counts found: "
            f"{invalid_count_count}"
        )

    invalid_rate_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE success_rate NOT BETWEEN 0 AND 1
           OR team_dropback_share NOT BETWEEN 0 AND 1
           OR sack_rate NOT BETWEEN 0 AND 1
           OR qb_hit_rate NOT BETWEEN 0 AND 1
           OR scramble_rate NOT BETWEEN 0 AND 1
           OR turnover_rate NOT BETWEEN 0 AND 1
           OR (
                completion_rate IS NOT NULL
                AND completion_rate NOT BETWEEN 0 AND 1
           )
           OR (
                interception_rate IS NOT NULL
                AND interception_rate NOT BETWEEN 0 AND 1
           )
        """
    ).fetchone()[0]

    if invalid_rate_count > 0:
        raise RuntimeError(
            "Invalid QB-game rates found: "
            f"{invalid_rate_count}"
        )

    invalid_assignment_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE team = opponent
           OR team NOT IN (home_team, away_team)
           OR opponent NOT IN (home_team, away_team)
           OR (
                is_home
                AND team <> home_team
           )
           OR (
                NOT is_home
                AND team <> away_team
           )
        """
    ).fetchone()[0]

    if invalid_assignment_count > 0:
        raise RuntimeError(
            "Invalid QB team assignments found: "
            f"{invalid_assignment_count}"
        )

    logger.info(
        "QB game performance validated successfully: %s rows.",
        target_row_count,
    )


def build_qb_game_performance(
    database_file: Path = DATABASE_FILE,
    pbp_directory: Path = PBP_DIRECTORY,
) -> None:
    """Build and validate quarterback game performance data."""

    validate_database_file(database_file)

    pbp_files = get_pbp_files(pbp_directory)
    parquet_source = build_parquet_source(pbp_files)

    logger.info(
        "Starting QB game performance build..."
    )

    with duckdb.connect(str(database_file)) as connection:
        validate_schedule_table(connection)
        validate_pbp_columns(
            connection,
            parquet_source,
        )

        connection.execute("BEGIN TRANSACTION")

        try:
            create_valid_qb_dropbacks(
                connection,
                parquet_source,
            )
            create_qb_game_aggregates(connection)
            create_qb_game_performance_table(connection)
            validate_qb_game_performance(connection)

            connection.execute("COMMIT")

            logger.info(
                "QB game performance transaction committed."
            )

        except Exception:
            connection.execute("ROLLBACK")

            logger.exception(
                "QB game performance build failed; "
                "transaction rolled back."
            )
            raise

    logger.info(
        "QB game performance build completed: %s.",
        TARGET_FULL_NAME,
    )


def main() -> None:
    """Run the quarterback game performance builder."""

    try:
        build_qb_game_performance()
    except Exception:
        logger.exception(
            "QB game performance builder failed."
        )
        raise


if __name__ == "__main__":
    main()