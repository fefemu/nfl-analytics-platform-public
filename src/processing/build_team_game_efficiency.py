"""
NFL Analytics Platform
Team-Game Efficiency Builder

Purpose:
    Build one team-level efficiency record per NFL game
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

SOURCE_SCHEMA = "processed"
SOURCE_TABLE = "schedule"

TARGET_SCHEMA = "processed"
TARGET_TABLE = "team_game_efficiency"

SOURCE_FULL_NAME = f"{SOURCE_SCHEMA}.{SOURCE_TABLE}"
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
    "down",
    "yardline_100",
    "wp",
    "play_type",
    "qb_kneel",
    "qb_spike",
    "aborted_play",
    "pass",
    "rush",
    "qb_dropback",
    "qb_scramble",
    "sack",
    "interception",
    "fumble_lost",
    "two_point_attempt",
    "special_teams_play",
    "yards_gained",
    "epa",
    "success",
}


def get_pbp_files(
    pbp_directory: Path = PBP_DIRECTORY,
) -> list[Path]:
    """Return the available season-level PBP Parquet files."""

    if not pbp_directory.exists():
        raise FileNotFoundError(
            f"PBP directory does not exist: {pbp_directory}"
        )

    pbp_files = sorted(pbp_directory.glob("pbp_*.parquet"))

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


def build_parquet_source(pbp_files: list[Path]) -> str:
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
    """Validate the columns required from the PBP Parquet files."""

    description = connection.execute(
        f"""
        SELECT *
        FROM read_parquet({parquet_source}, union_by_name = true)
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
            f"Missing required PBP columns: {missing_names}"
        )

    logger.info(
        "Required PBP columns validated successfully: %s columns.",
        len(REQUIRED_PBP_COLUMNS),
    )


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


def validate_schedule_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate the processed schedule source table."""

    table_exists = connection.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = ?
          AND table_name = ?
        """,
        [SOURCE_SCHEMA, SOURCE_TABLE],
    ).fetchone()[0]

    if table_exists == 0:
        raise RuntimeError(
            f"Source table does not exist: {SOURCE_FULL_NAME}"
        )

    required_columns = {
        "game_id",
        "home_score",
        "away_score",
        "home_win",
        "away_win",
        "is_tie",
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
            [SOURCE_SCHEMA, SOURCE_TABLE],
        ).fetchall()
    }

    missing_columns = sorted(
        required_columns - available_columns
    )

    if missing_columns:
        missing_names = ", ".join(missing_columns)
        raise RuntimeError(
            "Missing required processed schedule columns: "
            f"{missing_names}"
        )

    logger.info(
        "Schedule source validated: %s",
        SOURCE_FULL_NAME,
    )


def create_valid_offensive_plays(
    connection: duckdb.DuckDBPyConnection,
    parquet_source: str,
) -> None:
    """Create a temporary table containing valid offensive plays."""

    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE valid_offensive_plays AS
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
            down,
            yardline_100,
            CASE
                WHEN posteam = home_team THEN TRUE
                ELSE FALSE
            END AS is_home,
            wp,
            play_type,
            qb_dropback,
            qb_scramble,
            rush,
            sack,
            interception,
            fumble_lost,
            yards_gained,
            epa,
            success,
            CASE
                WHEN wp BETWEEN 0.10 AND 0.90 THEN TRUE
                ELSE FALSE
            END AS is_competitive_play,
            CASE
                WHEN down IN (1, 2) THEN TRUE
                ELSE FALSE
            END AS is_early_down_play,
            CASE
                WHEN yardline_100 <= 20 THEN TRUE
                ELSE FALSE
            END AS is_red_zone_play,
            CASE
                WHEN qb_dropback = 1 THEN TRUE
                ELSE FALSE
            END AS is_dropback,
            CASE
                WHEN rush = 1
                 AND COALESCE(qb_scramble, 0) = 0
                THEN TRUE
                ELSE FALSE
            END AS is_designed_rush,
            CASE
                WHEN qb_dropback = 1
                 AND yards_gained >= 20
                THEN TRUE
                WHEN rush = 1
                 AND COALESCE(qb_scramble, 0) = 0
                 AND yards_gained >= 10
                THEN TRUE
                ELSE FALSE
            END AS is_explosive_play
        FROM read_parquet(
            {parquet_source},
            union_by_name = true
        )
        WHERE game_id IS NOT NULL
          AND posteam IS NOT NULL
          AND defteam IS NOT NULL
          AND epa IS NOT NULL
          AND COALESCE(qb_kneel, 0) = 0
          AND COALESCE(qb_spike, 0) = 0
          AND COALESCE(aborted_play, 0) = 0
          AND COALESCE(two_point_attempt, 0) = 0
          AND COALESCE(special_teams_play, 0) = 0
          AND (
                qb_dropback = 1
                OR (
                    rush = 1
                    AND COALESCE(qb_scramble, 0) = 0
                )
          )
        """
    )

    play_count = connection.execute(
        """
        SELECT COUNT(*)
        FROM valid_offensive_plays
        """
    ).fetchone()[0]

    if play_count == 0:
        raise RuntimeError(
            "No valid offensive plays were created."
        )

    logger.info(
        "Valid offensive plays created: %s rows.",
        play_count,
    )


def create_team_game_offense(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Aggregate valid plays to one offensive row per team and game."""

    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE team_game_offense AS
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

            COUNT(*) AS offensive_plays,

            COUNT(*) FILTER (
                WHERE is_dropback
            ) AS dropbacks,

            COUNT(*) FILTER (
                WHERE is_designed_rush
            ) AS designed_rushes,

            COUNT(*) FILTER (
                WHERE is_competitive_play
            ) AS competitive_plays,

            COUNT(*) FILTER (
                WHERE is_early_down_play
            ) AS early_down_plays,

            COUNT(*) FILTER (
                WHERE is_red_zone_play
            ) AS red_zone_plays,

            AVG(epa) AS offensive_epa_per_play,

            AVG(epa) FILTER (
                WHERE is_competitive_play
            ) AS competitive_epa_per_play,

            AVG(epa) FILTER (
                WHERE is_dropback
            ) AS dropback_epa_per_play,

            AVG(epa) FILTER (
                WHERE is_designed_rush
            ) AS designed_rush_epa_per_play,

            AVG(epa) FILTER (
                WHERE is_early_down_play
            ) AS early_down_epa_per_play,

            AVG(success) AS success_rate,

            AVG(success) FILTER (
                WHERE is_dropback
            ) AS dropback_success_rate,

            AVG(success) FILTER (
                WHERE is_designed_rush
            ) AS designed_rush_success_rate,

            COUNT(*) FILTER (
                WHERE is_explosive_play
            ) AS explosive_plays,

            AVG(
                CASE
                    WHEN is_explosive_play THEN 1.0
                    ELSE 0.0
                END
            ) AS explosive_play_rate,

            SUM(COALESCE(sack, 0)) AS sacks_allowed,

            SUM(COALESCE(sack, 0))
                / NULLIF(
                    COUNT(*) FILTER (WHERE is_dropback),
                    0
                )::DOUBLE AS sack_rate,

            SUM(COALESCE(interception, 0))
                AS interceptions_thrown,

            SUM(COALESCE(fumble_lost, 0))
                AS fumbles_lost,

            SUM(
                COALESCE(interception, 0)
                + COALESCE(fumble_lost, 0)
            ) AS turnovers,

            SUM(
                COALESCE(interception, 0)
                + COALESCE(fumble_lost, 0)
            ) / NULLIF(COUNT(*), 0)::DOUBLE
                AS turnover_rate

        FROM valid_offensive_plays
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
            is_home
        """
    )

    row_count = connection.execute(
        """
        SELECT COUNT(*)
        FROM team_game_offense
        """
    ).fetchone()[0]

    if row_count == 0:
        raise RuntimeError(
            "No team-game offensive records were created."
        )

    logger.info(
        "Team-game offense created: %s rows.",
        row_count,
    )


def create_team_game_efficiency_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create the final processed team-game efficiency table."""

    connection.execute(
        f"""
        CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA};

        CREATE OR REPLACE TABLE {TARGET_FULL_NAME} AS
        SELECT
            offense.game_id,
            offense.season,
            offense.season_type,
            offense.week,
            offense.game_date,
            offense.team,
            offense.opponent,
            offense.home_team,
            offense.away_team,
            offense.is_home,

            CASE
                WHEN offense.is_home
                THEN schedule.home_score
                ELSE schedule.away_score
            END AS points_scored,

            CASE
                WHEN offense.is_home
                THEN schedule.away_score
                ELSE schedule.home_score
            END AS points_allowed,

            CASE
                WHEN offense.is_home
                THEN schedule.home_score - schedule.away_score
                ELSE schedule.away_score - schedule.home_score
            END AS point_differential,

            CASE
                WHEN offense.is_home
                THEN schedule.home_win
                ELSE schedule.away_win
            END AS team_win,

            schedule.is_tie,

            offense.offensive_plays,
            offense.dropbacks,
            offense.designed_rushes,
            offense.competitive_plays,
            offense.early_down_plays,
            offense.red_zone_plays,

            offense.offensive_epa_per_play,
            offense.competitive_epa_per_play,
            offense.dropback_epa_per_play,
            offense.designed_rush_epa_per_play,
            offense.early_down_epa_per_play,

            offense.success_rate,
            offense.dropback_success_rate,
            offense.designed_rush_success_rate,

            offense.explosive_plays,
            offense.explosive_play_rate,

            offense.sacks_allowed,
            offense.sack_rate,
            offense.interceptions_thrown,
            offense.fumbles_lost,
            offense.turnovers,
            offense.turnover_rate,

            defense.offensive_epa_per_play
                AS defensive_epa_allowed_per_play,

            defense.competitive_epa_per_play
                AS competitive_defensive_epa_allowed_per_play,

            defense.success_rate
                AS defensive_success_rate_allowed,

            defense.explosive_play_rate
                AS explosive_play_rate_allowed,

            defense.sack_rate
                AS sack_rate_generated,

            defense.turnover_rate
                AS turnover_rate_generated

        FROM team_game_offense AS offense

        INNER JOIN team_game_offense AS defense
            ON offense.game_id = defense.game_id
           AND offense.team = defense.opponent
           AND offense.opponent = defense.team

        INNER JOIN {SOURCE_FULL_NAME} AS schedule
            ON offense.game_id = schedule.game_id
        """
    )

    row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    logger.info(
        "Team-game efficiency table created: %s rows in %s.",
        row_count,
        TARGET_FULL_NAME,
    )


def validate_team_game_efficiency(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate the processed team-game efficiency table."""

    row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    source_row_count = connection.execute(
        """
        SELECT COUNT(*)
        FROM team_game_offense
        """
    ).fetchone()[0]

    if row_count == 0:
        raise RuntimeError(
            f"Target table is empty: {TARGET_FULL_NAME}"
        )

    if row_count != source_row_count:
        raise RuntimeError(
            "Team-game row count does not match the offensive "
            f"aggregation: target={row_count}, "
            f"source={source_row_count}"
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
            "Duplicate game-team business keys found: "
            f"{duplicate_count}"
        )

    invalid_game_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT
                game_id
            FROM {TARGET_FULL_NAME}
            GROUP BY game_id
            HAVING COUNT(*) <> 2
               OR COUNT(DISTINCT team) <> 2
        )
        """
    ).fetchone()[0]

    if invalid_game_count > 0:
        raise RuntimeError(
            "Games without exactly two team records found: "
            f"{invalid_game_count}"
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
            "Invalid team assignments found: "
            f"{invalid_assignment_count}"
        )

    missing_metric_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE points_scored IS NULL
           OR points_allowed IS NULL
           OR point_differential IS NULL
           OR offensive_plays IS NULL
           OR offensive_epa_per_play IS NULL
           OR success_rate IS NULL
           OR defensive_epa_allowed_per_play IS NULL
           OR defensive_success_rate_allowed IS NULL
        """
    ).fetchone()[0]

    if missing_metric_count > 0:
        raise RuntimeError(
            "Required team-game results or metrics are missing: "
            f"{missing_metric_count}"
        )

    inconsistent_result_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE point_differential
              <> points_scored - points_allowed
           OR (
                is_tie
                AND point_differential <> 0
           )
           OR (
                NOT is_tie
                AND point_differential = 0
           )
        """
    ).fetchone()[0]

    if inconsistent_result_count > 0:
        raise RuntimeError(
            "Inconsistent team-game results found: "
            f"{inconsistent_result_count}"
        )

    logger.info(
        "Team-game efficiency validated successfully: %s rows.",
        row_count,
    )


def build_team_game_efficiency(
    database_file: Path = DATABASE_FILE,
    pbp_directory: Path = PBP_DIRECTORY,
) -> None:
    """Build and validate processed team-game efficiency data."""

    validate_database_file(database_file)

    pbp_files = get_pbp_files(pbp_directory)
    parquet_source = build_parquet_source(pbp_files)

    logger.info(
        "Starting team-game efficiency build..."
    )

    with duckdb.connect(str(database_file)) as connection:
        validate_schedule_table(connection)
        validate_pbp_columns(
            connection,
            parquet_source,
        )

        connection.execute("BEGIN TRANSACTION")

        try:
            create_valid_offensive_plays(
                connection,
                parquet_source,
            )
            create_team_game_offense(connection)
            create_team_game_efficiency_table(connection)
            validate_team_game_efficiency(connection)

            connection.execute("COMMIT")

            logger.info(
                "Team-game efficiency transaction committed."
            )

        except Exception:
            connection.execute("ROLLBACK")

            logger.exception(
                "Team-game efficiency build failed; "
                "transaction rolled back."
            )
            raise

    logger.info(
        "Team-game efficiency build completed: %s.",
        TARGET_FULL_NAME,
    )


def main() -> None:
    """Run the team-game efficiency builder."""

    try:
        build_team_game_efficiency()
    except Exception:
        logger.exception(
            "Team-game efficiency builder failed."
        )
        raise


if __name__ == "__main__":
    main()