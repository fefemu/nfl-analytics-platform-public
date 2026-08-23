"""
NFL Analytics Platform
Player Injury-Impact Builder

Purpose:
    Convert player-game injury context into transparent,
    bounded player availability and importance scores.

Modeling note:
    These rule-based weights are first-generation candidate
    features. They must earn promotion through time-based
    ablation and are not assumed to be optimal.

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
SOURCE_TABLE = "player_game_injury_context"
SOURCE_FULL_NAME = (
    f"{SOURCE_SCHEMA}.{SOURCE_TABLE}"
)

TARGET_SCHEMA = "analytics"
TARGET_TABLE = "player_injury_impact"
TARGET_FULL_NAME = (
    f"{TARGET_SCHEMA}.{TARGET_TABLE}"
)

OUT_SEVERITY = 1.00
DOUBTFUL_SEVERITY = 0.75
QUESTIONABLE_SEVERITY = 0.35

DNP_MODIFIER = 0.10
LIMITED_MODIFIER = 0.05
FULL_MODIFIER = -0.05

STARTER_IMPORTANCE = 1.00
BACKUP_IMPORTANCE = 0.55
RESERVE_IMPORTANCE = 0.25
UNKNOWN_IMPORTANCE = 0.40

SPECIAL_TEAMS_USAGE_WEIGHT = 0.25
FULL_RELIABILITY_GAME_COUNT = 4.0

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
    "has_primary_backup_role",
    "has_reserve_role",
    "has_offense_role",
    "has_defense_role",
    "has_special_teams_role",
    "has_prior_snap_history",
    "snap_history_source",
    "days_since_prior_snap_history",
    "prior_snap_games_last_4",
    "prior_snap_games_last_8",
    "prior_offense_snap_share_last_4",
    "prior_defense_snap_share_last_4",
    "prior_special_teams_snap_share_last_4",
    "prior_offense_snap_share_last_8",
    "prior_defense_snap_share_last_8",
    "prior_special_teams_snap_share_last_8",
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
) -> set[str]:
    """Return available source-table columns."""

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
    """Validate the player injury-context source."""

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
        "Player injury-impact source validated: %s.",
        SOURCE_FULL_NAME,
    )


def count_source_rows(
    connection: duckdb.DuckDBPyConnection,
) -> int:
    """Count player-game injury-context records."""

    return connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {SOURCE_FULL_NAME}
        """
    ).fetchone()[0]


def create_player_injury_impact(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create analytics.player_injury_impact."""

    connection.execute(
        f"""
        CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA}
        """
    )

    connection.execute(
        f"""
        CREATE OR REPLACE TABLE {TARGET_FULL_NAME} AS
        WITH base_scores AS (
            SELECT
                *,

                CASE
                    WHEN report_status = 'Out'
                        THEN {OUT_SEVERITY}
                    WHEN report_status = 'Doubtful'
                        THEN {DOUBTFUL_SEVERITY}
                    WHEN report_status = 'Questionable'
                        THEN {QUESTIONABLE_SEVERITY}
                    ELSE 0.0
                END::DOUBLE AS report_severity_score,

                CASE
                    WHEN report_status IS NULL
                        THEN 0.0
                    WHEN did_not_practice
                        THEN {DNP_MODIFIER}
                    WHEN limited_practice
                        THEN {LIMITED_MODIFIER}
                    WHEN full_practice
                        THEN {FULL_MODIFIER}
                    ELSE 0.0
                END::DOUBLE AS practice_severity_modifier,

                CASE
                    WHEN depth_tier = 'STARTER'
                        THEN {STARTER_IMPORTANCE}
                    WHEN depth_tier = 'PRIMARY_BACKUP'
                        THEN {BACKUP_IMPORTANCE}
                    WHEN depth_tier = 'RESERVE'
                        THEN {RESERVE_IMPORTANCE}
                    ELSE {UNKNOWN_IMPORTANCE}
                END::DOUBLE AS depth_importance_score,

                CASE
                    WHEN has_prior_snap_history
                        THEN GREATEST(
                            COALESCE(
                                prior_offense_snap_share_last_4,
                                0.0
                            ),
                            COALESCE(
                                prior_defense_snap_share_last_4,
                                0.0
                            ),
                            {SPECIAL_TEAMS_USAGE_WEIGHT}
                            * COALESCE(
                                prior_special_teams_snap_share_last_4,
                                0.0
                            )
                        )
                    ELSE NULL
                END::DOUBLE AS observed_usage_score,

                CASE
                    WHEN has_prior_snap_history
                        THEN LEAST(
                            COALESCE(
                                prior_snap_games_last_4,
                                0
                            )
                            / {FULL_RELIABILITY_GAME_COUNT},
                            1.0
                        )
                    ELSE 0.0
                END::DOUBLE AS usage_reliability_score,

                (
                    position = 'QB'
                ) AS is_qb,

                (
                    has_offense_role
                    OR position IN (
                        'QB',
                        'RB',
                        'FB',
                        'WR',
                        'TE',
                        'T',
                        'OT',
                        'G',
                        'OG',
                        'C',
                        'OL'
                    )
                ) AS is_offensive_player,

                (
                    has_defense_role
                    OR position IN (
                        'DE',
                        'DT',
                        'DL',
                        'NT',
                        'LB',
                        'ILB',
                        'OLB',
                        'DB',
                        'CB',
                        'S',
                        'FS',
                        'SS'
                    )
                ) AS is_defensive_player,

                (
                    has_special_teams_role
                    OR position IN (
                        'K',
                        'PK',
                        'P',
                        'LS'
                    )
                ) AS is_special_teams_player

            FROM {SOURCE_FULL_NAME}
        ),
        availability_scores AS (
            SELECT
                *,

                CASE
                    WHEN report_status = 'Out'
                        THEN 1.0
                    WHEN report_status IS NULL
                        THEN 0.0
                    ELSE GREATEST(
                        0.0,
                        LEAST(
                            report_severity_score
                            + practice_severity_modifier,
                            1.0
                        )
                    )
                END::DOUBLE AS availability_severity_score,

                CASE
                    WHEN has_prior_snap_history
                        THEN
                            usage_reliability_score
                            * observed_usage_score
                            +
                            (
                                1.0
                                - usage_reliability_score
                            )
                            * depth_importance_score
                    ELSE depth_importance_score
                END::DOUBLE AS player_importance_score

            FROM base_scores
        )
        SELECT
            *,

            availability_severity_score
                * player_importance_score
                AS injury_impact_score,

            CASE
                WHEN is_qb
                    THEN 0.0
                ELSE
                    availability_severity_score
                    * player_importance_score
            END AS non_qb_injury_impact_score,

            CASE
                WHEN is_qb
                    THEN 0.0
                WHEN is_offensive_player
                    THEN
                        availability_severity_score
                        * player_importance_score
                ELSE 0.0
            END AS offense_injury_impact_score,

            CASE
                WHEN is_qb
                    THEN 0.0
                WHEN is_defensive_player
                    THEN
                        availability_severity_score
                        * player_importance_score
                ELSE 0.0
            END AS defense_injury_impact_score,

            CASE
                WHEN is_special_teams_player
                    THEN
                        availability_severity_score
                        * player_importance_score
                ELSE 0.0
            END AS special_teams_injury_impact_score

        FROM availability_scores
        """
    )


def validate_target_table(
    connection: duckdb.DuckDBPyConnection,
    expected_row_count: int,
) -> tuple[int, int, int, int, int]:
    """Validate player injury-impact scores."""

    row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    if row_count != expected_row_count:
        raise RuntimeError(
            "Player injury-impact row count does not "
            "match its source: "
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
            "Player injury impact contains "
            f"{duplicate_count} duplicate business keys."
        )

    invalid_score_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE report_severity_score NOT BETWEEN 0 AND 1
           OR practice_severity_modifier
                NOT BETWEEN -1 AND 1
           OR availability_severity_score
                NOT BETWEEN 0 AND 1
           OR depth_importance_score
                NOT BETWEEN 0 AND 1
           OR observed_usage_score
                NOT BETWEEN 0 AND 1
           OR usage_reliability_score
                NOT BETWEEN 0 AND 1
           OR player_importance_score
                NOT BETWEEN 0 AND 1
           OR injury_impact_score
                NOT BETWEEN 0 AND 1
           OR non_qb_injury_impact_score
                NOT BETWEEN 0 AND 1
           OR offense_injury_impact_score
                NOT BETWEEN 0 AND 1
           OR defense_injury_impact_score
                NOT BETWEEN 0 AND 1
           OR special_teams_injury_impact_score
                NOT BETWEEN 0 AND 1
        """
    ).fetchone()[0]

    if invalid_score_count > 0:
        raise RuntimeError(
            "Player injury impact contains "
            f"{invalid_score_count} invalid scores."
        )

    inconsistent_status_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE (
                report_status IS NULL
                AND availability_severity_score != 0
              )
           OR (
                report_status = 'Out'
                AND availability_severity_score != 1
              )
           OR (
                report_status = 'Doubtful'
                AND report_severity_score
                    != {DOUBTFUL_SEVERITY}
              )
           OR (
                report_status = 'Questionable'
                AND report_severity_score
                    != {QUESTIONABLE_SEVERITY}
              )
        """
    ).fetchone()[0]

    if inconsistent_status_count > 0:
        raise RuntimeError(
            "Player injury impact contains "
            f"{inconsistent_status_count} inconsistent "
            "status scores."
        )

    inconsistent_qb_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE is_qb
          AND (
                non_qb_injury_impact_score != 0
                OR offense_injury_impact_score != 0
              )
        """
    ).fetchone()[0]

    if inconsistent_qb_count > 0:
        raise RuntimeError(
            "Player injury impact contains "
            f"{inconsistent_qb_count} QB rows in "
            "generic offensive burden."
        )

    active_impact_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE injury_impact_score > 0
        """
    ).fetchone()[0]

    qb_impact_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE is_qb
          AND injury_impact_score > 0
        """
    ).fetchone()[0]

    non_qb_impact_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE NOT is_qb
          AND non_qb_injury_impact_score > 0
        """
    ).fetchone()[0]

    missing_snap_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE NOT has_prior_snap_history
        """
    ).fetchone()[0]

    return (
        row_count,
        active_impact_count,
        qb_impact_count,
        non_qb_impact_count,
        missing_snap_count,
    )


def build_player_injury_impact(
    database_file: Path = DATABASE_FILE,
) -> None:
    """Build analytics.player_injury_impact."""

    validate_database_file(
        database_file
    )

    logger.info(
        "Starting player injury-impact build..."
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
                create_player_injury_impact(
                    connection
                )

                (
                    row_count,
                    active_impact_count,
                    qb_impact_count,
                    non_qb_impact_count,
                    missing_snap_count,
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
            "Player injury-impact build failed."
        )
        raise

    logger.info(
        "Player injury-impact table validated: %s rows.",
        row_count,
    )
    logger.info(
        "Players with positive injury impact: %s.",
        active_impact_count,
    )
    logger.info(
        "QB rows with positive impact: %s.",
        qb_impact_count,
    )
    logger.info(
        "Non-QB rows with positive impact: %s.",
        non_qb_impact_count,
    )
    logger.info(
        "Rows using depth-only importance fallback: %s.",
        missing_snap_count,
    )
    logger.info(
        "Player injury-impact build completed successfully."
    )


def main() -> None:
    """Run the player injury-impact builder."""

    build_player_injury_impact()


if __name__ == "__main__":
    main()