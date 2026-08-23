"""
NFL Analytics Platform
Modeling Dataset Split Builder

Purpose:
    Create reproducible time-based train, validation
    and holdout assignments for NFL modeling games.

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

SOURCE_SCHEMA = "analytics"
SOURCE_TABLE = "game_modeling_dataset"
SOURCE_FULL_NAME = (
    f"{SOURCE_SCHEMA}.{SOURCE_TABLE}"
)

TARGET_SCHEMA = "analytics"
TARGET_TABLE = "modeling_game_splits"
TARGET_FULL_NAME = (
    f"{TARGET_SCHEMA}.{TARGET_TABLE}"
)

TRAIN_FIRST_SEASON = 2018
TRAIN_LAST_SEASON = 2022

VALIDATION_FIRST_SEASON = 2023
VALIDATION_LAST_SEASON = 2024

HOLDOUT_SEASON = 2025

VALID_SPLIT_NAMES = {
    "train",
    "validation",
    "holdout",
}

REQUIRED_SOURCE_COLUMNS = {
    "game_id",
    "season",
    "game_type",
    "week",
    "game_date",
    "target_home_win",
    "target_home_result",
    "both_short_windows_complete",
    "both_long_windows_complete",
    "both_listed_qb_ratings_available",
}


def assign_split_name(
    season: int,
) -> str:
    """Assign a season to a reproducible time-based split."""

    if TRAIN_FIRST_SEASON <= season <= TRAIN_LAST_SEASON:
        return "train"

    if (
        VALIDATION_FIRST_SEASON
        <= season
        <= VALIDATION_LAST_SEASON
    ):
        return "validation"

    if season == HOLDOUT_SEASON:
        return "holdout"

    raise ValueError(
        f"Season is outside the configured split range: {season}"
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


def validate_source_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate the game modeling dataset source."""

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
        REQUIRED_SOURCE_COLUMNS - available_columns
    )

    if missing_columns:
        missing_names = ", ".join(missing_columns)

        raise RuntimeError(
            f"Missing columns in {SOURCE_FULL_NAME}: "
            f"{missing_names}"
        )

    source_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {SOURCE_FULL_NAME}
        """
    ).fetchone()[0]

    if source_count == 0:
        raise RuntimeError(
            f"Source table is empty: {SOURCE_FULL_NAME}"
        )

    logger.info(
        "Modeling split source validated: %s rows in %s.",
        source_count,
        SOURCE_FULL_NAME,
    )


def create_modeling_splits_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create time-based modeling split assignments."""

    connection.execute(
        f"CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA}"
    )

    connection.execute(
        f"""
        CREATE OR REPLACE TABLE {TARGET_FULL_NAME} AS

        SELECT
            game_id,
            season,
            game_type,
            week,
            game_date,

            CASE
                WHEN season BETWEEN
                    {TRAIN_FIRST_SEASON}
                    AND {TRAIN_LAST_SEASON}
                    THEN 'train'

                WHEN season BETWEEN
                    {VALIDATION_FIRST_SEASON}
                    AND {VALIDATION_LAST_SEASON}
                    THEN 'validation'

                WHEN season = {HOLDOUT_SEASON}
                    THEN 'holdout'

                ELSE NULL
            END AS split_name,

            CASE
                WHEN season BETWEEN
                    {TRAIN_FIRST_SEASON}
                    AND {TRAIN_LAST_SEASON}
                    THEN 1

                WHEN season BETWEEN
                    {VALIDATION_FIRST_SEASON}
                    AND {VALIDATION_LAST_SEASON}
                    THEN 2

                WHEN season = {HOLDOUT_SEASON}
                    THEN 3

                ELSE NULL
            END AS split_order,

            target_home_win IS NOT NULL
                AS is_binary_target_eligible,

            both_short_windows_complete
                AS has_complete_short_history,

            both_long_windows_complete
                AS has_complete_long_history,

            both_listed_qb_ratings_available
                AS has_both_qb_ratings,

            (
                target_home_win IS NOT NULL
                AND both_short_windows_complete
                AND both_listed_qb_ratings_available
            ) AS is_core_model_eligible,

            (
                target_home_win IS NOT NULL
                AND both_long_windows_complete
                AND both_listed_qb_ratings_available
            ) AS is_extended_model_eligible

        FROM {SOURCE_FULL_NAME}

        WHERE season BETWEEN
            {TRAIN_FIRST_SEASON}
            AND {HOLDOUT_SEASON}

        ORDER BY
            game_date,
            game_id
        """
    )

    row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    logger.info(
        "Modeling game splits created: %s rows in %s.",
        row_count,
        TARGET_FULL_NAME,
    )


def validate_modeling_splits_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate modeling split assignments and eligibility flags."""

    row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    expected_row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {SOURCE_FULL_NAME}
        WHERE season BETWEEN
            {TRAIN_FIRST_SEASON}
            AND {HOLDOUT_SEASON}
        """
    ).fetchone()[0]

    if row_count != expected_row_count:
        raise RuntimeError(
            "Modeling split row count does not match: "
            f"expected {expected_row_count}, "
            f"found {row_count}."
        )

    duplicate_game_count = connection.execute(
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

    if duplicate_game_count > 0:
        raise RuntimeError(
            "Duplicate games found in modeling splits: "
            f"{duplicate_game_count}"
        )

    invalid_split_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE split_name NOT IN (
                'train',
                'validation',
                'holdout'
              )
           OR split_order NOT BETWEEN 1 AND 3

           OR split_name IS DISTINCT FROM (
                CASE
                    WHEN season BETWEEN
                        {TRAIN_FIRST_SEASON}
                        AND {TRAIN_LAST_SEASON}
                        THEN 'train'

                    WHEN season BETWEEN
                        {VALIDATION_FIRST_SEASON}
                        AND {VALIDATION_LAST_SEASON}
                        THEN 'validation'

                    WHEN season = {HOLDOUT_SEASON}
                        THEN 'holdout'

                    ELSE NULL
                END
           )

           OR split_order IS DISTINCT FROM (
                CASE
                    WHEN season BETWEEN
                        {TRAIN_FIRST_SEASON}
                        AND {TRAIN_LAST_SEASON}
                        THEN 1

                    WHEN season BETWEEN
                        {VALIDATION_FIRST_SEASON}
                        AND {VALIDATION_LAST_SEASON}
                        THEN 2

                    WHEN season = {HOLDOUT_SEASON}
                        THEN 3

                    ELSE NULL
                END
           )
        """
    ).fetchone()[0]

    if invalid_split_count > 0:
        raise RuntimeError(
            "Invalid modeling split assignments found: "
            f"{invalid_split_count}"
        )

    missing_split_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT split_name
            FROM {TARGET_FULL_NAME}
            GROUP BY split_name
        )
        """
    ).fetchone()[0]

    if missing_split_count != len(VALID_SPLIT_NAMES):
        raise RuntimeError(
            "Not every required modeling split is present."
        )

    chronology_violation_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT
                (
                    SELECT MAX(game_date)
                    FROM {TARGET_FULL_NAME}
                    WHERE split_name = 'train'
                ) AS train_last_date,

                (
                    SELECT MIN(game_date)
                    FROM {TARGET_FULL_NAME}
                    WHERE split_name = 'validation'
                ) AS validation_first_date,

                (
                    SELECT MAX(game_date)
                    FROM {TARGET_FULL_NAME}
                    WHERE split_name = 'validation'
                ) AS validation_last_date,

                (
                    SELECT MIN(game_date)
                    FROM {TARGET_FULL_NAME}
                    WHERE split_name = 'holdout'
                ) AS holdout_first_date
        )
        WHERE train_last_date >= validation_first_date
           OR validation_last_date >= holdout_first_date
        """
    ).fetchone()[0]

    if chronology_violation_count > 0:
        raise RuntimeError(
            "Modeling split chronology is invalid."
        )

    invalid_eligibility_count = connection.execute(
        f"""
        SELECT COUNT(*)

        FROM {TARGET_FULL_NAME} AS splits

        INNER JOIN {SOURCE_FULL_NAME} AS dataset
            ON splits.game_id = dataset.game_id

        WHERE splits.is_binary_target_eligible
                IS DISTINCT FROM (
                    dataset.target_home_win IS NOT NULL
                )

           OR splits.has_complete_short_history
                IS DISTINCT FROM (
                    dataset.both_short_windows_complete
                )

           OR splits.has_complete_long_history
                IS DISTINCT FROM (
                    dataset.both_long_windows_complete
                )

           OR splits.has_both_qb_ratings
                IS DISTINCT FROM (
                    dataset.both_listed_qb_ratings_available
                )

           OR splits.is_core_model_eligible
                IS DISTINCT FROM (
                    dataset.target_home_win IS NOT NULL
                    AND dataset.both_short_windows_complete
                    AND dataset.both_listed_qb_ratings_available
                )

           OR splits.is_extended_model_eligible
                IS DISTINCT FROM (
                    dataset.target_home_win IS NOT NULL
                    AND dataset.both_long_windows_complete
                    AND dataset.both_listed_qb_ratings_available
                )
        """
    ).fetchone()[0]

    if invalid_eligibility_count > 0:
        raise RuntimeError(
            "Invalid modeling eligibility flags found: "
            f"{invalid_eligibility_count}"
        )

    logger.info(
        "Modeling game splits validated successfully: %s rows.",
        row_count,
    )


def build_modeling_splits(
    database_file: Path = DATABASE_FILE,
) -> None:
    """Build reproducible time-based modeling splits."""

    validate_database_file(database_file)

    logger.info(
        "Starting modeling game splits build..."
    )

    with duckdb.connect(str(database_file)) as connection:
        validate_source_table(connection)

        connection.execute("BEGIN TRANSACTION")

        try:
            create_modeling_splits_table(connection)

            validate_modeling_splits_table(connection)

            connection.execute("COMMIT")

            logger.info(
                "Modeling game splits transaction committed."
            )

        except Exception:
            connection.execute("ROLLBACK")

            logger.exception(
                "Modeling game splits build failed; "
                "transaction rolled back."
            )
            raise

    logger.info(
        "Modeling game splits build completed: %s.",
        TARGET_FULL_NAME,
    )


def main() -> None:
    """Run the modeling game splits builder."""

    try:
        build_modeling_splits()

    except Exception:
        logger.exception(
            "Modeling game splits builder failed."
        )
        raise


if __name__ == "__main__":
    main()