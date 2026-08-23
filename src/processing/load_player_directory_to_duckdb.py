"""
NFL Analytics Platform
Player Directory DuckDB Loader

Purpose:
    Load the canonical nflverse player directory into the
    raw DuckDB layer for stable PFR, GSIS and ESPN identity
    resolution.

Leakage note:
    Stable identifiers may be used for historical joins.
    Current roster fields such as latest_team and status
    must not be used as historical pregame features.

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
PLAYER_DIRECTORY_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "players"
    / "player_directory.parquet"
)

TARGET_SCHEMA = "raw"
TARGET_TABLE = "player_directory"
TARGET_FULL_NAME = (
    f"{TARGET_SCHEMA}.{TARGET_TABLE}"
)

REQUIRED_PLAYER_COLUMNS = {
    "gsis_id",
    "display_name",
    "common_first_name",
    "first_name",
    "last_name",
    "short_name",
    "football_name",
    "suffix",
    "esb_id",
    "nfl_id",
    "pfr_id",
    "pff_id",
    "otc_id",
    "espn_id",
    "smart_id",
    "birth_date",
    "position_group",
    "position",
    "ngs_position_group",
    "ngs_position",
    "height",
    "weight",
    "headshot",
    "college_name",
    "college_conference",
    "jersey_number",
    "rookie_season",
    "last_season",
    "latest_team",
    "status",
    "ngs_status",
    "ngs_status_short_description",
    "years_of_experience",
    "pff_position",
    "pff_status",
    "draft_year",
    "draft_round",
    "draft_pick",
    "draft_team",
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


def validate_player_directory_file(
    player_directory_file: Path = PLAYER_DIRECTORY_FILE,
) -> None:
    """Validate the canonical player-directory Parquet file."""

    if not player_directory_file.exists():
        raise FileNotFoundError(
            "Player-directory file does not exist: "
            f"{player_directory_file}"
        )

    if not player_directory_file.is_file():
        raise RuntimeError(
            "Player-directory path is not a file: "
            f"{player_directory_file}"
        )

    logger.info(
        "Player-directory file validated: %s",
        player_directory_file,
    )


def escape_path(
    source_path: Path,
) -> str:
    """Return one SQL-safe resolved filesystem path."""

    return str(
        source_path.resolve()
    ).replace(
        "'",
        "''",
    )


def validate_parquet_columns(
    connection: duckdb.DuckDBPyConnection,
    player_directory_file: Path,
) -> None:
    """Validate player-directory Parquet columns."""

    escaped_path = escape_path(
        player_directory_file
    )

    description = connection.execute(
        f"""
        SELECT *
        FROM read_parquet(
            '{escaped_path}'
        )
        LIMIT 0
        """
    ).description

    available_columns = {
        column[0]
        for column in description
    }

    missing_columns = sorted(
        REQUIRED_PLAYER_COLUMNS
        - available_columns
    )

    if missing_columns:
        raise RuntimeError(
            "Missing required player-directory columns: "
            + ", ".join(
                missing_columns
            )
        )

    logger.info(
        "Required player-directory columns validated: %s.",
        len(REQUIRED_PLAYER_COLUMNS),
    )


def create_raw_player_directory(
    connection: duckdb.DuckDBPyConnection,
    player_directory_file: Path,
) -> None:
    """Create raw.player_directory."""

    escaped_path = escape_path(
        player_directory_file
    )

    connection.execute(
        f"""
        CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA}
        """
    )

    connection.execute(
        f"""
        CREATE OR REPLACE TABLE {TARGET_FULL_NAME} AS
        SELECT
            *,
            '{escaped_path}' AS source_file
        FROM read_parquet(
            '{escaped_path}'
        )
        """
    )


def count_duplicate_non_null_ids(
    connection: duckdb.DuckDBPyConnection,
    identifier_column: str,
) -> int:
    """Count duplicated non-null identifier groups."""

    return connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT
                {identifier_column}
            FROM {TARGET_FULL_NAME}
            WHERE {identifier_column} IS NOT NULL
            GROUP BY
                {identifier_column}
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]


def validate_loaded_table(
    connection: duckdb.DuckDBPyConnection,
    expected_row_count: int,
) -> tuple[int, int, int]:
    """Validate raw.player_directory."""

    row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    if row_count != expected_row_count:
        raise RuntimeError(
            "Raw player-directory row count does not match "
            f"the Parquet source: {row_count} != "
            f"{expected_row_count}."
        )

    null_gsis_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE gsis_id IS NULL
           OR TRIM(gsis_id) = ''
        """
    ).fetchone()[0]

    if null_gsis_count > 0:
        raise RuntimeError(
            "Raw player directory contains "
            f"{null_gsis_count} rows without a GSIS ID."
        )

    for identifier_column in (
        "gsis_id",
        "pfr_id",
        "espn_id",
    ):
        duplicate_count = count_duplicate_non_null_ids(
            connection=connection,
            identifier_column=identifier_column,
        )

        if duplicate_count > 0:
            raise RuntimeError(
                "Raw player directory contains "
                f"{duplicate_count} duplicate "
                f"{identifier_column} groups."
            )

    null_source_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE source_file IS NULL
        """
    ).fetchone()[0]

    if null_source_count > 0:
        raise RuntimeError(
            "Raw player directory contains null source files."
        )

    pfr_coverage_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE pfr_id IS NOT NULL
        """
    ).fetchone()[0]

    espn_coverage_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE espn_id IS NOT NULL
        """
    ).fetchone()[0]

    if pfr_coverage_count == 0:
        raise RuntimeError(
            "Raw player directory contains no PFR identifiers."
        )

    return (
        row_count,
        pfr_coverage_count,
        espn_coverage_count,
    )


def load_player_directory_to_duckdb(
    database_file: Path = DATABASE_FILE,
    player_directory_file: Path = PLAYER_DIRECTORY_FILE,
) -> None:
    """Create or replace raw.player_directory."""

    validate_database_file(
        database_file
    )
    validate_player_directory_file(
        player_directory_file
    )

    logger.info(
        "Loading the player directory into DuckDB..."
    )

    try:
        with duckdb.connect(
            str(database_file)
        ) as connection:
            validate_parquet_columns(
                connection=connection,
                player_directory_file=player_directory_file,
            )

            escaped_path = escape_path(
                player_directory_file
            )

            expected_row_count = connection.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet(
                    '{escaped_path}'
                )
                """
            ).fetchone()[0]

            connection.execute(
                "BEGIN TRANSACTION"
            )

            try:
                create_raw_player_directory(
                    connection=connection,
                    player_directory_file=player_directory_file,
                )

                (
                    row_count,
                    pfr_coverage_count,
                    espn_coverage_count,
                ) = validate_loaded_table(
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
            "Failed to load the player directory into DuckDB."
        )
        raise

    logger.info(
        "DuckDB player-directory load completed: %s rows.",
        row_count,
    )
    logger.info(
        "PFR identifier coverage: %s rows.",
        pfr_coverage_count,
    )
    logger.info(
        "ESPN identifier coverage: %s rows.",
        espn_coverage_count,
    )
    logger.info(
        "Database saved to: %s",
        database_file,
    )
    logger.info(
        "Table created: %s",
        TARGET_FULL_NAME,
    )


def main() -> None:
    """Run the raw player-directory loading workflow."""

    try:
        load_player_directory_to_duckdb()
    except Exception:
        logger.exception(
            "Raw player-directory load failed."
        )
        raise

    logger.info(
        "Raw player-directory load completed successfully."
    )


if __name__ == "__main__":
    main()