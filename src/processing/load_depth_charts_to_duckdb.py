"""
NFL Analytics Platform
Depth-Chart DuckDB Loader

Purpose:
    Load legacy weekly NFL depth charts and timestamped
    ESPN depth charts into separate raw DuckDB tables.

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

DEPTH_CHART_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "depth_charts"
)
LEGACY_DATA_DIR = (
    DEPTH_CHART_DATA_DIR
    / "legacy"
)
ESPN_DATA_DIR = (
    DEPTH_CHART_DATA_DIR
    / "espn"
)

TARGET_SCHEMA = "raw"

LEGACY_TABLE = "depth_charts_legacy"
LEGACY_FULL_NAME = (
    f"{TARGET_SCHEMA}.{LEGACY_TABLE}"
)

ESPN_TABLE = "depth_charts_espn"
ESPN_FULL_NAME = (
    f"{TARGET_SCHEMA}.{ESPN_TABLE}"
)

LEGACY_REQUIRED_COLUMNS = {
    "season",
    "club_code",
    "week",
    "game_type",
    "depth_team",
    "last_name",
    "first_name",
    "football_name",
    "formation",
    "gsis_id",
    "jersey_number",
    "position",
    "elias_id",
    "depth_position",
    "full_name",
}

ESPN_REQUIRED_COLUMNS = {
    "dt",
    "team",
    "player_name",
    "espn_id",
    "gsis_id",
    "pos_grp_id",
    "pos_grp",
    "pos_id",
    "pos_name",
    "pos_abb",
    "pos_slot",
    "pos_rank",
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


def get_depth_chart_files(
    source_directory: Path,
) -> list[Path]:
    """Return sorted source Parquet files."""

    if not source_directory.exists():
        raise FileNotFoundError(
            "Depth-chart source directory does not exist: "
            f"{source_directory}"
        )

    if not source_directory.is_dir():
        raise RuntimeError(
            "Depth-chart source path is not a directory: "
            f"{source_directory}"
        )

    source_files = sorted(
        source_directory.glob(
            "depth_charts_*.parquet"
        )
    )

    if not source_files:
        raise FileNotFoundError(
            "No depth-chart Parquet files found in: "
            f"{source_directory}"
        )

    return source_files


def build_parquet_source(
    source_files: list[Path],
) -> str:
    """Build a DuckDB-compatible Parquet file list."""

    if not source_files:
        raise ValueError(
            "Depth-chart file list must not be empty."
        )

    escaped_paths = [
        str(path.resolve()).replace(
            "'",
            "''",
        )
        for path in source_files
    ]

    quoted_paths = ", ".join(
        f"'{path}'"
        for path in escaped_paths
    )

    return f"[{quoted_paths}]"


def validate_parquet_columns(
    connection: duckdb.DuckDBPyConnection,
    parquet_source: str,
    required_columns: set[str],
    source_name: str,
) -> None:
    """Validate one source generation's Parquet columns."""

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
        required_columns
        - available_columns
    )

    if missing_columns:
        raise RuntimeError(
            f"{source_name} depth-chart source "
            "is missing columns: "
            + ", ".join(missing_columns)
        )

    logger.info(
        "%s depth-chart columns validated: %s.",
        source_name,
        len(required_columns),
    )


def create_legacy_raw_table(
    connection: duckdb.DuckDBPyConnection,
    parquet_source: str,
) -> None:
    """Create raw.depth_charts_legacy."""

    connection.execute(
        f"""
        CREATE OR REPLACE TABLE {LEGACY_FULL_NAME} AS
        SELECT
            * EXCLUDE(filename),
            filename AS source_file
        FROM read_parquet(
            {parquet_source},
            union_by_name = true,
            filename = true
        )
        """
    )


def create_espn_raw_table(
    connection: duckdb.DuckDBPyConnection,
    parquet_source: str,
) -> None:
    """Create raw.depth_charts_espn with source season."""

    connection.execute(
        f"""
        CREATE OR REPLACE TABLE {ESPN_FULL_NAME} AS
        SELECT
            CAST(
                REGEXP_EXTRACT(
                    filename,
                    'depth_charts_([0-9]{{4}})[.]parquet$',
                    1
                )
                AS INTEGER
            ) AS source_season,
            * EXCLUDE(filename),
            filename AS source_file
        FROM read_parquet(
            {parquet_source},
            union_by_name = true,
            filename = true
        )
        """
    )


def validate_loaded_table(
    connection: duckdb.DuckDBPyConnection,
    full_table_name: str,
    expected_row_count: int,
    expected_source_count: int,
) -> tuple[int, int]:
    """Validate one loaded raw depth-chart table."""

    row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {full_table_name}
        """
    ).fetchone()[0]

    if row_count != expected_row_count:
        raise RuntimeError(
            f"{full_table_name} row count does not match "
            f"the Parquet source: {row_count} != "
            f"{expected_row_count}."
        )

    source_count = connection.execute(
        f"""
        SELECT COUNT(DISTINCT source_file)
        FROM {full_table_name}
        """
    ).fetchone()[0]

    if source_count != expected_source_count:
        raise RuntimeError(
            f"{full_table_name} source-file count is invalid: "
            f"{source_count} != {expected_source_count}."
        )

    null_source_file_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {full_table_name}
        WHERE source_file IS NULL
        """
    ).fetchone()[0]

    if null_source_file_count > 0:
        raise RuntimeError(
            f"{full_table_name} contains null source files."
        )

    return (
        row_count,
        source_count,
    )


def validate_espn_source_seasons(
    connection: duckdb.DuckDBPyConnection,
) -> int:
    """Validate ESPN seasons derived from filenames."""

    null_season_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {ESPN_FULL_NAME}
        WHERE source_season IS NULL
        """
    ).fetchone()[0]

    if null_season_count > 0:
        raise RuntimeError(
            "ESPN raw depth charts contain "
            f"{null_season_count} null source seasons."
        )

    season_count = connection.execute(
        f"""
        SELECT COUNT(DISTINCT source_season)
        FROM {ESPN_FULL_NAME}
        """
    ).fetchone()[0]

    if season_count == 0:
        raise RuntimeError(
            "ESPN raw depth charts contain no seasons."
        )

    return season_count


def load_depth_charts_to_duckdb(
    database_file: Path = DATABASE_FILE,
    legacy_data_dir: Path = LEGACY_DATA_DIR,
    espn_data_dir: Path = ESPN_DATA_DIR,
) -> None:
    """Load both depth-chart generations into DuckDB."""

    validate_database_file(
        database_file
    )

    legacy_files = get_depth_chart_files(
        legacy_data_dir
    )
    espn_files = get_depth_chart_files(
        espn_data_dir
    )

    legacy_source = build_parquet_source(
        legacy_files
    )
    espn_source = build_parquet_source(
        espn_files
    )

    logger.info(
        "Loading raw depth-chart tables..."
    )

    try:
        with duckdb.connect(
            str(database_file)
        ) as connection:
            validate_parquet_columns(
                connection=connection,
                parquet_source=legacy_source,
                required_columns=LEGACY_REQUIRED_COLUMNS,
                source_name="Legacy",
            )
            validate_parquet_columns(
                connection=connection,
                parquet_source=espn_source,
                required_columns=ESPN_REQUIRED_COLUMNS,
                source_name="ESPN",
            )

            expected_legacy_rows = connection.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet(
                    {legacy_source},
                    union_by_name = true
                )
                """
            ).fetchone()[0]

            expected_espn_rows = connection.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet(
                    {espn_source},
                    union_by_name = true
                )
                """
            ).fetchone()[0]

            connection.execute(
                f"""
                CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA}
                """
            )

            connection.execute(
                "BEGIN TRANSACTION"
            )

            try:
                create_legacy_raw_table(
                    connection,
                    legacy_source,
                )
                create_espn_raw_table(
                    connection,
                    espn_source,
                )

                (
                    legacy_row_count,
                    legacy_source_count,
                ) = validate_loaded_table(
                    connection=connection,
                    full_table_name=LEGACY_FULL_NAME,
                    expected_row_count=expected_legacy_rows,
                    expected_source_count=len(
                        legacy_files
                    ),
                )

                (
                    espn_row_count,
                    espn_source_count,
                ) = validate_loaded_table(
                    connection=connection,
                    full_table_name=ESPN_FULL_NAME,
                    expected_row_count=expected_espn_rows,
                    expected_source_count=len(
                        espn_files
                    ),
                )

                espn_season_count = (
                    validate_espn_source_seasons(
                        connection
                    )
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
            "Raw depth-chart load failed."
        )
        raise

    logger.info(
        "Legacy raw depth charts loaded: "
        "%s rows from %s files.",
        legacy_row_count,
        legacy_source_count,
    )
    logger.info(
        "ESPN raw depth charts loaded: "
        "%s rows from %s files and %s seasons.",
        espn_row_count,
        espn_source_count,
        espn_season_count,
    )
    logger.info(
        "Tables created: %s and %s.",
        LEGACY_FULL_NAME,
        ESPN_FULL_NAME,
    )
    logger.info(
        "Raw depth-chart load completed successfully."
    )


def main() -> None:
    """Run the raw depth-chart loading workflow."""

    load_depth_charts_to_duckdb()


if __name__ == "__main__":
    main()