"""
NFL Analytics Platform
Injury Report DuckDB Loader

Purpose:
    Load canonical season-level injury-report Parquet files
    into the raw DuckDB layer.

Notes:
    Raw source values and timestamped status changes are
    preserved without model-facing cleaning.

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
INJURY_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "injuries"
DATABASE_FILE = PROJECT_ROOT / "data" / "nfl_analytics.duckdb"

SCHEMA_NAME = "raw"
TABLE_NAME = "injury_reports"
FULL_TABLE_NAME = f"{SCHEMA_NAME}.{TABLE_NAME}"

REQUIRED_INJURY_COLUMNS = {
    "season",
    "season_type",
    "game_type",
    "team",
    "week",
    "gsis_id",
    "position",
    "full_name",
    "first_name",
    "last_name",
    "report_primary_injury",
    "report_secondary_injury",
    "report_status",
    "practice_primary_injury",
    "practice_secondary_injury",
    "practice_status",
    "date_modified",
}


def get_injury_files(
    injury_data_dir: Path = INJURY_DATA_DIR,
) -> list[Path]:
    """Return canonical injury Parquet files in season order."""

    if not injury_data_dir.exists():
        raise FileNotFoundError(
            "Injury data directory does not exist: "
            f"{injury_data_dir}"
        )

    if not injury_data_dir.is_dir():
        raise RuntimeError(
            "Injury data path is not a directory: "
            f"{injury_data_dir}"
        )

    injury_files = sorted(
        injury_data_dir.glob(
            "injury_reports_*.parquet"
        )
    )

    if not injury_files:
        raise FileNotFoundError(
            "No injury-report Parquet files found in: "
            f"{injury_data_dir}"
        )

    logger.info(
        "Injury-report files discovered: %s file(s) in %s.",
        len(injury_files),
        injury_data_dir,
    )

    return injury_files


def build_parquet_source(
    injury_files: list[Path],
) -> str:
    """Build a DuckDB-compatible Parquet file-list expression."""

    if not injury_files:
        raise ValueError(
            "Injury file list must not be empty."
        )

    escaped_paths = [
        str(path.resolve()).replace(
            "'",
            "''",
        )
        for path in injury_files
    ]

    quoted_paths = ", ".join(
        f"'{path}'"
        for path in escaped_paths
    )

    return f"[{quoted_paths}]"


def validate_parquet_columns(
    connection: duckdb.DuckDBPyConnection,
    parquet_source: str,
) -> None:
    """Validate the canonical injury Parquet schema."""

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
        REQUIRED_INJURY_COLUMNS
        - available_columns
    )

    if missing_columns:
        raise RuntimeError(
            "Missing required injury-report columns: "
            + ", ".join(missing_columns)
        )

    logger.info(
        "Required injury-report columns validated: %s columns.",
        len(REQUIRED_INJURY_COLUMNS),
    )


def validate_loaded_table(
    connection: duckdb.DuckDBPyConnection,
    expected_row_count: int,
) -> tuple[int, int, int]:
    """Validate the loaded raw injury-report table."""

    row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {FULL_TABLE_NAME}
        """
    ).fetchone()[0]

    if row_count != expected_row_count:
        raise RuntimeError(
            "Raw injury-report row count does not match "
            f"the Parquet source: {row_count} != "
            f"{expected_row_count}."
        )

    column_count = connection.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_schema = ?
          AND table_name = ?
        """,
        [
            SCHEMA_NAME,
            TABLE_NAME,
        ],
    ).fetchone()[0]

    if column_count != len(
        REQUIRED_INJURY_COLUMNS
    ):
        raise RuntimeError(
            "Raw injury-report column count is invalid: "
            f"{column_count}."
        )

    season_count = connection.execute(
        f"""
        SELECT COUNT(DISTINCT season)
        FROM {FULL_TABLE_NAME}
        """
    ).fetchone()[0]

    if season_count == 0:
        raise RuntimeError(
            "Raw injury-report table contains no seasons."
        )

    return (
        row_count,
        column_count,
        season_count,
    )


def load_injury_reports_to_duckdb(
    database_file: Path = DATABASE_FILE,
    injury_data_dir: Path = INJURY_DATA_DIR,
) -> None:
    """Create or replace raw.injury_reports in DuckDB."""

    injury_files = get_injury_files(
        injury_data_dir
    )
    parquet_source = build_parquet_source(
        injury_files
    )

    logger.info(
        "Loading injury reports into DuckDB..."
    )

    try:
        with duckdb.connect(
            str(database_file)
        ) as connection:
            validate_parquet_columns(
                connection,
                parquet_source,
            )

            expected_row_count = connection.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet(
                    {parquet_source},
                    union_by_name = true
                )
                """
            ).fetchone()[0]

            connection.execute(
                f"""
                CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME}
                """
            )

            connection.execute(
                f"""
                CREATE OR REPLACE TABLE {FULL_TABLE_NAME} AS
                SELECT *
                FROM read_parquet(
                    {parquet_source},
                    union_by_name = true
                )
                """
            )

            (
                row_count,
                column_count,
                season_count,
            ) = validate_loaded_table(
                connection=connection,
                expected_row_count=expected_row_count,
            )

    except Exception:
        logger.exception(
            "Failed to load injury reports into DuckDB."
        )
        raise

    logger.info(
        "DuckDB injury load completed: "
        "%s rows, %s columns and %s seasons.",
        row_count,
        column_count,
        season_count,
    )
    logger.info(
        "Database saved to: %s",
        database_file,
    )
    logger.info(
        "Table created: %s",
        FULL_TABLE_NAME,
    )


def main() -> None:
    """Run the raw injury-report loading workflow."""

    try:
        load_injury_reports_to_duckdb()
    except Exception:
        logger.exception(
            "Raw injury-report load failed."
        )
        raise

    logger.info(
        "Raw injury-report load completed successfully."
    )


if __name__ == "__main__":
    main()