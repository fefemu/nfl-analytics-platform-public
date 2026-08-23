"""
NFL Analytics Platform
Player Snap-Count DuckDB Loader

Purpose:
    Load canonical season-level player snap-count Parquet
    files into the raw DuckDB layer.

Notes:
    Raw source participation values are preserved, including
    the documented 1.01 special-teams rounding edge cases.

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
SNAP_COUNT_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "snap_counts"
)

TARGET_SCHEMA = "raw"
TARGET_TABLE = "player_snap_counts"
TARGET_FULL_NAME = (
    f"{TARGET_SCHEMA}.{TARGET_TABLE}"
)

REQUIRED_SNAP_COUNT_COLUMNS = {
    "game_id",
    "pfr_game_id",
    "season",
    "game_type",
    "week",
    "player",
    "pfr_player_id",
    "position",
    "team",
    "opponent",
    "offense_snaps",
    "offense_pct",
    "defense_snaps",
    "defense_pct",
    "st_snaps",
    "st_pct",
}

BUSINESS_KEY_COLUMNS = (
    "game_id",
    "team",
    "pfr_player_id",
)


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


def get_snap_count_files(
    snap_count_data_dir: Path = SNAP_COUNT_DATA_DIR,
) -> list[Path]:
    """Return canonical snap-count Parquet files."""

    if not snap_count_data_dir.exists():
        raise FileNotFoundError(
            "Snap-count data directory does not exist: "
            f"{snap_count_data_dir}"
        )

    if not snap_count_data_dir.is_dir():
        raise RuntimeError(
            "Snap-count data path is not a directory: "
            f"{snap_count_data_dir}"
        )

    snap_count_files = sorted(
        snap_count_data_dir.glob(
            "snap_counts_*.parquet"
        )
    )

    if not snap_count_files:
        raise FileNotFoundError(
            "No snap-count Parquet files found in: "
            f"{snap_count_data_dir}"
        )

    logger.info(
        "Snap-count files discovered: %s file(s) in %s.",
        len(snap_count_files),
        snap_count_data_dir,
    )

    return snap_count_files


def build_parquet_source(
    snap_count_files: list[Path],
) -> str:
    """Build a DuckDB-compatible Parquet file list."""

    if not snap_count_files:
        raise ValueError(
            "Snap-count file list must not be empty."
        )

    escaped_paths = [
        str(path.resolve()).replace(
            "'",
            "''",
        )
        for path in snap_count_files
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
    """Validate the canonical snap-count Parquet schema."""

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
        REQUIRED_SNAP_COUNT_COLUMNS
        - available_columns
    )

    if missing_columns:
        raise RuntimeError(
            "Missing required snap-count columns: "
            + ", ".join(
                missing_columns
            )
        )

    logger.info(
        "Required snap-count columns validated: %s columns.",
        len(REQUIRED_SNAP_COUNT_COLUMNS),
    )


def count_parquet_rows(
    connection: duckdb.DuckDBPyConnection,
    parquet_source: str,
) -> int:
    """Count source records across all Parquet files."""

    return connection.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet(
            {parquet_source},
            union_by_name = true
        )
        """
    ).fetchone()[0]


def create_raw_snap_count_table(
    connection: duckdb.DuckDBPyConnection,
    parquet_source: str,
) -> None:
    """Create raw.player_snap_counts."""

    connection.execute(
        f"""
        CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA}
        """
    )

    connection.execute(
        f"""
        CREATE OR REPLACE TABLE {TARGET_FULL_NAME} AS
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


def validate_loaded_table(
    connection: duckdb.DuckDBPyConnection,
    expected_row_count: int,
    expected_source_count: int,
) -> tuple[int, int, int]:
    """Validate the loaded raw snap-count table."""

    row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    if row_count != expected_row_count:
        raise RuntimeError(
            "Raw snap-count row count does not match "
            f"the Parquet source: {row_count} != "
            f"{expected_row_count}."
        )

    source_count = connection.execute(
        f"""
        SELECT COUNT(DISTINCT source_file)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    if source_count != expected_source_count:
        raise RuntimeError(
            "Raw snap-count source-file count is invalid: "
            f"{source_count} != {expected_source_count}."
        )

    season_count = connection.execute(
        f"""
        SELECT COUNT(DISTINCT season)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    if season_count != expected_source_count:
        raise RuntimeError(
            "Raw snap-count season count does not match "
            f"the source-file count: {season_count} != "
            f"{expected_source_count}."
        )

    null_key_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE game_id IS NULL
           OR team IS NULL
           OR pfr_player_id IS NULL
           OR source_file IS NULL
        """
    ).fetchone()[0]

    if null_key_count > 0:
        raise RuntimeError(
            "Raw snap-count table contains "
            f"{null_key_count} rows with null business keys."
        )

    duplicate_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT
                game_id,
                team,
                pfr_player_id
            FROM {TARGET_FULL_NAME}
            GROUP BY
                game_id,
                team,
                pfr_player_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_count > 0:
        raise RuntimeError(
            "Raw snap-count table contains "
            f"{duplicate_count} duplicate player-team-game keys."
        )

    invalid_snap_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE offense_snaps < 0
           OR defense_snaps < 0
           OR st_snaps < 0
           OR offense_pct < 0
           OR offense_pct > 1.01
           OR defense_pct < 0
           OR defense_pct > 1.01
           OR st_pct < 0
           OR st_pct > 1.01
        """
    ).fetchone()[0]

    if invalid_snap_count > 0:
        raise RuntimeError(
            "Raw snap-count table contains "
            f"{invalid_snap_count} invalid participation values."
        )

    return (
        row_count,
        source_count,
        season_count,
    )


def load_snap_counts_to_duckdb(
    database_file: Path = DATABASE_FILE,
    snap_count_data_dir: Path = SNAP_COUNT_DATA_DIR,
) -> None:
    """Create or replace raw.player_snap_counts."""

    validate_database_file(
        database_file
    )

    snap_count_files = get_snap_count_files(
        snap_count_data_dir
    )
    parquet_source = build_parquet_source(
        snap_count_files
    )

    logger.info(
        "Loading player snap counts into DuckDB..."
    )

    try:
        with duckdb.connect(
            str(database_file)
        ) as connection:
            validate_parquet_columns(
                connection=connection,
                parquet_source=parquet_source,
            )

            expected_row_count = count_parquet_rows(
                connection=connection,
                parquet_source=parquet_source,
            )

            connection.execute(
                "BEGIN TRANSACTION"
            )

            try:
                create_raw_snap_count_table(
                    connection=connection,
                    parquet_source=parquet_source,
                )

                (
                    row_count,
                    source_count,
                    season_count,
                ) = validate_loaded_table(
                    connection=connection,
                    expected_row_count=expected_row_count,
                    expected_source_count=len(
                        snap_count_files
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
            "Failed to load player snap counts into DuckDB."
        )
        raise

    logger.info(
        "DuckDB snap-count load completed: "
        "%s rows from %s files and %s seasons.",
        row_count,
        source_count,
        season_count,
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
    """Run the raw snap-count loading workflow."""

    try:
        load_snap_counts_to_duckdb()
    except Exception:
        logger.exception(
            "Raw snap-count load failed."
        )
        raise

    logger.info(
        "Raw snap-count load completed successfully."
    )


if __name__ == "__main__":
    main()