"""
NFL Analytics Platform
SQL Runner

Purpose:
    Execute SQL files against the local DuckDB database.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import argparse
import logging
from pathlib import Path

import duckdb


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "data" / "nfl_analytics.duckdb"


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Execute a SQL file against the NFL Analytics DuckDB database."
    )

    parser.add_argument(
        "sql_file",
        type=Path,
        help="Path to the SQL file relative to the project root.",
    )

    return parser.parse_args()


def load_sql_file(sql_file: Path) -> str:
    """Load SQL statements from a file."""

    resolved_path = (
        sql_file
        if sql_file.is_absolute()
        else PROJECT_ROOT / sql_file
    )

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"SQL file not found: {resolved_path}"
        )

    if not resolved_path.is_file():
        raise ValueError(
            f"SQL path is not a file: {resolved_path}"
        )

    logger.info("Loading SQL file: %s", resolved_path)

    sql = resolved_path.read_text(encoding="utf-8").strip()

    if not sql:
        raise ValueError(
            f"SQL file is empty: {resolved_path}"
        )

    logger.info("SQL file loaded successfully.")

    return sql


def execute_sql(sql: str) -> None:
    """Execute a complete SQL file and print its final result."""

    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"DuckDB database not found: {DATABASE_PATH}"
        )

    logger.info("Connecting to DuckDB: %s", DATABASE_PATH)

    with duckdb.connect(str(DATABASE_PATH)) as connection:
        print("\n" + "=" * 80)
        print("NFL Analytics Platform - SQL Runner")
        print("=" * 80)

        logger.info("Executing SQL file against DuckDB...")

        result = connection.execute(sql)

        if result.description is not None:
            columns = [column[0] for column in result.description]
            rows = result.fetchall()

            print("\nResult")
            print("-" * 80)
            print(" | ".join(columns))
            print("-" * 80)

            for row in rows:
                print(
                    " | ".join(
                        "NULL" if value is None else str(value)
                        for value in row
                    )
                )

            print(f"\nRows returned: {len(rows)}")
        else:
            logger.info(
                "SQL executed successfully. No result set returned."
            )

        print("\n" + "=" * 80)
        print("Execution completed successfully.")
        print("=" * 80)


def main() -> None:
    """Run the SQL file provided through the command line."""

    args = parse_arguments()

    try:
        sql = load_sql_file(args.sql_file)
        execute_sql(sql)

    except Exception:
        logger.exception("SQL execution failed.")
        raise


if __name__ == "__main__":
    main()
