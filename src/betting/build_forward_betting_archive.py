"""Archive the current future betting board without overwriting prior snapshots."""

import logging
from pathlib import Path

import duckdb

from src.betting.forward_betting_archive import (
    prepare_forward_archive_rows,
    persist_forward_archive,
    validate_forward_archive,
)
from src.modeling.train_logistic_baseline import DATABASE_FILE, validate_database_file

logger = logging.getLogger(__name__)


def build_forward_betting_archive(
    refresh_run_id: str,
    database_file: Path = DATABASE_FILE,
) -> int:
    """Append the current board to immutable forward-test history."""
    validate_database_file(database_file)
    with duckdb.connect(str(database_file)) as connection:
        board = connection.execute("SELECT * FROM analytics.current_betting_board").fetchdf()
        rows = prepare_forward_archive_rows(board, refresh_run_id)
        connection.execute("BEGIN TRANSACTION")
        try:
            row_count = persist_forward_archive(connection, rows)
            validate_forward_archive(connection)
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
    logger.info("Forward betting archive validated: %s cumulative market rows.", row_count)
    return row_count
