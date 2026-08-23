"""Run one audited in-season modeling, odds, EV and forward-archive refresh."""

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import duckdb

from src.betting.build_forward_betting_archive import build_forward_betting_archive
from src.modeling.train_logistic_baseline import DATABASE_FILE, validate_database_file
from src.pipeline.run_modeling_pipeline import run_modeling_pipeline
from src.pipeline.run_odds_pipeline import run_odds_pipeline
from src.pipeline.run_odds_snapshot_pipeline import run_odds_snapshot_pipeline

logger = logging.getLogger(__name__)
RUN_TABLE = "analytics.refresh_run_history"


def create_refresh_run_id(started_at: datetime | None = None) -> str:
    """Create a sortable unique refresh identifier."""
    if started_at is None:
        started_at = datetime.now(timezone.utc)
    return f"refresh_{started_at.strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"


def _ensure_run_table(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        f"""
        CREATE SCHEMA IF NOT EXISTS analytics;
        CREATE TABLE IF NOT EXISTS {RUN_TABLE} (
            refresh_run_id VARCHAR PRIMARY KEY,
            refresh_mode VARCHAR NOT NULL,
            snapshot_file VARCHAR,
            started_at TIMESTAMPTZ NOT NULL,
            completed_at TIMESTAMPTZ,
            status VARCHAR NOT NULL,
            archived_market_row_count BIGINT,
            error_message VARCHAR
        )
        """
    )


def record_refresh_start(
    database_file: Path,
    refresh_run_id: str,
    refresh_mode: str,
    snapshot_file: Path | None,
    started_at: datetime,
) -> None:
    """Insert the immutable identity and starting state of a refresh."""
    with duckdb.connect(str(database_file)) as connection:
        _ensure_run_table(connection)
        connection.execute(
            f"INSERT INTO {RUN_TABLE} VALUES (?, ?, ?, ?, NULL, 'RUNNING', NULL, NULL)",
            [refresh_run_id, refresh_mode, str(snapshot_file) if snapshot_file else None, started_at],
        )


def record_refresh_completion(
    database_file: Path,
    refresh_run_id: str,
    status: str,
    archived_market_row_count: int | None = None,
    error_message: str | None = None,
) -> None:
    """Finalize one existing refresh audit row."""
    if status not in {"SUCCESS", "FAILED"}:
        raise ValueError("Refresh completion status must be SUCCESS or FAILED.")
    with duckdb.connect(str(database_file)) as connection:
        _ensure_run_table(connection)
        connection.execute(
            f"""
            UPDATE {RUN_TABLE}
            SET completed_at = ?, status = ?, archived_market_row_count = ?, error_message = ?
            WHERE refresh_run_id = ?
            """,
            [datetime.now(timezone.utc), status, archived_market_row_count, error_message, refresh_run_id],
        )


def run_in_season_refresh(
    database_file: Path = DATABASE_FILE,
    snapshot_file: Path | None = None,
) -> str:
    """Run modeling then online/offline odds processing and archive future markets."""
    validate_database_file(database_file)
    if snapshot_file is not None and not snapshot_file.is_file():
        raise FileNotFoundError(f"Odds snapshot does not exist: {snapshot_file}")
    started_at = datetime.now(timezone.utc)
    refresh_run_id = create_refresh_run_id(started_at)
    refresh_mode = "OFFLINE_SNAPSHOT" if snapshot_file is not None else "ONLINE_API"
    record_refresh_start(database_file, refresh_run_id, refresh_mode, snapshot_file, started_at)
    logger.info("In-season refresh started: %s (%s).", refresh_run_id, refresh_mode)
    try:
        run_modeling_pipeline(database_file=database_file)
        if snapshot_file is None:
            run_odds_pipeline()
        else:
            run_odds_snapshot_pipeline(snapshot_file=snapshot_file)
        archive_count = build_forward_betting_archive(
            refresh_run_id=refresh_run_id,
            database_file=database_file,
        )
        record_refresh_completion(database_file, refresh_run_id, "SUCCESS", archive_count)
    except Exception as error:
        record_refresh_completion(database_file, refresh_run_id, "FAILED", error_message=str(error)[:2000])
        logger.exception("In-season refresh failed: %s.", refresh_run_id)
        raise
    logger.info("In-season refresh completed successfully: %s.", refresh_run_id)
    return refresh_run_id


def parse_arguments() -> argparse.Namespace:
    """Require an explicit quota-consuming online or safe offline mode."""
    parser = argparse.ArgumentParser(description="Run an audited NFL in-season refresh.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--online", action="store_true", help="Download a new Odds API snapshot.")
    mode.add_argument("--snapshot", type=Path, help="Reuse an existing local odds JSON snapshot.")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    run_in_season_refresh(snapshot_file=args.snapshot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()
