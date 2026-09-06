"""Capture and publish a kickoff-near market snapshot without rerunning models."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from src.betting.build_forward_betting_archive import build_forward_betting_archive
from src.deployment.build_dashboard_snapshot import DEFAULT_OUTPUT_FILE, build_dashboard_snapshot
from src.deployment.publish_dashboard_release import publish_dashboard_release
from src.modeling.train_logistic_baseline import DATABASE_FILE, validate_database_file
from src.pipeline.run_in_season_refresh import (
    create_refresh_run_id,
    record_refresh_completion,
    record_refresh_start,
)
from src.pipeline.run_odds_pipeline import run_odds_pipeline

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_kickoff_odds_capture(
    database_file: Path = DATABASE_FILE,
    *,
    output_file: Path = DEFAULT_OUTPUT_FILE,
    publish: bool = False,
) -> str:
    """Refresh only market layers, archive them and optionally publish validated state."""
    validate_database_file(database_file)
    started_at = datetime.now(timezone.utc)
    refresh_run_id = create_refresh_run_id(started_at).replace(
        "refresh_", "kickoff_capture_", 1
    )
    record_refresh_start(database_file, refresh_run_id, "KICKOFF_ODDS_CAPTURE", None, started_at)
    try:
        run_odds_pipeline()
        archive_count = build_forward_betting_archive(refresh_run_id, database_file)
        record_refresh_completion(database_file, refresh_run_id, "SUCCESS", archive_count)
        build_dashboard_snapshot(source_file=database_file, output_file=output_file)
        if publish:
            publish_dashboard_release(output_file, operational_file=database_file)
    except Exception as error:
        record_refresh_completion(
            database_file, refresh_run_id, "FAILED", error_message=str(error)[:2000]
        )
        raise
    return refresh_run_id


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DATABASE_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    run_kickoff_odds_capture(args.database, output_file=args.output, publish=args.publish)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()
