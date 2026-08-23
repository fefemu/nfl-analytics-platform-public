"""
NFL Analytics Platform
Schedule Pipeline Runner

Purpose:
    Run the complete raw-to-processed schedule pipeline.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging

from src.processing.build_processed_schedule import (
    build_processed_schedule,
)
from src.processing.load_schedule_to_duckdb import (
    load_schedule_to_duckdb,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def run_schedule_pipeline() -> None:
    """Run the raw and processed schedule pipeline steps."""

    logger.info("Starting schedule pipeline...")

    logger.info("Step 1/2: Loading raw schedule data.")
    load_schedule_to_duckdb()

    logger.info("Step 2/2: Building processed schedule data.")
    build_processed_schedule()

    logger.info("Schedule pipeline completed successfully.")


def main() -> None:
    """Run the schedule pipeline workflow."""

    try:
        run_schedule_pipeline()
    except Exception:
        logger.exception("Schedule pipeline failed.")
        raise


if __name__ == "__main__":
    main()