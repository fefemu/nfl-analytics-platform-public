"""
NFL Analytics Platform
Historical Data Ingestion

Purpose:
    Download historical NFL schedule data from nflverse
    and store it in the raw data layer.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging
from pathlib import Path

import nflreadpy as nfl


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
SCHEDULE_FILE = RAW_DATA_DIR / "schedules.parquet"


def download_schedule() -> None:
    """Download NFL schedule data and save it to the raw data layer."""

    logger.info("Starting NFL schedule ingestion...")

    try:
        schedules = nfl.load_schedules()
    except Exception:
        logger.exception("Failed to download NFL schedule data.")
        raise

    if schedules.is_empty():
        raise ValueError("The downloaded schedule dataset is empty.")

    try:
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        schedules.write_parquet(SCHEDULE_FILE)
    except Exception:
        logger.exception("Failed to save NFL schedule data.")
        raise

    logger.info(
        "Schedule ingestion completed: %s rows and %s columns.",
        schedules.height,
        schedules.width,
    )
    logger.info("Dataset saved to: %s", SCHEDULE_FILE)


def main() -> None:
    """Run the historical NFL data ingestion workflow."""

    download_schedule()


if __name__ == "__main__":
    main()