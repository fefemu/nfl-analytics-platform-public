"""
NFL Analytics Platform
Schedule Data Profiling

Purpose:
    Profile the raw NFL schedule dataset and report
    initial data quality information.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging
from pathlib import Path

import polars as pl


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEDULE_FILE = PROJECT_ROOT / "data" / "raw" / "schedules.parquet"

# Required for the profiling process to work correctly
REQUIRED_COLUMNS = [
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "away_team",
    "home_team",
]

# Useful fields that may contain missing values
OPTIONAL_COLUMNS = [
    "weekday",
    "gametime",
    "away_score",
    "home_score",
    "location",
    "result",
    "total",
    "overtime",
    "away_rest",
    "home_rest",
    "roof",
    "surface",
    "temp",
    "wind",
    "away_moneyline",
    "home_moneyline",
]


def load_schedule() -> pl.DataFrame:
    """Load the raw NFL schedule dataset."""

    if not SCHEDULE_FILE.exists():
        raise FileNotFoundError(
            f"Schedule file does not exist: {SCHEDULE_FILE}"
        )

    try:
        return pl.read_parquet(SCHEDULE_FILE)
    except Exception:
        logger.exception("Failed to read the NFL schedule dataset.")
        raise


def check_required_columns(schedule: pl.DataFrame) -> None:
    """Check whether all columns required for profiling are available."""

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in schedule.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required schedule columns: {missing_columns}"
        )


def profile_schedule(schedule: pl.DataFrame) -> None:
    """Log basic structure and data quality information."""

    check_required_columns(schedule)

    available_profile_columns = [
        column
        for column in REQUIRED_COLUMNS + OPTIONAL_COLUMNS
        if column in schedule.columns
    ]

    unavailable_optional_columns = [
        column
        for column in OPTIONAL_COLUMNS
        if column not in schedule.columns
    ]

    logger.info(
        "Schedule dataset loaded: %s rows and %s columns.",
        schedule.height,
        schedule.width,
    )

    duplicate_game_ids = (
        schedule
        .filter(pl.col("game_id").is_not_null())
        .select(pl.col("game_id").is_duplicated().sum())
        .item()
    )

    missing_summary = (
        schedule
        .select(
            [
                pl.col(column).null_count().alias(column)
                for column in available_profile_columns
            ]
        )
        .transpose(
            include_header=True,
            header_name="column_name",
            column_names=["missing_values"],
        )
        .sort("missing_values", descending=True)
    )

    logger.info(
        "Season range: %s-%s.",
        schedule["season"].min(),
        schedule["season"].max(),
    )
    logger.info(
        "Missing game_id values: %s.",
        schedule["game_id"].null_count(),
    )
    logger.info(
        "Duplicate game_id values: %s.",
        duplicate_game_ids,
    )

    if unavailable_optional_columns:
        logger.warning(
            "Optional columns not available: %s.",
            unavailable_optional_columns,
        )

    logger.info("Game type distribution:")
    print(
        schedule
        .group_by("game_type")
        .len()
        .sort("game_type")
    )

    logger.info("Missing values in available profiling columns:")
    print(missing_summary)


def main() -> None:
    """Run the schedule profiling workflow."""

    logger.info("Starting NFL schedule data profiling...")

    try:
        schedule = load_schedule()
        profile_schedule(schedule)
    except Exception:
        logger.exception("NFL schedule data profiling failed.")
        raise

    logger.info("NFL schedule data profiling completed.")


if __name__ == "__main__":
    main()