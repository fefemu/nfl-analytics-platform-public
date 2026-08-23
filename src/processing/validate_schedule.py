"""
NFL Analytics Platform
Schedule Data Validation

Purpose:
    Validate the raw NFL schedule dataset against structural
    and business rules before downstream processing.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging
from dataclasses import dataclass, field
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

# Validation thresholds
MIN_SUPPORTED_SEASON = 1999
CURRENT_SEASON = 2026
MAX_FUTURE_SEASON_OFFSET = 1
MAX_REGULAR_SEASON_WEEK = 18
MAX_REASONABLE_SCORE = 80

REQUIRED_COLUMNS = [
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "away_team",
    "home_team",
    "away_score",
    "home_score",
]

ALLOWED_GAME_TYPES = {
    "REG",
    "WC",
    "DIV",
    "CON",
    "SB",
}


@dataclass
class ValidationResult:
    """Store validation errors, warnings and informational messages."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Return True when no blocking validation errors exist."""

        return not self.errors


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


def validate_required_columns(
    schedule: pl.DataFrame,
    result: ValidationResult,
) -> None:
    """Validate that all required columns are present."""

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in schedule.columns
    ]

    if missing_columns:
        result.errors.append(
            f"Missing required columns: {missing_columns}."
        )


def validate_identifiers(
    schedule: pl.DataFrame,
    result: ValidationResult,
) -> None:
    """Validate game identifiers."""

    missing_game_ids = schedule["game_id"].null_count()

    if missing_game_ids > 0:
        result.errors.append(
            f"game_id contains {missing_game_ids} missing values."
        )

    duplicate_game_ids = (
        schedule
        .filter(pl.col("game_id").is_not_null())
        .select(pl.col("game_id").is_duplicated().sum())
        .item()
    )

    if duplicate_game_ids > 0:
        result.errors.append(
            f"game_id contains {duplicate_game_ids} duplicate values."
        )


def validate_seasons(
    schedule: pl.DataFrame,
    result: ValidationResult,
) -> None:
    """Validate season values."""

    missing_seasons = schedule["season"].null_count()

    if missing_seasons > 0:
        result.errors.append(
            f"season contains {missing_seasons} missing values."
        )

    seasons_before_supported_range = schedule.filter(
        pl.col("season") < MIN_SUPPORTED_SEASON
    ).height

    if seasons_before_supported_range > 0:
        result.errors.append(
            f"Found {seasons_before_supported_range} records with season "
            f"before {MIN_SUPPORTED_SEASON}."
        )

    future_season_limit = CURRENT_SEASON + MAX_FUTURE_SEASON_OFFSET

    seasons_above_expected_range = schedule.filter(
        pl.col("season") > future_season_limit
    ).height

    if seasons_above_expected_range > 0:
        result.warnings.append(
            f"Found {seasons_above_expected_range} records with season "
            f"after {future_season_limit}."
        )

    result.info.append(
        f"Season range: {schedule['season'].min()}-"
        f"{schedule['season'].max()}."
    )


def validate_weeks(
    schedule: pl.DataFrame,
    result: ValidationResult,
) -> None:
    """Validate week values."""

    missing_weeks = schedule["week"].null_count()

    if missing_weeks > 0:
        result.errors.append(
            f"week contains {missing_weeks} missing values."
        )

    weeks_below_minimum = schedule.filter(
        pl.col("week") < 1
    ).height

    if weeks_below_minimum > 0:
        result.errors.append(
            f"Found {weeks_below_minimum} records with week below 1."
        )

    invalid_regular_season_weeks = schedule.filter(
        (pl.col("game_type") == "REG")
        & (pl.col("week") > MAX_REGULAR_SEASON_WEEK)
    ).height

    if invalid_regular_season_weeks > 0:
        result.errors.append(
            f"Found {invalid_regular_season_weeks} regular-season records "
            f"with week above {MAX_REGULAR_SEASON_WEEK}."
        )


def validate_game_types(
    schedule: pl.DataFrame,
    result: ValidationResult,
) -> None:
    """Validate game type values."""

    missing_game_types = schedule["game_type"].null_count()

    if missing_game_types > 0:
        result.errors.append(
            f"game_type contains {missing_game_types} missing values."
        )

    invalid_game_types = (
        schedule
        .filter(
            pl.col("game_type").is_not_null()
            & ~pl.col("game_type").is_in(ALLOWED_GAME_TYPES)
        )
        .select("game_type")
        .unique()
        .to_series()
        .to_list()
    )

    if invalid_game_types:
        result.errors.append(
            f"Unexpected game_type values: {invalid_game_types}."
        )


def validate_teams(
    schedule: pl.DataFrame,
    result: ValidationResult,
) -> None:
    """Validate home and away team values."""

    missing_teams = schedule.filter(
        pl.col("home_team").is_null()
        | pl.col("away_team").is_null()
    ).height

    if missing_teams > 0:
        result.errors.append(
            f"Found {missing_teams} records with a missing team."
        )

    same_team_games = schedule.filter(
        pl.col("home_team") == pl.col("away_team")
    ).height

    if same_team_games > 0:
        result.errors.append(
            f"Found {same_team_games} records where home and away teams "
            f"are identical."
        )


def validate_scores(
    schedule: pl.DataFrame,
    result: ValidationResult,
) -> None:
    """Validate score completeness and reasonable ranges."""

    inconsistent_score_nulls = schedule.filter(
        pl.col("home_score").is_null()
        != pl.col("away_score").is_null()
    ).height

    if inconsistent_score_nulls > 0:
        result.errors.append(
            f"Found {inconsistent_score_nulls} records where only one "
            f"team score is missing."
        )

    negative_scores = schedule.filter(
        (pl.col("home_score") < 0)
        | (pl.col("away_score") < 0)
    ).height

    if negative_scores > 0:
        result.errors.append(
            f"Found {negative_scores} records containing a negative score."
        )

    unusually_high_scores = schedule.filter(
        (pl.col("home_score") > MAX_REASONABLE_SCORE)
        | (pl.col("away_score") > MAX_REASONABLE_SCORE)
    ).height

    if unusually_high_scores > 0:
        result.warnings.append(
            f"Found {unusually_high_scores} records with a team score "
            f"above {MAX_REASONABLE_SCORE}."
        )

    games_without_scores = schedule.filter(
        pl.col("home_score").is_null()
        & pl.col("away_score").is_null()
    ).height

    result.info.append(
        f"Games without final scores: {games_without_scores}."
    )


def validate_schedule(schedule: pl.DataFrame) -> ValidationResult:
    """Run structural and business validation rules."""

    result = ValidationResult()

    validate_required_columns(schedule, result)

    if result.errors:
        return result

    validate_identifiers(schedule, result)
    validate_seasons(schedule, result)
    validate_weeks(schedule, result)
    validate_game_types(schedule, result)
    validate_teams(schedule, result)
    validate_scores(schedule, result)

    result.info.append(
        f"Validated dataset size: {schedule.height} rows and "
        f"{schedule.width} columns."
    )

    return result


def log_validation_result(result: ValidationResult) -> None:
    """Write validation findings to the application log."""

    for message in result.info:
        logger.info(message)

    for message in result.warnings:
        logger.warning(message)

    for message in result.errors:
        logger.error(message)


def main() -> None:
    """Run the schedule validation workflow."""

    logger.info("Starting NFL schedule data validation...")

    try:
        schedule = load_schedule()
        validation_result = validate_schedule(schedule)
    except Exception:
        logger.exception("NFL schedule data validation failed.")
        raise

    log_validation_result(validation_result)

    if not validation_result.is_valid:
        raise ValueError(
            f"Schedule validation failed with "
            f"{len(validation_result.errors)} blocking error(s)."
        )

    logger.info(
        "Schedule validation completed successfully with "
        "%s warning(s).",
        len(validation_result.warnings),
    )


if __name__ == "__main__":
    main()