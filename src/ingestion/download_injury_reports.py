"""
NFL Analytics Platform
Historical Injury Report Ingestion

Purpose:
    Download season-level NFL injury and practice reports
    from nflverse and store them as Parquet files.

Notes:
    The source contains one final weekly record per
    player-team-week. It does not contain a complete
    timestamped history of report changes.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import argparse
import logging
from pathlib import Path

import nflreadpy as nfl
import polars as pl


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INJURY_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "injuries"

FIRST_AVAILABLE_SEASON = 2009
FIRST_MODELING_SEASON = 2018
LAST_COMPLETED_SEASON = 2025

INJURY_KEY_COLUMNS = (
    "season",
    "team",
    "week",
    "gsis_id",
)

INJURY_RECORD_KEY_COLUMNS = (
    *INJURY_KEY_COLUMNS,
    "date_modified",
)

CANONICAL_INJURY_COLUMNS = (
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
)

CANONICAL_INJURY_SCHEMA = {
    "season": pl.Int32,
    "season_type": pl.String,
    "game_type": pl.String,
    "team": pl.String,
    "week": pl.Int32,
    "gsis_id": pl.String,
    "position": pl.String,
    "full_name": pl.String,
    "first_name": pl.String,
    "last_name": pl.String,
    "report_primary_injury": pl.String,
    "report_secondary_injury": pl.String,
    "report_status": pl.String,
    "practice_primary_injury": pl.String,
    "practice_secondary_injury": pl.String,
    "practice_status": pl.String,
    "date_modified": pl.Datetime(
        time_unit="us",
        time_zone="UTC",
    ),
}

REQUIRED_SOURCE_COLUMNS = {
    "season",
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
}

def validate_season(
    season: int,
) -> None:
    """Validate that a season is available for injury ingestion."""

    if type(season) is not int:
        raise TypeError(
            "Season must be an integer."
        )

    if not (
        FIRST_AVAILABLE_SEASON
        <= season
        <= LAST_COMPLETED_SEASON
    ):
        raise ValueError(
            "Season must be between "
            f"{FIRST_AVAILABLE_SEASON} and "
            f"{LAST_COMPLETED_SEASON}."
        )


def build_season_range(
    start_season: int,
    end_season: int,
) -> list[int]:
    """Build an inclusive, validated season range."""

    validate_season(start_season)
    validate_season(end_season)

    if start_season > end_season:
        raise ValueError(
            "Start season must not be later than end season."
        )

    return list(
        range(
            start_season,
            end_season + 1,
        )
    )


def get_season_file(
    season: int,
) -> Path:
    """Return the local Parquet path for one injury season."""

    validate_season(season)

    return (
        INJURY_DATA_DIR
        / f"injury_reports_{season}.parquet"
    )


def normalize_injury_schema(
    injury_data: pl.DataFrame,
) -> pl.DataFrame:
    """Normalize historical nflverse injury schema versions."""

    missing_source_columns = (
        REQUIRED_SOURCE_COLUMNS
        - set(injury_data.columns)
    )

    if missing_source_columns:
        missing_names = ", ".join(
            sorted(missing_source_columns)
        )
        raise RuntimeError(
            "Downloaded injury dataset is missing columns: "
            f"{missing_names}"
        )

    normalized_data = injury_data

    if "season_type" not in normalized_data.columns:
        normalized_data = normalized_data.with_columns(
            pl.when(
                pl.col("game_type") == "REG"
            )
            .then(
                pl.lit("REG")
            )
            .otherwise(
                pl.lit("POST")
            )
            .alias("season_type")
        )

    if "date_modified" not in normalized_data.columns:
        normalized_data = normalized_data.with_columns(
            pl.lit(
                None,
                dtype=pl.Datetime(
                    time_unit="us",
                    time_zone="UTC",
                ),
            ).alias("date_modified")
        )

    return (
        normalized_data
        .select(
            list(CANONICAL_INJURY_COLUMNS)
        )
        .with_columns(
            [
                pl.col(column).cast(data_type)
                for column, data_type
                in CANONICAL_INJURY_SCHEMA.items()
            ]
        )
    )


def validate_injury_data(
    injury_data: pl.DataFrame,
    season: int,
) -> None:
    """Validate one downloaded season of injury reports."""

    if injury_data.is_empty():
        raise ValueError(
            f"Downloaded injury dataset is empty for season {season}."
        )

    missing_columns = (
        set(CANONICAL_INJURY_COLUMNS)
        - set(injury_data.columns)
    )

    if missing_columns:
        missing_names = ", ".join(
            sorted(missing_columns)
        )
        raise RuntimeError(
            "Downloaded injury dataset is missing columns: "
            f"{missing_names}"
        )

    unexpected_seasons = (
        injury_data
        .filter(
            pl.col("season") != season
        )
        .height
    )

    if unexpected_seasons:
        raise RuntimeError(
            "Downloaded injury dataset contains records "
            f"outside season {season}."
        )

    null_key_counts = injury_data.select(
        [
            pl.col(column)
            .null_count()
            .alias(column)
            for column in INJURY_KEY_COLUMNS
        ]
    ).row(0)

    null_key_columns = [
        column
        for column, null_count
        in zip(
            INJURY_KEY_COLUMNS,
            null_key_counts,
            strict=True,
        )
        if null_count > 0
    ]

    if null_key_columns:
        raise RuntimeError(
            "Downloaded injury dataset contains null key columns: "
            + ", ".join(null_key_columns)
        )

    duplicate_records = (
        injury_data
        .group_by(
            list(INJURY_RECORD_KEY_COLUMNS)
        )
        .len()
        .filter(
            pl.col("len") > 1
        )
    )

    if not duplicate_records.is_empty():
        raise RuntimeError(
            "Downloaded injury dataset contains duplicate "
            "player-team-week snapshot records."
        )


def download_season_injuries(
    season: int,
    overwrite: bool = False,
) -> Path:
    """Download and save one season of injury reports."""

    season_file = get_season_file(season)

    if season_file.is_file() and not overwrite:
        logger.info(
            "Injury file already exists; skipping download: %s",
            season_file,
        )
        return season_file

    logger.info(
        "Starting NFL injury report download for season %s...",
        season,
    )

    try:
        injury_data = nfl.load_injuries(season)
    except Exception:
        logger.exception(
            "Failed to download injury reports for season %s.",
            season,
        )
        raise

    injury_data = normalize_injury_schema(
        injury_data
    )

    validate_injury_data(
        injury_data=injury_data,
        season=season,
    )

    INJURY_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file = season_file.with_suffix(
        ".tmp.parquet"
    )

    try:
        injury_data.write_parquet(
            temporary_file
        )
        temporary_file.replace(
            season_file
        )
    except Exception:
        temporary_file.unlink(
            missing_ok=True
        )
        logger.exception(
            "Failed to save injury reports for season %s.",
            season,
        )
        raise

    logger.info(
        "Injury ingestion completed for season %s: "
        "%s rows and %s columns.",
        season,
        injury_data.height,
        injury_data.width,
    )
    logger.info(
        "Dataset saved to: %s",
        season_file,
    )

    return season_file


def download_injury_reports(
    start_season: int = FIRST_MODELING_SEASON,
    end_season: int = LAST_COMPLETED_SEASON,
    overwrite: bool = False,
) -> list[Path]:
    """Download an inclusive range of injury-report seasons."""

    seasons = build_season_range(
        start_season=start_season,
        end_season=end_season,
    )

    logger.info(
        "Starting injury ingestion for %s season(s): %s-%s.",
        len(seasons),
        seasons[0],
        seasons[-1],
    )

    season_files = [
        download_season_injuries(
            season=season,
            overwrite=overwrite,
        )
        for season in seasons
    ]

    logger.info(
        "Injury ingestion completed for %s season(s).",
        len(season_files),
    )

    return season_files


def parse_arguments(
    arguments: list[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Download season-level NFL injury and "
            "practice reports from nflverse."
        )
    )

    parser.add_argument(
        "--start-season",
        type=int,
        default=FIRST_MODELING_SEASON,
        help=(
            "First NFL season to download. "
            f"Default: {FIRST_MODELING_SEASON}."
        ),
    )
    parser.add_argument(
        "--end-season",
        type=int,
        default=LAST_COMPLETED_SEASON,
        help=(
            "Last NFL season to download. "
            f"Default: {LAST_COMPLETED_SEASON}."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing season files.",
    )

    return parser.parse_args(arguments)


def main() -> None:
    """Run injury-report ingestion from the command line."""

    arguments = parse_arguments()

    try:
        download_injury_reports(
            start_season=arguments.start_season,
            end_season=arguments.end_season,
            overwrite=arguments.overwrite,
        )
    except Exception:
        logger.exception(
            "Injury report ingestion failed."
        )
        raise

    logger.info(
        "Injury report ingestion completed successfully."
    )


if __name__ == "__main__":
    main()