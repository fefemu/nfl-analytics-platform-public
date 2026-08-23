"""
NFL Analytics Platform
Depth-Chart Data Ingestion

Purpose:
    Download historical and current NFL depth-chart data
    from nflverse and preserve the two source generations
    as separate season-level Parquet datasets.

Source generations:
    2018-2024:
        Weekly legacy NFL depth charts.

    2025 onward:
        Timestamped ESPN depth-chart snapshots.

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
DEPTH_CHART_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "depth_charts"
)
LEGACY_DATA_DIR = (
    DEPTH_CHART_DATA_DIR
    / "legacy"
)
ESPN_DATA_DIR = (
    DEPTH_CHART_DATA_DIR
    / "espn"
)

FIRST_MODELING_SEASON = 2018
LAST_LEGACY_SEASON = 2024
FIRST_ESPN_SEASON = 2025
CURRENT_SEASON = 2026

LEGACY_GENERATION = "legacy"
ESPN_GENERATION = "espn"

LEGACY_REQUIRED_COLUMNS = {
    "season",
    "club_code",
    "week",
    "game_type",
    "depth_team",
    "last_name",
    "first_name",
    "football_name",
    "formation",
    "gsis_id",
    "jersey_number",
    "position",
    "elias_id",
    "depth_position",
    "full_name",
}

LEGACY_KEY_COLUMNS = (
    "season",
    "club_code",
    "game_type",
    "gsis_id",
)

ESPN_REQUIRED_COLUMNS = {
    "dt",
    "team",
    "player_name",
    "espn_id",
    "gsis_id",
    "pos_grp_id",
    "pos_grp",
    "pos_id",
    "pos_name",
    "pos_abb",
    "pos_slot",
    "pos_rank",
}

ESPN_KEY_COLUMNS = (
    "dt",
    "team",
    "espn_id",
    "pos_grp",
    "pos_name",
    "pos_slot",
    "pos_rank",
)


def validate_season(
    season: int,
) -> None:
    """Validate a supported depth-chart season."""

    if type(season) is not int:
        raise TypeError(
            "Season must be an integer."
        )

    if not (
        FIRST_MODELING_SEASON
        <= season
        <= CURRENT_SEASON
    ):
        raise ValueError(
            "Season must be between "
            f"{FIRST_MODELING_SEASON} and "
            f"{CURRENT_SEASON}."
        )


def get_source_generation(
    season: int,
) -> str:
    """Return the source generation for one season."""

    validate_season(season)

    if season <= LAST_LEGACY_SEASON:
        return LEGACY_GENERATION

    return ESPN_GENERATION


def build_season_range(
    start_season: int,
    end_season: int,
) -> list[int]:
    """Build an inclusive validated season range."""

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
    """Return the raw Parquet path for one season."""

    source_generation = get_source_generation(
        season
    )

    if source_generation == LEGACY_GENERATION:
        return (
            LEGACY_DATA_DIR
            / f"depth_charts_{season}.parquet"
        )

    return (
        ESPN_DATA_DIR
        / f"depth_charts_{season}.parquet"
    )


def validate_required_columns(
    depth_chart_data: pl.DataFrame,
    required_columns: set[str],
) -> None:
    """Validate required columns for one source generation."""

    missing_columns = (
        required_columns
        - set(depth_chart_data.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Downloaded depth-chart dataset is missing columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )


def count_null_key_rows(
    depth_chart_data: pl.DataFrame,
    key_columns: tuple[str, ...],
) -> int:
    """Count rows containing a null business-key field."""

    return depth_chart_data.filter(
        pl.any_horizontal(
            [
                pl.col(column).is_null()
                for column in key_columns
            ]
        )
    ).height


def validate_legacy_data(
    depth_chart_data: pl.DataFrame,
    season: int,
) -> None:
    """Validate a weekly legacy depth-chart season."""

    if depth_chart_data.is_empty():
        raise ValueError(
            "Downloaded legacy depth-chart dataset "
            f"is empty for season {season}."
        )

    validate_required_columns(
        depth_chart_data=depth_chart_data,
        required_columns=LEGACY_REQUIRED_COLUMNS,
    )

    unexpected_season_count = (
        depth_chart_data
        .filter(
            pl.col("season") != season
        )
        .height
    )

    if unexpected_season_count > 0:
        raise RuntimeError(
            "Legacy depth-chart dataset contains records "
            f"outside season {season}."
        )

    null_key_count = count_null_key_rows(
        depth_chart_data=depth_chart_data,
        key_columns=LEGACY_KEY_COLUMNS,
    )

    if null_key_count > 0:
        raise RuntimeError(
            "Legacy depth-chart dataset contains "
            f"{null_key_count} null business-key rows."
        )

    invalid_null_week_count = (
        depth_chart_data
        .filter(
            pl.col("week").is_null()
            & (
                pl.col("game_type")
                != "SBBYE"
            )
        )
        .height
    )

    if invalid_null_week_count > 0:
        raise RuntimeError(
            "Legacy depth-chart dataset contains "
            f"{invalid_null_week_count} unexpected null weeks."
        )

    invalid_rank_count = (
        depth_chart_data
        .filter(
            ~pl.col("depth_team").is_in(
                [
                    "1",
                    "2",
                    "3",
                ]
            )
        )
        .height
    )

    if invalid_rank_count > 0:
        raise RuntimeError(
            "Legacy depth-chart dataset contains "
            f"{invalid_rank_count} invalid depth-team values."
        )


def validate_espn_data(
    depth_chart_data: pl.DataFrame,
    season: int,
) -> None:
    """Validate a timestamped ESPN depth-chart season."""

    if depth_chart_data.is_empty():
        raise ValueError(
            "Downloaded ESPN depth-chart dataset "
            f"is empty for season {season}."
        )

    validate_required_columns(
        depth_chart_data=depth_chart_data,
        required_columns=ESPN_REQUIRED_COLUMNS,
    )

    null_key_count = count_null_key_rows(
        depth_chart_data=depth_chart_data,
        key_columns=ESPN_KEY_COLUMNS,
    )

    if null_key_count > 0:
        raise RuntimeError(
            "ESPN depth-chart dataset contains "
            f"{null_key_count} null business-key rows."
        )

    duplicate_key_count = (
        depth_chart_data
        .group_by(
            list(ESPN_KEY_COLUMNS)
        )
        .len()
        .filter(
            pl.col("len") > 1
        )
        .height
    )

    if duplicate_key_count > 0:
        raise RuntimeError(
            "ESPN depth-chart dataset contains "
            f"{duplicate_key_count} duplicate business-key groups."
        )

    invalid_rank_count = (
        depth_chart_data
        .filter(
            pl.col("pos_rank") < 1
        )
        .height
    )

    if invalid_rank_count > 0:
        raise RuntimeError(
            "ESPN depth-chart dataset contains "
            f"{invalid_rank_count} invalid position ranks."
        )

    invalid_timestamp_count = (
        depth_chart_data
        .select(
            pl.col("dt")
            .str.to_datetime(
                format="%Y-%m-%dT%H:%M:%SZ",
                time_zone="UTC",
                strict=False,
            )
            .is_null()
            .sum()
        )
        .item()
    )

    if invalid_timestamp_count > 0:
        raise RuntimeError(
            "ESPN depth-chart dataset contains "
            f"{invalid_timestamp_count} invalid timestamps."
        )


def validate_depth_chart_data(
    depth_chart_data: pl.DataFrame,
    season: int,
) -> None:
    """Validate one downloaded depth-chart season."""

    source_generation = get_source_generation(
        season
    )

    if source_generation == LEGACY_GENERATION:
        validate_legacy_data(
            depth_chart_data=depth_chart_data,
            season=season,
        )
        return

    validate_espn_data(
        depth_chart_data=depth_chart_data,
        season=season,
    )


def download_season_depth_charts(
    season: int,
    overwrite: bool = False,
) -> Path:
    """Download and save one depth-chart season."""

    season_file = get_season_file(
        season
    )
    source_generation = get_source_generation(
        season
    )

    if season_file.is_file() and not overwrite:
        logger.info(
            "Depth-chart file already exists; "
            "skipping download: %s",
            season_file,
        )
        return season_file

    logger.info(
        "Starting %s depth-chart download "
        "for season %s...",
        source_generation,
        season,
    )

    try:
        depth_chart_data = (
            nfl.load_depth_charts(
                season
            )
        )
    except Exception:
        logger.exception(
            "Failed to download depth charts "
            "for season %s.",
            season,
        )
        raise

    validate_depth_chart_data(
        depth_chart_data=depth_chart_data,
        season=season,
    )

    season_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file = season_file.with_suffix(
        ".tmp.parquet"
    )

    try:
        depth_chart_data.write_parquet(
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
            "Failed to save depth charts "
            "for season %s.",
            season,
        )
        raise

    logger.info(
        "Depth-chart ingestion completed "
        "for season %s: %s rows and %s columns.",
        season,
        depth_chart_data.height,
        depth_chart_data.width,
    )
    logger.info(
        "Source generation: %s.",
        source_generation,
    )
    logger.info(
        "Dataset saved to: %s",
        season_file,
    )

    return season_file


def download_depth_charts(
    start_season: int = FIRST_MODELING_SEASON,
    end_season: int = CURRENT_SEASON,
    overwrite: bool = False,
) -> list[Path]:
    """Download an inclusive depth-chart season range."""

    seasons = build_season_range(
        start_season=start_season,
        end_season=end_season,
    )

    logger.info(
        "Starting depth-chart ingestion "
        "for %s season(s): %s-%s.",
        len(seasons),
        seasons[0],
        seasons[-1],
    )

    season_files = [
        download_season_depth_charts(
            season=season,
            overwrite=overwrite,
        )
        for season in seasons
    ]

    logger.info(
        "Depth-chart ingestion completed "
        "for %s season(s).",
        len(season_files),
    )

    return season_files


def parse_arguments(
    arguments: list[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Download legacy and ESPN NFL depth charts "
            "from nflverse."
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
        default=CURRENT_SEASON,
        help=(
            "Last NFL season to download. "
            f"Default: {CURRENT_SEASON}."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing season files.",
    )

    return parser.parse_args(
        arguments
    )


def main() -> None:
    """Run depth-chart ingestion."""

    arguments = parse_arguments()

    try:
        download_depth_charts(
            start_season=arguments.start_season,
            end_season=arguments.end_season,
            overwrite=arguments.overwrite,
        )
    except Exception:
        logger.exception(
            "Depth-chart ingestion failed."
        )
        raise

    logger.info(
        "Depth-chart ingestion completed successfully."
    )


if __name__ == "__main__":
    main()