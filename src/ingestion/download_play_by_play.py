"""
NFL Analytics Platform
Play-by-Play Data Ingestion

Purpose:
    Download season-level NFL play-by-play data
    from nflverse and store it as Parquet files.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import argparse
import logging
from pathlib import Path

import nflreadpy as nfl


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PBP_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "pbp"

FIRST_AVAILABLE_SEASON = 1999
LAST_COMPLETED_SEASON = 2025

REQUIRED_PBP_COLUMNS = {
    "game_id",
    "play_id",
    "season",
}


def validate_season(
    season: int,
) -> None:
    """Validate that a season is available for PBP ingestion."""

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


def get_season_file(
    season: int,
) -> Path:
    """Return the local Parquet path for one NFL season."""

    validate_season(season)

    return PBP_DATA_DIR / f"pbp_{season}.parquet"


def build_season_range(
    start_season: int,
    end_season: int,
) -> list[int]:
    """Build an inclusive, validated range of NFL seasons."""

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


def download_season_pbp(
    season: int,
    overwrite: bool = False,
) -> Path:
    """Download and save one season of NFL play-by-play data."""

    season_file = get_season_file(season)

    if season_file.is_file() and not overwrite:
        logger.info(
            "PBP file already exists; skipping download: %s",
            season_file,
        )
        return season_file

    logger.info(
        "Starting NFL play-by-play download for season %s...",
        season,
    )

    try:
        play_by_play = nfl.load_pbp(season)
    except Exception:
        logger.exception(
            "Failed to download play-by-play data "
            "for season %s.",
            season,
        )
        raise

    if play_by_play.is_empty():
        raise ValueError(
            f"Downloaded PBP dataset is empty for season {season}."
        )

    missing_columns = (
        REQUIRED_PBP_COLUMNS - set(play_by_play.columns)
    )

    if missing_columns:
        missing_names = ", ".join(
            sorted(missing_columns)
        )
        raise RuntimeError(
            "Downloaded PBP dataset is missing columns: "
            f"{missing_names}"
        )

    PBP_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file = season_file.with_suffix(
        ".tmp.parquet"
    )

    try:
        play_by_play.write_parquet(
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
            "Failed to save play-by-play data "
            "for season %s.",
            season,
        )
        raise

    logger.info(
        "PBP ingestion completed for season %s: "
        "%s rows and %s columns.",
        season,
        play_by_play.height,
        play_by_play.width,
    )
    logger.info(
        "Dataset saved to: %s",
        season_file,
    )

    return season_file


def download_play_by_play(
    start_season: int = LAST_COMPLETED_SEASON,
    end_season: int | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Download an inclusive range of NFL PBP seasons."""

    resolved_end_season = (
        start_season
        if end_season is None
        else end_season
    )

    seasons = build_season_range(
        start_season=start_season,
        end_season=resolved_end_season,
    )

    logger.info(
        "Starting PBP ingestion for %s season(s): %s-%s.",
        len(seasons),
        seasons[0],
        seasons[-1],
    )

    season_files = [
        download_season_pbp(
            season=season,
            overwrite=overwrite,
        )
        for season in seasons
    ]

    logger.info(
        "PBP ingestion completed for %s season(s).",
        len(season_files),
    )

    return season_files


def parse_arguments(
    arguments: list[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments for PBP ingestion."""

    parser = argparse.ArgumentParser(
        description=(
            "Download season-level NFL play-by-play "
            "data from nflverse."
        )
    )

    parser.add_argument(
        "--start-season",
        type=int,
        default=LAST_COMPLETED_SEASON,
        help=(
            "First NFL season to download. "
            f"Default: {LAST_COMPLETED_SEASON}."
        ),
    )
    parser.add_argument(
        "--end-season",
        type=int,
        default=None,
        help=(
            "Last NFL season to download. "
            "Defaults to the start season."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Download and replace existing season files."
        ),
    )

    return parser.parse_args(arguments)


def main() -> None:
    """Run play-by-play data ingestion."""

    arguments = parse_arguments()

    try:
        download_play_by_play(
            start_season=arguments.start_season,
            end_season=arguments.end_season,
            overwrite=arguments.overwrite,
        )
    except Exception:
        logger.exception(
            "NFL play-by-play ingestion failed."
        )
        raise


if __name__ == "__main__":
    main()