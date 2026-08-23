"""
NFL Analytics Platform
Historical Player Snap-Count Ingestion

Purpose:
    Download season-level NFL player snap counts from
    nflverse and store canonical Parquet files locally.

Leakage note:
    Snap counts describe actual participation in a completed
    game. The current game's snap counts must never be used
    as pregame features for that same game. Later feature
    builders must shift all rolling participation measures
    so that they use completed prior games only.

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
SNAP_COUNT_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "snap_counts"
)

FIRST_MODELING_SEASON = 2018
LAST_COMPLETED_SEASON = 2025

CANONICAL_SNAP_COUNT_COLUMNS = (
    "game_id",
    "pfr_game_id",
    "season",
    "game_type",
    "week",
    "player",
    "pfr_player_id",
    "position",
    "team",
    "opponent",
    "offense_snaps",
    "offense_pct",
    "defense_snaps",
    "defense_pct",
    "st_snaps",
    "st_pct",
)

CANONICAL_SNAP_COUNT_SCHEMA = {
    "game_id": pl.String,
    "pfr_game_id": pl.String,
    "season": pl.Int32,
    "game_type": pl.String,
    "week": pl.Int32,
    "player": pl.String,
    "pfr_player_id": pl.String,
    "position": pl.String,
    "team": pl.String,
    "opponent": pl.String,
    "offense_snaps": pl.Float64,
    "offense_pct": pl.Float64,
    "defense_snaps": pl.Float64,
    "defense_pct": pl.Float64,
    "st_snaps": pl.Float64,
    "st_pct": pl.Float64,
}

REQUIRED_SOURCE_COLUMNS = set(
    CANONICAL_SNAP_COUNT_COLUMNS
)

SNAP_COUNT_KEY_COLUMNS = (
    "game_id",
    "team",
    "pfr_player_id",
)

SNAP_COLUMNS = (
    "offense_snaps",
    "defense_snaps",
    "st_snaps",
)

SNAP_SHARE_COLUMNS = (
    "offense_pct",
    "defense_pct",
    "st_pct",
)

MAX_SOURCE_SNAP_SHARE = 1.01


def validate_season(
    season: int,
) -> None:
    """Validate a supported snap-count season."""

    if type(season) is not int:
        raise TypeError(
            "Season must be an integer."
        )

    if not (
        FIRST_MODELING_SEASON
        <= season
        <= LAST_COMPLETED_SEASON
    ):
        raise ValueError(
            "Season must be between "
            f"{FIRST_MODELING_SEASON} and "
            f"{LAST_COMPLETED_SEASON}."
        )


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
    """Return the local Parquet path for one season."""

    validate_season(season)

    return (
        SNAP_COUNT_DATA_DIR
        / f"snap_counts_{season}.parquet"
    )


def normalize_snap_count_schema(
    snap_count_data: pl.DataFrame,
) -> pl.DataFrame:
    """Return snap counts with one canonical typed schema."""

    missing_columns = (
        REQUIRED_SOURCE_COLUMNS
        - set(snap_count_data.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Downloaded snap-count dataset is missing columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    return (
        snap_count_data
        .select(
            list(CANONICAL_SNAP_COUNT_COLUMNS)
        )
        .with_columns(
            [
                pl.col(column).cast(
                    data_type,
                    strict=True,
                )
                for column, data_type
                in CANONICAL_SNAP_COUNT_SCHEMA.items()
            ]
        )
    )


def count_null_key_rows(
    snap_count_data: pl.DataFrame,
) -> int:
    """Count rows with a null snap-count business key."""

    return snap_count_data.filter(
        pl.any_horizontal(
            [
                pl.col(column).is_null()
                for column
                in SNAP_COUNT_KEY_COLUMNS
            ]
        )
    ).height


def count_duplicate_player_games(
    snap_count_data: pl.DataFrame,
) -> int:
    """Count duplicate player-team-game business-key groups."""

    return (
        snap_count_data
        .group_by(
            list(SNAP_COUNT_KEY_COLUMNS)
        )
        .len()
        .filter(
            pl.col("len") > 1
        )
        .height
    )


def validate_snap_count_data(
    snap_count_data: pl.DataFrame,
    season: int,
) -> None:
    """Validate one canonical season of player snap counts."""

    if snap_count_data.is_empty():
        raise ValueError(
            "Downloaded snap-count dataset is empty "
            f"for season {season}."
        )

    missing_columns = (
        set(CANONICAL_SNAP_COUNT_COLUMNS)
        - set(snap_count_data.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Downloaded snap-count dataset is missing columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    unexpected_season_count = snap_count_data.filter(
        pl.col("season") != season
    ).height

    if unexpected_season_count > 0:
        raise RuntimeError(
            "Downloaded snap-count dataset contains records "
            f"outside season {season}."
        )

    null_key_count = count_null_key_rows(
        snap_count_data
    )

    if null_key_count > 0:
        raise RuntimeError(
            "Downloaded snap-count dataset contains "
            f"{null_key_count} rows with null business keys."
        )

    duplicate_count = count_duplicate_player_games(
        snap_count_data
    )

    if duplicate_count > 0:
        raise RuntimeError(
            "Downloaded snap-count dataset contains "
            f"{duplicate_count} duplicate player-team-game keys."
        )

    invalid_snap_count = snap_count_data.filter(
        pl.any_horizontal(
            [
                pl.col(column).is_not_null()
                & (
                    ~pl.col(column).is_finite()
                    | (pl.col(column) < 0)
                )
                for column in SNAP_COLUMNS
            ]
        )
    ).height

    if invalid_snap_count > 0:
        raise RuntimeError(
            "Downloaded snap-count dataset contains "
            "negative or non-finite snap counts."
        )

    invalid_snap_share_count = snap_count_data.filter(
        pl.any_horizontal(
            [
                pl.col(column).is_not_null()
                & (
                    ~pl.col(column).is_finite()
                    | (pl.col(column) < 0)
                    | (
                        pl.col(column)
                        > MAX_SOURCE_SNAP_SHARE
                    )
                )
                for column in SNAP_SHARE_COLUMNS
            ]
        )
    ).height

    if invalid_snap_share_count > 0:
        raise RuntimeError(
            "Downloaded snap-count dataset contains "
            "Snap percentages outside the accepted source range "
            f"0 to {MAX_SOURCE_SNAP_SHARE}."
        )

    invalid_team_game_count = (
        snap_count_data
        .group_by(
            [
                "game_id",
                "team",
            ]
        )
        .agg(
            pl.col("opponent")
            .n_unique()
            .alias("opponent_count")
        )
        .filter(
            pl.col("opponent_count") != 1
        )
        .height
    )

    if invalid_team_game_count > 0:
        raise RuntimeError(
            "Downloaded snap-count dataset contains "
            "team-games with inconsistent opponents."
        )


def download_season_snap_counts(
    season: int,
    overwrite: bool = False,
) -> Path:
    """Download and save one season of player snap counts."""

    season_file = get_season_file(
        season
    )

    if season_file.is_file() and not overwrite:
        logger.info(
            "Snap-count file already exists; "
            "skipping download: %s",
            season_file,
        )
        return season_file

    logger.info(
        "Starting NFL snap-count download for season %s...",
        season,
    )

    try:
        snap_count_data = nfl.load_snap_counts(
            season
        )
    except Exception:
        logger.exception(
            "Failed to download snap counts for season %s.",
            season,
        )
        raise

    snap_count_data = normalize_snap_count_schema(
        snap_count_data
    )

    validate_snap_count_data(
        snap_count_data=snap_count_data,
        season=season,
    )

    SNAP_COUNT_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file = season_file.with_suffix(
        ".tmp.parquet"
    )

    try:
        snap_count_data.write_parquet(
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
            "Failed to save snap counts for season %s.",
            season,
        )
        raise

    logger.info(
        "Snap-count ingestion completed for season %s: "
        "%s rows and %s columns.",
        season,
        snap_count_data.height,
        snap_count_data.width,
    )
    logger.info(
        "Dataset saved to: %s",
        season_file,
    )

    return season_file


def download_snap_counts(
    start_season: int = FIRST_MODELING_SEASON,
    end_season: int = LAST_COMPLETED_SEASON,
    overwrite: bool = False,
) -> list[Path]:
    """Download an inclusive range of snap-count seasons."""

    seasons = build_season_range(
        start_season=start_season,
        end_season=end_season,
    )

    logger.info(
        "Starting snap-count ingestion for "
        "%s season(s): %s-%s.",
        len(seasons),
        seasons[0],
        seasons[-1],
    )

    season_files = [
        download_season_snap_counts(
            season=season,
            overwrite=overwrite,
        )
        for season in seasons
    ]

    logger.info(
        "Snap-count ingestion completed for %s season(s).",
        len(season_files),
    )

    return season_files


def parse_arguments(
    arguments: list[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Download season-level NFL player snap counts "
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

    return parser.parse_args(
        arguments
    )


def main() -> None:
    """Run snap-count ingestion from the command line."""

    arguments = parse_arguments()

    try:
        download_snap_counts(
            start_season=arguments.start_season,
            end_season=arguments.end_season,
            overwrite=arguments.overwrite,
        )
    except Exception:
        logger.exception(
            "Snap-count ingestion failed."
        )
        raise

    logger.info(
        "Snap-count ingestion completed successfully."
    )


if __name__ == "__main__":
    main()