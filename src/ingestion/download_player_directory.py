"""
NFL Analytics Platform
NFL Player Directory Ingestion

Purpose:
    Download the nflverse player directory and preserve
    stable player identifiers and source metadata locally.

Leakage note:
    The player directory is a current identity registry.
    Stable identifiers may be used for historical joins.
    Current fields such as latest_team and status must not
    be treated as historical pregame attributes.

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
PLAYER_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "players"
)
PLAYER_DIRECTORY_FILE = (
    PLAYER_DATA_DIR
    / "player_directory.parquet"
)

CANONICAL_PLAYER_COLUMNS = (
    "gsis_id",
    "display_name",
    "common_first_name",
    "first_name",
    "last_name",
    "short_name",
    "football_name",
    "suffix",
    "esb_id",
    "nfl_id",
    "pfr_id",
    "pff_id",
    "otc_id",
    "espn_id",
    "smart_id",
    "birth_date",
    "position_group",
    "position",
    "ngs_position_group",
    "ngs_position",
    "height",
    "weight",
    "headshot",
    "college_name",
    "college_conference",
    "jersey_number",
    "rookie_season",
    "last_season",
    "latest_team",
    "status",
    "ngs_status",
    "ngs_status_short_description",
    "years_of_experience",
    "pff_position",
    "pff_status",
    "draft_year",
    "draft_round",
    "draft_pick",
    "draft_team",
)

CANONICAL_PLAYER_SCHEMA = {
    "gsis_id": pl.String,
    "display_name": pl.String,
    "common_first_name": pl.String,
    "first_name": pl.String,
    "last_name": pl.String,
    "short_name": pl.String,
    "football_name": pl.String,
    "suffix": pl.String,
    "esb_id": pl.String,
    "nfl_id": pl.String,
    "pfr_id": pl.String,
    "pff_id": pl.String,
    "otc_id": pl.String,
    "espn_id": pl.String,
    "smart_id": pl.String,
    "birth_date": pl.String,
    "position_group": pl.String,
    "position": pl.String,
    "ngs_position_group": pl.String,
    "ngs_position": pl.String,
    "height": pl.Int32,
    "weight": pl.Int32,
    "headshot": pl.String,
    "college_name": pl.String,
    "college_conference": pl.String,
    "jersey_number": pl.String,
    "rookie_season": pl.Int32,
    "last_season": pl.Int32,
    "latest_team": pl.String,
    "status": pl.String,
    "ngs_status": pl.String,
    "ngs_status_short_description": pl.String,
    "years_of_experience": pl.Int32,
    "pff_position": pl.String,
    "pff_status": pl.String,
    "draft_year": pl.Int32,
    "draft_round": pl.Int32,
    "draft_pick": pl.Int32,
    "draft_team": pl.String,
}

REQUIRED_SOURCE_COLUMNS = set(
    CANONICAL_PLAYER_COLUMNS
)


def get_player_directory_file() -> Path:
    """Return the canonical local player-directory path."""

    return PLAYER_DIRECTORY_FILE


def normalize_player_directory_schema(
    player_data: pl.DataFrame,
) -> pl.DataFrame:
    """Return the player directory with a canonical schema."""

    missing_columns = (
        REQUIRED_SOURCE_COLUMNS
        - set(player_data.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Downloaded player directory is missing columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    return (
        player_data
        .select(
            list(CANONICAL_PLAYER_COLUMNS)
        )
        .with_columns(
            [
                pl.col(column).cast(
                    data_type,
                    strict=True,
                )
                for column, data_type
                in CANONICAL_PLAYER_SCHEMA.items()
            ]
        )
    )


def count_duplicate_non_null_ids(
    player_data: pl.DataFrame,
    identifier_column: str,
) -> int:
    """Count duplicated non-null player identifier groups."""

    return (
        player_data
        .filter(
            pl.col(identifier_column).is_not_null()
        )
        .group_by(
            identifier_column
        )
        .len()
        .filter(
            pl.col("len") > 1
        )
        .height
    )


def validate_player_directory(
    player_data: pl.DataFrame,
) -> None:
    """Validate the canonical player identity registry."""

    if player_data.is_empty():
        raise ValueError(
            "Downloaded player directory is empty."
        )

    missing_columns = (
        set(CANONICAL_PLAYER_COLUMNS)
        - set(player_data.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Downloaded player directory is missing columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    null_gsis_count = player_data.filter(
        pl.col("gsis_id").is_null()
        | (
            pl.col("gsis_id")
            .str.strip_chars()
            == ""
        )
    ).height

    if null_gsis_count > 0:
        raise RuntimeError(
            "Downloaded player directory contains "
            f"{null_gsis_count} rows without a GSIS ID."
        )

    for identifier_column in (
        "gsis_id",
        "pfr_id",
        "espn_id",
    ):
        duplicate_count = count_duplicate_non_null_ids(
            player_data=player_data,
            identifier_column=identifier_column,
        )

        if duplicate_count > 0:
            raise RuntimeError(
                "Downloaded player directory contains "
                f"{duplicate_count} duplicate "
                f"{identifier_column} groups."
            )

    pfr_coverage_count = player_data.filter(
        pl.col("pfr_id").is_not_null()
    ).height

    if pfr_coverage_count == 0:
        raise RuntimeError(
            "Downloaded player directory contains "
            "no PFR identifiers."
        )

    invalid_season_count = player_data.filter(
        (
            pl.col("rookie_season").is_not_null()
            & (
                pl.col("rookie_season") < 1900
            )
        )
        | (
            pl.col("last_season").is_not_null()
            & (
                pl.col("last_season") < 1900
            )
        )
        | (
            pl.col("rookie_season").is_not_null()
            & pl.col("last_season").is_not_null()
            & (
                pl.col("rookie_season")
                > pl.col("last_season")
            )
        )
    ).height

    if invalid_season_count > 0:
        raise RuntimeError(
            "Downloaded player directory contains "
            f"{invalid_season_count} invalid career ranges."
        )


def download_player_directory(
    overwrite: bool = False,
) -> Path:
    """Download and save the nflverse player directory."""

    player_directory_file = get_player_directory_file()

    if (
        player_directory_file.is_file()
        and not overwrite
    ):
        logger.info(
            "Player-directory file already exists; "
            "skipping download: %s",
            player_directory_file,
        )
        return player_directory_file

    logger.info(
        "Starting NFL player-directory download..."
    )

    try:
        player_data = nfl.load_players()
    except Exception:
        logger.exception(
            "Failed to download the NFL player directory."
        )
        raise

    player_data = normalize_player_directory_schema(
        player_data
    )

    validate_player_directory(
        player_data
    )

    PLAYER_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file = player_directory_file.with_suffix(
        ".tmp.parquet"
    )

    try:
        player_data.write_parquet(
            temporary_file
        )
        temporary_file.replace(
            player_directory_file
        )
    except Exception:
        temporary_file.unlink(
            missing_ok=True
        )
        logger.exception(
            "Failed to save the NFL player directory."
        )
        raise

    logger.info(
        "Player-directory ingestion completed: "
        "%s rows and %s columns.",
        player_data.height,
        player_data.width,
    )
    logger.info(
        "PFR identifier coverage: %s rows.",
        player_data.filter(
            pl.col("pfr_id").is_not_null()
        ).height,
    )
    logger.info(
        "ESPN identifier coverage: %s rows.",
        player_data.filter(
            pl.col("espn_id").is_not_null()
        ).height,
    )
    logger.info(
        "Dataset saved to: %s",
        player_directory_file,
    )

    return player_directory_file


def parse_arguments(
    arguments: list[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Download the nflverse NFL player directory."
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the existing local player directory.",
    )

    return parser.parse_args(
        arguments
    )


def main() -> None:
    """Run player-directory ingestion."""

    arguments = parse_arguments()

    try:
        download_player_directory(
            overwrite=arguments.overwrite,
        )
    except Exception:
        logger.exception(
            "Player-directory ingestion failed."
        )
        raise

    logger.info(
        "Player-directory ingestion completed successfully."
    )


if __name__ == "__main__":
    main()