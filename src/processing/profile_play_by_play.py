"""
NFL Analytics Platform
Play-by-Play Data Profiler

Purpose:
    Inspect a local NFL play-by-play Parquet file
    without loading it into the project database.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PBP_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "pbp"
    / "pbp_2025.parquet"
)

REQUIRED_PROFILE_COLUMNS = {
    "game_id",
    "play_id",
    "season",
    "week",
    "game_date",
    "posteam",
    "defteam",
    "play_type",
    "epa",
}

CANDIDATE_FEATURE_COLUMNS = (
    # Identifiers and time
    "game_id",
    "play_id",
    "season",
    "season_type",
    "week",
    "game_date",
    "qtr",
    "drive",
    "game_seconds_remaining",
    "half_seconds_remaining",

    # Team assignment
    "posteam",
    "defteam",
    "home_team",
    "away_team",

    # Play situation
    "down",
    "ydstogo",
    "yardline_100",
    "goal_to_go",
    "score_differential",
    "posteam_score",
    "defteam_score",
    "wp",

    # Play classification
    "play_type",
    "qb_kneel",
    "qb_spike",
    "aborted_play",
    "pass",
    "rush",
    "qb_dropback",
    "qb_scramble",
    "passer_player_id",
    "passer_player_name",
    "rusher_player_id",
    "rusher_player_name",
    "pass_attempt",
    "incomplete_pass",
    "passing_yards",
    "qb_hit",
    "complete_pass",
    "sack",
    "interception",
    "fumble_lost",
    "penalty",
    "touchdown",
    "first_down",
    "two_point_attempt",
    "special_teams_play",

    # Performance values
    "yards_gained",
    "air_yards",
    "yards_after_catch",
    "epa",
    "wpa",
    "success",
    "cpoe",
    "series_success",
)


@dataclass(frozen=True)
class PbpFileProfile:
    """Store structural information about one PBP file."""

    file_path: Path
    file_size_bytes: int
    row_count: int
    column_count: int
    game_count: int
    minimum_season: int
    maximum_season: int
    minimum_game_date: str
    maximum_game_date: str


def validate_pbp_file(
    pbp_file: Path,
) -> None:
    """Validate that a local PBP Parquet file exists."""

    if not pbp_file.is_file():
        raise FileNotFoundError(
            f"PBP Parquet file does not exist: {pbp_file}"
        )

    if pbp_file.suffix.lower() != ".parquet":
        raise ValueError(
            f"PBP file must use the Parquet format: {pbp_file}"
        )


def load_pbp_column_names(
    connection: duckdb.DuckDBPyConnection,
    pbp_file: Path,
) -> list[str]:
    """Read PBP column names without loading all rows."""

    connection.execute(
        """
        SELECT *
        FROM read_parquet(?)
        LIMIT 0
        """,
        [str(pbp_file)],
    )

    return [
        description[0]
        for description in connection.description
    ]


def validate_profile_columns(
    column_names: list[str],
) -> None:
    """Validate columns required by the PBP profiler."""

    missing_columns = (
        REQUIRED_PROFILE_COLUMNS
        - set(column_names)
    )

    if missing_columns:
        missing_names = ", ".join(
            sorted(missing_columns)
        )
        raise RuntimeError(
            "PBP file is missing required columns: "
            f"{missing_names}"
        )


def profile_pbp_file(
    pbp_file: Path = DEFAULT_PBP_FILE,
) -> PbpFileProfile:
    """Profile one local PBP Parquet file."""

    validate_pbp_file(pbp_file)

    logger.info(
        "Starting PBP file profiling: %s",
        pbp_file,
    )

    with duckdb.connect(":memory:") as connection:
        column_names = load_pbp_column_names(
            connection=connection,
            pbp_file=pbp_file,
        )
        validate_profile_columns(
            column_names=column_names,
        )

        profile_row = connection.execute(
            """
            SELECT
                COUNT(*) AS row_count,
                COUNT(DISTINCT game_id) AS game_count,
                MIN(season) AS minimum_season,
                MAX(season) AS maximum_season,
                MIN(game_date) AS minimum_game_date,
                MAX(game_date) AS maximum_game_date
            FROM read_parquet(?)
            """,
            [str(pbp_file)],
        ).fetchone()

    profile = PbpFileProfile(
        file_path=pbp_file,
        file_size_bytes=pbp_file.stat().st_size,
        row_count=profile_row[0],
        column_count=len(column_names),
        game_count=profile_row[1],
        minimum_season=profile_row[2],
        maximum_season=profile_row[3],
        minimum_game_date=str(profile_row[4]),
        maximum_game_date=str(profile_row[5]),
    )

    return profile


def inspect_candidate_feature_columns(
    pbp_file: Path,
) -> tuple[list[str], list[str]]:
    """Identify available and missing candidate PBP columns."""

    validate_pbp_file(pbp_file)

    with duckdb.connect(":memory:") as connection:
        column_names = load_pbp_column_names(
            connection=connection,
            pbp_file=pbp_file,
        )

    available_columns = [
        column_name
        for column_name in CANDIDATE_FEATURE_COLUMNS
        if column_name in column_names
    ]
    missing_columns = [
        column_name
        for column_name in CANDIDATE_FEATURE_COLUMNS
        if column_name not in column_names
    ]

    return (
        available_columns,
        missing_columns,
    )


def log_pbp_profile(
    profile: PbpFileProfile,
) -> None:
    """Log the structural PBP file profile."""

    file_size_megabytes = (
        profile.file_size_bytes
        / 1024
        / 1024
    )

    logger.info(
        "PBP file size: %.2f MiB",
        file_size_megabytes,
    )
    logger.info(
        "PBP rows: %s",
        profile.row_count,
    )
    logger.info(
        "PBP columns: %s",
        profile.column_count,
    )
    logger.info(
        "NFL games: %s",
        profile.game_count,
    )
    logger.info(
        "Season range: %s-%s",
        profile.minimum_season,
        profile.maximum_season,
    )
    logger.info(
        "Game date range: %s to %s",
        profile.minimum_game_date,
        profile.maximum_game_date,
    )


def log_candidate_feature_columns(
    available_columns: list[str],
    missing_columns: list[str],
) -> None:
    """Log candidate PBP feature availability."""

    logger.info(
        "Candidate feature columns available: %s/%s",
        len(available_columns),
        len(CANDIDATE_FEATURE_COLUMNS),
    )
    logger.info(
        "Available candidate columns: %s",
        ", ".join(available_columns),
    )

    if missing_columns:
        logger.warning(
            "Missing candidate columns: %s",
            ", ".join(missing_columns),
        )
    else:
        logger.info(
            "All candidate feature columns are available."
        )


def parse_arguments(
    arguments: list[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments for PBP profiling."""

    parser = argparse.ArgumentParser(
        description=(
            "Profile a local NFL play-by-play Parquet file."
        )
    )
    parser.add_argument(
        "pbp_file",
        nargs="?",
        type=Path,
        default=DEFAULT_PBP_FILE,
        help=(
            "PBP Parquet file to profile. "
            "Defaults to the local 2025 dataset."
        ),
    )

    return parser.parse_args(arguments)


def main() -> None:
    """Run play-by-play file profiling."""

    arguments = parse_arguments()

    try:
        profile = profile_pbp_file(
            pbp_file=arguments.pbp_file,
        )
        (
            available_columns,
            missing_columns,
        ) = inspect_candidate_feature_columns(
            pbp_file=arguments.pbp_file,
        )

        log_pbp_profile(
            profile=profile,
        )
        log_candidate_feature_columns(
            available_columns=available_columns,
            missing_columns=missing_columns,
        )

        logger.info(
            "PBP file profiling completed successfully."
        )
    except Exception:
        logger.exception(
            "PBP file profiling failed."
        )
        raise


if __name__ == "__main__":
    main()