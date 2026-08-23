"""Tests for play-by-play file profiling."""

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from src.processing.profile_play_by_play import (
    inspect_candidate_feature_columns,
    profile_pbp_file,
    validate_pbp_file,
    validate_profile_columns,
)


def create_test_pbp_file(
    pbp_file: Path,
) -> None:
    """Create a small valid PBP Parquet test file."""

    test_data = pl.DataFrame(
        {
            "game_id": [
                "2025_01_DAL_PHI",
                "2025_01_DAL_PHI",
                "2025_01_KC_LAC",
            ],
            "play_id": [
                1,
                2,
                1,
            ],
            "season": [
                2025,
                2025,
                2025,
            ],
            "week": [
                1,
                1,
                1,
            ],
            "game_date": [
                date(2025, 9, 4),
                date(2025, 9, 4),
                date(2025, 9, 5),
            ],
            "posteam": [
                "PHI",
                "DAL",
                "KC",
            ],
            "defteam": [
                "DAL",
                "PHI",
                "LAC",
            ],
            "play_type": [
                "run",
                "pass",
                "pass",
            ],
            "epa": [
                0.10,
                -0.20,
                0.30,
            ],
        }
    )

    test_data.write_parquet(pbp_file)


def test_validate_pbp_file_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """Reject a PBP file that does not exist."""

    missing_file = tmp_path / "missing.parquet"

    with pytest.raises(
        FileNotFoundError,
        match="does not exist",
    ):
        validate_pbp_file(missing_file)


def test_validate_pbp_file_rejects_non_parquet_file(
    tmp_path: Path,
) -> None:
    """Reject an existing file without a Parquet extension."""

    csv_file = tmp_path / "pbp.csv"
    csv_file.write_text(
        "game_id,play_id",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Parquet format",
    ):
        validate_pbp_file(csv_file)


def test_validate_profile_columns_rejects_missing_column() -> None:
    """Reject PBP columns without the required profile fields."""

    with pytest.raises(
        RuntimeError,
        match="missing required columns",
    ):
        validate_profile_columns(
            column_names=[
                "game_id",
                "play_id",
            ]
        )


def test_profile_pbp_file_returns_expected_summary(
    tmp_path: Path,
) -> None:
    """Return structural metrics for a valid PBP file."""

    pbp_file = tmp_path / "pbp_2025.parquet"
    create_test_pbp_file(pbp_file)

    profile = profile_pbp_file(pbp_file)

    assert profile.file_path == pbp_file
    assert profile.file_size_bytes > 0
    assert profile.row_count == 3
    assert profile.column_count == 9
    assert profile.game_count == 2
    assert profile.minimum_season == 2025
    assert profile.maximum_season == 2025
    assert profile.minimum_game_date == "2025-09-04"
    assert profile.maximum_game_date == "2025-09-05"


def test_inspect_candidate_feature_columns_separates_columns(
    tmp_path: Path,
) -> None:
    """Separate available and missing candidate feature columns."""

    pbp_file = tmp_path / "pbp_2025.parquet"
    create_test_pbp_file(pbp_file)

    (
        available_columns,
        missing_columns,
    ) = inspect_candidate_feature_columns(
        pbp_file=pbp_file,
    )

    assert "game_id" in available_columns
    assert "posteam" in available_columns
    assert "epa" in available_columns

    assert "wp" in missing_columns
    assert "success" in missing_columns

    assert not (
        set(available_columns)
        & set(missing_columns)
    )