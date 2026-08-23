"""Tests for play-by-play data ingestion."""

from pathlib import Path

import pytest
import polars as pl

import src.ingestion.download_play_by_play as pbp_ingestion

from src.ingestion.download_play_by_play import (
    FIRST_AVAILABLE_SEASON,
    LAST_COMPLETED_SEASON,
    build_season_range,
    download_play_by_play,
    download_season_pbp,
    get_season_file,
    parse_arguments,
    validate_season,
)


def test_validate_season_accepts_available_boundaries() -> None:
    """Accept the first and last configured PBP seasons."""

    validate_season(FIRST_AVAILABLE_SEASON)
    validate_season(LAST_COMPLETED_SEASON)


@pytest.mark.parametrize(
    "season",
    [
        FIRST_AVAILABLE_SEASON - 1,
        LAST_COMPLETED_SEASON + 1,
    ],
)
def test_validate_season_rejects_unavailable_season(
    season: int,
) -> None:
    """Reject seasons outside the configured PBP range."""

    with pytest.raises(
        ValueError,
        match="Season must be between",
    ):
        validate_season(season)


def test_validate_season_rejects_non_integer() -> None:
    """Reject a season value that is not an integer."""

    with pytest.raises(
        TypeError,
        match="Season must be an integer",
    ):
        validate_season("2025")  # type: ignore[arg-type]


def test_get_season_file_returns_expected_parquet_path() -> None:
    """Build the expected season-level Parquet filename."""

    season_file = get_season_file(2025)

    assert isinstance(season_file, Path)
    assert season_file.name == "pbp_2025.parquet"
    assert season_file.parent.name == "pbp"
    assert season_file.suffix == ".parquet"


def test_download_season_pbp_saves_parquet_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Download valid PBP data and save it as Parquet."""

    play_by_play = pl.DataFrame(
        {
            "game_id": ["2025_01_DAL_PHI"],
            "play_id": [1],
            "season": [2025],
            "epa": [0.25],
        }
    )

    monkeypatch.setattr(
        pbp_ingestion,
        "PBP_DATA_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        pbp_ingestion.nfl,
        "load_pbp",
        lambda season: play_by_play,
    )

    season_file = download_season_pbp(2025)

    saved_data = pl.read_parquet(season_file)

    assert season_file == tmp_path / "pbp_2025.parquet"
    assert season_file.is_file()
    assert saved_data.height == 1
    assert saved_data.columns == play_by_play.columns


def test_download_season_pbp_skips_existing_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Do not download a season whose file already exists."""

    season_file = tmp_path / "pbp_2025.parquet"
    season_file.write_bytes(b"existing data")

    def fail_if_called(
        season: int,
    ) -> pl.DataFrame:
        raise AssertionError(
            f"Unexpected download for season {season}."
        )

    monkeypatch.setattr(
        pbp_ingestion,
        "PBP_DATA_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        pbp_ingestion.nfl,
        "load_pbp",
        fail_if_called,
    )

    returned_file = download_season_pbp(2025)

    assert returned_file == season_file
    assert season_file.read_bytes() == b"existing data"


def test_download_season_pbp_rejects_missing_columns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject downloaded PBP data without required identifiers."""

    invalid_data = pl.DataFrame(
        {
            "season": [2025],
            "epa": [0.25],
        }
    )

    monkeypatch.setattr(
        pbp_ingestion,
        "PBP_DATA_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        pbp_ingestion.nfl,
        "load_pbp",
        lambda season: invalid_data,
    )

    with pytest.raises(
        RuntimeError,
        match="missing columns",
    ):
        download_season_pbp(2025)

    assert not (
        tmp_path / "pbp_2025.parquet"
    ).exists()


def test_build_season_range_is_inclusive() -> None:
    """Include both the first and last requested seasons."""

    seasons = build_season_range(
        start_season=2023,
        end_season=2025,
    )

    assert seasons == [
        2023,
        2024,
        2025,
    ]


def test_build_season_range_accepts_single_season() -> None:
    """Build a range containing one requested season."""

    seasons = build_season_range(
        start_season=2025,
        end_season=2025,
    )

    assert seasons == [2025]


def test_build_season_range_rejects_reversed_period() -> None:
    """Reject a start season later than the end season."""

    with pytest.raises(
        ValueError,
        match="Start season must not be later",
    ):
        build_season_range(
            start_season=2025,
            end_season=2023,
        )


def test_download_play_by_play_processes_each_season(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Download every season in the requested inclusive range."""

    calls = []

    def fake_download_season_pbp(
        season: int,
        overwrite: bool = False,
    ) -> Path:
        calls.append(
            (
                season,
                overwrite,
            )
        )
        return tmp_path / f"pbp_{season}.parquet"

    monkeypatch.setattr(
        pbp_ingestion,
        "download_season_pbp",
        fake_download_season_pbp,
    )

    season_files = download_play_by_play(
        start_season=2023,
        end_season=2025,
        overwrite=True,
    )

    assert calls == [
        (2023, True),
        (2024, True),
        (2025, True),
    ]
    assert season_files == [
        tmp_path / "pbp_2023.parquet",
        tmp_path / "pbp_2024.parquet",
        tmp_path / "pbp_2025.parquet",
    ]


def test_parse_arguments_uses_safe_defaults() -> None:
    """Default to the latest completed season without overwrite."""

    arguments = parse_arguments([])

    assert (
        arguments.start_season
        == LAST_COMPLETED_SEASON
    )
    assert arguments.end_season is None
    assert arguments.overwrite is False


def test_parse_arguments_reads_requested_period() -> None:
    """Parse an explicit season range and overwrite flag."""

    arguments = parse_arguments(
        [
            "--start-season",
            "2023",
            "--end-season",
            "2025",
            "--overwrite",
        ]
    )

    assert arguments.start_season == 2023
    assert arguments.end_season == 2025
    assert arguments.overwrite is True