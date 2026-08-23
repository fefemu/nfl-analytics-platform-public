"""Tests for historical injury-report ingestion."""

from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

import src.ingestion.download_injury_reports as injury_ingestion

from src.ingestion.download_injury_reports import (
    CANONICAL_INJURY_SCHEMA,
    FIRST_AVAILABLE_SEASON,
    LAST_COMPLETED_SEASON,
    build_season_range,
    download_injury_reports,
    download_season_injuries,
    get_season_file,
    parse_arguments,
    normalize_injury_schema,
    validate_injury_data,
    validate_season,
)


def create_injury_frame(
    season: int = 2025,
) -> pl.DataFrame:
    """Create a valid injury-report test frame."""

    return pl.DataFrame(
        {
            "season": [
                season,
                season,
            ],
            "season_type": [
                "REG",
                "REG",
            ],
            "game_type": [
                "REG",
                "REG",
            ],
            "team": [
                "NE",
                "NE",
            ],
            "week": [
                1,
                1,
            ],
            "gsis_id": [
                "00-0031234",
                "00-0035678",
            ],
            "position": [
                "QB",
                "WR",
            ],
            "full_name": [
                "Test Quarterback",
                "Test Receiver",
            ],
            "first_name": [
                "Test",
                "Test",
            ],
            "last_name": [
                "Quarterback",
                "Receiver",
            ],
            "report_primary_injury": [
                "Shoulder",
                None,
            ],
            "report_secondary_injury": [
                None,
                None,
            ],
            "report_status": [
                "Questionable",
                None,
            ],
            "practice_primary_injury": [
                "Shoulder",
                "Hamstring",
            ],
            "practice_secondary_injury": [
                None,
                None,
            ],
            "practice_status": [
                "Limited Participation in Practice",
                "Full Participation in Practice",
            ],
            "date_modified": [
                datetime(
                    2025,
                    9,
                    5,
                    12,
                    0,
                    tzinfo=timezone.utc,
                ),
                datetime(
                    2025,
                    9,
                    5,
                    12,
                    5,
                    tzinfo=timezone.utc,
                ),
            ],
        }
    )


def test_validate_season_accepts_available_boundaries() -> None:
    """Accept the configured injury-season boundaries."""

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
    """Reject seasons outside the configured range."""

    with pytest.raises(
        ValueError,
        match="Season must be between",
    ):
        validate_season(season)


def test_validate_season_rejects_non_integer() -> None:
    """Reject non-integer season values."""

    with pytest.raises(
        TypeError,
        match="Season must be an integer",
    ):
        validate_season("2025")  # type: ignore[arg-type]


def test_build_season_range_is_inclusive() -> None:
    """Include both requested season boundaries."""

    seasons = build_season_range(
        start_season=2023,
        end_season=2025,
    )

    assert seasons == [
        2023,
        2024,
        2025,
    ]


def test_build_season_range_rejects_reversed_range() -> None:
    """Reject a start season later than the end season."""

    with pytest.raises(
        ValueError,
        match="Start season must not be later",
    ):
        build_season_range(
            start_season=2025,
            end_season=2023,
        )


def test_get_season_file_returns_expected_path() -> None:
    """Build the expected season-level Parquet path."""

    season_file = get_season_file(2025)

    assert isinstance(
        season_file,
        Path,
    )
    assert (
        season_file.name
        == "injury_reports_2025.parquet"
    )
    assert season_file.parent.name == "injuries"
    assert season_file.suffix == ".parquet"


def test_normalize_injury_schema_casts_canonical_types() -> None:
    """Cast legacy numeric columns to canonical integer types."""

    legacy_data = (
        create_injury_frame()
        .with_columns(
            pl.col("season").cast(
                pl.Float64
            ),
            pl.col("week").cast(
                pl.Float64
            ),
        )
    )

    normalized_data = normalize_injury_schema(
        legacy_data
    )

    assert (
        normalized_data.schema
        == pl.Schema(
            CANONICAL_INJURY_SCHEMA
        )
    )


def test_normalize_injury_schema_adds_season_type() -> None:
    """Normalize the 2018-2024 nflverse injury schema."""

    historical_data = create_injury_frame().drop(
        "season_type"
    )

    normalized_data = normalize_injury_schema(
        historical_data
    )

    assert "season_type" in normalized_data.columns
    assert (
        normalized_data
        .get_column("season_type")
        .unique()
        .to_list()
        == ["REG"]
    )
    assert (
        normalized_data.schema["date_modified"]
        == pl.Datetime(
            time_unit="us",
            time_zone="UTC",
        )
    )


def test_normalize_injury_schema_adds_nullable_timestamp() -> None:
    """Normalize the 2025 nflverse injury schema."""

    current_schema_data = create_injury_frame().drop(
        "date_modified"
    )

    normalized_data = normalize_injury_schema(
        current_schema_data
    )

    assert "date_modified" in normalized_data.columns
    assert (
        normalized_data
        .get_column("date_modified")
        .null_count()
        == normalized_data.height
    )
    assert (
        normalized_data.schema["date_modified"]
        == pl.Datetime(
            time_unit="us",
            time_zone="UTC",
        )
    )


def test_validate_injury_data_accepts_valid_frame() -> None:
    """Accept valid player-team-week injury records."""

    injury_data = create_injury_frame()

    validate_injury_data(
        injury_data=injury_data,
        season=2025,
    )


def test_validate_injury_data_rejects_empty_frame() -> None:
    """Reject an empty injury dataset."""

    with pytest.raises(
        ValueError,
        match="is empty",
    ):
        validate_injury_data(
            injury_data=pl.DataFrame(),
            season=2025,
        )


def test_validate_injury_data_rejects_missing_columns() -> None:
    """Reject injury data without its required schema."""

    injury_data = create_injury_frame().drop(
        "report_status"
    )

    with pytest.raises(
        RuntimeError,
        match="missing columns",
    ):
        validate_injury_data(
            injury_data=injury_data,
            season=2025,
        )


def test_validate_injury_data_rejects_unexpected_season() -> None:
    """Reject records from a different NFL season."""

    injury_data = create_injury_frame().with_columns(
        pl.when(
            pl.col("gsis_id") == "00-0035678"
        )
        .then(
            pl.lit(2024)
        )
        .otherwise(
            pl.col("season")
        )
        .alias("season")
    )

    with pytest.raises(
        RuntimeError,
        match="outside season 2025",
    ):
        validate_injury_data(
            injury_data=injury_data,
            season=2025,
        )


def test_validate_injury_data_rejects_null_key() -> None:
    """Reject null player-team-week identifiers."""

    injury_data = create_injury_frame().with_columns(
        pl.when(
            pl.col("gsis_id") == "00-0035678"
        )
        .then(
            pl.lit(None, dtype=pl.String)
        )
        .otherwise(
            pl.col("gsis_id")
        )
        .alias("gsis_id")
    )

    with pytest.raises(
        RuntimeError,
        match="null key columns: gsis_id",
    ):
        validate_injury_data(
            injury_data=injury_data,
            season=2025,
        )


def test_validate_injury_data_rejects_duplicate_snapshot() -> None:
    """Reject duplicate records with the same snapshot time."""

    injury_data = pl.concat(
        [
            create_injury_frame(),
            create_injury_frame().head(1),
        ]
    )

    with pytest.raises(
        RuntimeError,
        match="duplicate player-team-week snapshot",
    ):
        validate_injury_data(
            injury_data=injury_data,
            season=2025,
        )


def test_validate_injury_data_accepts_status_history() -> None:
    """Accept multiple timestamped snapshots for one weekly key."""

    first_snapshot = create_injury_frame().head(1)

    second_snapshot = (
        first_snapshot
        .with_columns(
            pl.lit("Out").alias(
                "report_status"
            ),
            pl.lit(
                datetime(
                    2025,
                    9,
                    6,
                    12,
                    0,
                    tzinfo=timezone.utc,
                )
            ).alias(
                "date_modified"
            ),
        )
    )

    injury_data = pl.concat(
        [
            first_snapshot,
            second_snapshot,
        ]
    )

    validate_injury_data(
        injury_data=injury_data,
        season=2025,
    )


def test_download_season_injuries_saves_parquet(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Download, validate and save one injury season."""

    injury_data = create_injury_frame()

    monkeypatch.setattr(
        injury_ingestion,
        "INJURY_DATA_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        injury_ingestion.nfl,
        "load_injuries",
        lambda season: injury_data,
    )

    season_file = download_season_injuries(2025)
    saved_data = pl.read_parquet(season_file)

    assert (
        season_file
        == tmp_path / "injury_reports_2025.parquet"
    )
    assert season_file.is_file()
    assert saved_data.equals(injury_data)


def test_download_season_injuries_skips_existing_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep an existing file unless overwrite is requested."""

    season_file = (
        tmp_path
        / "injury_reports_2025.parquet"
    )
    season_file.write_bytes(
        b"existing injury data"
    )

    def fail_if_called(
        season: int,
    ) -> pl.DataFrame:
        raise AssertionError(
            f"Unexpected download for season {season}."
        )

    monkeypatch.setattr(
        injury_ingestion,
        "INJURY_DATA_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        injury_ingestion.nfl,
        "load_injuries",
        fail_if_called,
    )

    returned_file = download_season_injuries(2025)

    assert returned_file == season_file
    assert (
        season_file.read_bytes()
        == b"existing injury data"
    )


def test_download_season_injuries_overwrites_existing_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Replace an existing file when overwrite is enabled."""

    season_file = (
        tmp_path
        / "injury_reports_2025.parquet"
    )
    season_file.write_bytes(
        b"old injury data"
    )

    injury_data = create_injury_frame()

    monkeypatch.setattr(
        injury_ingestion,
        "INJURY_DATA_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        injury_ingestion.nfl,
        "load_injuries",
        lambda season: injury_data,
    )

    returned_file = download_season_injuries(
        season=2025,
        overwrite=True,
    )

    saved_data = pl.read_parquet(
        returned_file
    )

    assert saved_data.equals(injury_data)


def test_download_injury_reports_processes_each_season(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Process every season in the requested range."""

    calls: list[tuple[int, bool]] = []

    def fake_download(
        season: int,
        overwrite: bool = False,
    ) -> Path:
        calls.append(
            (
                season,
                overwrite,
            )
        )
        return (
            tmp_path
            / f"injury_reports_{season}.parquet"
        )

    monkeypatch.setattr(
        injury_ingestion,
        "download_season_injuries",
        fake_download,
    )

    season_files = download_injury_reports(
        start_season=2023,
        end_season=2025,
        overwrite=True,
    )

    assert calls == [
        (
            2023,
            True,
        ),
        (
            2024,
            True,
        ),
        (
            2025,
            True,
        ),
    ]
    assert len(season_files) == 3


def test_parse_arguments_uses_modeling_defaults() -> None:
    """Use the configured historical modeling period."""

    arguments = parse_arguments([])

    assert (
        arguments.start_season
        == injury_ingestion.FIRST_MODELING_SEASON
    )
    assert (
        arguments.end_season
        == LAST_COMPLETED_SEASON
    )
    assert arguments.overwrite is False


def test_parse_arguments_accepts_overrides() -> None:
    """Accept custom seasons and overwrite mode."""

    arguments = parse_arguments(
        [
            "--start-season",
            "2024",
            "--end-season",
            "2025",
            "--overwrite",
        ]
    )

    assert arguments.start_season == 2024
    assert arguments.end_season == 2025
    assert arguments.overwrite is True