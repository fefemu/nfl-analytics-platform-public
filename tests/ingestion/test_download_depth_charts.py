"""Tests for legacy and ESPN depth-chart ingestion."""

from pathlib import Path

import polars as pl
import pytest

import src.ingestion.download_depth_charts as depth_ingestion

from src.ingestion.download_depth_charts import (
    CURRENT_SEASON,
    ESPN_GENERATION,
    FIRST_ESPN_SEASON,
    FIRST_MODELING_SEASON,
    LAST_LEGACY_SEASON,
    LEGACY_GENERATION,
    build_season_range,
    count_null_key_rows,
    download_depth_charts,
    download_season_depth_charts,
    get_season_file,
    get_source_generation,
    parse_arguments,
    validate_depth_chart_data,
    validate_espn_data,
    validate_legacy_data,
    validate_season,
)


def create_legacy_frame(
    season: int = 2024,
) -> pl.DataFrame:
    """Create a valid weekly legacy depth-chart frame."""

    return pl.DataFrame(
        {
            "season": [
                season,
                season,
            ],
            "club_code": [
                "NE",
                "NE",
            ],
            "week": [
                1,
                1,
            ],
            "game_type": [
                "REG",
                "REG",
            ],
            "depth_team": [
                "1",
                "2",
            ],
            "last_name": [
                "Starter",
                "Backup",
            ],
            "first_name": [
                "Test",
                "Test",
            ],
            "football_name": [
                "Test Starter",
                "Test Backup",
            ],
            "formation": [
                "Offense",
                "Offense",
            ],
            "gsis_id": [
                "00-0000001",
                "00-0000002",
            ],
            "jersey_number": [
                "10",
                "11",
            ],
            "position": [
                "QB",
                "QB",
            ],
            "elias_id": [
                "STA000001",
                "BAC000002",
            ],
            "depth_position": [
                "QB",
                "QB",
            ],
            "full_name": [
                "Test Starter",
                "Test Backup",
            ],
        },
        schema_overrides={
            "season": pl.Int32,
            "week": pl.Int32,
        },
    )


def create_espn_frame() -> pl.DataFrame:
    """Create a valid timestamped ESPN depth-chart frame."""

    return pl.DataFrame(
        {
            "dt": [
                "2026-08-03T10:36:38Z",
                "2026-08-03T10:36:38Z",
            ],
            "team": [
                "NE",
                "NE",
            ],
            "player_name": [
                "Test Starter",
                "Test Backup",
            ],
            "espn_id": [
                "1000001",
                "1000002",
            ],
            "gsis_id": [
                "00-0000001",
                None,
            ],
            "pos_grp_id": [
                "1",
                "1",
            ],
            "pos_grp": [
                "3WR 1TE",
                "3WR 1TE",
            ],
            "pos_id": [
                "1",
                "1",
            ],
            "pos_name": [
                "Quarterback",
                "Quarterback",
            ],
            "pos_abb": [
                "QB",
                "QB",
            ],
            "pos_slot": [
                1,
                1,
            ],
            "pos_rank": [
                1,
                2,
            ],
        },
        schema_overrides={
            "pos_slot": pl.Int32,
            "pos_rank": pl.Int32,
        },
    )


def test_validate_season_accepts_boundaries() -> None:
    """Accept the configured depth-chart boundaries."""

    validate_season(
        FIRST_MODELING_SEASON
    )
    validate_season(
        CURRENT_SEASON
    )


@pytest.mark.parametrize(
    "season",
    [
        FIRST_MODELING_SEASON - 1,
        CURRENT_SEASON + 1,
    ],
)
def test_validate_season_rejects_outside_range(
    season: int,
) -> None:
    """Reject unsupported depth-chart seasons."""

    with pytest.raises(
        ValueError,
        match="Season must be between",
    ):
        validate_season(
            season
        )


def test_validate_season_rejects_non_integer() -> None:
    """Reject non-integer season values."""

    with pytest.raises(
        TypeError,
        match="Season must be an integer",
    ):
        validate_season(
            "2025"  # type: ignore[arg-type]
        )


def test_get_source_generation_uses_legacy_boundary() -> None:
    """Use the legacy source through the 2024 season."""

    assert (
        get_source_generation(
            LAST_LEGACY_SEASON
        )
        == LEGACY_GENERATION
    )


def test_get_source_generation_uses_espn_boundary() -> None:
    """Use ESPN beginning with the 2025 season."""

    assert (
        get_source_generation(
            FIRST_ESPN_SEASON
        )
        == ESPN_GENERATION
    )


def test_build_season_range_is_inclusive() -> None:
    """Include both requested season boundaries."""

    assert build_season_range(
        start_season=2024,
        end_season=2026,
    ) == [
        2024,
        2025,
        2026,
    ]


def test_build_season_range_rejects_reversed_range() -> None:
    """Reject a reversed season range."""

    with pytest.raises(
        ValueError,
        match="Start season must not be later",
    ):
        build_season_range(
            start_season=2026,
            end_season=2025,
        )


def test_get_season_file_routes_legacy_source() -> None:
    """Route a legacy season to its own raw directory."""

    season_file = get_season_file(
        2024
    )

    assert (
        season_file.name
        == "depth_charts_2024.parquet"
    )
    assert season_file.parent.name == "legacy"


def test_get_season_file_routes_espn_source() -> None:
    """Route an ESPN season to its own raw directory."""

    season_file = get_season_file(
        2025
    )

    assert (
        season_file.name
        == "depth_charts_2025.parquet"
    )
    assert season_file.parent.name == "espn"


def test_count_null_key_rows_counts_any_null() -> None:
    """Count rows with any null business-key field."""

    data = pl.DataFrame(
        {
            "team": [
                "NE",
                None,
                "BUF",
            ],
            "player_id": [
                "1",
                "2",
                None,
            ],
        }
    )

    result = count_null_key_rows(
        depth_chart_data=data,
        key_columns=(
            "team",
            "player_id",
        ),
    )

    assert result == 2


def test_validate_legacy_data_accepts_valid_frame() -> None:
    """Accept a valid weekly legacy source."""

    validate_legacy_data(
        depth_chart_data=create_legacy_frame(),
        season=2024,
    )


def test_validate_legacy_data_accepts_exact_duplicates() -> None:
    """Preserve exact legacy source duplicates in raw data."""

    legacy_data = pl.concat(
        [
            create_legacy_frame(),
            create_legacy_frame().head(1),
        ]
    )

    validate_legacy_data(
        depth_chart_data=legacy_data,
        season=2024,
    )


def test_validate_legacy_data_rejects_wrong_season() -> None:
    """Reject legacy records from another season."""

    legacy_data = (
        create_legacy_frame()
        .with_columns(
            pl.when(
                pl.col("gsis_id")
                == "00-0000002"
            )
            .then(
                pl.lit(2023)
            )
            .otherwise(
                pl.col("season")
            )
            .alias("season")
        )
    )

    with pytest.raises(
        RuntimeError,
        match="outside season 2024",
    ):
        validate_legacy_data(
            depth_chart_data=legacy_data,
            season=2024,
        )


def test_validate_legacy_data_rejects_null_key() -> None:
    """Reject null legacy business-key fields."""

    legacy_data = (
        create_legacy_frame()
        .with_columns(
            pl.when(
                pl.col("gsis_id")
                == "00-0000002"
            )
            .then(
                pl.lit(
                    None,
                    dtype=pl.String,
                )
            )
            .otherwise(
                pl.col("gsis_id")
            )
            .alias("gsis_id")
        )
    )

    with pytest.raises(
        RuntimeError,
        match="null business-key rows",
    ):
        validate_legacy_data(
            depth_chart_data=legacy_data,
            season=2024,
        )


def test_validate_legacy_data_accepts_super_bowl_bye(
) -> None:
    """Allow a null week for Super Bowl bye snapshots."""

    legacy_data = (
        create_legacy_frame()
        .head(1)
        .with_columns(
            pl.lit(
                None,
                dtype=pl.Int32,
            ).alias("week"),
            pl.lit("SBBYE").alias(
                "game_type"
            ),
        )
    )

    validate_legacy_data(
        depth_chart_data=legacy_data,
        season=2024,
    )


def test_validate_legacy_data_rejects_other_null_week(
) -> None:
    """Reject a null week outside a Super Bowl bye."""

    legacy_data = (
        create_legacy_frame()
        .head(1)
        .with_columns(
            pl.lit(
                None,
                dtype=pl.Int32,
            ).alias("week")
        )
    )

    with pytest.raises(
        RuntimeError,
        match="unexpected null weeks",
    ):
        validate_legacy_data(
            depth_chart_data=legacy_data,
            season=2024,
        )


def test_validate_legacy_data_rejects_invalid_rank() -> None:
    """Reject an unsupported legacy depth rank."""

    legacy_data = (
        create_legacy_frame()
        .with_columns(
            pl.lit("4").alias(
                "depth_team"
            )
        )
    )

    with pytest.raises(
        RuntimeError,
        match="invalid depth-team values",
    ):
        validate_legacy_data(
            depth_chart_data=legacy_data,
            season=2024,
        )


def test_validate_espn_data_accepts_valid_frame() -> None:
    """Accept a valid timestamped ESPN source."""

    validate_espn_data(
        depth_chart_data=create_espn_frame(),
        season=2026,
    )


def test_validate_espn_data_accepts_missing_gsis() -> None:
    """Allow ESPN records without a GSIS identifier."""

    espn_data = create_espn_frame()

    assert (
        espn_data
        .get_column("gsis_id")
        .null_count()
        == 1
    )

    validate_espn_data(
        depth_chart_data=espn_data,
        season=2026,
    )


def test_validate_espn_data_rejects_duplicate_key() -> None:
    """Reject duplicate ESPN business-key records."""

    espn_data = pl.concat(
        [
            create_espn_frame(),
            create_espn_frame().head(1),
        ]
    )

    with pytest.raises(
        RuntimeError,
        match="duplicate business-key groups",
    ):
        validate_espn_data(
            depth_chart_data=espn_data,
            season=2026,
        )


def test_validate_espn_data_rejects_invalid_timestamp() -> None:
    """Reject an invalid ESPN snapshot timestamp."""

    espn_data = (
        create_espn_frame()
        .with_columns(
            pl.lit(
                "not-a-timestamp"
            ).alias("dt")
        )
    )

    with pytest.raises(
        RuntimeError,
        match="invalid timestamps",
    ):
        validate_espn_data(
            depth_chart_data=espn_data,
            season=2026,
        )


def test_validate_espn_data_rejects_invalid_rank() -> None:
    """Reject non-positive ESPN position ranks."""

    espn_data = (
        create_espn_frame()
        .with_columns(
            pl.lit(0).alias(
                "pos_rank"
            )
        )
    )

    with pytest.raises(
        RuntimeError,
        match="invalid position ranks",
    ):
        validate_espn_data(
            depth_chart_data=espn_data,
            season=2026,
        )


def test_validate_depth_chart_data_routes_generations() -> None:
    """Route validation based on the requested season."""

    validate_depth_chart_data(
        depth_chart_data=create_legacy_frame(),
        season=2024,
    )
    validate_depth_chart_data(
        depth_chart_data=create_espn_frame(),
        season=2025,
    )


def test_download_legacy_season_saves_parquet(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Download and save one legacy season."""

    legacy_directory = tmp_path / "legacy"

    monkeypatch.setattr(
        depth_ingestion,
        "LEGACY_DATA_DIR",
        legacy_directory,
    )
    monkeypatch.setattr(
        depth_ingestion.nfl,
        "load_depth_charts",
        lambda season: create_legacy_frame(
            season
        ),
    )

    season_file = (
        download_season_depth_charts(
            2024
        )
    )

    saved_data = pl.read_parquet(
        season_file
    )

    assert (
        season_file
        == legacy_directory
        / "depth_charts_2024.parquet"
    )
    assert saved_data.equals(
        create_legacy_frame()
    )


def test_download_espn_season_saves_parquet(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Download and save one ESPN season."""

    espn_directory = tmp_path / "espn"

    monkeypatch.setattr(
        depth_ingestion,
        "ESPN_DATA_DIR",
        espn_directory,
    )
    monkeypatch.setattr(
        depth_ingestion.nfl,
        "load_depth_charts",
        lambda season: create_espn_frame(),
    )

    season_file = (
        download_season_depth_charts(
            2026
        )
    )

    saved_data = pl.read_parquet(
        season_file
    )

    assert (
        season_file
        == espn_directory
        / "depth_charts_2026.parquet"
    )
    assert saved_data.equals(
        create_espn_frame()
    )


def test_download_season_skips_existing_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep an existing season file without overwrite."""

    espn_directory = tmp_path / "espn"
    espn_directory.mkdir()

    season_file = (
        espn_directory
        / "depth_charts_2026.parquet"
    )
    season_file.write_bytes(
        b"existing depth chart"
    )

    def fail_if_called(
        season: int,
    ) -> pl.DataFrame:
        raise AssertionError(
            f"Unexpected download for season {season}."
        )

    monkeypatch.setattr(
        depth_ingestion,
        "ESPN_DATA_DIR",
        espn_directory,
    )
    monkeypatch.setattr(
        depth_ingestion.nfl,
        "load_depth_charts",
        fail_if_called,
    )

    returned_file = (
        download_season_depth_charts(
            2026
        )
    )

    assert returned_file == season_file
    assert (
        season_file.read_bytes()
        == b"existing depth chart"
    )


def test_download_depth_charts_processes_range(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Process every requested depth-chart season."""

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
            / f"depth_charts_{season}.parquet"
        )

    monkeypatch.setattr(
        depth_ingestion,
        "download_season_depth_charts",
        fake_download,
    )

    season_files = download_depth_charts(
        start_season=2024,
        end_season=2026,
        overwrite=True,
    )

    assert calls == [
        (
            2024,
            True,
        ),
        (
            2025,
            True,
        ),
        (
            2026,
            True,
        ),
    ]
    assert len(season_files) == 3


def test_parse_arguments_uses_defaults() -> None:
    """Use the complete configured depth-chart range."""

    arguments = parse_arguments([])

    assert (
        arguments.start_season
        == FIRST_MODELING_SEASON
    )
    assert (
        arguments.end_season
        == CURRENT_SEASON
    )
    assert arguments.overwrite is False


def test_parse_arguments_accepts_overrides() -> None:
    """Accept a custom range and overwrite mode."""

    arguments = parse_arguments(
        [
            "--start-season",
            "2025",
            "--end-season",
            "2026",
            "--overwrite",
        ]
    )

    assert arguments.start_season == 2025
    assert arguments.end_season == 2026
    assert arguments.overwrite is True