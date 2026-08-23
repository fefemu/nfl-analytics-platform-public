"""Tests for historical player snap-count ingestion."""

from pathlib import Path

import polars as pl
import pytest

import src.ingestion.download_snap_counts as snap_ingestion

from src.ingestion.download_snap_counts import (
    CANONICAL_SNAP_COUNT_COLUMNS,
    CANONICAL_SNAP_COUNT_SCHEMA,
    FIRST_MODELING_SEASON,
    LAST_COMPLETED_SEASON,
    build_season_range,
    count_duplicate_player_games,
    count_null_key_rows,
    download_season_snap_counts,
    download_snap_counts,
    get_season_file,
    normalize_snap_count_schema,
    parse_arguments,
    validate_season,
    validate_snap_count_data,
)


def create_snap_count_frame(
    season: int = 2025,
) -> pl.DataFrame:
    """Create a valid player snap-count frame."""

    return pl.DataFrame(
        {
            "game_id": [
                f"{season}_01_NE_BUF",
                f"{season}_01_NE_BUF",
            ],
            "pfr_game_id": [
                f"{season}09070buf",
                f"{season}09070buf",
            ],
            "season": [
                season,
                season,
            ],
            "game_type": [
                "REG",
                "REG",
            ],
            "week": [
                1,
                1,
            ],
            "player": [
                "Test Quarterback",
                "Test Receiver",
            ],
            "pfr_player_id": [
                "TestQu00",
                "TestRe00",
            ],
            "position": [
                "QB",
                "WR",
            ],
            "team": [
                "NE",
                "NE",
            ],
            "opponent": [
                "BUF",
                "BUF",
            ],
            "offense_snaps": [
                65.0,
                52.0,
            ],
            "offense_pct": [
                1.0,
                0.8,
            ],
            "defense_snaps": [
                0.0,
                0.0,
            ],
            "defense_pct": [
                0.0,
                0.0,
            ],
            "st_snaps": [
                0.0,
                3.0,
            ],
            "st_pct": [
                0.0,
                0.12,
            ],
        }
    )


def test_validate_season_accepts_boundaries() -> None:
    """Accept the configured snap-count season boundaries."""

    validate_season(
        FIRST_MODELING_SEASON
    )
    validate_season(
        LAST_COMPLETED_SEASON
    )


@pytest.mark.parametrize(
    "season",
    [
        FIRST_MODELING_SEASON - 1,
        LAST_COMPLETED_SEASON + 1,
    ],
)
def test_validate_season_rejects_outside_range(
    season: int,
) -> None:
    """Reject unsupported snap-count seasons."""

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


def test_build_season_range_is_inclusive() -> None:
    """Include both requested season boundaries."""

    assert build_season_range(
        start_season=2023,
        end_season=2025,
    ) == [
        2023,
        2024,
        2025,
    ]


def test_build_season_range_rejects_reverse_range() -> None:
    """Reject a start season later than the end season."""

    with pytest.raises(
        ValueError,
        match="must not be later",
    ):
        build_season_range(
            start_season=2025,
            end_season=2024,
        )


def test_get_season_file_uses_expected_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build the canonical season-level Parquet path."""

    monkeypatch.setattr(
        snap_ingestion,
        "SNAP_COUNT_DATA_DIR",
        tmp_path,
    )

    assert get_season_file(
        2025
    ) == (
        tmp_path
        / "snap_counts_2025.parquet"
    )


def test_normalize_snap_count_schema_selects_canonical_columns(
) -> None:
    """Select canonical columns and discard extras."""

    snap_data = create_snap_count_frame().with_columns(
        pl.lit(
            "unused"
        ).alias(
            "extra_column"
        )
    )

    normalized_data = normalize_snap_count_schema(
        snap_data
    )

    assert normalized_data.columns == list(
        CANONICAL_SNAP_COUNT_COLUMNS
    )
    assert (
        "extra_column"
        not in normalized_data.columns
    )


def test_normalize_snap_count_schema_casts_types() -> None:
    """Cast every source field to the canonical type."""

    normalized_data = normalize_snap_count_schema(
        create_snap_count_frame()
    )

    assert normalized_data.schema == (
        CANONICAL_SNAP_COUNT_SCHEMA
    )


def test_normalize_snap_count_schema_rejects_missing_column(
) -> None:
    """Reject a source frame missing a required column."""

    snap_data = create_snap_count_frame().drop(
        "offense_pct"
    )

    with pytest.raises(
        RuntimeError,
        match="missing columns: offense_pct",
    ):
        normalize_snap_count_schema(
            snap_data
        )


def test_count_null_key_rows_counts_any_missing_key() -> None:
    """Count records missing any business-key field."""

    snap_data = create_snap_count_frame().with_columns(
        pl.when(
            pl.col("pfr_player_id") == "TestRe00"
        )
        .then(
            pl.lit(None)
        )
        .otherwise(
            pl.col("pfr_player_id")
        )
        .alias("pfr_player_id")
    )

    assert count_null_key_rows(
        snap_data
    ) == 1


def test_count_duplicate_player_games_counts_key_groups() -> None:
    """Count duplicated player-team-game key groups."""

    snap_data = pl.concat(
        [
            create_snap_count_frame(),
            create_snap_count_frame().head(1),
        ]
    )

    assert count_duplicate_player_games(
        snap_data
    ) == 1


def test_validate_snap_count_data_accepts_valid_frame() -> None:
    """Accept a valid canonical snap-count dataset."""

    snap_data = normalize_snap_count_schema(
        create_snap_count_frame()
    )

    validate_snap_count_data(
        snap_count_data=snap_data,
        season=2025,
    )


def test_validate_snap_count_data_accepts_source_rounding(
) -> None:
    """Accept the documented 1.01 source rounding edge case."""

    snap_data = (
        create_snap_count_frame()
        .with_columns(
            pl.when(
                pl.col("pfr_player_id") == "TestRe00"
            )
            .then(
                pl.lit(1.01)
            )
            .otherwise(
                pl.col("st_pct")
            )
            .alias("st_pct")
        )
    )

    snap_data = normalize_snap_count_schema(
        snap_data
    )

    validate_snap_count_data(
        snap_count_data=snap_data,
        season=2025,
    )


def test_validate_snap_count_data_rejects_empty_frame() -> None:
    """Reject an empty season dataset."""

    snap_data = normalize_snap_count_schema(
        create_snap_count_frame()
    ).clear()

    with pytest.raises(
        ValueError,
        match="is empty",
    ):
        validate_snap_count_data(
            snap_count_data=snap_data,
            season=2025,
        )


def test_validate_snap_count_data_rejects_wrong_season() -> None:
    """Reject records outside the requested season."""

    snap_data = normalize_snap_count_schema(
        create_snap_count_frame(
            season=2024
        )
    )

    with pytest.raises(
        RuntimeError,
        match="outside season 2025",
    ):
        validate_snap_count_data(
            snap_count_data=snap_data,
            season=2025,
        )


def test_validate_snap_count_data_rejects_null_key() -> None:
    """Reject a null player-team-game key."""

    snap_data = (
        create_snap_count_frame()
        .with_columns(
            pl.when(
                pl.col("pfr_player_id") == "TestRe00"
            )
            .then(
                pl.lit(None)
            )
            .otherwise(
                pl.col("pfr_player_id")
            )
            .alias("pfr_player_id")
        )
    )

    snap_data = normalize_snap_count_schema(
        snap_data
    )

    with pytest.raises(
        RuntimeError,
        match="null business keys",
    ):
        validate_snap_count_data(
            snap_count_data=snap_data,
            season=2025,
        )


def test_validate_snap_count_data_rejects_duplicate_key() -> None:
    """Reject duplicate player-team-game records."""

    snap_data = pl.concat(
        [
            create_snap_count_frame(),
            create_snap_count_frame().head(1),
        ]
    )

    snap_data = normalize_snap_count_schema(
        snap_data
    )

    with pytest.raises(
        RuntimeError,
        match="duplicate player-team-game keys",
    ):
        validate_snap_count_data(
            snap_count_data=snap_data,
            season=2025,
        )


def test_validate_snap_count_data_rejects_negative_snaps() -> None:
    """Reject negative actual snap counts."""

    snap_data = (
        create_snap_count_frame()
        .with_columns(
            pl.when(
                pl.col("pfr_player_id") == "TestRe00"
            )
            .then(
                pl.lit(-1.0)
            )
            .otherwise(
                pl.col("offense_snaps")
            )
            .alias("offense_snaps")
        )
    )

    snap_data = normalize_snap_count_schema(
        snap_data
    )

    with pytest.raises(
        RuntimeError,
        match="negative or non-finite",
    ):
        validate_snap_count_data(
            snap_count_data=snap_data,
            season=2025,
        )


@pytest.mark.parametrize(
    "invalid_share",
    [
        -0.01,
        1.02,
        float("nan"),
    ],
)
def test_validate_snap_count_data_rejects_invalid_share(
    invalid_share: float,
) -> None:
    """Reject invalid snap-share values."""

    snap_data = (
        create_snap_count_frame()
        .with_columns(
            pl.when(
                pl.col("pfr_player_id") == "TestRe00"
            )
            .then(
                pl.lit(invalid_share)
            )
            .otherwise(
                pl.col("offense_pct")
            )
            .alias("offense_pct")
        )
    )

    snap_data = normalize_snap_count_schema(
        snap_data
    )

    with pytest.raises(
        RuntimeError,
        match="outside the accepted source range",
    ):
        validate_snap_count_data(
            snap_count_data=snap_data,
            season=2025,
        )


def test_validate_snap_count_data_rejects_inconsistent_opponent(
) -> None:
    """Reject one team-game containing multiple opponents."""

    snap_data = (
        create_snap_count_frame()
        .with_columns(
            pl.when(
                pl.col("pfr_player_id") == "TestRe00"
            )
            .then(
                pl.lit("MIA")
            )
            .otherwise(
                pl.col("opponent")
            )
            .alias("opponent")
        )
    )

    snap_data = normalize_snap_count_schema(
        snap_data
    )

    with pytest.raises(
        RuntimeError,
        match="inconsistent opponents",
    ):
        validate_snap_count_data(
            snap_count_data=snap_data,
            season=2025,
        )


def test_download_season_snap_counts_saves_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Download, validate and save one season."""

    monkeypatch.setattr(
        snap_ingestion,
        "SNAP_COUNT_DATA_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        snap_ingestion.nfl,
        "load_snap_counts",
        lambda season: create_snap_count_frame(
            season=season
        ),
    )

    output_file = download_season_snap_counts(
        season=2025,
    )

    assert output_file.is_file()

    saved_data = pl.read_parquet(
        output_file
    )

    assert saved_data.shape == (
        2,
        16,
    )
    assert saved_data.schema == (
        CANONICAL_SNAP_COUNT_SCHEMA
    )


def test_download_season_snap_counts_skips_existing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip an existing file unless overwrite is requested."""

    monkeypatch.setattr(
        snap_ingestion,
        "SNAP_COUNT_DATA_DIR",
        tmp_path,
    )

    existing_file = (
        tmp_path
        / "snap_counts_2025.parquet"
    )
    existing_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    existing_file.write_bytes(
        b"existing"
    )

    def fail_if_called(
        season: int,
    ) -> pl.DataFrame:
        raise AssertionError(
            f"Unexpected download for season {season}."
        )

    monkeypatch.setattr(
        snap_ingestion.nfl,
        "load_snap_counts",
        fail_if_called,
    )

    output_file = download_season_snap_counts(
        season=2025,
        overwrite=False,
    )

    assert output_file == existing_file
    assert output_file.read_bytes() == b"existing"


def test_download_season_snap_counts_overwrites_existing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replace an existing season file when requested."""

    monkeypatch.setattr(
        snap_ingestion,
        "SNAP_COUNT_DATA_DIR",
        tmp_path,
    )

    existing_file = (
        tmp_path
        / "snap_counts_2025.parquet"
    )
    existing_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    existing_file.write_bytes(
        b"existing"
    )

    monkeypatch.setattr(
        snap_ingestion.nfl,
        "load_snap_counts",
        lambda season: create_snap_count_frame(
            season=season
        ),
    )

    output_file = download_season_snap_counts(
        season=2025,
        overwrite=True,
    )

    saved_data = pl.read_parquet(
        output_file
    )

    assert saved_data.height == 2


def test_download_snap_counts_processes_inclusive_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Process every requested season in order."""

    downloaded_seasons: list[int] = []

    def fake_download(
        season: int,
        overwrite: bool = False,
    ) -> Path:
        downloaded_seasons.append(
            season
        )
        return (
            tmp_path
            / f"snap_counts_{season}.parquet"
        )

    monkeypatch.setattr(
        snap_ingestion,
        "download_season_snap_counts",
        fake_download,
    )

    files = download_snap_counts(
        start_season=2023,
        end_season=2025,
        overwrite=True,
    )

    assert downloaded_seasons == [
        2023,
        2024,
        2025,
    ]
    assert len(files) == 3


def test_parse_arguments_uses_defaults() -> None:
    """Use project modeling boundaries by default."""

    arguments = parse_arguments(
        []
    )

    assert (
        arguments.start_season
        == FIRST_MODELING_SEASON
    )
    assert (
        arguments.end_season
        == LAST_COMPLETED_SEASON
    )
    assert arguments.overwrite is False


def test_parse_arguments_accepts_overrides() -> None:
    """Accept explicit CLI season and overwrite options."""

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