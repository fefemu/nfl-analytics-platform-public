"""Tests for NFL player-directory ingestion."""

from pathlib import Path

import polars as pl
import pytest

import src.ingestion.download_player_directory as player_ingestion

from src.ingestion.download_player_directory import (
    CANONICAL_PLAYER_COLUMNS,
    CANONICAL_PLAYER_SCHEMA,
    count_duplicate_non_null_ids,
    download_player_directory,
    get_player_directory_file,
    normalize_player_directory_schema,
    parse_arguments,
    validate_player_directory,
)


def create_player_directory_frame() -> pl.DataFrame:
    """Create a valid canonical player-directory frame."""

    player_data: dict[str, list[object]] = {
        column: [
            None,
            None,
        ]
        for column in CANONICAL_PLAYER_COLUMNS
    }

    player_data.update(
        {
            "gsis_id": [
                "00-0000001",
                "00-0000002",
            ],
            "display_name": [
                "Test Quarterback",
                "Test Receiver",
            ],
            "common_first_name": [
                "Test",
                "Test",
            ],
            "first_name": [
                "Test",
                "Test",
            ],
            "last_name": [
                "Quarterback",
                "Receiver",
            ],
            "short_name": [
                "T. Quarterback",
                "T. Receiver",
            ],
            "football_name": [
                "Test Quarterback",
                "Test Receiver",
            ],
            "pfr_id": [
                "TestQu00",
                "TestRe00",
            ],
            "espn_id": [
                "1000001",
                "1000002",
            ],
            "position_group": [
                "QB",
                "WR",
            ],
            "position": [
                "QB",
                "WR",
            ],
            "height": [
                75,
                72,
            ],
            "weight": [
                220,
                195,
            ],
            "rookie_season": [
                2020,
                2021,
            ],
            "last_season": [
                2025,
                2025,
            ],
            "latest_team": [
                "NE",
                "NE",
            ],
            "status": [
                "ACT",
                "ACT",
            ],
            "years_of_experience": [
                5,
                4,
            ],
            "draft_year": [
                2020,
                2021,
            ],
            "draft_round": [
                1,
                2,
            ],
            "draft_pick": [
                10,
                45,
            ],
            "draft_team": [
                "NE",
                "NE",
            ],
        }
    )

    return pl.DataFrame(
        player_data
    )


def canonical_player_frame() -> pl.DataFrame:
    """Return a typed valid player-directory frame."""

    return normalize_player_directory_schema(
        create_player_directory_frame()
    )


def test_get_player_directory_file_returns_configured_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return the configured canonical player file."""

    expected_file = (
        tmp_path
        / "player_directory.parquet"
    )

    monkeypatch.setattr(
        player_ingestion,
        "PLAYER_DIRECTORY_FILE",
        expected_file,
    )

    assert get_player_directory_file() == expected_file


def test_normalize_player_directory_schema_selects_columns(
) -> None:
    """Select canonical columns and discard extra fields."""

    player_data = create_player_directory_frame().with_columns(
        pl.lit(
            "unused"
        ).alias(
            "extra_column"
        )
    )

    normalized_data = normalize_player_directory_schema(
        player_data
    )

    assert normalized_data.columns == list(
        CANONICAL_PLAYER_COLUMNS
    )
    assert (
        "extra_column"
        not in normalized_data.columns
    )


def test_normalize_player_directory_schema_casts_types(
) -> None:
    """Cast every field to the canonical player schema."""

    normalized_data = normalize_player_directory_schema(
        create_player_directory_frame()
    )

    assert normalized_data.schema == (
        CANONICAL_PLAYER_SCHEMA
    )


def test_normalize_player_directory_schema_rejects_missing_column(
) -> None:
    """Reject a source missing a required player column."""

    player_data = create_player_directory_frame().drop(
        "pfr_id"
    )

    with pytest.raises(
        RuntimeError,
        match="missing columns: pfr_id",
    ):
        normalize_player_directory_schema(
            player_data
        )


def test_count_duplicate_non_null_ids_ignores_nulls() -> None:
    """Ignore repeated null values in identifier checks."""

    player_data = canonical_player_frame().with_columns(
        pl.lit(
            None,
            dtype=pl.String,
        ).alias(
            "espn_id"
        )
    )

    assert count_duplicate_non_null_ids(
        player_data=player_data,
        identifier_column="espn_id",
    ) == 0


def test_count_duplicate_non_null_ids_counts_groups() -> None:
    """Count duplicated non-null identifier groups."""

    player_data = canonical_player_frame().with_columns(
        pl.lit(
            "duplicate"
        ).alias(
            "espn_id"
        )
    )

    assert count_duplicate_non_null_ids(
        player_data=player_data,
        identifier_column="espn_id",
    ) == 1


def test_validate_player_directory_accepts_valid_frame() -> None:
    """Accept a valid typed player directory."""

    validate_player_directory(
        canonical_player_frame()
    )


def test_validate_player_directory_rejects_empty_frame() -> None:
    """Reject an empty player directory."""

    with pytest.raises(
        ValueError,
        match="is empty",
    ):
        validate_player_directory(
            canonical_player_frame().clear()
        )


@pytest.mark.parametrize(
    "invalid_gsis_id",
    [
        None,
        "",
        "   ",
    ],
)
def test_validate_player_directory_rejects_missing_gsis(
    invalid_gsis_id: str | None,
) -> None:
    """Reject null or blank GSIS identifiers."""

    player_data = canonical_player_frame().with_columns(
        pl.when(
            pl.col("gsis_id") == "00-0000002"
        )
        .then(
            pl.lit(invalid_gsis_id)
        )
        .otherwise(
            pl.col("gsis_id")
        )
        .alias("gsis_id")
    )

    with pytest.raises(
        RuntimeError,
        match="without a GSIS ID",
    ):
        validate_player_directory(
            player_data
        )


@pytest.mark.parametrize(
    "identifier_column",
    [
        "gsis_id",
        "pfr_id",
        "espn_id",
    ],
)
def test_validate_player_directory_rejects_duplicate_identifier(
    identifier_column: str,
) -> None:
    """Reject duplicated stable player identifiers."""

    player_data = canonical_player_frame().with_columns(
        pl.when(
            pl.col("gsis_id") == "00-0000002"
        )
        .then(
            pl.col(identifier_column).first()
        )
        .otherwise(
            pl.col(identifier_column)
        )
        .alias(identifier_column)
    )

    first_value = player_data.get_column(
        identifier_column
    )[0]

    player_data = player_data.with_columns(
        pl.lit(
            first_value
        ).alias(
            identifier_column
        )
    )

    with pytest.raises(
        RuntimeError,
        match=f"duplicate {identifier_column} groups",
    ):
        validate_player_directory(
            player_data
        )


def test_validate_player_directory_requires_pfr_coverage() -> None:
    """Require at least one PFR identifier."""

    player_data = canonical_player_frame().with_columns(
        pl.lit(
            None,
            dtype=pl.String,
        ).alias(
            "pfr_id"
        )
    )

    with pytest.raises(
        RuntimeError,
        match="no PFR identifiers",
    ):
        validate_player_directory(
            player_data
        )


@pytest.mark.parametrize(
    (
        "rookie_season",
        "last_season",
    ),
    [
        (
            1899,
            2025,
        ),
        (
            2020,
            1899,
        ),
        (
            2025,
            2024,
        ),
    ],
)
def test_validate_player_directory_rejects_invalid_career_range(
    rookie_season: int,
    last_season: int,
) -> None:
    """Reject impossible source career ranges."""

    player_data = canonical_player_frame().with_columns(
        [
            pl.when(
                pl.col("gsis_id") == "00-0000002"
            )
            .then(
                pl.lit(rookie_season)
            )
            .otherwise(
                pl.col("rookie_season")
            )
            .alias("rookie_season"),
            pl.when(
                pl.col("gsis_id") == "00-0000002"
            )
            .then(
                pl.lit(last_season)
            )
            .otherwise(
                pl.col("last_season")
            )
            .alias("last_season"),
        ]
    )

    with pytest.raises(
        RuntimeError,
        match="invalid career ranges",
    ):
        validate_player_directory(
            player_data
        )


def test_download_player_directory_saves_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Download, validate and save the player directory."""

    output_file = (
        tmp_path
        / "player_directory.parquet"
    )

    monkeypatch.setattr(
        player_ingestion,
        "PLAYER_DATA_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        player_ingestion,
        "PLAYER_DIRECTORY_FILE",
        output_file,
    )
    monkeypatch.setattr(
        player_ingestion.nfl,
        "load_players",
        create_player_directory_frame,
    )

    result_file = download_player_directory()

    assert result_file == output_file
    assert result_file.is_file()

    saved_data = pl.read_parquet(
        result_file
    )

    assert saved_data.shape == (
        2,
        39,
    )
    assert saved_data.schema == (
        CANONICAL_PLAYER_SCHEMA
    )


def test_download_player_directory_skips_existing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip an existing player file without overwrite."""

    output_file = (
        tmp_path
        / "player_directory.parquet"
    )
    output_file.write_bytes(
        b"existing"
    )

    monkeypatch.setattr(
        player_ingestion,
        "PLAYER_DIRECTORY_FILE",
        output_file,
    )

    def fail_if_called() -> pl.DataFrame:
        raise AssertionError(
            "Unexpected player-directory download."
        )

    monkeypatch.setattr(
        player_ingestion.nfl,
        "load_players",
        fail_if_called,
    )

    result_file = download_player_directory(
        overwrite=False
    )

    assert result_file == output_file
    assert result_file.read_bytes() == b"existing"


def test_download_player_directory_overwrites_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replace an existing file when overwrite is requested."""

    output_file = (
        tmp_path
        / "player_directory.parquet"
    )
    output_file.write_bytes(
        b"existing"
    )

    monkeypatch.setattr(
        player_ingestion,
        "PLAYER_DATA_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        player_ingestion,
        "PLAYER_DIRECTORY_FILE",
        output_file,
    )
    monkeypatch.setattr(
        player_ingestion.nfl,
        "load_players",
        create_player_directory_frame,
    )

    download_player_directory(
        overwrite=True
    )

    saved_data = pl.read_parquet(
        output_file
    )

    assert saved_data.height == 2


def test_parse_arguments_uses_default() -> None:
    """Do not overwrite by default."""

    arguments = parse_arguments(
        []
    )

    assert arguments.overwrite is False


def test_parse_arguments_accepts_overwrite() -> None:
    """Accept the explicit overwrite option."""

    arguments = parse_arguments(
        [
            "--overwrite",
        ]
    )

    assert arguments.overwrite is True