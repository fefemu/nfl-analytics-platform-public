"""Tests for the raw depth-chart DuckDB loader."""

from pathlib import Path

import duckdb
import polars as pl
import pytest

from src.processing.load_depth_charts_to_duckdb import (
    build_parquet_source,
    get_depth_chart_files,
    load_depth_charts_to_duckdb,
    validate_database_file,
    validate_parquet_columns,
)


def create_legacy_frame(
    season: int = 2024,
    team: str = "NE",
) -> pl.DataFrame:
    """Create a representative legacy depth-chart frame."""

    return pl.DataFrame(
        {
            "season": [
                season,
            ],
            "club_code": [
                team,
            ],
            "week": [
                1,
            ],
            "game_type": [
                "REG",
            ],
            "depth_team": [
                "1",
            ],
            "last_name": [
                "Starter",
            ],
            "first_name": [
                "Test",
            ],
            "football_name": [
                "Test",
            ],
            "formation": [
                "Offense",
            ],
            "gsis_id": [
                "00-0000001",
            ],
            "jersey_number": [
                "10",
            ],
            "position": [
                "QB",
            ],
            "elias_id": [
                "STA000001",
            ],
            "depth_position": [
                "QB",
            ],
            "full_name": [
                "Test Starter",
            ],
        },
        schema_overrides={
            "season": pl.Int32,
            "week": pl.Int32,
        },
    )


def create_espn_frame(
    player_suffix: str = "1",
    gsis_id: str | None = "00-0000001",
) -> pl.DataFrame:
    """Create a representative ESPN depth-chart frame."""

    return pl.DataFrame(
        {
            "dt": [
                "2026-08-03T10:36:38Z",
            ],
            "team": [
                "NE",
            ],
            "player_name": [
                f"Test Player {player_suffix}",
            ],
            "espn_id": [
                f"100000{player_suffix}",
            ],
            "gsis_id": [
                gsis_id,
            ],
            "pos_grp_id": [
                "1",
            ],
            "pos_grp": [
                "3WR 1TE",
            ],
            "pos_id": [
                "1",
            ],
            "pos_name": [
                "Quarterback",
            ],
            "pos_abb": [
                "QB",
            ],
            "pos_slot": [
                1,
            ],
            "pos_rank": [
                1,
            ],
        },
        schema_overrides={
            "pos_slot": pl.Int32,
            "pos_rank": pl.Int32,
        },
    )


def create_database_file(
    database_file: Path,
) -> None:
    """Create an empty DuckDB database file."""

    with duckdb.connect(
        str(database_file)
    ):
        pass


def test_validate_database_file_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """Reject a missing DuckDB file."""

    with pytest.raises(
        FileNotFoundError,
        match="Database file does not exist",
    ):
        validate_database_file(
            tmp_path / "missing.duckdb"
        )


def test_validate_database_file_accepts_database(
    tmp_path: Path,
) -> None:
    """Accept an existing DuckDB file."""

    database_file = tmp_path / "test.duckdb"

    create_database_file(
        database_file
    )

    validate_database_file(
        database_file
    )


def test_get_depth_chart_files_rejects_missing_directory(
    tmp_path: Path,
) -> None:
    """Reject a missing depth-chart directory."""

    with pytest.raises(
        FileNotFoundError,
        match="source directory does not exist",
    ):
        get_depth_chart_files(
            tmp_path / "missing"
        )


def test_get_depth_chart_files_rejects_empty_directory(
    tmp_path: Path,
) -> None:
    """Reject a directory without matching Parquet files."""

    with pytest.raises(
        FileNotFoundError,
        match="No depth-chart Parquet files found",
    ):
        get_depth_chart_files(
            tmp_path
        )


def test_get_depth_chart_files_returns_sorted_files(
    tmp_path: Path,
) -> None:
    """Return only canonical files in filename order."""

    later_file = (
        tmp_path
        / "depth_charts_2026.parquet"
    )
    earlier_file = (
        tmp_path
        / "depth_charts_2025.parquet"
    )
    ignored_file = (
        tmp_path
        / "notes.txt"
    )

    later_file.touch()
    earlier_file.touch()
    ignored_file.touch()

    files = get_depth_chart_files(
        tmp_path
    )

    assert files == [
        earlier_file,
        later_file,
    ]


def test_build_parquet_source_rejects_empty_list() -> None:
    """Reject an empty Parquet file list."""

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        build_parquet_source([])


def test_build_parquet_source_preserves_order(
    tmp_path: Path,
) -> None:
    """Preserve input order in the DuckDB expression."""

    first_file = (
        tmp_path
        / "depth_charts_2025.parquet"
    )
    second_file = (
        tmp_path
        / "depth_charts_2026.parquet"
    )

    parquet_source = build_parquet_source(
        [
            first_file,
            second_file,
        ]
    )

    first_path = str(
        first_file.resolve()
    )
    second_path = str(
        second_file.resolve()
    )

    assert first_path in parquet_source
    assert second_path in parquet_source
    assert (
        parquet_source.index(first_path)
        < parquet_source.index(second_path)
    )


def test_validate_parquet_columns_rejects_missing_columns(
    tmp_path: Path,
) -> None:
    """Reject a source without its required schema."""

    parquet_file = (
        tmp_path
        / "depth_charts_2025.parquet"
    )

    pl.DataFrame(
        {
            "team": [
                "NE",
            ],
        }
    ).write_parquet(
        parquet_file
    )

    parquet_source = build_parquet_source(
        [
            parquet_file,
        ]
    )

    with duckdb.connect(
        ":memory:"
    ) as connection:
        with pytest.raises(
            RuntimeError,
            match="is missing columns",
        ):
            validate_parquet_columns(
                connection=connection,
                parquet_source=parquet_source,
                required_columns={
                    "team",
                    "espn_id",
                },
                source_name="ESPN",
            )


def test_load_depth_charts_creates_both_raw_tables(
    tmp_path: Path,
) -> None:
    """Load legacy and ESPN sources into separate tables."""

    legacy_directory = tmp_path / "legacy"
    espn_directory = tmp_path / "espn"
    legacy_directory.mkdir()
    espn_directory.mkdir()

    database_file = tmp_path / "test.duckdb"
    create_database_file(
        database_file
    )

    create_legacy_frame().write_parquet(
        legacy_directory
        / "depth_charts_2024.parquet"
    )
    create_espn_frame().write_parquet(
        espn_directory
        / "depth_charts_2026.parquet"
    )

    load_depth_charts_to_duckdb(
        database_file=database_file,
        legacy_data_dir=legacy_directory,
        espn_data_dir=espn_directory,
    )

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        legacy_rows = connection.execute(
            """
            SELECT
                season,
                club_code,
                gsis_id
            FROM raw.depth_charts_legacy
            """
        ).fetchall()

        espn_rows = connection.execute(
            """
            SELECT
                source_season,
                team,
                espn_id,
                gsis_id
            FROM raw.depth_charts_espn
            """
        ).fetchall()

    assert legacy_rows == [
        (
            2024,
            "NE",
            "00-0000001",
        ),
    ]
    assert espn_rows == [
        (
            2026,
            "NE",
            "1000001",
            "00-0000001",
        ),
    ]


def test_load_depth_charts_preserves_source_files(
    tmp_path: Path,
) -> None:
    """Add source-file provenance to both raw tables."""

    legacy_directory = tmp_path / "legacy"
    espn_directory = tmp_path / "espn"
    legacy_directory.mkdir()
    espn_directory.mkdir()

    database_file = tmp_path / "test.duckdb"
    create_database_file(
        database_file
    )

    create_legacy_frame().write_parquet(
        legacy_directory
        / "depth_charts_2024.parquet"
    )
    create_espn_frame().write_parquet(
        espn_directory
        / "depth_charts_2025.parquet"
    )

    load_depth_charts_to_duckdb(
        database_file=database_file,
        legacy_data_dir=legacy_directory,
        espn_data_dir=espn_directory,
    )

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        source_files = connection.execute(
            """
            SELECT source_file
            FROM raw.depth_charts_legacy

            UNION ALL

            SELECT source_file
            FROM raw.depth_charts_espn
            """
        ).fetchall()

    assert len(source_files) == 2
    assert all(
        row[0].endswith(
            ".parquet"
        )
        for row in source_files
    )


def test_load_depth_charts_preserves_legacy_super_bowl_bye(
    tmp_path: Path,
) -> None:
    """Preserve a null-week SBBYE legacy source row."""

    legacy_directory = tmp_path / "legacy"
    espn_directory = tmp_path / "espn"
    legacy_directory.mkdir()
    espn_directory.mkdir()

    database_file = tmp_path / "test.duckdb"
    create_database_file(
        database_file
    )

    legacy_data = (
        create_legacy_frame()
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

    legacy_data.write_parquet(
        legacy_directory
        / "depth_charts_2024.parquet"
    )
    create_espn_frame().write_parquet(
        espn_directory
        / "depth_charts_2025.parquet"
    )

    load_depth_charts_to_duckdb(
        database_file=database_file,
        legacy_data_dir=legacy_directory,
        espn_data_dir=espn_directory,
    )

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        result = connection.execute(
            """
            SELECT
                week,
                game_type
            FROM raw.depth_charts_legacy
            """
        ).fetchone()

    assert result == (
        None,
        "SBBYE",
    )


def test_load_depth_charts_allows_missing_espn_gsis(
    tmp_path: Path,
) -> None:
    """Preserve ESPN rows without GSIS identifiers."""

    legacy_directory = tmp_path / "legacy"
    espn_directory = tmp_path / "espn"
    legacy_directory.mkdir()
    espn_directory.mkdir()

    database_file = tmp_path / "test.duckdb"
    create_database_file(
        database_file
    )

    create_legacy_frame().write_parquet(
        legacy_directory
        / "depth_charts_2024.parquet"
    )
    create_espn_frame(
        gsis_id=None
    ).write_parquet(
        espn_directory
        / "depth_charts_2025.parquet"
    )

    load_depth_charts_to_duckdb(
        database_file=database_file,
        legacy_data_dir=legacy_directory,
        espn_data_dir=espn_directory,
    )

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        result = connection.execute(
            """
            SELECT
                espn_id,
                gsis_id
            FROM raw.depth_charts_espn
            """
        ).fetchone()

    assert result == (
        "1000001",
        None,
    )


def test_load_depth_charts_replaces_existing_tables(
    tmp_path: Path,
) -> None:
    """Replace both raw tables during a repeated refresh."""

    legacy_directory = tmp_path / "legacy"
    espn_directory = tmp_path / "espn"
    legacy_directory.mkdir()
    espn_directory.mkdir()

    database_file = tmp_path / "test.duckdb"
    create_database_file(
        database_file
    )

    legacy_file = (
        legacy_directory
        / "depth_charts_2024.parquet"
    )
    espn_file = (
        espn_directory
        / "depth_charts_2026.parquet"
    )

    create_legacy_frame(
        team="NE"
    ).write_parquet(
        legacy_file
    )
    create_espn_frame(
        player_suffix="1"
    ).write_parquet(
        espn_file
    )

    load_depth_charts_to_duckdb(
        database_file=database_file,
        legacy_data_dir=legacy_directory,
        espn_data_dir=espn_directory,
    )

    create_legacy_frame(
        team="BUF"
    ).write_parquet(
        legacy_file
    )
    create_espn_frame(
        player_suffix="9",
        gsis_id="00-0000009",
    ).write_parquet(
        espn_file
    )

    load_depth_charts_to_duckdb(
        database_file=database_file,
        legacy_data_dir=legacy_directory,
        espn_data_dir=espn_directory,
    )

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        legacy_team = connection.execute(
            """
            SELECT club_code
            FROM raw.depth_charts_legacy
            """
        ).fetchone()[0]

        espn_player = connection.execute(
            """
            SELECT espn_id
            FROM raw.depth_charts_espn
            """
        ).fetchone()[0]

    assert legacy_team == "BUF"
    assert espn_player == "1000009"