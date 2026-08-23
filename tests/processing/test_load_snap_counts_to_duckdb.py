"""Tests for the raw player snap-count DuckDB loader."""

from pathlib import Path

import duckdb
import polars as pl
import pytest

from src.processing.load_snap_counts_to_duckdb import (
    REQUIRED_SNAP_COUNT_COLUMNS,
    TARGET_FULL_NAME,
    build_parquet_source,
    count_parquet_rows,
    create_raw_snap_count_table,
    get_snap_count_files,
    load_snap_counts_to_duckdb,
    validate_database_file,
    validate_loaded_table,
    validate_parquet_columns,
)


def create_snap_count_frame(
    season: int,
) -> pl.DataFrame:
    """Create a valid canonical snap-count frame."""

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
                f"TestQu{season}",
                f"TestRe{season}",
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
        },
        schema_overrides={
            "season": pl.Int32,
            "week": pl.Int32,
            "offense_snaps": pl.Float64,
            "offense_pct": pl.Float64,
            "defense_snaps": pl.Float64,
            "defense_pct": pl.Float64,
            "st_snaps": pl.Float64,
            "st_pct": pl.Float64,
        },
    )


def write_snap_count_files(
    data_directory: Path,
) -> list[Path]:
    """Write two canonical season-level test files."""

    data_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    files: list[Path] = []

    for season in [
        2024,
        2025,
    ]:
        season_file = (
            data_directory
            / f"snap_counts_{season}.parquet"
        )

        create_snap_count_frame(
            season
        ).write_parquet(
            season_file
        )

        files.append(
            season_file
        )

    return files


def create_loaded_table(
    connection: duckdb.DuckDBPyConnection,
    tmp_path: Path,
) -> tuple[list[Path], str]:
    """Create one valid raw test table."""

    source_files = write_snap_count_files(
        tmp_path
    )
    parquet_source = build_parquet_source(
        source_files
    )

    create_raw_snap_count_table(
        connection=connection,
        parquet_source=parquet_source,
    )

    return (
        source_files,
        parquet_source,
    )


def test_validate_database_file_accepts_file(
    tmp_path: Path,
) -> None:
    """Accept an existing database file."""

    database_file = (
        tmp_path
        / "test.duckdb"
    )

    with duckdb.connect(
        str(database_file)
    ):
        pass

    validate_database_file(
        database_file
    )


def test_validate_database_file_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """Reject a missing database file."""

    with pytest.raises(
        FileNotFoundError,
        match="does not exist",
    ):
        validate_database_file(
            tmp_path
            / "missing.duckdb"
        )


def test_validate_database_file_rejects_directory(
    tmp_path: Path,
) -> None:
    """Reject a directory used as a database path."""

    with pytest.raises(
        RuntimeError,
        match="is not a file",
    ):
        validate_database_file(
            tmp_path
        )


def test_get_snap_count_files_returns_sorted_files(
    tmp_path: Path,
) -> None:
    """Return canonical source files in season order."""

    files = write_snap_count_files(
        tmp_path
    )

    assert get_snap_count_files(
        tmp_path
    ) == files


def test_get_snap_count_files_rejects_missing_directory(
    tmp_path: Path,
) -> None:
    """Reject a missing source directory."""

    with pytest.raises(
        FileNotFoundError,
        match="does not exist",
    ):
        get_snap_count_files(
            tmp_path
            / "missing"
        )


def test_get_snap_count_files_rejects_empty_directory(
    tmp_path: Path,
) -> None:
    """Reject a source directory without Parquet files."""

    with pytest.raises(
        FileNotFoundError,
        match="No snap-count Parquet files",
    ):
        get_snap_count_files(
            tmp_path
        )


def test_build_parquet_source_rejects_empty_list() -> None:
    """Reject an empty Parquet source list."""

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        build_parquet_source(
            []
        )


def test_build_parquet_source_contains_every_file(
    tmp_path: Path,
) -> None:
    """Include every resolved source path."""

    files = write_snap_count_files(
        tmp_path
    )

    parquet_source = build_parquet_source(
        files
    )

    for source_file in files:
        assert (
            str(source_file.resolve())
            in parquet_source
        )


def test_validate_parquet_columns_accepts_canonical_schema(
    tmp_path: Path,
) -> None:
    """Accept the canonical 16-column source schema."""

    source_files = write_snap_count_files(
        tmp_path
    )
    parquet_source = build_parquet_source(
        source_files
    )

    with duckdb.connect(
        ":memory:"
    ) as connection:
        validate_parquet_columns(
            connection=connection,
            parquet_source=parquet_source,
        )


def test_validate_parquet_columns_rejects_missing_column(
    tmp_path: Path,
) -> None:
    """Reject a Parquet source missing a required column."""

    invalid_file = (
        tmp_path
        / "snap_counts_2025.parquet"
    )

    create_snap_count_frame(
        2025
    ).drop(
        "st_pct"
    ).write_parquet(
        invalid_file
    )

    parquet_source = build_parquet_source(
        [
            invalid_file,
        ]
    )

    with duckdb.connect(
        ":memory:"
    ) as connection:
        with pytest.raises(
            RuntimeError,
            match="Missing required snap-count columns: st_pct",
        ):
            validate_parquet_columns(
                connection=connection,
                parquet_source=parquet_source,
            )


def test_count_parquet_rows_counts_all_files(
    tmp_path: Path,
) -> None:
    """Count records across every season file."""

    source_files = write_snap_count_files(
        tmp_path
    )
    parquet_source = build_parquet_source(
        source_files
    )

    with duckdb.connect(
        ":memory:"
    ) as connection:
        assert count_parquet_rows(
            connection=connection,
            parquet_source=parquet_source,
        ) == 4


def test_create_raw_snap_count_table_adds_source_file(
    tmp_path: Path,
) -> None:
    """Create the raw table with source-file provenance."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_loaded_table(
            connection=connection,
            tmp_path=tmp_path,
        )

        columns = {
            row[0]
            for row in connection.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'raw'
                  AND table_name = 'player_snap_counts'
                """
            ).fetchall()
        }

        assert (
            REQUIRED_SNAP_COUNT_COLUMNS
            <= columns
        )
        assert "source_file" in columns


def test_validate_loaded_table_accepts_valid_table(
    tmp_path: Path,
) -> None:
    """Accept a valid loaded raw table."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_loaded_table(
            connection=connection,
            tmp_path=tmp_path,
        )

        result = validate_loaded_table(
            connection=connection,
            expected_row_count=4,
            expected_source_count=2,
        )

        assert result == (
            4,
            2,
            2,
        )


def test_validate_loaded_table_rejects_row_mismatch(
    tmp_path: Path,
) -> None:
    """Reject a loaded row count differing from the source."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_loaded_table(
            connection=connection,
            tmp_path=tmp_path,
        )

        with pytest.raises(
            RuntimeError,
            match="row count does not match",
        ):
            validate_loaded_table(
                connection=connection,
                expected_row_count=5,
                expected_source_count=2,
            )


def test_validate_loaded_table_rejects_duplicate_key(
    tmp_path: Path,
) -> None:
    """Reject duplicate player-team-game business keys."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_loaded_table(
            connection=connection,
            tmp_path=tmp_path,
        )

        connection.execute(
            f"""
            INSERT INTO {TARGET_FULL_NAME}
            SELECT *
            FROM {TARGET_FULL_NAME}
            LIMIT 1
            """
        )

        with pytest.raises(
            RuntimeError,
            match="duplicate player-team-game keys",
        ):
            validate_loaded_table(
                connection=connection,
                expected_row_count=5,
                expected_source_count=2,
            )


def test_validate_loaded_table_rejects_invalid_share(
    tmp_path: Path,
) -> None:
    """Reject a snap percentage above source tolerance."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_loaded_table(
            connection=connection,
            tmp_path=tmp_path,
        )

        connection.execute(
            f"""
            UPDATE {TARGET_FULL_NAME}
            SET st_pct = 1.02
            WHERE pfr_player_id = 'TestRe2025'
            """
        )

        with pytest.raises(
            RuntimeError,
            match="invalid participation values",
        ):
            validate_loaded_table(
                connection=connection,
                expected_row_count=4,
                expected_source_count=2,
            )


def test_load_snap_counts_to_duckdb_creates_table(
    tmp_path: Path,
) -> None:
    """Run the complete raw loading workflow."""

    source_directory = (
        tmp_path
        / "snap_counts"
    )
    write_snap_count_files(
        source_directory
    )

    database_file = (
        tmp_path
        / "test.duckdb"
    )

    with duckdb.connect(
        str(database_file)
    ):
        pass

    load_snap_counts_to_duckdb(
        database_file=database_file,
        snap_count_data_dir=source_directory,
    )

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        row_count = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM {TARGET_FULL_NAME}
            """
        ).fetchone()[0]

        source_count = connection.execute(
            f"""
            SELECT COUNT(DISTINCT source_file)
            FROM {TARGET_FULL_NAME}
            """
        ).fetchone()[0]

    assert row_count == 4
    assert source_count == 2