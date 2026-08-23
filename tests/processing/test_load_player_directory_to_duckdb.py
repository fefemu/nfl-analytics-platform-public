"""Tests for the raw player-directory DuckDB loader."""

from pathlib import Path

import duckdb
import polars as pl
import pytest

from src.ingestion.download_player_directory import (
    CANONICAL_PLAYER_COLUMNS,
    normalize_player_directory_schema,
)
from src.processing.load_player_directory_to_duckdb import (
    REQUIRED_PLAYER_COLUMNS,
    TARGET_FULL_NAME,
    count_duplicate_non_null_ids,
    create_raw_player_directory,
    escape_path,
    load_player_directory_to_duckdb,
    validate_database_file,
    validate_loaded_table,
    validate_parquet_columns,
    validate_player_directory_file,
)


def create_player_frame() -> pl.DataFrame:
    """Create a valid typed player-directory frame."""

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
            "first_name": [
                "Test",
                "Test",
            ],
            "last_name": [
                "Quarterback",
                "Receiver",
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
        }
    )

    return normalize_player_directory_schema(
        pl.DataFrame(
            player_data
        )
    )


def write_player_file(
    tmp_path: Path,
) -> Path:
    """Write one canonical player-directory Parquet file."""

    player_file = (
        tmp_path
        / "player_directory.parquet"
    )

    create_player_frame().write_parquet(
        player_file
    )

    return player_file


def test_validate_database_file_accepts_file(
    tmp_path: Path,
) -> None:
    """Accept an existing DuckDB file."""

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
    """Reject a missing DuckDB file."""

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
    """Reject a directory as the database path."""

    with pytest.raises(
        RuntimeError,
        match="is not a file",
    ):
        validate_database_file(
            tmp_path
        )


def test_validate_player_directory_file_accepts_file(
    tmp_path: Path,
) -> None:
    """Accept an existing player-directory file."""

    player_file = write_player_file(
        tmp_path
    )

    validate_player_directory_file(
        player_file
    )


def test_validate_player_directory_file_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """Reject a missing player-directory file."""

    with pytest.raises(
        FileNotFoundError,
        match="does not exist",
    ):
        validate_player_directory_file(
            tmp_path
            / "missing.parquet"
        )


def test_validate_player_directory_file_rejects_directory(
    tmp_path: Path,
) -> None:
    """Reject a directory as the player source path."""

    with pytest.raises(
        RuntimeError,
        match="is not a file",
    ):
        validate_player_directory_file(
            tmp_path
        )


def test_escape_path_resolves_path(
    tmp_path: Path,
) -> None:
    """Return an absolute SQL-safe source path."""

    source_file = (
        tmp_path
        / "players.parquet"
    )

    assert escape_path(
        source_file
    ) == str(
        source_file.resolve()
    )


def test_validate_parquet_columns_accepts_schema(
    tmp_path: Path,
) -> None:
    """Accept the canonical player-directory schema."""

    player_file = write_player_file(
        tmp_path
    )

    with duckdb.connect(
        ":memory:"
    ) as connection:
        validate_parquet_columns(
            connection=connection,
            player_directory_file=player_file,
        )


def test_validate_parquet_columns_rejects_missing_column(
    tmp_path: Path,
) -> None:
    """Reject a player source missing a required column."""

    player_file = (
        tmp_path
        / "player_directory.parquet"
    )

    create_player_frame().drop(
        "pfr_id"
    ).write_parquet(
        player_file
    )

    with duckdb.connect(
        ":memory:"
    ) as connection:
        with pytest.raises(
            RuntimeError,
            match="Missing required player-directory columns: pfr_id",
        ):
            validate_parquet_columns(
                connection=connection,
                player_directory_file=player_file,
            )


def test_create_raw_player_directory_adds_source_file(
    tmp_path: Path,
) -> None:
    """Create the raw player table with source provenance."""

    player_file = write_player_file(
        tmp_path
    )

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_raw_player_directory(
            connection=connection,
            player_directory_file=player_file,
        )

        columns = {
            row[0]
            for row in connection.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'raw'
                  AND table_name = 'player_directory'
                """
            ).fetchall()
        }

    assert REQUIRED_PLAYER_COLUMNS <= columns
    assert "source_file" in columns


def test_count_duplicate_non_null_ids_ignores_nulls(
    tmp_path: Path,
) -> None:
    """Ignore repeated null identifier values."""

    player_file = write_player_file(
        tmp_path
    )

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_raw_player_directory(
            connection=connection,
            player_directory_file=player_file,
        )

        connection.execute(
            f"""
            UPDATE {TARGET_FULL_NAME}
            SET espn_id = NULL
            """
        )

        duplicate_count = count_duplicate_non_null_ids(
            connection=connection,
            identifier_column="espn_id",
        )

    assert duplicate_count == 0


def test_validate_loaded_table_accepts_valid_table(
    tmp_path: Path,
) -> None:
    """Accept a valid raw player directory."""

    player_file = write_player_file(
        tmp_path
    )

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_raw_player_directory(
            connection=connection,
            player_directory_file=player_file,
        )

        result = validate_loaded_table(
            connection=connection,
            expected_row_count=2,
        )

    assert result == (
        2,
        2,
        2,
    )


def test_validate_loaded_table_rejects_row_mismatch(
    tmp_path: Path,
) -> None:
    """Reject a row count differing from the Parquet source."""

    player_file = write_player_file(
        tmp_path
    )

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_raw_player_directory(
            connection=connection,
            player_directory_file=player_file,
        )

        with pytest.raises(
            RuntimeError,
            match="row count does not match",
        ):
            validate_loaded_table(
                connection=connection,
                expected_row_count=3,
            )


def test_validate_loaded_table_rejects_null_gsis(
    tmp_path: Path,
) -> None:
    """Reject a player without a GSIS identifier."""

    player_file = write_player_file(
        tmp_path
    )

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_raw_player_directory(
            connection=connection,
            player_directory_file=player_file,
        )

        connection.execute(
            f"""
            UPDATE {TARGET_FULL_NAME}
            SET gsis_id = NULL
            WHERE pfr_id = 'TestRe00'
            """
        )

        with pytest.raises(
            RuntimeError,
            match="without a GSIS ID",
        ):
            validate_loaded_table(
                connection=connection,
                expected_row_count=2,
            )


@pytest.mark.parametrize(
    "identifier_column",
    [
        "gsis_id",
        "pfr_id",
        "espn_id",
    ],
)
def test_validate_loaded_table_rejects_duplicate_id(
    identifier_column: str,
    tmp_path: Path,
) -> None:
    """Reject duplicated stable player identifiers."""

    player_file = write_player_file(
        tmp_path
    )

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_raw_player_directory(
            connection=connection,
            player_directory_file=player_file,
        )

        first_identifier = connection.execute(
            f"""
            SELECT {identifier_column}
            FROM {TARGET_FULL_NAME}
            LIMIT 1
            """
        ).fetchone()[0]

        connection.execute(
            f"""
            UPDATE {TARGET_FULL_NAME}
            SET {identifier_column} = ?
            """,
            [
                first_identifier,
            ],
        )

        with pytest.raises(
            RuntimeError,
            match=f"duplicate {identifier_column} groups",
        ):
            validate_loaded_table(
                connection=connection,
                expected_row_count=2,
            )


def test_load_player_directory_to_duckdb_creates_table(
    tmp_path: Path,
) -> None:
    """Run the complete raw player-directory workflow."""

    player_file = write_player_file(
        tmp_path
    )
    database_file = (
        tmp_path
        / "test.duckdb"
    )

    with duckdb.connect(
        str(database_file)
    ):
        pass

    load_player_directory_to_duckdb(
        database_file=database_file,
        player_directory_file=player_file,
    )

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        result = connection.execute(
            f"""
            SELECT
                COUNT(*),
                COUNT(pfr_id),
                COUNT(espn_id)
            FROM {TARGET_FULL_NAME}
            """
        ).fetchone()

    assert result == (
        2,
        2,
        2,
    )