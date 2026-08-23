"""Tests for the raw injury-report DuckDB loader."""

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import polars as pl
import pytest

from src.processing.load_injury_reports_to_duckdb import (
    build_parquet_source,
    get_injury_files,
    load_injury_reports_to_duckdb,
    validate_parquet_columns,
)


def create_injury_frame(
    season: int,
    player_suffix: str = "1",
) -> pl.DataFrame:
    """Create a canonical injury-report test frame."""

    return pl.DataFrame(
        {
            "season": [
                season,
            ],
            "season_type": [
                "REG",
            ],
            "game_type": [
                "REG",
            ],
            "team": [
                "NE",
            ],
            "week": [
                1,
            ],
            "gsis_id": [
                f"00-000000{player_suffix}",
            ],
            "position": [
                "QB",
            ],
            "full_name": [
                f"Test Player {player_suffix}",
            ],
            "first_name": [
                "Test",
            ],
            "last_name": [
                f"Player {player_suffix}",
            ],
            "report_primary_injury": [
                "Shoulder",
            ],
            "report_secondary_injury": [
                None,
            ],
            "report_status": [
                "Questionable",
            ],
            "practice_primary_injury": [
                "Shoulder",
            ],
            "practice_secondary_injury": [
                None,
            ],
            "practice_status": [
                "Limited Participation in Practice",
            ],
            "date_modified": [
                datetime(
                    season,
                    9,
                    5,
                    12,
                    0,
                    tzinfo=timezone.utc,
                ),
            ],
        },
        schema_overrides={
            "season": pl.Int32,
            "week": pl.Int32,
            "date_modified": pl.Datetime(
                time_unit="us",
                time_zone="UTC",
            ),
        },
    )


def test_get_injury_files_rejects_missing_directory(
    tmp_path: Path,
) -> None:
    """Reject a missing injury source directory."""

    missing_directory = tmp_path / "missing"

    with pytest.raises(
        FileNotFoundError,
        match="Injury data directory does not exist",
    ):
        get_injury_files(
            missing_directory
        )


def test_get_injury_files_rejects_file_path(
    tmp_path: Path,
) -> None:
    """Reject a source path that is not a directory."""

    source_file = tmp_path / "injuries"
    source_file.touch()

    with pytest.raises(
        RuntimeError,
        match="is not a directory",
    ):
        get_injury_files(
            source_file
        )


def test_get_injury_files_rejects_empty_directory(
    tmp_path: Path,
) -> None:
    """Reject a directory without injury Parquet files."""

    with pytest.raises(
        FileNotFoundError,
        match="No injury-report Parquet files found",
    ):
        get_injury_files(
            tmp_path
        )


def test_get_injury_files_returns_sorted_files(
    tmp_path: Path,
) -> None:
    """Return only canonical files in deterministic order."""

    later_file = (
        tmp_path
        / "injury_reports_2025.parquet"
    )
    earlier_file = (
        tmp_path
        / "injury_reports_2024.parquet"
    )
    ignored_file = (
        tmp_path
        / "notes.txt"
    )

    later_file.touch()
    earlier_file.touch()
    ignored_file.touch()

    injury_files = get_injury_files(
        tmp_path
    )

    assert injury_files == [
        earlier_file,
        later_file,
    ]


def test_build_parquet_source_rejects_empty_list() -> None:
    """Reject an empty DuckDB Parquet source."""

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        build_parquet_source([])


def test_build_parquet_source_preserves_order(
    tmp_path: Path,
) -> None:
    """Preserve file order in the DuckDB expression."""

    first_file = (
        tmp_path
        / "injury_reports_2024.parquet"
    )
    second_file = (
        tmp_path
        / "injury_reports_2025.parquet"
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

    assert parquet_source.startswith("[")
    assert parquet_source.endswith("]")
    assert first_path in parquet_source
    assert second_path in parquet_source
    assert (
        parquet_source.index(first_path)
        < parquet_source.index(second_path)
    )


def test_validate_parquet_columns_rejects_missing_columns(
    tmp_path: Path,
) -> None:
    """Reject a Parquet source without canonical columns."""

    parquet_file = (
        tmp_path
        / "injury_reports_2025.parquet"
    )

    pl.DataFrame(
        {
            "season": [
                2025,
            ],
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
            match="Missing required injury-report columns",
        ):
            validate_parquet_columns(
                connection,
                parquet_source,
            )


def test_load_injury_reports_creates_raw_table(
    tmp_path: Path,
) -> None:
    """Load multiple seasons into raw.injury_reports."""

    injury_directory = tmp_path / "injuries"
    injury_directory.mkdir()

    first_file = (
        injury_directory
        / "injury_reports_2024.parquet"
    )
    second_file = (
        injury_directory
        / "injury_reports_2025.parquet"
    )
    database_file = tmp_path / "test.duckdb"

    create_injury_frame(
        season=2024,
        player_suffix="1",
    ).write_parquet(
        first_file
    )
    create_injury_frame(
        season=2025,
        player_suffix="2",
    ).write_parquet(
        second_file
    )

    load_injury_reports_to_duckdb(
        database_file=database_file,
        injury_data_dir=injury_directory,
    )

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        rows = connection.execute(
            """
            SELECT
                season,
                gsis_id,
                report_status
            FROM raw.injury_reports
            ORDER BY season
            """
        ).fetchall()

        column_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.columns
            WHERE table_schema = 'raw'
              AND table_name = 'injury_reports'
            """
        ).fetchone()[0]

    assert rows == [
        (
            2024,
            "00-0000001",
            "Questionable",
        ),
        (
            2025,
            "00-0000002",
            "Questionable",
        ),
    ]
    assert column_count == 17


def test_load_injury_reports_preserves_status_history(
    tmp_path: Path,
) -> None:
    """Preserve multiple timestamped weekly snapshots."""

    injury_directory = tmp_path / "injuries"
    injury_directory.mkdir()

    parquet_file = (
        injury_directory
        / "injury_reports_2024.parquet"
    )
    database_file = tmp_path / "test.duckdb"

    first_snapshot = create_injury_frame(
        season=2024,
    )

    second_snapshot = (
        first_snapshot
        .with_columns(
            pl.lit("Out").alias(
                "report_status"
            ),
            pl.lit(
                datetime(
                    2024,
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

    pl.concat(
        [
            first_snapshot,
            second_snapshot,
        ]
    ).write_parquet(
        parquet_file
    )

    load_injury_reports_to_duckdb(
        database_file=database_file,
        injury_data_dir=injury_directory,
    )

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        statuses = connection.execute(
            """
            SELECT report_status
            FROM raw.injury_reports
            ORDER BY date_modified
            """
        ).fetchall()

    assert statuses == [
        (
            "Questionable",
        ),
        (
            "Out",
        ),
    ]


def test_load_injury_reports_replaces_existing_table(
    tmp_path: Path,
) -> None:
    """Replace the raw table during a repeated full refresh."""

    injury_directory = tmp_path / "injuries"
    injury_directory.mkdir()

    parquet_file = (
        injury_directory
        / "injury_reports_2025.parquet"
    )
    database_file = tmp_path / "test.duckdb"

    create_injury_frame(
        season=2025,
        player_suffix="1",
    ).write_parquet(
        parquet_file
    )

    load_injury_reports_to_duckdb(
        database_file=database_file,
        injury_data_dir=injury_directory,
    )

    create_injury_frame(
        season=2025,
        player_suffix="9",
    ).write_parquet(
        parquet_file
    )

    load_injury_reports_to_duckdb(
        database_file=database_file,
        injury_data_dir=injury_directory,
    )

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        player_ids = connection.execute(
            """
            SELECT gsis_id
            FROM raw.injury_reports
            """
        ).fetchall()

    assert player_ids == [
        (
            "00-0000009",
        ),
    ]