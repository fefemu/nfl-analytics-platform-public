from pathlib import Path

import duckdb
import polars as pl
import pytest

from src.processing.load_schedule_to_duckdb import (
    load_schedule_to_duckdb,
    validate_source_file,
)


def test_validate_source_file_rejects_missing_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail validation when the raw schedule Parquet file is missing."""

    missing_file = tmp_path / "missing.parquet"

    monkeypatch.setattr(
        "src.processing.load_schedule_to_duckdb.SCHEDULE_FILE",
        missing_file,
    )

    with pytest.raises(
        FileNotFoundError,
        match="Schedule file does not exist",
    ):
        validate_source_file()


def test_validate_source_file_accepts_existing_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass validation when the raw schedule Parquet file exists."""

    schedule_file = tmp_path / "schedules.parquet"
    schedule_file.touch()

    monkeypatch.setattr(
        "src.processing.load_schedule_to_duckdb.SCHEDULE_FILE",
        schedule_file,
    )

    validate_source_file()


def test_load_schedule_to_duckdb_creates_raw_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load a temporary Parquet file into raw.schedule."""

    schedule_file = tmp_path / "schedules.parquet"
    database_file = tmp_path / "test.duckdb"

    schedule = pl.DataFrame(
        {
            "game_id": [
                "2026_01_BUF_NYJ",
                "2026_01_KC_LV",
            ],
            "season": [
                2026,
                2026,
            ],
            "week": [
                1,
                1,
            ],
        }
    )
    schedule.write_parquet(schedule_file)

    monkeypatch.setattr(
        "src.processing.load_schedule_to_duckdb.SCHEDULE_FILE",
        schedule_file,
    )
    monkeypatch.setattr(
        "src.processing.load_schedule_to_duckdb.DATABASE_FILE",
        database_file,
    )

    load_schedule_to_duckdb()

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        rows = connection.execute(
            """
            SELECT game_id, season, week
            FROM raw.schedule
            ORDER BY game_id
            """
        ).fetchall()

        column_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.columns
            WHERE table_schema = 'raw'
              AND table_name = 'schedule'
            """
        ).fetchone()[0]

    assert rows == [
        ("2026_01_BUF_NYJ", 2026, 1),
        ("2026_01_KC_LV", 2026, 1),
    ]
    assert column_count == 3


def test_load_schedule_to_duckdb_replaces_existing_raw_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replace raw.schedule during a repeated full-refresh load."""

    schedule_file = tmp_path / "schedules.parquet"
    database_file = tmp_path / "test.duckdb"

    first_schedule = pl.DataFrame(
        {
            "game_id": [
                "old_game_1",
                "old_game_2",
            ],
            "season": [
                2025,
                2025,
            ],
        }
    )
    first_schedule.write_parquet(schedule_file)

    monkeypatch.setattr(
        "src.processing.load_schedule_to_duckdb.SCHEDULE_FILE",
        schedule_file,
    )
    monkeypatch.setattr(
        "src.processing.load_schedule_to_duckdb.DATABASE_FILE",
        database_file,
    )

    load_schedule_to_duckdb()

    replacement_schedule = pl.DataFrame(
        {
            "game_id": [
                "new_game",
            ],
            "season": [
                2026,
            ],
        }
    )
    replacement_schedule.write_parquet(schedule_file)

    load_schedule_to_duckdb()

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        rows = connection.execute(
            """
            SELECT game_id, season
            FROM raw.schedule
            """
        ).fetchall()

    assert rows == [
        ("new_game", 2026),
    ]
