"""Tests for the quarterback game performance builder."""

from pathlib import Path

import duckdb
import pytest

from src.processing.build_qb_game_performance import (
    build_parquet_source,
    create_qb_game_aggregates,
    get_pbp_files,
    validate_pbp_columns,
)


def test_get_pbp_files_rejects_missing_directory(
    tmp_path: Path,
) -> None:
    """Fail when the configured PBP directory is missing."""

    missing_directory = tmp_path / "missing"

    with pytest.raises(
        FileNotFoundError,
        match="PBP directory does not exist",
    ):
        get_pbp_files(missing_directory)


def test_get_pbp_files_rejects_empty_directory(
    tmp_path: Path,
) -> None:
    """Fail when no season PBP files are available."""

    with pytest.raises(
        FileNotFoundError,
        match="No PBP Parquet files found",
    ):
        get_pbp_files(tmp_path)


def test_get_pbp_files_returns_sorted_files(
    tmp_path: Path,
) -> None:
    """Return matching PBP files in deterministic order."""

    later_file = tmp_path / "pbp_2025.parquet"
    earlier_file = tmp_path / "pbp_2024.parquet"
    ignored_file = tmp_path / "notes.txt"

    later_file.touch()
    earlier_file.touch()
    ignored_file.touch()

    result = get_pbp_files(tmp_path)

    assert result == [
        earlier_file,
        later_file,
    ]


def test_validate_pbp_columns_rejects_missing_columns(
    tmp_path: Path,
) -> None:
    """Fail when required QB columns are missing."""

    parquet_file = tmp_path / "pbp_2025.parquet"
    escaped_path = str(parquet_file).replace("'", "''")

    with duckdb.connect(":memory:") as connection:
        connection.execute(
            f"""
            COPY (
                SELECT
                    'game_1'::VARCHAR AS game_id
            )
            TO '{escaped_path}'
            (FORMAT PARQUET)
            """
        )

        parquet_source = build_parquet_source(
            [parquet_file]
        )

        with pytest.raises(
            RuntimeError,
            match="Missing required QB PBP columns",
        ):
            validate_pbp_columns(
                connection,
                parquet_source,
            )


def test_create_qb_game_aggregates_calculates_metrics(
) -> None:
    """Aggregate dropbacks into one QB-game performance row."""

    with duckdb.connect(":memory:") as connection:
        connection.execute(
            """
            CREATE TEMP TABLE valid_qb_dropbacks AS
            SELECT *
            FROM (
                VALUES
                    (
                        'game_1', 2025, 'REG', 1,
                        DATE '2025-09-07',
                        'BUF', 'MIA', 'BUF', 'MIA',
                        TRUE, 'qb_1', 'Test QB',
                        TRUE, TRUE,
                        1, 0, 20, 0.5, 1.0,
                        5.0, 10.0,
                        0, 0, 0, 0, 0
                    ),
                    (
                        'game_1', 2025, 'REG', 1,
                        DATE '2025-09-07',
                        'BUF', 'MIA', 'BUF', 'MIA',
                        TRUE, 'qb_1', 'Test QB',
                        TRUE, TRUE,
                        0, 1, 0, -1.0, 0.0,
                        -10.0, 15.0,
                        0, 0, 0, 1, 0
                    ),
                    (
                        'game_1', 2025, 'REG', 1,
                        DATE '2025-09-07',
                        'BUF', 'MIA', 'BUF', 'MIA',
                        TRUE, 'qb_1', 'Test QB',
                        TRUE, FALSE,
                        0, 0, 0, -1.5, 0.0,
                        NULL, NULL,
                        1, 1, 0, 0, 0
                    ),
                    (
                        'game_1', 2025, 'REG', 1,
                        DATE '2025-09-07',
                        'BUF', 'MIA', 'BUF', 'MIA',
                        TRUE, 'qb_1', 'Test QB',
                        FALSE, FALSE,
                        0, 0, 0, 0.4, 1.0,
                        NULL, NULL,
                        0, 0, 1, 0, 1
                    )
            ) AS dropbacks(
                game_id,
                season,
                season_type,
                week,
                game_date,
                team,
                opponent,
                home_team,
                away_team,
                is_home,
                qb_id,
                qb_name,
                is_competitive_dropback,
                is_throw_attempt,
                complete_pass,
                incomplete_pass,
                passing_yards,
                epa,
                success,
                cpoe,
                air_yards,
                sack,
                qb_hit,
                qb_scramble,
                interception,
                fumble_lost
            )
            """
        )

        create_qb_game_aggregates(connection)

        result = connection.execute(
            """
            SELECT
                dropbacks,
                competitive_dropbacks,
                throw_attempts,
                completions,
                incompletions,
                passing_yards,
                epa_per_dropback,
                competitive_epa_per_dropback,
                success_rate,
                completion_rate,
                cpoe,
                air_yards_per_attempt,
                sacks,
                sack_rate,
                qb_hits,
                qb_hit_rate,
                scrambles,
                scramble_rate,
                interceptions,
                interception_rate,
                fumbles_lost,
                turnovers,
                turnover_rate
            FROM qb_game_aggregates
            """
        ).fetchone()

    assert result[0:6] == (
        4,
        3,
        2,
        1,
        1,
        20,
    )

    assert result[6] == pytest.approx(-0.4)
    assert result[7] == pytest.approx(-2.0 / 3.0)
    assert result[8] == pytest.approx(0.5)
    assert result[9] == pytest.approx(0.5)
    assert result[10] == pytest.approx(-2.5)
    assert result[11] == pytest.approx(12.5)

    assert result[12] == 1
    assert result[13] == pytest.approx(0.25)
    assert result[14] == 1
    assert result[15] == pytest.approx(0.25)
    assert result[16] == 1
    assert result[17] == pytest.approx(0.25)

    assert result[18] == 1
    assert result[19] == pytest.approx(0.5)
    assert result[20] == 1
    assert result[21] == 2
    assert result[22] == pytest.approx(0.5)