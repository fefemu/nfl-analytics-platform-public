"""Tests for the team-game efficiency builder."""

from pathlib import Path

import duckdb
import pytest

from src.processing.build_team_game_efficiency import (
    build_parquet_source,
    create_team_game_offense,
    create_valid_offensive_plays,
    get_pbp_files,
    validate_pbp_columns,
)


def test_get_pbp_files_rejects_missing_directory(
    tmp_path: Path,
) -> None:
    """Fail when the configured PBP directory does not exist."""

    missing_directory = tmp_path / "missing"

    with pytest.raises(
        FileNotFoundError,
        match="PBP directory does not exist",
    ):
        get_pbp_files(missing_directory)


def test_get_pbp_files_rejects_empty_directory(
    tmp_path: Path,
) -> None:
    """Fail when no season-level PBP files are available."""

    with pytest.raises(
        FileNotFoundError,
        match="No PBP Parquet files found",
    ):
        get_pbp_files(tmp_path)


def test_get_pbp_files_returns_sorted_parquet_files(
    tmp_path: Path,
) -> None:
    """Return only matching PBP files in deterministic order."""

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


def test_build_parquet_source_preserves_file_order(
    tmp_path: Path,
) -> None:
    """Create a DuckDB file-list expression in input order."""

    first_file = tmp_path / "pbp_2024.parquet"
    second_file = tmp_path / "pbp_2025.parquet"

    result = build_parquet_source(
        [first_file, second_file]
    )

    first_path = str(first_file.resolve())
    second_path = str(second_file.resolve())

    assert result.startswith("[")
    assert result.endswith("]")
    assert first_path in result
    assert second_path in result
    assert result.index(first_path) < result.index(second_path)


def test_validate_pbp_columns_rejects_missing_columns(
    tmp_path: Path,
) -> None:
    """Fail when a PBP file lacks required feature columns."""

    parquet_file = tmp_path / "pbp_2025.parquet"
    escaped_path = str(parquet_file).replace("'", "''")

    with duckdb.connect(":memory:") as connection:
        connection.execute(
            f"""
            COPY (
                SELECT
                    '2025_01_TEST'::VARCHAR AS game_id
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
            match="Missing required PBP columns",
        ):
            validate_pbp_columns(
                connection,
                parquet_source,
            )


def test_create_team_game_offense_calculates_metrics() -> None:
    """Aggregate play-level inputs into team-game metrics."""

    with duckdb.connect(":memory:") as connection:
        connection.execute(
            """
            CREATE TEMP TABLE valid_offensive_plays AS
            SELECT *
            FROM (
                VALUES
                    (
                        'game_1', 2025, 'REG', 1,
                        DATE '2025-09-07',
                        'BUF', 'MIA', 'BUF', 'MIA', TRUE,
                        0.5, 1.0,
                        TRUE, TRUE, FALSE,
                        TRUE, FALSE, TRUE,
                        0, 0, 0
                    ),
                    (
                        'game_1', 2025, 'REG', 1,
                        DATE '2025-09-07',
                        'BUF', 'MIA', 'BUF', 'MIA', TRUE,
                        -1.0, 0.0,
                        FALSE, TRUE, FALSE,
                        TRUE, FALSE, FALSE,
                        1, 0, 0
                    ),
                    (
                        'game_1', 2025, 'REG', 1,
                        DATE '2025-09-07',
                        'BUF', 'MIA', 'BUF', 'MIA', TRUE,
                        0.2, 1.0,
                        TRUE, FALSE, TRUE,
                        FALSE, TRUE, TRUE,
                        0, 0, 1
                    ),
                    (
                        'game_1', 2025, 'REG', 1,
                        DATE '2025-09-07',
                        'BUF', 'MIA', 'BUF', 'MIA', TRUE,
                        -0.1, 0.0,
                        TRUE, FALSE, TRUE,
                        TRUE, FALSE, FALSE,
                        0, 0, 0
                    )
            ) AS plays(
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
                epa,
                success,
                is_competitive_play,
                is_dropback,
                is_designed_rush,
                is_early_down_play,
                is_red_zone_play,
                is_explosive_play,
                sack,
                interception,
                fumble_lost
            )
            """
        )

        create_team_game_offense(connection)

        result = connection.execute(
            """
            SELECT
                offensive_plays,
                dropbacks,
                designed_rushes,
                competitive_plays,
                early_down_plays,
                red_zone_plays,
                offensive_epa_per_play,
                competitive_epa_per_play,
                dropback_epa_per_play,
                designed_rush_epa_per_play,
                early_down_epa_per_play,
                success_rate,
                explosive_plays,
                explosive_play_rate,
                sacks_allowed,
                sack_rate,
                turnovers,
                turnover_rate
            FROM team_game_offense
            """
        ).fetchone()

    assert result[0] == 4
    assert result[1] == 2
    assert result[2] == 2
    assert result[3] == 3
    assert result[4] == 3
    assert result[5] == 1

    assert result[6] == pytest.approx(-0.1)
    assert result[7] == pytest.approx(0.2)
    assert result[8] == pytest.approx(-0.25)
    assert result[9] == pytest.approx(0.05)
    assert result[10] == pytest.approx(-0.2)
    assert result[11] == pytest.approx(0.5)

    assert result[12] == 2
    assert result[13] == pytest.approx(0.5)
    assert result[14] == 1
    assert result[15] == pytest.approx(0.5)
    assert result[16] == 1
    assert result[17] == pytest.approx(0.25)


def test_create_valid_offensive_plays_applies_filters(
    tmp_path: Path,
) -> None:
    """Keep valid dropbacks and designed rushes only."""

    parquet_file = tmp_path / "pbp_2025.parquet"
    escaped_path = str(parquet_file).replace("'", "''")

    with duckdb.connect(":memory:") as connection:
        connection.execute(
            f"""
            COPY (
                SELECT *
                FROM (
                    VALUES
                        (
                            'game_1', 2025, 'REG', 1,
                            DATE '2025-09-07',
                            'BUF', 'MIA', 'BUF', 'MIA',
                            1, 75.0, 0.50, 'pass',
                            1, 0, 0, 0, 0, 0,
                            25.0, 0.50, 1.0,
                            0, 0, 0, 0, 0
                        ),
                        (
                            'game_1', 2025, 'REG', 1,
                            DATE '2025-09-07',
                            'BUF', 'MIA', 'BUF', 'MIA',
                            2, 60.0, 0.50, 'run',
                            0, 0, 1, 0, 0, 0,
                            12.0, 0.20, 1.0,
                            0, 0, 0, 0, 0
                        ),
                        (
                            'game_1', 2025, 'REG', 1,
                            DATE '2025-09-07',
                            'BUF', 'MIA', 'BUF', 'MIA',
                            3, 50.0, 0.50, 'run',
                            0, 0, 1, 0, 0, 0,
                            -1.0, -0.10, 0.0,
                            1, 0, 0, 0, 0
                        ),
                        (
                            'game_1', 2025, 'REG', 1,
                            DATE '2025-09-07',
                            'BUF', 'MIA', 'BUF', 'MIA',
                            1, 50.0, 0.50, 'pass',
                            1, 0, 0, 0, 0, 0,
                            0.0, 0.00, 0.0,
                            0, 1, 0, 0, 0
                        ),
                        (
                            'game_1', 2025, 'REG', 1,
                            DATE '2025-09-07',
                            'BUF', 'MIA', 'BUF', 'MIA',
                            1, 2.0, 0.50, 'pass',
                            1, 0, 0, 0, 0, 0,
                            2.0, 0.10, 1.0,
                            0, 0, 0, 1, 0
                        ),
                        (
                            'game_1', 2025, 'REG', 1,
                            DATE '2025-09-07',
                            'BUF', 'MIA', 'BUF', 'MIA',
                            1, 80.0, 0.50, 'kickoff',
                            0, 0, 0, 0, 0, 0,
                            0.0, 0.00, 0.0,
                            0, 0, 0, 0, 1
                        )
                ) AS plays(
                    game_id,
                    season,
                    season_type,
                    week,
                    game_date,
                    posteam,
                    defteam,
                    home_team,
                    away_team,
                    down,
                    yardline_100,
                    wp,
                    play_type,
                    qb_dropback,
                    qb_scramble,
                    rush,
                    sack,
                    interception,
                    fumble_lost,
                    yards_gained,
                    epa,
                    success,
                    qb_kneel,
                    qb_spike,
                    aborted_play,
                    two_point_attempt,
                    special_teams_play
                )
            )
            TO '{escaped_path}'
            (FORMAT PARQUET)
            """
        )

        parquet_source = build_parquet_source(
            [parquet_file]
        )

        create_valid_offensive_plays(
            connection,
            parquet_source,
        )

        result = connection.execute(
            """
            SELECT
                COUNT(*) AS play_count,
                SUM(is_dropback::INTEGER) AS dropbacks,
                SUM(is_designed_rush::INTEGER)
                    AS designed_rushes,
                SUM(is_explosive_play::INTEGER)
                    AS explosive_plays
            FROM valid_offensive_plays
            """
        ).fetchone()

    assert result == (2, 1, 1, 2)