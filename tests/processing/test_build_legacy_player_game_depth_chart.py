"""Tests for the legacy player-game depth-chart builder."""

from pathlib import Path

import duckdb
import pytest

from src.processing.build_legacy_player_game_depth_chart import (
    build_legacy_player_game_depth_chart,
    count_scheduled_team_games,
    create_legacy_player_game_depth_chart,
    validate_database_file,
    validate_source_tables,
    validate_target_table,
)


def create_source_tables(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create representative legacy and schedule sources."""

    connection.execute(
        """
        CREATE SCHEMA raw;
        CREATE SCHEMA processed;

        CREATE TABLE processed.schedule AS
        SELECT *
        FROM (
            VALUES
                (
                    '2024_01_BUF_NE',
                    2024,
                    'REG',
                    1,
                    DATE '2024-09-08',
                    'BUF',
                    'NE'
                )
        ) AS schedule(
            game_id,
            season,
            game_type,
            week,
            gameday,
            away_team,
            home_team
        );

        CREATE TABLE raw.depth_charts_legacy AS
        SELECT *
        FROM (
            VALUES
                (
                    2024,
                    'NE',
                    1,
                    'REG',
                    '2',
                    'Starter',
                    'Test',
                    'Test',
                    'Offense',
                    '00-0000001',
                    '10',
                    'QB',
                    'STA000001',
                    'QB',
                    'Test Starter',
                    'legacy_file'
                ),
                (
                    2024,
                    'NE',
                    1,
                    'REG',
                    '1',
                    'Starter',
                    'Test',
                    'Test',
                    'Offense',
                    '00-0000001',
                    '10',
                    'QB',
                    'STA000001',
                    'QB',
                    'Test Starter',
                    'legacy_file'
                ),
                (
                    2024,
                    'NE',
                    1,
                    'REG',
                    '1',
                    'Starter',
                    'Test',
                    'Test',
                    'Offense',
                    '00-0000001',
                    '10',
                    'QB',
                    'STA000001',
                    'QB',
                    'Test Starter',
                    'legacy_file'
                ),
                (
                    2024,
                    'BUF',
                    1,
                    'REG',
                    '1',
                    'Opponent',
                    'Test',
                    'Test',
                    'Defense',
                    '00-0000002',
                    '20',
                    'CB',
                    'OPP000002',
                    'LCB',
                    'Test Opponent',
                    'legacy_file'
                ),
                (
                    2024,
                    'NE',
                    NULL,
                    'SBBYE',
                    '1',
                    'Bye',
                    'Test',
                    'Test',
                    'Offense',
                    '00-0000003',
                    '30',
                    'WR',
                    'BYE000003',
                    'WR',
                    'Test Bye Player',
                    'legacy_file'
                )
        ) AS depth_chart(
            season,
            club_code,
            week,
            game_type,
            depth_team,
            last_name,
            first_name,
            football_name,
            formation,
            gsis_id,
            jersey_number,
            position,
            elias_id,
            depth_position,
            full_name,
            source_file
        );
        """
    )


def test_validate_database_file_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """Reject a missing DuckDB database."""

    with pytest.raises(
        FileNotFoundError,
        match="Database file does not exist",
    ):
        validate_database_file(
            tmp_path / "missing.duckdb"
        )


def test_validate_source_tables_rejects_missing_table() -> None:
    """Reject missing source tables."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        with pytest.raises(
            RuntimeError,
            match="Source table does not exist",
        ):
            validate_source_tables(
                connection
            )


def test_count_scheduled_team_games_counts_both_sides() -> None:
    """Count one team-game for each side of a game."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )

        result = count_scheduled_team_games(
            connection
        )

    assert result == 2


def test_create_legacy_depth_chart_selects_best_rank() -> None:
    """Use the smallest rank across conflicting source rows."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )

        create_legacy_player_game_depth_chart(
            connection
        )

        result = connection.execute(
            """
            SELECT
                depth_rank,
                is_starter,
                is_primary_backup,
                is_reserve,
                source_record_count,
                source_rank_count,
                has_conflicting_ranks
            FROM processed.player_game_depth_chart_legacy
            WHERE player_key = '00-0000001'
            """
        ).fetchone()

    assert result == (
        1,
        True,
        False,
        False,
        3,
        2,
        True,
    )


def test_create_legacy_depth_chart_maps_game_context() -> None:
    """Map team, opponent and home status from schedule."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )

        create_legacy_player_game_depth_chart(
            connection
        )

        rows = connection.execute(
            """
            SELECT
                player_key,
                game_id,
                team,
                opponent,
                is_home
            FROM processed.player_game_depth_chart_legacy
            ORDER BY player_key
            """
        ).fetchall()

    assert rows == [
        (
            "00-0000001",
            "2024_01_BUF_NE",
            "NE",
            "BUF",
            True,
        ),
        (
            "00-0000002",
            "2024_01_BUF_NE",
            "BUF",
            "NE",
            False,
        ),
    ]


def test_create_legacy_depth_chart_excludes_unplayed_rows() -> None:
    """Exclude SBBYE records without a scheduled game."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )

        create_legacy_player_game_depth_chart(
            connection
        )

        bye_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM processed.player_game_depth_chart_legacy
            WHERE player_key = '00-0000003'
            """
        ).fetchone()[0]

    assert bye_count == 0


def test_validate_target_table_accepts_valid_result() -> None:
    """Validate unique roles and complete team-game coverage."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )

        create_legacy_player_game_depth_chart(
            connection
        )

        result = validate_target_table(
            connection
        )

    assert result == (
        2,
        2,
        1,
    )


def test_validate_target_table_rejects_missing_team_game() -> None:
    """Reject incomplete scheduled team-game coverage."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )

        connection.execute(
            """
            DELETE FROM raw.depth_charts_legacy
            WHERE club_code = 'BUF'
            """
        )

        create_legacy_player_game_depth_chart(
            connection
        )

        with pytest.raises(
            RuntimeError,
            match="coverage does not match",
        ):
            validate_target_table(
                connection
            )


def test_build_legacy_player_game_depth_chart(
    tmp_path: Path,
) -> None:
    """Run the complete legacy builder."""

    database_file = tmp_path / "test.duckdb"

    with duckdb.connect(
        str(database_file)
    ) as connection:
        create_source_tables(
            connection
        )

    build_legacy_player_game_depth_chart(
        database_file
    )

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        rows = connection.execute(
            """
            SELECT
                player_key,
                depth_rank,
                source_generation
            FROM processed.player_game_depth_chart_legacy
            ORDER BY player_key
            """
        ).fetchall()

    assert rows == [
        (
            "00-0000001",
            1,
            "legacy_nfl",
        ),
        (
            "00-0000002",
            1,
            "legacy_nfl",
        ),
    ]