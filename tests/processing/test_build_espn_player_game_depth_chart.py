"""Tests for the ESPN player-game depth-chart builder."""

from pathlib import Path

import duckdb
import pytest

from src.processing.build_espn_player_game_depth_chart import (
    build_espn_player_game_depth_chart,
    count_scheduled_team_games,
    create_espn_player_game_depth_chart,
    validate_database_file,
    validate_source_tables,
    validate_target_table,
)


def create_source_tables(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create representative ESPN and schedule sources."""

    connection.execute(
        """
        CREATE SCHEMA raw;
        CREATE SCHEMA processed;

        CREATE TABLE processed.schedule AS
        SELECT *
        FROM (
            VALUES
                (
                    '2025_01_BUF_NE',
                    2025,
                    'REG',
                    1,
                    DATE '2025-09-07',
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

        CREATE TABLE raw.depth_charts_espn AS
        SELECT *
        FROM (
            VALUES
                (
                    2025,
                    '2025-08-03T10:00:00Z',
                    'NE',
                    'Test Quarterback',
                    '1000001',
                    '00-0000001',
                    '1',
                    '3WR 1TE',
                    '1',
                    'Quarterback',
                    'QB',
                    1,
                    2,
                    'espn_2025'
                ),
                (
                    2025,
                    '2025-09-07T10:00:00Z',
                    'NE',
                    'Test Quarterback',
                    '1000001',
                    '00-0000001',
                    '1',
                    '3WR 1TE',
                    '1',
                    'Quarterback',
                    'QB',
                    1,
                    1,
                    'espn_2025'
                ),
                (
                    2025,
                    '2025-09-07T10:00:00Z',
                    'NE',
                    'Test Receiver',
                    '1000002',
                    NULL,
                    '1',
                    '3WR 1TE',
                    '2',
                    'Wide Receiver',
                    'WR',
                    2,
                    1,
                    'espn_2025'
                ),
                (
                    2025,
                    '2025-09-08T10:00:00Z',
                    'NE',
                    'Future Player',
                    '1000003',
                    '00-0000003',
                    '1',
                    '3WR 1TE',
                    '3',
                    'Running Back',
                    'RB',
                    3,
                    1,
                    'espn_2025'
                ),
                (
                    2025,
                    '2025-09-06T10:00:00Z',
                    'BUF',
                    'Test Cornerback',
                    '1000004',
                    '00-0000004',
                    '2',
                    'Base 4-3 D',
                    '4',
                    'Left Cornerback',
                    'LCB',
                    1,
                    1,
                    'espn_2025'
                )
        ) AS depth_chart(
            source_season,
            dt,
            team,
            player_name,
            espn_id,
            gsis_id,
            pos_grp_id,
            pos_grp,
            pos_id,
            pos_name,
            pos_abb,
            pos_slot,
            pos_rank,
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
    """Reject missing ESPN or schedule source tables."""

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
    """Count one team-game for both sides of a game."""

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


def test_create_espn_depth_chart_selects_latest_snapshot() -> None:
    """Select the latest snapshot no later than game day."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )

        create_espn_player_game_depth_chart(
            connection
        )

        result = connection.execute(
            """
            SELECT
                depth_rank,
                is_starter,
                source_snapshot_at =
                    TIMESTAMPTZ
                    '2025-09-07 10:00:00+00'
                    AS expected_snapshot
            FROM processed.player_game_depth_chart_espn
            WHERE player_key = '00-0000001'
            """
        ).fetchone()

    assert result == (
        1,
        True,
        True,
    )


def test_create_espn_depth_chart_excludes_future_snapshot() -> None:
    """Do not use a snapshot after the scheduled game date."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )

        create_espn_player_game_depth_chart(
            connection
        )

        future_player_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM processed.player_game_depth_chart_espn
            WHERE espn_id = '1000003'
            """
        ).fetchone()[0]

    assert future_player_count == 0


def test_create_espn_depth_chart_uses_espn_fallback_key() -> None:
    """Use ESPN ID when a GSIS identifier is unavailable."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )

        create_espn_player_game_depth_chart(
            connection
        )

        result = connection.execute(
            """
            SELECT
                player_key,
                gsis_id,
                espn_id
            FROM processed.player_game_depth_chart_espn
            WHERE espn_id = '1000002'
            """
        ).fetchone()

    assert result == (
        "ESPN:1000002",
        None,
        "1000002",
    )


def test_create_espn_depth_chart_maps_game_and_formation(
) -> None:
    """Map schedule context and ESPN formation groups."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )

        create_espn_player_game_depth_chart(
            connection
        )

        rows = connection.execute(
            """
            SELECT
                player_key,
                team,
                opponent,
                is_home,
                formation
            FROM processed.player_game_depth_chart_espn
            ORDER BY player_key
            """
        ).fetchall()

    assert rows == [
        (
            "00-0000001",
            "NE",
            "BUF",
            True,
            "Offense",
        ),
        (
            "00-0000004",
            "BUF",
            "NE",
            False,
            "Defense",
        ),
        (
            "ESPN:1000002",
            "NE",
            "BUF",
            True,
            "Offense",
        ),
    ]


def test_validate_target_table_accepts_valid_result() -> None:
    """Validate unique roles and full team-game coverage."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )

        create_espn_player_game_depth_chart(
            connection
        )

        result = validate_target_table(
            connection
        )

    assert result == (
        3,
        2,
        1,
        0,
    )


def test_validate_target_table_rejects_missing_team_snapshot(
) -> None:
    """Reject a scheduled team without a usable snapshot."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )

        connection.execute(
            """
            DELETE FROM raw.depth_charts_espn
            WHERE team = 'BUF'
            """
        )

        create_espn_player_game_depth_chart(
            connection
        )

        with pytest.raises(
            RuntimeError,
            match="coverage does not match",
        ):
            validate_target_table(
                connection
            )


def test_build_espn_player_game_depth_chart(
    tmp_path: Path,
) -> None:
    """Run the complete ESPN builder."""

    database_file = tmp_path / "test.duckdb"

    with duckdb.connect(
        str(database_file)
    ) as connection:
        create_source_tables(
            connection
        )

    build_espn_player_game_depth_chart(
        database_file
    )

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        result = connection.execute(
            """
            SELECT
                COUNT(*),
                COUNT(DISTINCT game_id || team),
                SUM(
                    CASE
                        WHEN gsis_id IS NULL THEN 1
                        ELSE 0
                    END
                )
            FROM processed.player_game_depth_chart_espn
            """
        ).fetchone()

    assert result == (
        3,
        2,
        1,
    )