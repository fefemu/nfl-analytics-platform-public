"""Tests for player snap-share history."""

from datetime import date
from pathlib import Path

import duckdb
import pytest

from src.analytics.build_player_snap_share_history import (
    TARGET_FULL_NAME,
    build_player_snap_share_history,
    count_source_rows,
    create_player_snap_share_history,
    validate_source_table,
    validate_target_table,
)


def create_source_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create deterministic player-game snap history."""

    connection.execute(
        """
        CREATE SCHEMA processed;

        CREATE TABLE processed.player_game_snap_counts AS
        SELECT
            source.game_id,
            source.season,
            'REG'::VARCHAR AS game_type,
            source.week,
            source.gameday,
            source.team,
            source.opponent,
            source.is_home,

            source.player_key,
            source.player_key AS gsis_id,
            source.pfr_player_id,
            source.espn_id,
            source.player_name,

            source.position AS source_position,
            source.position AS directory_position,
            source.position_group
                AS directory_position_group,

            source.offense_snap_share,
            source.defense_snap_share,
            source.special_teams_snap_share,
            source.total_snaps,

            TRUE AS has_player_directory_match,
            'GSIS'::VARCHAR
                AS player_identifier_source

        FROM (
            VALUES
                (
                    '2025_01_NE_MIA',
                    2025,
                    1,
                    DATE '2025-09-07',
                    'NE',
                    'MIA',
                    TRUE,
                    '00-0000001',
                    'PlayerA00',
                    '1000001',
                    'Player A',
                    'WR',
                    'WR',
                    0.10,
                    0.00,
                    0.10,
                    10.0
                ),
                (
                    '2025_02_NE_BUF',
                    2025,
                    2,
                    DATE '2025-09-14',
                    'NE',
                    'BUF',
                    TRUE,
                    '00-0000001',
                    'PlayerA00',
                    '1000001',
                    'Player A',
                    'WR',
                    'WR',
                    0.20,
                    0.00,
                    0.20,
                    20.0
                ),
                (
                    '2025_03_NYJ_NE',
                    2025,
                    3,
                    DATE '2025-09-21',
                    'NE',
                    'NYJ',
                    FALSE,
                    '00-0000001',
                    'PlayerA00',
                    '1000001',
                    'Player A',
                    'WR',
                    'WR',
                    0.30,
                    0.00,
                    0.30,
                    30.0
                ),
                (
                    '2025_04_NE_CAR',
                    2025,
                    4,
                    DATE '2025-09-28',
                    'NE',
                    'CAR',
                    TRUE,
                    '00-0000001',
                    'PlayerA00',
                    '1000001',
                    'Player A',
                    'WR',
                    'WR',
                    0.40,
                    0.00,
                    0.40,
                    40.0
                ),
                (
                    '2025_05_NE_PIT',
                    2025,
                    5,
                    DATE '2025-10-05',
                    'NE',
                    'PIT',
                    TRUE,
                    '00-0000001',
                    'PlayerA00',
                    '1000001',
                    'Player A',
                    'WR',
                    'WR',
                    0.50,
                    0.00,
                    0.50,
                    50.0
                ),
                (
                    '2025_06_BUF_ATL',
                    2025,
                    6,
                    DATE '2025-10-12',
                    'BUF',
                    'ATL',
                    TRUE,
                    '00-0000001',
                    'PlayerA00',
                    '1000001',
                    'Player A',
                    'WR',
                    'WR',
                    0.90,
                    0.00,
                    0.90,
                    90.0
                ),
                (
                    '2025_01_NE_MIA',
                    2025,
                    1,
                    DATE '2025-09-07',
                    'NE',
                    'MIA',
                    TRUE,
                    '00-0000002',
                    'PlayerB00',
                    '1000002',
                    'Player B',
                    'CB',
                    'DB',
                    0.00,
                    0.80,
                    0.20,
                    70.0
                )
        ) AS source(
            game_id,
            season,
            week,
            gameday,
            team,
            opponent,
            is_home,
            player_key,
            pfr_player_id,
            espn_id,
            player_name,
            position,
            position_group,
            offense_snap_share,
            defense_snap_share,
            special_teams_snap_share,
            total_snaps
        )
        """
    )


def test_validate_source_table_accepts_valid_source() -> None:
    """Accept the processed player-game snap source."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )

        validate_source_table(
            connection
        )


def test_validate_source_table_rejects_missing_table() -> None:
    """Reject a missing processed snap source."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        with pytest.raises(
            RuntimeError,
            match="Source table does not exist",
        ):
            validate_source_table(
                connection
            )


def test_validate_source_table_rejects_missing_column() -> None:
    """Reject a source missing a required column."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
                processed.player_game_snap_counts AS
            SELECT * EXCLUDE(offense_snap_share)
            FROM processed.player_game_snap_counts
            """
        )

        with pytest.raises(
            RuntimeError,
            match="missing columns: offense_snap_share",
        ):
            validate_source_table(
                connection
            )


def test_count_source_rows_counts_every_appearance() -> None:
    """Count each player-team-game appearance."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )

        assert count_source_rows(
            connection
        ) == 7


def test_create_history_preserves_source_grain() -> None:
    """Create one history row per source appearance."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )

        create_player_snap_share_history(
            connection
        )

        row_count = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM {TARGET_FULL_NAME}
            """
        ).fetchone()[0]

    assert row_count == 7


def test_create_history_calculates_last_four() -> None:
    """Calculate the rolling four-appearance average."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )
        create_player_snap_share_history(
            connection
        )

        result = connection.execute(
            f"""
            SELECT
                career_appearance_count,
                team_appearance_count,
                career_games_last_4,
                team_games_last_4,
                career_offense_snap_share_last_4,
                team_offense_snap_share_last_4
            FROM {TARGET_FULL_NAME}
            WHERE player_key = '00-0000001'
              AND game_id = '2025_05_NE_PIT'
            """
        ).fetchone()

    assert result[:4] == (
        5,
        5,
        4,
        4,
    )
    assert result[4] == pytest.approx(
        0.35
    )
    assert result[5] == pytest.approx(
        0.35
    )


def test_create_history_separates_team_and_career_windows(
) -> None:
    """Continue career history but restart team history."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )
        create_player_snap_share_history(
            connection
        )

        result = connection.execute(
            f"""
            SELECT
                career_appearance_count,
                team_appearance_count,
                career_games_last_4,
                team_games_last_4,
                career_offense_snap_share_last_4,
                team_offense_snap_share_last_4,
                previous_career_game_id,
                previous_team_game_id,
                days_since_previous_career_appearance,
                days_since_previous_team_appearance
            FROM {TARGET_FULL_NAME}
            WHERE player_key = '00-0000001'
              AND game_id = '2025_06_BUF_ATL'
            """
        ).fetchone()

    assert result[:4] == (
        6,
        1,
        4,
        1,
    )
    assert result[4] == pytest.approx(
        0.525
    )
    assert result[5] == pytest.approx(
        0.90
    )
    assert result[6:] == (
        "2025_05_NE_PIT",
        None,
        7,
        None,
    )


def test_create_history_calculates_long_window_count() -> None:
    """Use all available rows before reaching eight games."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )
        create_player_snap_share_history(
            connection
        )

        result = connection.execute(
            f"""
            SELECT
                career_games_last_8,
                team_games_last_8
            FROM {TARGET_FULL_NAME}
            WHERE player_key = '00-0000001'
              AND game_id = '2025_06_BUF_ATL'
            """
        ).fetchone()

    assert result == (
        6,
        1,
    )


def test_create_history_sets_availability_date() -> None:
    """Expose history only after the represented game date."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )
        create_player_snap_share_history(
            connection
        )

        result = connection.execute(
            f"""
            SELECT
                gameday,
                available_after_gameday
            FROM {TARGET_FULL_NAME}
            WHERE player_key = '00-0000001'
              AND game_id = '2025_06_BUF_ATL'
            """
        ).fetchone()

    assert result == (
        date(
            2025,
            10,
            12,
        ),
        date(
            2025,
            10,
            12,
        ),
    )


def test_validate_target_table_accepts_valid_history() -> None:
    """Accept valid rolling player history."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )
        create_player_snap_share_history(
            connection
        )

        result = validate_target_table(
            connection=connection,
            expected_row_count=7,
        )

    assert result == (
        7,
        2,
        3,
        1,
    )


def test_validate_target_table_rejects_row_mismatch() -> None:
    """Reject a history row-count mismatch."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )
        create_player_snap_share_history(
            connection
        )

        with pytest.raises(
            RuntimeError,
            match="does not match its source",
        ):
            validate_target_table(
                connection=connection,
                expected_row_count=8,
            )


def test_validate_target_table_rejects_duplicate_key() -> None:
    """Reject duplicate player-team-game history rows."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )
        create_player_snap_share_history(
            connection
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
            match="duplicate business keys",
        ):
            validate_target_table(
                connection=connection,
                expected_row_count=8,
            )


def test_validate_target_table_rejects_invalid_window() -> None:
    """Reject an invalid rolling window count."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )
        create_player_snap_share_history(
            connection
        )

        connection.execute(
            f"""
            UPDATE {TARGET_FULL_NAME}
            SET career_games_last_4 = 5
            WHERE player_key = '00-0000001'
              AND game_id = '2025_05_NE_PIT'
            """
        )

        with pytest.raises(
            RuntimeError,
            match="invalid rolling windows",
        ):
            validate_target_table(
                connection=connection,
                expected_row_count=7,
            )


def test_validate_target_table_rejects_invalid_share() -> None:
    """Reject a rolling share outside zero to one."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )
        create_player_snap_share_history(
            connection
        )

        connection.execute(
            f"""
            UPDATE {TARGET_FULL_NAME}
            SET team_offense_snap_share_last_4 = 1.01
            WHERE player_key = '00-0000001'
              AND game_id = '2025_05_NE_PIT'
            """
        )

        with pytest.raises(
            RuntimeError,
            match="outside the range 0 to 1",
        ):
            validate_target_table(
                connection=connection,
                expected_row_count=7,
            )


def test_build_player_snap_share_history_creates_table(
    tmp_path: Path,
) -> None:
    """Run the complete player history workflow."""

    database_file = (
        tmp_path
        / "test.duckdb"
    )

    with duckdb.connect(
        str(database_file)
    ) as connection:
        create_source_table(
            connection
        )

    build_player_snap_share_history(
        database_file=database_file
    )

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        result = connection.execute(
            f"""
            SELECT
                COUNT(*),
                MAX(career_appearance_count),
                MAX(team_appearance_count)
            FROM {TARGET_FULL_NAME}
            """
        ).fetchone()

    assert result == (
        7,
        6,
        5,
    )