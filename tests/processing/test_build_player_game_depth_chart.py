"""Tests for the unified player-game depth-chart builder."""

from pathlib import Path

import duckdb
import pytest

from src.processing.build_player_game_depth_chart import (
    build_player_game_depth_chart,
    count_source_rows,
    create_player_game_depth_chart,
    validate_database_file,
    validate_source_tables,
    validate_target_table,
)


def create_source_tables(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create representative processed source tables."""

    connection.execute(
        """
        CREATE SCHEMA processed;

        CREATE TABLE
            processed.player_game_depth_chart_legacy
        AS
        SELECT *
        FROM (
            VALUES
                (
                    '2024_01_BUF_NE',
                    2024,
                    'REG',
                    1,
                    DATE '2024-09-08',
                    'NE',
                    'BUF',
                    TRUE,
                    '00-0000001',
                    '00-0000001',
                    NULL,
                    'Legacy Starter',
                    'QB',
                    'Offense',
                    'QB',
                    1,
                    TRUE,
                    FALSE,
                    FALSE,
                    2,
                    1,
                    FALSE,
                    'legacy_nfl',
                    NULL::TIMESTAMPTZ
                )
        ) AS legacy(
            game_id,
            season,
            game_type,
            week,
            gameday,
            team,
            opponent,
            is_home,
            player_key,
            gsis_id,
            espn_id,
            player_name,
            player_position,
            formation,
            depth_position,
            depth_rank,
            is_starter,
            is_primary_backup,
            is_reserve,
            source_record_count,
            source_rank_count,
            has_conflicting_ranks,
            source_generation,
            source_snapshot_at
        );

        CREATE TABLE
            processed.player_game_depth_chart_espn
        AS
        SELECT *
        FROM (
            VALUES
                (
                    '2025_01_MIA_BUF',
                    2025,
                    'REG',
                    1,
                    DATE '2025-09-07',
                    'BUF',
                    'MIA',
                    TRUE,
                    '00-0000002',
                    '00-0000002',
                    '1000002',
                    'ESPN Backup',
                    'WR',
                    'Offense',
                    'Wide Receiver',
                    2,
                    2,
                    FALSE,
                    TRUE,
                    FALSE,
                    1,
                    1,
                    FALSE,
                    'espn',
                    TIMESTAMPTZ
                        '2025-09-07 10:00:00+00'
                ),
                (
                    '2025_01_MIA_BUF',
                    2025,
                    'REG',
                    1,
                    DATE '2025-09-07',
                    'BUF',
                    'MIA',
                    TRUE,
                    'ESPN:1000003',
                    NULL,
                    '1000003',
                    'ESPN Reserve',
                    'CB',
                    'Defense',
                    'Left Cornerback',
                    1,
                    3,
                    FALSE,
                    FALSE,
                    TRUE,
                    1,
                    1,
                    FALSE,
                    'espn',
                    TIMESTAMPTZ
                        '2025-09-07 10:00:00+00'
                )
        ) AS espn(
            game_id,
            season,
            game_type,
            week,
            gameday,
            team,
            opponent,
            is_home,
            player_key,
            gsis_id,
            espn_id,
            player_name,
            player_position,
            formation,
            depth_position,
            pos_slot,
            depth_rank,
            is_starter,
            is_primary_backup,
            is_reserve,
            source_record_count,
            source_rank_count,
            has_conflicting_ranks,
            source_generation,
            source_snapshot_at
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
    """Reject missing processed source tables."""

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


def test_count_source_rows_sums_both_generations() -> None:
    """Count all legacy and ESPN role rows."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )

        result = count_source_rows(
            connection
        )

    assert result == 3


def test_create_unified_depth_chart_combines_sources() -> None:
    """Combine legacy and ESPN rows without loss."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )

        create_player_game_depth_chart(
            connection
        )

        rows = connection.execute(
            """
            SELECT
                season,
                player_key,
                source_generation,
                position_slot
            FROM processed.player_game_depth_chart
            ORDER BY
                season,
                player_key
            """
        ).fetchall()

    assert rows == [
        (
            2024,
            "00-0000001",
            "legacy_nfl",
            None,
        ),
        (
            2025,
            "00-0000002",
            "espn",
            2,
        ),
        (
            2025,
            "ESPN:1000003",
            "espn",
            1,
        ),
    ]


def test_create_unified_depth_chart_derives_tiers() -> None:
    """Derive stable starter, backup and reserve tiers."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )

        create_player_game_depth_chart(
            connection
        )

        rows = connection.execute(
            """
            SELECT
                player_key,
                depth_tier
            FROM processed.player_game_depth_chart
            ORDER BY player_key
            """
        ).fetchall()

    assert rows == [
        (
            "00-0000001",
            "STARTER",
        ),
        (
            "00-0000002",
            "PRIMARY_BACKUP",
        ),
        (
            "ESPN:1000003",
            "RESERVE",
        ),
    ]


def test_create_unified_depth_chart_derives_identifier_fields(
) -> None:
    """Identify GSIS and ESPN-fallback player keys."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )

        create_player_game_depth_chart(
            connection
        )

        rows = connection.execute(
            """
            SELECT
                player_key,
                has_gsis_id,
                player_identifier_source
            FROM processed.player_game_depth_chart
            ORDER BY player_key
            """
        ).fetchall()

    assert rows == [
        (
            "00-0000001",
            True,
            "GSIS",
        ),
        (
            "00-0000002",
            True,
            "GSIS",
        ),
        (
            "ESPN:1000003",
            False,
            "ESPN",
        ),
    ]


def test_create_unified_depth_chart_derives_role_flags() -> None:
    """Derive offense, defense and timestamp flags."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )

        create_player_game_depth_chart(
            connection
        )

        rows = connection.execute(
            """
            SELECT
                player_key,
                is_offense_role,
                is_defense_role,
                is_special_teams_role,
                has_timestamped_snapshot
            FROM processed.player_game_depth_chart
            ORDER BY player_key
            """
        ).fetchall()

    assert rows == [
        (
            "00-0000001",
            True,
            False,
            False,
            False,
        ),
        (
            "00-0000002",
            True,
            False,
            False,
            True,
        ),
        (
            "ESPN:1000003",
            False,
            True,
            False,
            True,
        ),
    ]


def test_validate_target_table_accepts_valid_result() -> None:
    """Validate the complete unified source union."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )

        expected_row_count = count_source_rows(
            connection
        )

        create_player_game_depth_chart(
            connection
        )

        result = validate_target_table(
            connection=connection,
            expected_row_count=expected_row_count,
        )

    assert result == (
        3,
        2,
        1,
        1,
    )


def test_validate_target_table_rejects_row_loss() -> None:
    """Reject a unified table that lost a source row."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )

        create_player_game_depth_chart(
            connection
        )

        with pytest.raises(
            RuntimeError,
            match="does not match its processed sources",
        ):
            validate_target_table(
                connection=connection,
                expected_row_count=4,
            )


def test_build_player_game_depth_chart(
    tmp_path: Path,
) -> None:
    """Run the complete unified builder."""

    database_file = tmp_path / "test.duckdb"

    with duckdb.connect(
        str(database_file)
    ) as connection:
        create_source_tables(
            connection
        )

    build_player_game_depth_chart(
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
                COUNT(
                    DISTINCT
                    game_id || team
                ),
                SUM(
                    CASE
                        WHEN is_starter THEN 1
                        ELSE 0
                    END
                )
            FROM processed.player_game_depth_chart
            """
        ).fetchone()

    assert result == (
        3,
        2,
        1,
    )