"""Tests for processed player-game snap counts."""

from pathlib import Path
from datetime import date

import duckdb
import pytest

from src.processing.build_player_game_snap_counts import (
    TARGET_FULL_NAME,
    build_player_game_snap_counts,
    count_source_rows,
    create_player_game_snap_counts,
    validate_schedule_coverage,
    validate_source_tables,
    validate_target_table,
)


def create_source_tables(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create valid snap, player and schedule sources."""

    connection.execute(
        """
        CREATE SCHEMA raw;
        CREATE SCHEMA processed;

        CREATE TABLE processed.schedule (
            game_id VARCHAR,
            season INTEGER,
            game_type VARCHAR,
            week INTEGER,
            gameday DATE,
            home_team VARCHAR,
            away_team VARCHAR
        );

        INSERT INTO processed.schedule
        VALUES (
            '2025_01_NE_BUF',
            2025,
            'REG',
            1,
            DATE '2025-09-07',
            'NE',
            'BUF'
        );

        CREATE TABLE raw.player_directory (
            gsis_id VARCHAR,
            pfr_id VARCHAR,
            espn_id VARCHAR,
            display_name VARCHAR,
            position_group VARCHAR,
            position VARCHAR
        );

        INSERT INTO raw.player_directory
        VALUES
            (
                '00-0000001',
                'TestQu00',
                '1000001',
                'Test Quarterback',
                'QB',
                'QB'
            ),
            (
                '00-0000002',
                'TestDe00',
                '1000002',
                'Test Defender',
                'DB',
                'CB'
            );

        CREATE TABLE raw.player_snap_counts (
            game_id VARCHAR,
            pfr_game_id VARCHAR,
            season INTEGER,
            game_type VARCHAR,
            week INTEGER,
            player VARCHAR,
            pfr_player_id VARCHAR,
            position VARCHAR,
            team VARCHAR,
            opponent VARCHAR,
            offense_snaps DOUBLE,
            offense_pct DOUBLE,
            defense_snaps DOUBLE,
            defense_pct DOUBLE,
            st_snaps DOUBLE,
            st_pct DOUBLE,
            source_file VARCHAR
        );

        INSERT INTO raw.player_snap_counts
        VALUES
            (
                '2025_01_NE_BUF',
                '202509070buf',
                2025,
                'REG',
                1,
                'Test Quarterback',
                'TestQu00',
                'QB',
                'NE',
                'BUF',
                65,
                1.0,
                0,
                0.0,
                0,
                0.0,
                'snap_counts_2025.parquet'
            ),
            (
                '2025_01_NE_BUF',
                '202509070buf',
                2025,
                'REG',
                1,
                'Unknown Specialist',
                'UnknSp00',
                'DB',
                'NE',
                'BUF',
                0,
                0.0,
                0,
                0.0,
                22,
                1.01,
                'snap_counts_2025.parquet'
            ),
            (
                '2025_01_NE_BUF',
                '202509070buf',
                2025,
                'REG',
                1,
                'Test Defender',
                'TestDe00',
                'CB',
                'BUF',
                'NE',
                0,
                0.0,
                58,
                0.91,
                4,
                0.18,
                'snap_counts_2025.parquet'
            );
        """
    )


def test_validate_source_tables_accepts_valid_sources() -> None:
    """Accept all required source tables and columns."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )

        validate_source_tables(
            connection
        )


def test_validate_source_tables_rejects_missing_table() -> None:
    """Reject a missing source table."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        connection.execute(
            """
            CREATE SCHEMA raw;
            CREATE SCHEMA processed;
            """
        )

        with pytest.raises(
            RuntimeError,
            match="Source table does not exist",
        ):
            validate_source_tables(
                connection
            )


def test_validate_source_tables_rejects_missing_column() -> None:
    """Reject a source missing a required column."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE raw.player_directory AS
            SELECT * EXCLUDE(espn_id)
            FROM raw.player_directory
            """
        )

        with pytest.raises(
            RuntimeError,
            match="missing columns: espn_id",
        ):
            validate_source_tables(
                connection
            )


def test_count_source_rows_counts_raw_records() -> None:
    """Count every raw snap-count record."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )

        assert count_source_rows(
            connection
        ) == 3


def test_validate_schedule_coverage_accepts_valid_context(
) -> None:
    """Accept valid game, team and opponent assignments."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )

        validate_schedule_coverage(
            connection
        )


def test_validate_schedule_coverage_rejects_missing_game(
) -> None:
    """Reject a snap record without a schedule game."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )

        connection.execute(
            """
            UPDATE raw.player_snap_counts
            SET game_id = '2025_99_UNKNOWN'
            WHERE pfr_player_id = 'UnknSp00'
            """
        )

        with pytest.raises(
            RuntimeError,
            match="have no game",
        ):
            validate_schedule_coverage(
                connection
            )


def test_validate_schedule_coverage_rejects_invalid_opponent(
) -> None:
    """Reject an invalid team-opponent assignment."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )

        connection.execute(
            """
            UPDATE raw.player_snap_counts
            SET opponent = 'MIA'
            WHERE pfr_player_id = 'TestQu00'
            """
        )

        with pytest.raises(
            RuntimeError,
            match="invalid team-opponent assignments",
        ):
            validate_schedule_coverage(
                connection
            )


def test_create_player_game_snap_counts_resolves_identity(
) -> None:
    """Use GSIS identity and explicit PFR fallback."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )

        create_player_game_snap_counts(
            connection
        )

        rows = connection.execute(
            f"""
            SELECT
                pfr_player_id,
                player_key,
                gsis_id,
                player_identifier_source,
                has_player_directory_match
            FROM {TARGET_FULL_NAME}
            ORDER BY pfr_player_id
            """
        ).fetchall()

    assert rows == [
        (
            "TestDe00",
            "00-0000002",
            "00-0000002",
            "GSIS",
            True,
        ),
        (
            "TestQu00",
            "00-0000001",
            "00-0000001",
            "GSIS",
            True,
        ),
        (
            "UnknSp00",
            "PFR:UnknSp00",
            None,
            "PFR",
            False,
        ),
    ]


def test_create_player_game_snap_counts_normalizes_rounding(
) -> None:
    """Preserve source share and cap processed share at one."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )

        create_player_game_snap_counts(
            connection
        )

        result = connection.execute(
            f"""
            SELECT
                source_special_teams_snap_share,
                special_teams_snap_share,
                has_source_rounding_adjustment,
                played_special_teams
            FROM {TARGET_FULL_NAME}
            WHERE pfr_player_id = 'UnknSp00'
            """
        ).fetchone()

    assert result == (
        1.01,
        1.0,
        True,
        True,
    )


def test_create_player_game_snap_counts_derives_context(
) -> None:
    """Derive home/away and participation attributes."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )

        create_player_game_snap_counts(
            connection
        )

        home_row = connection.execute(
            f"""
            SELECT
                season,
                week,
                gameday,
                team,
                opponent,
                is_home,
                played_offense,
                played_defense,
                total_snaps
            FROM {TARGET_FULL_NAME}
            WHERE pfr_player_id = 'TestQu00'
            """
        ).fetchone()

        away_row = connection.execute(
            f"""
            SELECT
                team,
                opponent,
                is_home,
                played_defense
            FROM {TARGET_FULL_NAME}
            WHERE pfr_player_id = 'TestDe00'
            """
        ).fetchone()

    assert home_row == (
        2025,
        1,
        date(
            2025,
            9,
            7,
        ),
        "NE",
        "BUF",
        True,
        True,
        False,
        65.0,
    )
    assert away_row == (
        "BUF",
        "NE",
        False,
        True,
    )


def test_validate_target_table_accepts_valid_result() -> None:
    """Accept the canonical processed snap table."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )
        create_player_game_snap_counts(
            connection
        )

        result = validate_target_table(
            connection=connection,
            expected_row_count=3,
        )

    assert result == (
        3,
        1,
        1,
        1,
    )


def test_validate_target_table_rejects_duplicate_key() -> None:
    """Reject duplicate player-team-game records."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )
        create_player_game_snap_counts(
            connection
        )

        connection.execute(
            f"""
            INSERT INTO {TARGET_FULL_NAME}
            SELECT *
            FROM {TARGET_FULL_NAME}
            WHERE pfr_player_id = 'TestQu00'
            """
        )

        with pytest.raises(
            RuntimeError,
            match="duplicate business keys",
        ):
            validate_target_table(
                connection=connection,
                expected_row_count=4,
            )


def test_validate_target_table_rejects_invalid_share() -> None:
    """Reject normalized shares outside zero to one."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )
        create_player_game_snap_counts(
            connection
        )

        connection.execute(
            f"""
            UPDATE {TARGET_FULL_NAME}
            SET offense_snap_share = 1.02
            WHERE pfr_player_id = 'TestQu00'
            """
        )

        with pytest.raises(
            RuntimeError,
            match="outside the range 0 to 1",
        ):
            validate_target_table(
                connection=connection,
                expected_row_count=3,
            )


def test_validate_target_table_rejects_inconsistent_flag(
) -> None:
    """Reject an inconsistent derived participation flag."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )
        create_player_game_snap_counts(
            connection
        )

        connection.execute(
            f"""
            UPDATE {TARGET_FULL_NAME}
            SET played_offense = FALSE
            WHERE pfr_player_id = 'TestQu00'
            """
        )

        with pytest.raises(
            RuntimeError,
            match="inconsistent flags",
        ):
            validate_target_table(
                connection=connection,
                expected_row_count=3,
            )


def test_build_player_game_snap_counts_creates_table(
    tmp_path: Path,
) -> None:
    """Run the complete processed snap-count workflow."""

    database_file = (
        tmp_path
        / "test.duckdb"
    )

    with duckdb.connect(
        str(database_file)
    ) as connection:
        create_source_tables(
            connection
        )

    build_player_game_snap_counts(
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
                COUNT(*) FILTER (
                    WHERE has_player_directory_match
                ),
                COUNT(*) FILTER (
                    WHERE has_source_rounding_adjustment
                )
            FROM {TARGET_FULL_NAME}
            """
        ).fetchone()

    assert result == (
        3,
        2,
        1,
    )