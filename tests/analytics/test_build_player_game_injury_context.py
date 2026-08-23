"""Tests for player-game injury context."""

from datetime import date
from pathlib import Path

import duckdb
import pytest

from src.analytics.build_player_game_injury_context import (
    TARGET_FULL_NAME,
    build_player_game_injury_context,
    count_source_rows,
    create_player_game_injury_context,
    validate_source_tables,
    validate_target_table,
)


def create_source_tables(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create injury, depth and snap-history test sources."""

    connection.execute(
        """
        CREATE SCHEMA processed;
        CREATE SCHEMA analytics;

        CREATE TABLE processed.player_game_injury_status AS
        SELECT
            source.game_id,
            2025 AS season,
            'REG'::VARCHAR AS season_type,
            'REG'::VARCHAR AS game_type,
            3 AS week,
            DATE '2025-09-21' AS gameday,
            source.team,
            source.opponent,
            source.is_home,
            source.gsis_id,
            source.position,
            source.full_name,

            'Knee'::VARCHAR
                AS report_primary_injury,
            NULL::VARCHAR
                AS report_secondary_injury,
            source.report_status,

            'Knee'::VARCHAR
                AS practice_primary_injury,
            NULL::VARCHAR
                AS practice_secondary_injury,
            source.practice_status,

            source.report_status = 'Out'
                AS is_out,
            source.report_status = 'Doubtful'
                AS is_doubtful,
            source.report_status = 'Questionable'
                AS is_questionable,

            source.practice_status =
                'Did Not Participate In Practice'
                AS did_not_practice,
            source.practice_status =
                'Limited Participation in Practice'
                AS limited_practice,
            source.practice_status =
                'Full Participation in Practice'
                AS full_practice,

            TIMESTAMPTZ '2025-09-20 12:00:00+00'
                AS source_date_modified,
            TRUE AS has_source_timestamp,
            1 AS source_snapshot_count

        FROM (
            VALUES
                (
                    '2025_03_NE_NYJ',
                    'NE',
                    'NYJ',
                    TRUE,
                    '00-0000001',
                    'WR',
                    'Player A',
                    'Out',
                    'Did Not Participate In Practice'
                ),
                (
                    '2025_03_NE_NYJ',
                    'NE',
                    'NYJ',
                    TRUE,
                    '00-0000002',
                    'RB',
                    'Player B',
                    'Questionable',
                    'Limited Participation in Practice'
                ),
                (
                    '2025_03_BUF_MIA',
                    'BUF',
                    'MIA',
                    TRUE,
                    '00-0000003',
                    'CB',
                    'Player C',
                    'Doubtful',
                    'Did Not Participate In Practice'
                )
        ) AS source(
            game_id,
            team,
            opponent,
            is_home,
            gsis_id,
            position,
            full_name,
            report_status,
            practice_status
        );

        CREATE TABLE processed.player_game_depth_chart (
            game_id VARCHAR,
            team VARCHAR,
            gsis_id VARCHAR,
            formation VARCHAR,
            depth_position VARCHAR,
            depth_rank INTEGER,
            is_starter BOOLEAN,
            is_primary_backup BOOLEAN,
            is_reserve BOOLEAN,
            source_generation VARCHAR
        );

        INSERT INTO processed.player_game_depth_chart
        VALUES
            (
                '2025_03_NE_NYJ',
                'NE',
                '00-0000001',
                'Offense',
                'WR',
                1,
                TRUE,
                FALSE,
                FALSE,
                'espn'
            ),
            (
                '2025_03_NE_NYJ',
                'NE',
                '00-0000001',
                'Special Teams',
                'KR',
                1,
                TRUE,
                FALSE,
                FALSE,
                'espn'
            );

        CREATE TABLE analytics.player_snap_share_history (
            game_id VARCHAR,
            gameday DATE,
            available_after_gameday DATE,
            team VARCHAR,
            player_key VARCHAR,

            career_appearance_count INTEGER,
            team_appearance_count INTEGER,

            career_games_last_4 INTEGER,
            career_games_last_8 INTEGER,
            team_games_last_4 INTEGER,
            team_games_last_8 INTEGER,

            career_offense_snap_share_last_4 DOUBLE,
            career_defense_snap_share_last_4 DOUBLE,
            career_special_teams_snap_share_last_4 DOUBLE,

            career_offense_snap_share_last_8 DOUBLE,
            career_defense_snap_share_last_8 DOUBLE,
            career_special_teams_snap_share_last_8 DOUBLE,

            team_offense_snap_share_last_4 DOUBLE,
            team_defense_snap_share_last_4 DOUBLE,
            team_special_teams_snap_share_last_4 DOUBLE,

            team_offense_snap_share_last_8 DOUBLE,
            team_defense_snap_share_last_8 DOUBLE,
            team_special_teams_snap_share_last_8 DOUBLE
        );

        INSERT INTO analytics.player_snap_share_history
        VALUES
            (
                '2025_01_NE_MIA',
                DATE '2025-09-07',
                DATE '2025-09-07',
                'NE',
                '00-0000001',
                1,
                1,
                1,
                1,
                1,
                1,
                0.70,
                0.00,
                0.20,
                0.70,
                0.00,
                0.20,
                0.70,
                0.00,
                0.20,
                0.70,
                0.00,
                0.20
            ),
            (
                '2025_02_NE_BUF',
                DATE '2025-09-14',
                DATE '2025-09-14',
                'NE',
                '00-0000001',
                2,
                2,
                2,
                2,
                2,
                2,
                0.75,
                0.00,
                0.25,
                0.72,
                0.00,
                0.22,
                0.80,
                0.00,
                0.30,
                0.76,
                0.00,
                0.26
            ),
            (
                '2025_02_MIA_ATL',
                DATE '2025-09-14',
                DATE '2025-09-14',
                'MIA',
                '00-0000002',
                5,
                2,
                4,
                5,
                2,
                2,
                0.60,
                0.05,
                0.10,
                0.55,
                0.04,
                0.08,
                0.90,
                0.00,
                0.00,
                0.85,
                0.00,
                0.00
            ),
            (
                '2025_03_BUF_MIA',
                DATE '2025-09-21',
                DATE '2025-09-21',
                'BUF',
                '00-0000003',
                3,
                3,
                3,
                3,
                3,
                3,
                0.00,
                0.80,
                0.20,
                0.00,
                0.75,
                0.18,
                0.00,
                0.80,
                0.20,
                0.00,
                0.75,
                0.18
            );
        """
    )


def test_validate_source_tables_accepts_sources() -> None:
    """Accept valid injury-context sources."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )

        validate_source_tables(
            connection
        )


def test_validate_source_tables_rejects_missing_source() -> None:
    """Reject a missing injury-context source."""

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


def test_count_source_rows_counts_injury_players() -> None:
    """Count injury player-game records."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )

        assert count_source_rows(
            connection
        ) == 3


def test_create_context_preserves_injury_grain() -> None:
    """Create one context row per injury player-game."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )
        create_player_game_injury_context(
            connection
        )

        row_count = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM {TARGET_FULL_NAME}
            """
        ).fetchone()[0]

    assert row_count == 3


def test_create_context_consolidates_depth_roles() -> None:
    """Consolidate multiple depth roles to one player row."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )
        create_player_game_injury_context(
            connection
        )

        result = connection.execute(
            f"""
            SELECT
                has_depth_chart_match,
                depth_role_count,
                depth_formation_count,
                best_depth_rank,
                has_starter_role,
                has_offense_role,
                has_special_teams_role,
                depth_tier
            FROM {TARGET_FULL_NAME}
            WHERE gsis_id = '00-0000001'
            """
        ).fetchone()

    assert result == (
        True,
        2,
        2,
        1,
        True,
        True,
        True,
        "STARTER",
    )


def test_create_context_uses_same_team_history() -> None:
    """Use team rolling history when latest team matches."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )
        create_player_game_injury_context(
            connection
        )

        result = connection.execute(
            f"""
            SELECT
                prior_snap_history_game_id,
                prior_snap_history_gameday,
                prior_snap_history_team,
                snap_history_source,
                days_since_prior_snap_history,
                prior_selected_appearance_count,
                prior_offense_snap_share_last_4
            FROM {TARGET_FULL_NAME}
            WHERE gsis_id = '00-0000001'
            """
        ).fetchone()

    assert result[:6] == (
        "2025_02_NE_BUF",
        date(
            2025,
            9,
            14,
        ),
        "NE",
        "TEAM",
        7,
        2,
    )
    assert result[6] == pytest.approx(
        0.80
    )


def test_create_context_uses_career_fallback() -> None:
    """Use career rolling history after a team change."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )
        create_player_game_injury_context(
            connection
        )

        result = connection.execute(
            f"""
            SELECT
                prior_snap_history_team,
                prior_snap_history_same_team,
                snap_history_source,
                prior_selected_appearance_count,
                prior_snap_games_last_4,
                prior_offense_snap_share_last_4
            FROM {TARGET_FULL_NAME}
            WHERE gsis_id = '00-0000002'
            """
        ).fetchone()

    assert result[:5] == (
        "MIA",
        False,
        "CAREER",
        5,
        4,
    )
    assert result[5] == pytest.approx(
        0.60
    )


def test_create_context_excludes_same_day_history() -> None:
    """Exclude current-game snap history from pregame context."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )
        create_player_game_injury_context(
            connection
        )

        result = connection.execute(
            f"""
            SELECT
                has_prior_snap_history,
                snap_history_source,
                prior_snap_history_game_id,
                prior_snap_history_gameday,
                prior_snap_games_last_4
            FROM {TARGET_FULL_NAME}
            WHERE gsis_id = '00-0000003'
            """
        ).fetchone()

    assert result == (
        False,
        "NONE",
        None,
        None,
        None,
    )


def test_create_context_represents_missing_depth() -> None:
    """Represent an unmatched depth player explicitly."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )
        create_player_game_injury_context(
            connection
        )

        result = connection.execute(
            f"""
            SELECT
                has_depth_chart_match,
                depth_role_count,
                depth_formation_count,
                best_depth_rank,
                depth_tier
            FROM {TARGET_FULL_NAME}
            WHERE gsis_id = '00-0000002'
            """
        ).fetchone()

    assert result == (
        False,
        0,
        0,
        None,
        "UNKNOWN",
    )


def test_validate_target_table_accepts_valid_context() -> None:
    """Accept valid player-game injury context."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )
        create_player_game_injury_context(
            connection
        )

        result = validate_target_table(
            connection=connection,
            expected_row_count=3,
        )

    assert result == (
        3,
        1,
        2,
        1,
        1,
    )


def test_validate_target_table_rejects_future_history() -> None:
    """Reject same-day or future snap history."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )
        create_player_game_injury_context(
            connection
        )

        connection.execute(
            f"""
            UPDATE {TARGET_FULL_NAME}
            SET prior_snap_history_gameday = gameday
            WHERE gsis_id = '00-0000001'
            """
        )

        with pytest.raises(
            RuntimeError,
            match="non-pregame snap-history",
        ):
            validate_target_table(
                connection=connection,
                expected_row_count=3,
            )


def test_validate_target_table_rejects_invalid_share() -> None:
    """Reject a prior snap share outside zero to one."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )
        create_player_game_injury_context(
            connection
        )

        connection.execute(
            f"""
            UPDATE {TARGET_FULL_NAME}
            SET prior_offense_snap_share_last_4 = 1.01
            WHERE gsis_id = '00-0000001'
            """
        )

        with pytest.raises(
            RuntimeError,
            match="invalid prior snap shares",
        ):
            validate_target_table(
                connection=connection,
                expected_row_count=3,
            )


def test_validate_target_table_rejects_depth_inconsistency(
) -> None:
    """Reject inconsistent depth-chart flags."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_tables(
            connection
        )
        create_player_game_injury_context(
            connection
        )

        connection.execute(
            f"""
            UPDATE {TARGET_FULL_NAME}
            SET depth_tier = 'RESERVE'
            WHERE gsis_id = '00-0000001'
            """
        )

        with pytest.raises(
            RuntimeError,
            match="inconsistent depth-chart fields",
        ):
            validate_target_table(
                connection=connection,
                expected_row_count=3,
            )


def test_build_player_game_injury_context_creates_table(
    tmp_path: Path,
) -> None:
    """Run the complete player injury-context workflow."""

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

    build_player_game_injury_context(
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
                    WHERE has_depth_chart_match
                ),
                COUNT(*) FILTER (
                    WHERE has_prior_snap_history
                )
            FROM {TARGET_FULL_NAME}
            """
        ).fetchone()

    assert result == (
        3,
        1,
        2,
    )