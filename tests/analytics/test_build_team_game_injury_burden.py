"""Tests for team-game injury burden aggregation."""

from pathlib import Path

import duckdb
import pytest

from src.analytics.build_team_game_injury_burden import (
    TARGET_FULL_NAME,
    build_team_game_injury_burden,
    count_schedule_team_games,
    create_team_game_injury_burden,
    validate_schedule_table,
    validate_source_table,
    validate_target_table,
)


def create_source_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create deterministic player injury-impact data."""

    connection.execute(
        """
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
        VALUES
            (
                '2025_03_NE_BUF',
                2025,
                'REG',
                3,
                DATE '2025-09-21',
                'NE',
                'BUF'
            ),
            (
                '2025_03_MIA_NYJ',
                2025,
                'REG',
                3,
                DATE '2025-09-21',
                'MIA',
                'NYJ'
            );

        CREATE SCHEMA analytics;

        CREATE TABLE analytics.player_injury_impact (
            game_id VARCHAR,
            season INTEGER,
            game_type VARCHAR,
            week INTEGER,
            gameday DATE,
            team VARCHAR,
            opponent VARCHAR,
            is_home BOOLEAN,
            gsis_id VARCHAR,
            position VARCHAR,
            full_name VARCHAR,
            report_status VARCHAR,
            practice_status VARCHAR,
            is_out BOOLEAN,
            is_doubtful BOOLEAN,
            is_questionable BOOLEAN,
            did_not_practice BOOLEAN,
            limited_practice BOOLEAN,
            full_practice BOOLEAN,
            has_depth_chart_match BOOLEAN,
            depth_tier VARCHAR,
            has_starter_role BOOLEAN,
            has_prior_snap_history BOOLEAN,
            snap_history_source VARCHAR,
            is_qb BOOLEAN,
            availability_severity_score DOUBLE,
            player_importance_score DOUBLE,
            injury_impact_score DOUBLE,
            non_qb_injury_impact_score DOUBLE,
            offense_injury_impact_score DOUBLE,
            defense_injury_impact_score DOUBLE,
            special_teams_injury_impact_score DOUBLE
        );

        INSERT INTO analytics.player_injury_impact
        VALUES
            (
                '2025_03_NE_BUF',
                2025,
                'REG',
                3,
                DATE '2025-09-21',
                'NE',
                'BUF',
                TRUE,
                '00-0000001',
                'WR',
                'Out Receiver',
                'Out',
                'Did Not Participate In Practice',
                TRUE,
                FALSE,
                FALSE,
                TRUE,
                FALSE,
                FALSE,
                TRUE,
                'STARTER',
                TRUE,
                TRUE,
                'TEAM',
                FALSE,
                1.00,
                0.80,
                0.80,
                0.80,
                0.80,
                0.00,
                0.00
            ),
            (
                '2025_03_NE_BUF',
                2025,
                'REG',
                3,
                DATE '2025-09-21',
                'NE',
                'BUF',
                TRUE,
                '00-0000002',
                'QB',
                'Doubtful Quarterback',
                'Doubtful',
                'Did Not Participate In Practice',
                FALSE,
                TRUE,
                FALSE,
                TRUE,
                FALSE,
                FALSE,
                TRUE,
                'STARTER',
                TRUE,
                TRUE,
                'TEAM',
                TRUE,
                0.85,
                1.00,
                0.85,
                0.00,
                0.00,
                0.00,
                0.00
            ),
            (
                '2025_03_NE_BUF',
                2025,
                'REG',
                3,
                DATE '2025-09-21',
                'NE',
                'BUF',
                TRUE,
                '00-0000003',
                'CB',
                'Questionable Cornerback',
                'Questionable',
                'Full Participation in Practice',
                FALSE,
                FALSE,
                TRUE,
                FALSE,
                FALSE,
                TRUE,
                TRUE,
                'PRIMARY_BACKUP',
                FALSE,
                TRUE,
                'CAREER',
                FALSE,
                0.30,
                0.40,
                0.12,
                0.12,
                0.00,
                0.12,
                0.00
            ),
            (
                '2025_03_NE_BUF',
                2025,
                'REG',
                3,
                DATE '2025-09-21',
                'NE',
                'BUF',
                TRUE,
                '00-0000004',
                'LB',
                'Cleared Linebacker',
                NULL,
                'Full Participation in Practice',
                FALSE,
                FALSE,
                FALSE,
                FALSE,
                FALSE,
                TRUE,
                FALSE,
                'UNKNOWN',
                FALSE,
                FALSE,
                'NONE',
                FALSE,
                0.00,
                0.40,
                0.00,
                0.00,
                0.00,
                0.00,
                0.00
            ),
            (
                '2025_03_NE_BUF',
                2025,
                'REG',
                3,
                DATE '2025-09-21',
                'BUF',
                'NE',
                FALSE,
                '00-0000005',
                'TE',
                'Healthy Tight End',
                NULL,
                'Full Participation in Practice',
                FALSE,
                FALSE,
                FALSE,
                FALSE,
                FALSE,
                TRUE,
                TRUE,
                'STARTER',
                TRUE,
                TRUE,
                'TEAM',
                FALSE,
                0.00,
                0.90,
                0.00,
                0.00,
                0.00,
                0.00,
                0.00
            );
        """
    )


def test_validate_source_table_accepts_valid_source() -> None:
    """Accept valid player injury-impact data."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )

        validate_source_table(
            connection
        )


def test_validate_source_table_rejects_missing_source() -> None:
    """Reject a missing player injury-impact source."""

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


def test_validate_schedule_table_accepts_valid_schedule(
) -> None:
    """Accept a valid schedule source."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )

        validate_schedule_table(
            connection
        )


def test_count_schedule_team_games_counts_both_teams(
) -> None:
    """Count both team rows for every scheduled game."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )

        assert count_schedule_team_games(
            connection
        ) == 4


def test_create_burden_aggregates_status_counts() -> None:
    """Aggregate player and status counts by team-game."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )
        create_team_game_injury_burden(
            connection
        )

        result = connection.execute(
            f"""
            SELECT
                injury_report_player_count,
                game_status_player_count,
                out_player_count,
                doubtful_player_count,
                questionable_player_count,
                starter_game_status_count,
                starter_out_count,
                qb_game_status_count,
                qb_out_count,
                non_qb_game_status_count,
                missing_depth_chart_count,
                missing_snap_history_count,
                career_snap_fallback_count
            FROM {TARGET_FULL_NAME}
            WHERE team = 'NE'
            """
        ).fetchone()

    assert result == (
        4,
        3,
        1,
        1,
        1,
        2,
        1,
        1,
        0,
        2,
        1,
        1,
        1,
    )


def test_create_burden_separates_qb_and_non_qb() -> None:
    """Keep QB burden separate from generic unit burden."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )
        create_team_game_injury_burden(
            connection
        )

        result = connection.execute(
            f"""
            SELECT
                total_injury_burden,
                qb_injury_burden,
                non_qb_injury_burden,
                offense_injury_burden,
                defense_injury_burden,
                special_teams_injury_burden,
                maximum_player_injury_impact
            FROM {TARGET_FULL_NAME}
            WHERE team = 'NE'
            """
        ).fetchone()

    assert result[0] == pytest.approx(
        1.77
    )
    assert result[1] == pytest.approx(
        0.85
    )
    assert result[2] == pytest.approx(
        0.92
    )
    assert result[3] == pytest.approx(
        0.80
    )
    assert result[4] == pytest.approx(
        0.12
    )
    assert result[5] == pytest.approx(
        0.00
    )
    assert result[6] == pytest.approx(
        0.85
    )


def test_create_burden_calculates_coverage_rates() -> None:
    """Calculate depth and snap-history match rates."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )
        create_team_game_injury_burden(
            connection
        )

        result = connection.execute(
            f"""
            SELECT
                depth_chart_match_rate,
                snap_history_match_rate
            FROM {TARGET_FULL_NAME}
            WHERE team = 'NE'
            """
        ).fetchone()

    assert result[0] == pytest.approx(
        0.75
    )
    assert result[1] == pytest.approx(
        0.75
    )


def test_create_burden_selects_top_player() -> None:
    """Select the largest-impact player for explanation."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )
        create_team_game_injury_burden(
            connection
        )

        result = connection.execute(
            f"""
            SELECT
                top_impact_player_name,
                top_impact_player_gsis_id,
                top_impact_player_position,
                top_impact_player_status,
                top_impact_player_score
            FROM {TARGET_FULL_NAME}
            WHERE team = 'NE'
            """
        ).fetchone()

    assert result[:4] == (
        "Doubtful Quarterback",
        "00-0000002",
        "QB",
        "Doubtful",
    )
    assert result[4] == pytest.approx(
        0.85
    )


def test_create_burden_handles_zero_burden_team() -> None:
    """Leave top-player fields empty when burden is zero."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )
        create_team_game_injury_burden(
            connection
        )

        result = connection.execute(
            f"""
            SELECT
                total_injury_burden,
                top_impact_player_name,
                top_impact_player_score
            FROM {TARGET_FULL_NAME}
            WHERE team = 'BUF'
            """
        ).fetchone()

    assert result == (
        0.0,
        None,
        None,
    )


def test_create_burden_marks_missing_injury_data(
) -> None:
    """Keep source absence separate from known zero burden."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )
        create_team_game_injury_burden(
            connection
        )

        result = connection.execute(
            f"""
            SELECT
                has_injury_report_data,
                injury_report_player_count,
                total_injury_burden,
                depth_chart_match_rate,
                maximum_player_injury_impact,
                top_impact_player_name
            FROM {TARGET_FULL_NAME}
            WHERE game_id = '2025_03_MIA_NYJ'
              AND team = 'MIA'
            """
        ).fetchone()

    assert result == (
        False,
        0,
        0.0,
        None,
        None,
        None,
    )


def test_validate_target_table_accepts_valid_burden() -> None:
    """Accept valid team-game injury burden."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )
        create_team_game_injury_burden(
            connection
        )

        result = validate_target_table(
            connection=connection,
            expected_team_game_count=4,
        )

    assert result == (
        4,
        1,
        1,
        1,
        2,
    )


def test_validate_target_table_rejects_row_mismatch() -> None:
    """Reject a team-game row-count mismatch."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )
        create_team_game_injury_burden(
            connection
        )

        with pytest.raises(
            RuntimeError,
            match="does not match scheduled team-games",
        ):
            validate_target_table(
                connection=connection,
                expected_team_game_count=5,
            )


def test_validate_target_table_rejects_inconsistent_count(
) -> None:
    """Reject inconsistent aggregated status counts."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )
        create_team_game_injury_burden(
            connection
        )

        connection.execute(
            f"""
            UPDATE {TARGET_FULL_NAME}
            SET game_status_player_count = 99
            WHERE team = 'NE'
            """
        )

        with pytest.raises(
            RuntimeError,
            match="inconsistent counts",
        ):
            validate_target_table(
                connection=connection,
                expected_team_game_count=4,
            )


def test_validate_target_table_rejects_invalid_score() -> None:
    """Reject a negative burden score."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )
        create_team_game_injury_burden(
            connection
        )

        connection.execute(
            f"""
            UPDATE {TARGET_FULL_NAME}
            SET non_qb_injury_burden = -0.1
            WHERE team = 'NE'
            """
        )

        with pytest.raises(
            RuntimeError,
            match="invalid scores",
        ):
            validate_target_table(
                connection=connection,
                expected_team_game_count=4,
            )


def test_validate_target_table_rejects_top_player_mismatch(
) -> None:
    """Reject a top-player score differing from maximum."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )
        create_team_game_injury_burden(
            connection
        )

        connection.execute(
            f"""
            UPDATE {TARGET_FULL_NAME}
            SET top_impact_player_score = 0.50
            WHERE team = 'NE'
            """
        )

        with pytest.raises(
            RuntimeError,
            match="inconsistent top-player fields",
        ):
            validate_target_table(
                connection=connection,
                expected_team_game_count=4,
            )


def test_build_team_game_injury_burden_creates_table(
    tmp_path: Path,
) -> None:
    """Run the complete team injury-burden workflow."""

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

    build_team_game_injury_burden(
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
                    WHERE total_injury_burden > 0
                ),
                COUNT(*) FILTER (
                    WHERE NOT has_injury_report_data
                )
            FROM {TARGET_FULL_NAME}
            """
        ).fetchone()

    assert result == (
        4,
        1,
        2,
    )