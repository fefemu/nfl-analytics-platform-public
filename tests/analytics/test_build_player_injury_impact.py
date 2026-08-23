"""Tests for rule-based player injury impact."""

from pathlib import Path

import duckdb
import pytest

from src.analytics.build_player_injury_impact import (
    TARGET_FULL_NAME,
    build_player_injury_impact,
    count_source_rows,
    create_player_injury_impact,
    validate_source_table,
    validate_target_table,
)


def create_source_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create deterministic player injury context."""

    connection.execute(
        """
        CREATE SCHEMA analytics;

        CREATE TABLE analytics.player_game_injury_context AS
        SELECT
            source.game_id,
            2025 AS season,
            'REG'::VARCHAR AS game_type,
            3 AS week,
            DATE '2025-09-21' AS gameday,
            'NE'::VARCHAR AS team,
            'NYJ'::VARCHAR AS opponent,
            TRUE AS is_home,

            source.gsis_id AS player_key,
            source.gsis_id,
            source.position,
            source.full_name,

            source.report_status,
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

            source.depth_tier != 'UNKNOWN'
                AS has_depth_chart_match,
            source.depth_tier,

            source.depth_tier = 'STARTER'
                AS has_starter_role,
            source.depth_tier = 'PRIMARY_BACKUP'
                AS has_primary_backup_role,
            source.depth_tier = 'RESERVE'
                AS has_reserve_role,

            source.has_offense_role,
            source.has_defense_role,
            source.has_special_teams_role,

            source.has_prior_snap_history,
            CASE
                WHEN source.has_prior_snap_history
                    THEN 'TEAM'
                ELSE 'NONE'
            END AS snap_history_source,

            CASE
                WHEN source.has_prior_snap_history
                    THEN 7
                ELSE NULL
            END AS days_since_prior_snap_history,

            CASE
                WHEN source.has_prior_snap_history
                    THEN source.prior_games
                ELSE NULL
            END AS prior_snap_games_last_4,

            CASE
                WHEN source.has_prior_snap_history
                    THEN source.prior_games
                ELSE NULL
            END AS prior_snap_games_last_8,

            source.offense_share
                AS prior_offense_snap_share_last_4,
            source.defense_share
                AS prior_defense_snap_share_last_4,
            source.special_teams_share
                AS prior_special_teams_snap_share_last_4,

            source.offense_share
                AS prior_offense_snap_share_last_8,
            source.defense_share
                AS prior_defense_snap_share_last_8,
            source.special_teams_share
                AS prior_special_teams_snap_share_last_8

        FROM (
            VALUES
                (
                    '2025_03_NE_NYJ',
                    '00-0000001',
                    'WR',
                    'Out Starter',
                    'Out',
                    'Did Not Participate In Practice',
                    'STARTER',
                    TRUE,
                    FALSE,
                    FALSE,
                    TRUE,
                    4,
                    0.80,
                    0.00,
                    0.10
                ),
                (
                    '2025_03_NE_NYJ',
                    '00-0000002',
                    'RB',
                    'Questionable Backup',
                    'Questionable',
                    'Limited Participation in Practice',
                    'PRIMARY_BACKUP',
                    TRUE,
                    FALSE,
                    FALSE,
                    TRUE,
                    2,
                    0.60,
                    0.00,
                    0.10
                ),
                (
                    '2025_03_NE_NYJ',
                    '00-0000003',
                    'LB',
                    'Cleared Starter',
                    NULL,
                    'Did Not Participate In Practice',
                    'STARTER',
                    FALSE,
                    TRUE,
                    FALSE,
                    FALSE,
                    NULL,
                    NULL,
                    NULL,
                    NULL
                ),
                (
                    '2025_03_NE_NYJ',
                    '00-0000004',
                    'QB',
                    'Doubtful Quarterback',
                    'Doubtful',
                    'Did Not Participate In Practice',
                    'STARTER',
                    TRUE,
                    FALSE,
                    FALSE,
                    TRUE,
                    4,
                    1.00,
                    0.00,
                    0.00
                ),
                (
                    '2025_03_NE_NYJ',
                    '00-0000005',
                    'CB',
                    'Unknown Cornerback',
                    'Questionable',
                    'Full Participation in Practice',
                    'UNKNOWN',
                    FALSE,
                    FALSE,
                    FALSE,
                    FALSE,
                    NULL,
                    NULL,
                    NULL,
                    NULL
                )
        ) AS source(
            game_id,
            gsis_id,
            position,
            full_name,
            report_status,
            practice_status,
            depth_tier,
            has_offense_role,
            has_defense_role,
            has_special_teams_role,
            has_prior_snap_history,
            prior_games,
            offense_share,
            defense_share,
            special_teams_share
        )
        """
    )


def test_validate_source_table_accepts_valid_source() -> None:
    """Accept a valid player injury-context source."""

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
    """Reject a missing player injury-context source."""

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
    """Reject a source missing a required field."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
                analytics.player_game_injury_context AS
            SELECT * EXCLUDE(depth_tier)
            FROM analytics.player_game_injury_context
            """
        )

        with pytest.raises(
            RuntimeError,
            match="missing columns: depth_tier",
        ):
            validate_source_table(
                connection
            )


def test_count_source_rows_counts_players() -> None:
    """Count each player-game injury context."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )

        assert count_source_rows(
            connection
        ) == 5


def test_create_impact_scores_out_starter() -> None:
    """Score an out high-usage starter."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )
        create_player_injury_impact(
            connection
        )

        result = connection.execute(
            f"""
            SELECT
                report_severity_score,
                practice_severity_modifier,
                availability_severity_score,
                depth_importance_score,
                observed_usage_score,
                usage_reliability_score,
                player_importance_score,
                injury_impact_score,
                offense_injury_impact_score
            FROM {TARGET_FULL_NAME}
            WHERE gsis_id = '00-0000001'
            """
        ).fetchone()

    assert result[0] == pytest.approx(
        1.00
    )
    assert result[1] == pytest.approx(
        0.10
    )
    assert result[2] == pytest.approx(
        1.00
    )
    assert result[3] == pytest.approx(
        1.00
    )
    assert result[4] == pytest.approx(
        0.80
    )
    assert result[5] == pytest.approx(
        1.00
    )
    assert result[6] == pytest.approx(
        0.80
    )
    assert result[7] == pytest.approx(
        0.80
    )
    assert result[8] == pytest.approx(
        0.80
    )


def test_create_impact_shrinks_small_sample() -> None:
    """Shrink a two-game usage estimate toward depth prior."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )
        create_player_injury_impact(
            connection
        )

        result = connection.execute(
            f"""
            SELECT
                availability_severity_score,
                depth_importance_score,
                observed_usage_score,
                usage_reliability_score,
                player_importance_score,
                injury_impact_score
            FROM {TARGET_FULL_NAME}
            WHERE gsis_id = '00-0000002'
            """
        ).fetchone()

    assert result[0] == pytest.approx(
        0.40
    )
    assert result[1] == pytest.approx(
        0.55
    )
    assert result[2] == pytest.approx(
        0.60
    )
    assert result[3] == pytest.approx(
        0.50
    )
    assert result[4] == pytest.approx(
        0.575
    )
    assert result[5] == pytest.approx(
        0.23
    )


def test_create_impact_ignores_practice_only_player() -> None:
    """Assign zero impact without a final game status."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )
        create_player_injury_impact(
            connection
        )

        result = connection.execute(
            f"""
            SELECT
                report_severity_score,
                practice_severity_modifier,
                availability_severity_score,
                injury_impact_score
            FROM {TARGET_FULL_NAME}
            WHERE gsis_id = '00-0000003'
            """
        ).fetchone()

    assert result == pytest.approx(
        (
            0.0,
            0.0,
            0.0,
            0.0,
        )
    )


def test_create_impact_excludes_qb_from_generic_burden() -> None:
    """Keep QB impact separate from non-QB offense burden."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )
        create_player_injury_impact(
            connection
        )

        result = connection.execute(
            f"""
            SELECT
                is_qb,
                availability_severity_score,
                injury_impact_score,
                non_qb_injury_impact_score,
                offense_injury_impact_score
            FROM {TARGET_FULL_NAME}
            WHERE gsis_id = '00-0000004'
            """
        ).fetchone()

    assert result[0] is True
    assert result[1] == pytest.approx(
        0.85
    )
    assert result[2] == pytest.approx(
        0.85
    )
    assert result[3] == pytest.approx(
        0.0
    )
    assert result[4] == pytest.approx(
        0.0
    )


def test_create_impact_uses_depth_fallback() -> None:
    """Use depth prior when no snap history exists."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )
        create_player_injury_impact(
            connection
        )

        result = connection.execute(
            f"""
            SELECT
                is_defensive_player,
                availability_severity_score,
                depth_importance_score,
                observed_usage_score,
                usage_reliability_score,
                player_importance_score,
                injury_impact_score,
                defense_injury_impact_score
            FROM {TARGET_FULL_NAME}
            WHERE gsis_id = '00-0000005'
            """
        ).fetchone()

    assert result[0] is True
    assert result[1] == pytest.approx(
        0.30
    )
    assert result[2] == pytest.approx(
        0.40
    )
    assert result[3] is None
    assert result[4] == pytest.approx(
        0.0
    )
    assert result[5] == pytest.approx(
        0.40
    )
    assert result[6] == pytest.approx(
        0.12
    )
    assert result[7] == pytest.approx(
        0.12
    )


def test_validate_target_table_accepts_valid_scores() -> None:
    """Accept valid bounded player-impact scores."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )
        create_player_injury_impact(
            connection
        )

        result = validate_target_table(
            connection=connection,
            expected_row_count=5,
        )

    assert result == (
        5,
        4,
        1,
        3,
        2,
    )


def test_validate_target_table_rejects_invalid_score() -> None:
    """Reject a score outside zero to one."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )
        create_player_injury_impact(
            connection
        )

        connection.execute(
            f"""
            UPDATE {TARGET_FULL_NAME}
            SET player_importance_score = 1.01
            WHERE gsis_id = '00-0000001'
            """
        )

        with pytest.raises(
            RuntimeError,
            match="invalid scores",
        ):
            validate_target_table(
                connection=connection,
                expected_row_count=5,
            )


def test_validate_target_table_rejects_status_inconsistency(
) -> None:
    """Reject an invalid status severity."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )
        create_player_injury_impact(
            connection
        )

        connection.execute(
            f"""
            UPDATE {TARGET_FULL_NAME}
            SET availability_severity_score = 0.5
            WHERE report_status = 'Out'
            """
        )

        with pytest.raises(
            RuntimeError,
            match="inconsistent status scores",
        ):
            validate_target_table(
                connection=connection,
                expected_row_count=5,
            )


def test_validate_target_table_rejects_qb_generic_impact(
) -> None:
    """Reject QB impact in generic offensive burden."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )
        create_player_injury_impact(
            connection
        )

        connection.execute(
            f"""
            UPDATE {TARGET_FULL_NAME}
            SET offense_injury_impact_score = 0.5
            WHERE is_qb
            """
        )

        with pytest.raises(
            RuntimeError,
            match="QB rows in generic offensive burden",
        ):
            validate_target_table(
                connection=connection,
                expected_row_count=5,
            )


def test_build_player_injury_impact_creates_table(
    tmp_path: Path,
) -> None:
    """Run the complete player injury-impact workflow."""

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

    build_player_injury_impact(
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
                    WHERE injury_impact_score > 0
                ),
                COUNT(*) FILTER (
                    WHERE is_qb
                      AND injury_impact_score > 0
                )
            FROM {TARGET_FULL_NAME}
            """
        ).fetchone()

    assert result == (
        5,
        4,
        1,
    )