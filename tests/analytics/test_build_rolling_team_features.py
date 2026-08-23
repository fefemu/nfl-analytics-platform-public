"""Tests for the rolling team features builder."""

import duckdb
import pytest

from src.analytics.build_rolling_team_features import (
    build_rolling_average_expressions,
    create_rolling_team_features_table,
    validate_source_table,
)


def create_team_efficiency_source(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create a minimal synthetic team-game efficiency source."""

    connection.execute(
        """
        CREATE SCHEMA processed;

        CREATE TABLE processed.team_game_efficiency AS
        WITH games(
            game_id,
            season,
            season_type,
            week,
            game_date,
            team,
            opponent,
            is_home,
            metric_value
        ) AS (
            VALUES
                (
                    'game_1', 2025, 'REG', 1,
                    DATE '2025-09-07',
                    'BUF', 'MIA', TRUE, 1.0
                ),
                (
                    'game_2', 2025, 'REG', 2,
                    DATE '2025-09-14',
                    'BUF', 'NYJ', FALSE, 3.0
                ),
                (
                    'game_3', 2025, 'REG', 3,
                    DATE '2025-09-21',
                    'BUF', 'NE', TRUE, 5.0
                ),
                (
                    'game_4', 2026, 'REG', 1,
                    DATE '2026-09-06',
                    'BUF', 'MIA', TRUE, 100.0
                )
        )
        SELECT
            game_id,
            season,
            season_type,
            week,
            game_date,
            team,
            opponent,
            is_home,
            60.0 + metric_value AS offensive_plays,
            20.0 + metric_value AS points_scored,
            15.0 + metric_value AS points_allowed,
            metric_value AS offensive_epa_per_play,
            metric_value AS competitive_epa_per_play,
            metric_value AS dropback_epa_per_play,
            metric_value AS designed_rush_epa_per_play,
            metric_value AS early_down_epa_per_play,
            metric_value AS success_rate,
            metric_value AS dropback_success_rate,
            metric_value AS designed_rush_success_rate,
            metric_value AS explosive_play_rate,
            metric_value AS sack_rate,
            metric_value AS turnover_rate,
            metric_value AS defensive_epa_allowed_per_play,
            metric_value
                AS competitive_defensive_epa_allowed_per_play,
            metric_value AS defensive_success_rate_allowed,
            metric_value AS explosive_play_rate_allowed,
            metric_value AS sack_rate_generated,
            metric_value AS turnover_rate_generated
        FROM games
        """
    )


def test_build_rolling_average_expressions_rejects_invalid_window(
) -> None:
    """Reject zero and negative rolling windows."""

    with pytest.raises(
        ValueError,
        match="Rolling window size must be positive",
    ):
        build_rolling_average_expressions(0)


def test_rolling_expression_excludes_current_game() -> None:
    """Build a SQL frame ending at the previous game."""

    result = build_rolling_average_expressions(4)

    assert "ROWS BETWEEN 4 PRECEDING" in result
    assert "AND 1 PRECEDING" in result
    assert "CURRENT ROW" not in result


def test_create_rolling_features_uses_previous_games_only() -> None:
    """Calculate pregame features without using the current game."""

    with duckdb.connect(":memory:") as connection:
        create_team_efficiency_source(connection)
        validate_source_table(connection)
        create_rolling_team_features_table(connection)

        results = connection.execute(
            """
            SELECT
                game_id,
                season_games_played_before,
                short_window_games,
                pregame_offensive_epa_per_play_last_4
            FROM analytics.rolling_team_features
            WHERE season = 2025
            ORDER BY game_date
            """
        ).fetchall()

    assert results[0] == (
        "game_1",
        0,
        0,
        None,
    )
    assert results[1][0:3] == (
        "game_2",
        1,
        1,
    )
    assert results[1][3] == pytest.approx(1.0)

    assert results[2][0:3] == (
        "game_3",
        2,
        2,
    )
    assert results[2][3] == pytest.approx(2.0)


def test_create_rolling_pace_and_scoring_features(
) -> None:
    """Use prior games for pace and scoring form."""

    with duckdb.connect(":memory:") as connection:
        create_team_efficiency_source(connection)
        create_rolling_team_features_table(
            connection
        )

        result = connection.execute(
            """
            SELECT
                pregame_offensive_plays_last_4,
                pregame_points_scored_last_4,
                pregame_points_allowed_last_4
            FROM analytics.rolling_team_features
            WHERE game_id = 'game_3'
            """
        ).fetchone()

    assert result[0] == pytest.approx(62.0)
    assert result[1] == pytest.approx(22.0)
    assert result[2] == pytest.approx(17.0)


def test_create_rolling_features_resets_for_new_season() -> None:
    """Do not mix the previous season into current-season windows."""

    with duckdb.connect(":memory:") as connection:
        create_team_efficiency_source(connection)
        create_rolling_team_features_table(connection)

        result = connection.execute(
            """
            SELECT
                season_games_played_before,
                short_window_games,
                long_window_games,
                pregame_offensive_epa_per_play_last_4,
                pregame_offensive_epa_per_play_last_8
            FROM analytics.rolling_team_features
            WHERE game_id = 'game_4'
            """
        ).fetchone()

    assert result == (
        0,
        0,
        0,
        None,
        None,
    )