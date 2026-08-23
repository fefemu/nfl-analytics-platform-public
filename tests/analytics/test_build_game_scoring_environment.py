"""Tests for league scoring-environment features."""

import duckdb
import pytest

from src.analytics.build_game_scoring_environment import (
    TARGET_FULL_NAME,
    create_game_scoring_environment_table,
    create_window_lateral_sql,
    validate_game_scoring_environment,
    validate_scoring_environment_source,
)


def create_schedule_source(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create chronological schedule results."""

    connection.execute(
        """
        CREATE SCHEMA processed;

        CREATE TABLE processed.schedule (
            game_id VARCHAR,
            season INTEGER,
            game_type VARCHAR,
            gameday DATE,
            home_team VARCHAR,
            away_team VARCHAR,
            is_completed BOOLEAN,
            total_points INTEGER
        );

        INSERT INTO processed.schedule
        VALUES
            (
                'game_1',
                2024,
                'REG',
                DATE '2024-09-01',
                'A',
                'B',
                TRUE,
                40
            ),
            (
                'game_2',
                2024,
                'REG',
                DATE '2024-09-08',
                'C',
                'D',
                TRUE,
                50
            ),
            (
                'game_3',
                2024,
                'REG',
                DATE '2024-09-15',
                'E',
                'F',
                TRUE,
                60
            ),
            (
                'same_day_game',
                2024,
                'REG',
                DATE '2024-09-15',
                'G',
                'H',
                TRUE,
                80
            ),
            (
                'future_game',
                2025,
                'REG',
                DATE '2025-09-01',
                'I',
                'J',
                FALSE,
                NULL
            );
        """
    )


def test_create_window_sql_rejects_invalid_size(
) -> None:
    """Reject non-positive scoring windows."""

    with pytest.raises(
        ValueError,
        match="must be positive",
    ):
        create_window_lateral_sql(0)


def test_create_and_validate_scoring_environment(
) -> None:
    """Create one feature row per schedule game."""

    with duckdb.connect(":memory:") as connection:
        create_schedule_source(connection)

        validate_scoring_environment_source(
            connection
        )

        create_game_scoring_environment_table(
            connection
        )

        validate_game_scoring_environment(
            connection
        )

        row_count = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM {TARGET_FULL_NAME}
            """
        ).fetchone()[0]

    assert row_count == 5


def test_first_game_has_no_prior_history() -> None:
    """Keep the first available game explicit."""

    with duckdb.connect(":memory:") as connection:
        create_schedule_source(connection)

        create_game_scoring_environment_table(
            connection
        )

        result = connection.execute(
            f"""
            SELECT
                league_game_count_last_32,
                league_average_total_last_32
            FROM {TARGET_FULL_NAME}
            WHERE game_id = 'game_1'
            """
        ).fetchone()

    assert result == (
        0,
        None,
    )


def test_same_day_games_do_not_leak() -> None:
    """Exclude every result from the target game date."""

    with duckdb.connect(":memory:") as connection:
        create_schedule_source(connection)

        create_game_scoring_environment_table(
            connection
        )

        results = connection.execute(
            f"""
            SELECT
                game_id,
                league_game_count_last_32,
                league_average_total_last_32
            FROM {TARGET_FULL_NAME}
            WHERE game_id IN (
                'game_3',
                'same_day_game'
            )
            ORDER BY game_id
            """
        ).fetchall()

    assert results == [
        (
            "game_3",
            2,
            45.0,
        ),
        (
            "same_day_game",
            2,
            45.0,
        ),
    ]


def test_future_game_uses_only_completed_history(
) -> None:
    """Use completed games before a future kickoff."""

    with duckdb.connect(":memory:") as connection:
        create_schedule_source(connection)

        create_game_scoring_environment_table(
            connection
        )

        result = connection.execute(
            f"""
            SELECT
                league_game_count_last_64,
                league_average_total_last_64
            FROM {TARGET_FULL_NAME}
            WHERE game_id = 'future_game'
            """
        ).fetchone()

    assert result[0] == 4
    assert result[1] == pytest.approx(57.5)


def test_postseason_history_is_included() -> None:
    """Include completed REG and POST games only."""

    with duckdb.connect(":memory:") as connection:
        create_schedule_source(connection)

        connection.execute(
            """
            INSERT INTO processed.schedule
            VALUES (
                'post_game',
                2024,
                'POST',
                DATE '2024-12-01',
                'K',
                'L',
                TRUE,
                30
            )
            """
        )

        create_game_scoring_environment_table(
            connection
        )

        result = connection.execute(
            f"""
            SELECT
                league_game_count_last_32
            FROM {TARGET_FULL_NAME}
            WHERE game_id = 'future_game'
            """
        ).fetchone()[0]

    assert result == 5