"""Tests for the Elo ratings DuckDB builder."""
from pathlib import Path
from datetime import date, time

import duckdb
import pytest

from src.analytics.build_elo_ratings import (
    build_elo_ratings,
    create_current_ratings_table,
    create_history_table,
    load_historical_games,
    validate_source_columns,
    validate_source_table,
)

from src.models.elo_history import EloHistoryRecord


def create_schedule_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create the minimal processed schedule table for Elo tests."""

    connection.execute("CREATE SCHEMA processed")
    connection.execute(
        """
        CREATE TABLE processed.schedule (
            game_id VARCHAR,
            season INTEGER,
            game_type VARCHAR,
            week INTEGER,
            gameday DATE,
            gametime TIME,
            home_team VARCHAR,
            away_team VARCHAR,
            home_score INTEGER,
            away_score INTEGER,
            location VARCHAR,
            is_completed BOOLEAN
        )
        """
    )


def test_validate_source_table_rejects_missing_table() -> None:
    """Reject a database without the processed schedule table."""

    with duckdb.connect(":memory:") as connection:
        with pytest.raises(
            RuntimeError,
            match="Source table does not exist",
        ):
            validate_source_table(connection)


def test_validate_source_table_accepts_existing_table() -> None:
    """Accept an existing processed schedule table."""

    with duckdb.connect(":memory:") as connection:
        create_schedule_table(connection)

        validate_source_table(connection)


def test_validate_source_columns_rejects_missing_columns() -> None:
    """Reject a processed schedule with an incomplete schema."""

    with duckdb.connect(":memory:") as connection:
        connection.execute("CREATE SCHEMA processed")
        connection.execute(
            """
            CREATE TABLE processed.schedule (
                game_id VARCHAR
            )
            """
        )

        with pytest.raises(
            RuntimeError,
            match="Missing Elo source columns",
        ):
            validate_source_columns(connection)


def test_load_historical_games_filters_and_converts_games() -> None:
    """Load completed regular and playoff games only."""

    with duckdb.connect(":memory:") as connection:
        create_schedule_table(connection)

        connection.executemany(
            """
            INSERT INTO processed.schedule
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "completed_game",
                    2025,
                    "REG",
                    1,
                    date(2025, 9, 7),
                    time(13, 0),
                    "CHI",
                    "GB",
                    24,
                    17,
                    "Home",
                    True,
                ),
                (
                    "preseason_game",
                    2025,
                    "PRE",
                    1,
                    date(2025, 8, 10),
                    time(13, 0),
                    "CHI",
                    "MIA",
                    20,
                    17,
                    "Home",
                    True,
                ),
                (
                    "future_game",
                    2025,
                    "REG",
                    2,
                    date(2025, 9, 14),
                    time(13, 0),
                    "CHI",
                    "DET",
                    None,
                    None,
                    "Home",
                    False,
                ),
            ],
        )

        games = load_historical_games(connection)

    assert len(games) == 1
    assert games[0].game_id == "completed_game"
    assert games[0].home_team == "CHI"
    assert games[0].away_team == "GB"
    assert games[0].home_score == 24
    assert games[0].away_score == 17
    assert games[0].is_neutral is False


def test_create_history_table_loads_elo_record() -> None:
    """Create the Elo history table and load a prediction record."""

    history_record = EloHistoryRecord(
        game_id="test_game",
        season=2025,
        game_type="REG",
        week=1,
        gameday=date(2025, 9, 7),
        home_team="CHI",
        away_team="GB",
        home_franchise="CHI",
        away_franchise="GB",
        is_neutral=False,
        home_advantage=50.0,
        home_rating_pre=1500.0,
        away_rating_pre=1500.0,
        home_win_probability=0.571463,
        away_win_probability=0.428537,
        actual_home_score=1.0,
        home_rating_post=1508.57074,
        away_rating_post=1491.42926,
        home_rating_change=8.57074,
    )

    with duckdb.connect(":memory:") as connection:
        create_history_table(
            connection=connection,
            history_records=[history_record],
        )

        loaded_record = connection.execute(
            """
            SELECT
                game_id,
                home_rating_pre,
                home_win_probability,
                home_rating_post
            FROM analytics.elo_game_predictions
            """
        ).fetchone()

    assert loaded_record[0] == "test_game"
    assert loaded_record[1] == pytest.approx(1500.0)
    assert loaded_record[2] == pytest.approx(0.571463)
    assert loaded_record[3] == pytest.approx(1508.57074)


def test_create_current_ratings_table_ranks_teams() -> None:
    """Create current ratings ordered by descending Elo rating."""

    history_record = EloHistoryRecord(
        game_id="test_game",
        season=2025,
        game_type="REG",
        week=1,
        gameday=date(2025, 9, 7),
        home_team="CHI",
        away_team="GB",
        home_franchise="CHI",
        away_franchise="GB",
        is_neutral=False,
        home_advantage=0.0,
        home_rating_pre=1500.0,
        away_rating_pre=1500.0,
        home_win_probability=0.5,
        away_win_probability=0.5,
        actual_home_score=1.0,
        home_rating_post=1510.0,
        away_rating_post=1490.0,
        home_rating_change=10.0,
    )

    with duckdb.connect(":memory:") as connection:
        create_current_ratings_table(
            connection=connection,
            history_records=[history_record],
            final_ratings={
                "CHI": 1510.0,
                "GB": 1490.0,
            },
        )

        rows = connection.execute(
            """
            SELECT
                elo_rank,
                team,
                elo_rating,
                games_played,
                last_game_id,
                as_of_gameday,
                last_completed_season
            FROM analytics.current_elo_ratings
            ORDER BY elo_rank
            """
        ).fetchall()

    assert len(rows) == 2

    assert rows[0][0] == 1
    assert rows[0][1] == "CHI"
    assert rows[0][2] == pytest.approx(1510.0)
    assert rows[0][3] == 1
    assert rows[0][4] == "test_game"
    assert rows[0][5] == date(2025, 9, 7)
    assert rows[0][6] == 2025

    assert rows[1][0] == 2
    assert rows[1][1] == "GB"
    assert rows[1][2] == pytest.approx(1490.0)


def test_build_elo_ratings_creates_target_tables(
    tmp_path: Path,
) -> None:
    """Build both Elo target tables from processed schedule data."""

    database_file = tmp_path / "test_elo.duckdb"

    with duckdb.connect(str(database_file)) as connection:
        create_schedule_table(connection)

        connection.execute(
            """
            INSERT INTO processed.schedule
            VALUES (
                'test_game',
                2025,
                'REG',
                1,
                DATE '2025-09-07',
                TIME '13:00:00',
                'CHI',
                'GB',
                24,
                17,
                'Home',
                TRUE
            )
            """
        )

    build_elo_ratings(database_file)

    with duckdb.connect(str(database_file)) as connection:
        history_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM analytics.elo_game_predictions
            """
        ).fetchone()[0]

        current_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM analytics.current_elo_ratings
            """
        ).fetchone()[0]

        prediction = connection.execute(
            """
            SELECT
                home_team,
                away_team,
                home_win_probability,
                actual_home_score
            FROM analytics.elo_game_predictions
            """
        ).fetchone()

    assert history_count == 1
    assert current_count == 2
    assert prediction[0] == "CHI"
    assert prediction[1] == "GB"
    assert prediction[2] > 0.5
    assert prediction[3] == pytest.approx(1.0)


def test_failed_build_preserves_previous_elo_tables(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Roll back both Elo tables when final validation fails."""

    database_file = tmp_path / "rollback_elo.duckdb"

    with duckdb.connect(str(database_file)) as connection:
        create_schedule_table(connection)

        connection.execute(
            """
            INSERT INTO processed.schedule
            VALUES (
                'test_game',
                2025,
                'REG',
                1,
                DATE '2025-09-07',
                TIME '13:00:00',
                'CHI',
                'GB',
                24,
                17,
                'Home',
                TRUE
            )
            """
        )

        connection.execute("CREATE SCHEMA analytics")
        connection.execute(
            """
            CREATE TABLE analytics.elo_game_predictions (
                marker VARCHAR
            )
            """
        )
        connection.execute(
            """
            INSERT INTO analytics.elo_game_predictions
            VALUES ('previous_history')
            """
        )

        connection.execute(
            """
            CREATE TABLE analytics.current_elo_ratings (
                marker VARCHAR
            )
            """
        )
        connection.execute(
            """
            INSERT INTO analytics.current_elo_ratings
            VALUES ('previous_current')
            """
        )

    def fail_validation(
        connection: duckdb.DuckDBPyConnection,
        expected_team_count: int,
    ) -> None:
        """Force a failure after both target tables are replaced."""

        del connection
        del expected_team_count

        raise RuntimeError(
            "Forced current Elo validation failure."
        )

    monkeypatch.setattr(
        "src.analytics.build_elo_ratings."
        "validate_current_ratings_table",
        fail_validation,
    )

    with pytest.raises(
        RuntimeError,
        match="Forced current Elo validation failure",
    ):
        build_elo_ratings(database_file)

    with duckdb.connect(str(database_file)) as connection:
        history_marker = connection.execute(
            """
            SELECT marker
            FROM analytics.elo_game_predictions
            """
        ).fetchone()[0]

        current_marker = connection.execute(
            """
            SELECT marker
            FROM analytics.current_elo_ratings
            """
        ).fetchone()[0]

    assert history_marker == "previous_history"
    assert current_marker == "previous_current"