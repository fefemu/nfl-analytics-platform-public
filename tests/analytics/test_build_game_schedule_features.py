"""
Tests for game schedule feature building.
"""

from collections.abc import Iterator

import duckdb
import pytest

from src.analytics.build_game_schedule_features import (
    TARGET_FULL_NAME,
    create_game_schedule_features_table,
    validate_game_schedule_features_table,
    validate_source_table,
)


@pytest.fixture
def connection() -> Iterator[duckdb.DuckDBPyConnection]:
    """Create an in-memory processed schedule."""

    with duckdb.connect(":memory:") as database:
        create_source_table(database)
        yield database


def create_source_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create representative schedule rest values."""

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
            away_team VARCHAR,
            home_rest INTEGER,
            away_rest INTEGER
        );

        INSERT INTO processed.schedule
        VALUES
            (
                '2024_01_A_B',
                2024,
                'REG',
                1,
                DATE '2024-09-05',
                'A',
                'B',
                7,
                7
            ),
            (
                '2024_02_C_D',
                2024,
                'REG',
                2,
                DATE '2024-09-12',
                'C',
                'D',
                4,
                7
            ),
            (
                '2024_03_E_F',
                2024,
                'REG',
                3,
                DATE '2024-09-22',
                'E',
                'F',
                10,
                6
            ),
            (
                '2024_04_G_H',
                2024,
                'REG',
                4,
                DATE '2024-10-06',
                'G',
                'H',
                14,
                7
            );
        """
    )


def test_validate_source_table_accepts_rest_data(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Accept complete non-negative rest values."""

    validate_source_table(connection)


def test_create_schedule_features_builds_rest_differences(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Build raw rest-day differences."""

    create_game_schedule_features_table(connection)

    rows = connection.execute(
        f"""
        SELECT
            game_id,
            home_rest_days,
            away_rest_days,
            rest_days_difference
        FROM {TARGET_FULL_NAME}
        ORDER BY game_id
        """
    ).fetchall()

    assert rows == [
        (
            "2024_01_A_B",
            7,
            7,
            0,
        ),
        (
            "2024_02_C_D",
            4,
            7,
            -3,
        ),
        (
            "2024_03_E_F",
            10,
            6,
            4,
        ),
        (
            "2024_04_G_H",
            14,
            7,
            7,
        ),
    ]


def test_create_schedule_features_builds_context_flags(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Classify short, extended and post-bye rest."""

    create_game_schedule_features_table(connection)

    rows = connection.execute(
        f"""
        SELECT
            game_id,
            home_short_week,
            away_short_week,
            home_extended_rest,
            away_extended_rest,
            home_post_bye,
            away_post_bye
        FROM {TARGET_FULL_NAME}
        ORDER BY game_id
        """
    ).fetchall()

    assert rows == [
        (
            "2024_01_A_B",
            False,
            False,
            False,
            False,
            False,
            False,
        ),
        (
            "2024_02_C_D",
            True,
            False,
            False,
            False,
            False,
            False,
        ),
        (
            "2024_03_E_F",
            False,
            True,
            True,
            False,
            False,
            False,
        ),
        (
            "2024_04_G_H",
            False,
            False,
            True,
            False,
            True,
            False,
        ),
    ]


def test_created_schedule_features_pass_validation(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Accept correctly derived schedule features."""

    create_game_schedule_features_table(connection)

    validate_game_schedule_features_table(
        connection
    )


def test_validate_source_table_rejects_missing_rest() -> None:
    """Reject schedule data with missing rest values."""

    with duckdb.connect(":memory:") as database:
        database.execute(
            """
            CREATE SCHEMA processed;

            CREATE TABLE processed.schedule (
                game_id VARCHAR,
                season INTEGER,
                game_type VARCHAR,
                week INTEGER,
                gameday DATE,
                home_team VARCHAR,
                away_team VARCHAR,
                home_rest INTEGER,
                away_rest INTEGER
            );

            INSERT INTO processed.schedule
            VALUES
                (
                    '2024_01_A_B',
                    2024,
                    'REG',
                    1,
                    DATE '2024-09-05',
                    'A',
                    'B',
                    NULL,
                    7
                );
            """
        )

        with pytest.raises(
            RuntimeError,
            match="Invalid rest values",
        ):
            validate_source_table(database)