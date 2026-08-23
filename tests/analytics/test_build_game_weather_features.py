"""Tests for game weather feature construction."""

import duckdb
import pytest

from src.analytics.build_game_weather_features import (
    TARGET_FULL_NAME,
    create_game_weather_features_table,
    validate_game_weather_features,
    validate_weather_source,
)


def create_schedule_source(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create representative schedule weather rows."""

    connection.execute(
        """
        CREATE SCHEMA processed;

        CREATE TABLE processed.schedule (
            game_id VARCHAR,
            season INTEGER,
            gameday DATE,
            home_team VARCHAR,
            away_team VARCHAR,
            roof VARCHAR,
            surface VARCHAR,
            stadium_id VARCHAR,
            stadium VARCHAR,
            temp INTEGER,
            wind INTEGER
        );

        INSERT INTO processed.schedule
        VALUES
            (
                'cold_outdoor',
                2025,
                DATE '2025-12-20',
                'BUF',
                'NYJ',
                'outdoors',
                'grass',
                'stadium_1',
                'Cold Stadium',
                25,
                18
            ),
            (
                'indoor',
                2025,
                DATE '2025-10-10',
                'DET',
                'MIN',
                'dome',
                'fieldturf',
                'stadium_2',
                'Indoor Stadium',
                NULL,
                NULL
            ),
            (
                'hot_open',
                2025,
                DATE '2025-09-07',
                'ARI',
                'SEA',
                'open',
                'grass',
                'stadium_3',
                'Open Stadium',
                95,
                6
            ),
            (
                'missing_outdoor_weather',
                2026,
                DATE '2026-09-10',
                'NE',
                'MIA',
                'outdoors',
                'grass',
                'stadium_4',
                'Future Stadium',
                NULL,
                NULL
            ),
            (
                'unknown_roof',
                2026,
                DATE '2026-09-11',
                'KC',
                'LV',
                NULL,
                'grass',
                'stadium_5',
                'Unknown Stadium',
                NULL,
                NULL
            );
        """
    )


def test_create_weather_features() -> None:
    """Create one normalized row per game."""

    with duckdb.connect(":memory:") as connection:
        create_schedule_source(connection)
        validate_weather_source(connection)

        create_game_weather_features_table(
            connection
        )

        validate_game_weather_features(
            connection
        )

        row_count = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM {TARGET_FULL_NAME}
            """
        ).fetchone()[0]

    assert row_count == 5


def test_outdoor_extreme_weather() -> None:
    """Derive cold and high-wind features."""

    with duckdb.connect(":memory:") as connection:
        create_schedule_source(connection)

        create_game_weather_features_table(
            connection
        )

        result = connection.execute(
            f"""
            SELECT
                is_weather_exposed,
                has_game_weather,
                modeled_temperature_f,
                modeled_wind_mph,
                is_freezing,
                is_high_wind,
                cold_degrees_below_50,
                wind_mph_above_10
            FROM {TARGET_FULL_NAME}
            WHERE game_id = 'cold_outdoor'
            """
        ).fetchone()

    assert result == (
        True,
        True,
        25.0,
        18.0,
        True,
        True,
        25.0,
        8.0,
    )


def test_indoor_uses_neutral_weather() -> None:
    """Use controlled neutral indoor conditions."""

    with duckdb.connect(":memory:") as connection:
        create_schedule_source(connection)

        create_game_weather_features_table(
            connection
        )

        result = connection.execute(
            f"""
            SELECT
                is_indoor,
                is_weather_exposed,
                has_game_weather,
                modeled_temperature_f,
                modeled_wind_mph,
                is_freezing,
                is_high_wind
            FROM {TARGET_FULL_NAME}
            WHERE game_id = 'indoor'
            """
        ).fetchone()

    assert result == (
        True,
        False,
        False,
        65.0,
        0.0,
        False,
        False,
    )


def test_hot_weather_features() -> None:
    """Derive extreme-heat exposure."""

    with duckdb.connect(":memory:") as connection:
        create_schedule_source(connection)

        create_game_weather_features_table(
            connection
        )

        result = connection.execute(
            f"""
            SELECT
                is_extreme_heat,
                heat_degrees_above_80
            FROM {TARGET_FULL_NAME}
            WHERE game_id = 'hot_open'
            """
        ).fetchone()

    assert result[0] is True
    assert result[1] == pytest.approx(15.0)


def test_missing_outdoor_weather_is_explicit(
) -> None:
    """Preserve exposed games with missing weather."""

    with duckdb.connect(":memory:") as connection:
        create_schedule_source(connection)

        create_game_weather_features_table(
            connection
        )

        result = connection.execute(
            f"""
            SELECT
                is_weather_exposed,
                has_game_weather,
                modeled_temperature_f,
                modeled_wind_mph
            FROM {TARGET_FULL_NAME}
            WHERE game_id
                = 'missing_outdoor_weather'
            """
        ).fetchone()

    assert result == (
        True,
        False,
        65.0,
        0.0,
    )


def test_unknown_roof_is_preserved() -> None:
    """Preserve games without a known roof type."""

    with duckdb.connect(":memory:") as connection:
        create_schedule_source(connection)

        create_game_weather_features_table(
            connection
        )

        result = connection.execute(
            f"""
            SELECT
                roof_type,
                is_indoor,
                is_weather_exposed,
                has_game_weather,
                modeled_temperature_f,
                modeled_wind_mph
            FROM {TARGET_FULL_NAME}
            WHERE game_id = 'unknown_roof'
            """
        ).fetchone()

    assert result == (
        "unknown",
        False,
        False,
        False,
        65.0,
        0.0,
    )


def test_invalid_raw_weather_is_rejected() -> None:
    """Reject physically implausible weather values."""

    with duckdb.connect(":memory:") as connection:
        create_schedule_source(connection)

        connection.execute(
            """
            UPDATE processed.schedule
            SET wind = -5
            WHERE game_id = 'cold_outdoor'
            """
        )

        create_game_weather_features_table(
            connection
        )

        with pytest.raises(
            RuntimeError,
            match="Invalid raw game weather",
        ):
            validate_game_weather_features(
                connection
            )