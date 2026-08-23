"""
Tests for the game quarterback features builder.
"""

from collections.abc import Iterator

import duckdb
import pytest

from src.analytics.build_game_qb_features import (
    AUDIT_FULL_NAME,
    FEATURE_FULL_NAME,
    create_game_qb_audit_table,
    create_game_qb_features_table,
    validate_game_qb_audit_table,
    validate_game_qb_features_table,
    validate_source_tables,
)


@pytest.fixture
def connection() -> Iterator[duckdb.DuckDBPyConnection]:
    """Create an in-memory DuckDB connection."""

    with duckdb.connect(":memory:") as database:
        create_source_tables(database)
        yield database


def create_source_tables(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create minimal source tables for game QB feature tests."""

    connection.execute(
        """
        CREATE SCHEMA processed;
        CREATE SCHEMA analytics;

        CREATE TABLE processed.schedule (
            game_id VARCHAR,
            season INTEGER,
            game_type VARCHAR,
            week INTEGER,
            gameday DATE,
            home_team VARCHAR,
            away_team VARCHAR,
            home_qb_id VARCHAR,
            home_qb_name VARCHAR,
            away_qb_id VARCHAR,
            away_qb_name VARCHAR,
            is_completed BOOLEAN
        );

        CREATE TABLE processed.qb_game_performance (
            game_id VARCHAR,
            game_date DATE,
            team VARCHAR,
            opponent VARCHAR,
            qb_id VARCHAR,
            qb_name VARCHAR,
            is_primary_qb BOOLEAN,
            is_listed_starter BOOLEAN,
            dropbacks INTEGER
        );

        CREATE TABLE analytics.qb_rating_history (
            game_id VARCHAR,
            game_date DATE,
            team VARCHAR,
            opponent VARCHAR,
            qb_id VARCHAR,
            qb_name VARCHAR,
            pregame_effective_dropbacks DOUBLE,
            pregame_qb_rating DOUBLE,
            pregame_prior_weight DOUBLE,
            pregame_rating_standard_error DOUBLE
        );
        """
    )

    connection.execute(
        """
        INSERT INTO processed.schedule
        VALUES
            (
                '2025_01_A_B',
                2025,
                'REG',
                1,
                DATE '2025-09-07',
                'A',
                'B',
                'HOME_LISTED',
                'Listed Home QB',
                'AWAY_STARTER',
                'Away Starter',
                TRUE
            ),
            (
                '2019_01_DEN_OAK',
                2019,
                'REG',
                1,
                DATE '2019-09-09',
                'OAK',
                'DEN',
                'RAIDERS_QB',
                'Raiders QB',
                'DENVER_QB',
                'Denver QB',
                TRUE
            );
        """
    )

    connection.execute(
        """
        INSERT INTO processed.qb_game_performance
        VALUES
            (
                '2025_01_A_B',
                DATE '2025-09-07',
                'A',
                'B',
                'HOME_ACTUAL',
                'Actual Home QB',
                TRUE,
                FALSE,
                34
            ),
            (
                '2025_01_A_B',
                DATE '2025-09-07',
                'B',
                'A',
                'AWAY_STARTER',
                'Away Starter',
                TRUE,
                TRUE,
                31
            ),
            (
                '2019_01_DEN_OAK',
                DATE '2019-09-09',
                'LV',
                'DEN',
                'RAIDERS_QB',
                'Raiders QB',
                TRUE,
                TRUE,
                36
            ),
            (
                '2019_01_DEN_OAK',
                DATE '2019-09-09',
                'DEN',
                'LV',
                'DENVER_QB',
                'Denver QB',
                TRUE,
                TRUE,
                33
            );
        """
    )

    connection.execute(
        """
        INSERT INTO analytics.qb_rating_history
        VALUES
            (
                '2025_01_A_B',
                DATE '2025-09-07',
                'A',
                'B',
                'HOME_ACTUAL',
                'Actual Home QB',
                300.0,
                104.0,
                0.40,
                0.80
            ),
            (
                '2025_01_A_B',
                DATE '2025-09-07',
                'B',
                'A',
                'AWAY_STARTER',
                'Away Starter',
                500.0,
                101.0,
                0.29,
                0.65
            ),
            (
                '2019_01_DEN_OAK',
                DATE '2019-09-09',
                'LV',
                'DEN',
                'RAIDERS_QB',
                'Raiders QB',
                450.0,
                103.0,
                0.31,
                0.70
            ),
            (
                '2019_01_DEN_OAK',
                DATE '2019-09-09',
                'DEN',
                'LV',
                'DENVER_QB',
                'Denver QB',
                420.0,
                99.0,
                0.32,
                0.72
            );
        """
    )


def build_target_tables(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Build both target tables for a test."""

    create_game_qb_features_table(connection)
    create_game_qb_audit_table(connection)


def test_validate_source_tables_accepts_valid_sources(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Accept source tables containing every required column."""

    validate_source_tables(connection)


def test_feature_table_uses_only_listed_starter_rating(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Do not substitute the postgame primary QB for the listed QB."""

    create_game_qb_features_table(connection)

    row = connection.execute(
        f"""
        SELECT
            home_listed_qb_id,
            home_listed_qb_rating,
            home_listed_qb_rating_available,
            away_listed_qb_rating,
            away_listed_qb_rating_available,
            both_listed_qb_ratings_available,
            listed_qb_rating_difference
        FROM {FEATURE_FULL_NAME}
        WHERE game_id = '2025_01_A_B'
        """
    ).fetchone()

    assert row == (
        "HOME_LISTED",
        None,
        False,
        101.0,
        True,
        False,
        None,
    )


def test_audit_table_identifies_actual_primary_qb(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Keep actual primary QB information in the audit table only."""

    build_target_tables(connection)

    row = connection.execute(
        f"""
        SELECT
            home_listed_qb_id,
            home_actual_primary_qb_id,
            home_listed_qb_matches_actual_primary,
            home_actual_primary_qb_pregame_rating,
            away_listed_qb_matches_actual_primary,
            both_listed_qbs_match_actual_primary
        FROM {AUDIT_FULL_NAME}
        WHERE game_id = '2025_01_A_B'
        """
    ).fetchone()

    assert row == (
        "HOME_LISTED",
        "HOME_ACTUAL",
        False,
        104.0,
        True,
        False,
    )


def test_historical_team_alias_matches_pbp_team_code(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Match historical OAK schedule data to normalized LV PBP data."""

    build_target_tables(connection)

    feature_row = connection.execute(
        f"""
        SELECT
            home_team,
            home_listed_qb_rating,
            away_listed_qb_rating,
            both_listed_qb_ratings_available,
            listed_qb_rating_difference
        FROM {FEATURE_FULL_NAME}
        WHERE game_id = '2019_01_DEN_OAK'
        """
    ).fetchone()

    audit_row = connection.execute(
        f"""
        SELECT
            home_actual_primary_qb_id,
            home_listed_qb_matches_actual_primary,
            both_listed_qbs_match_actual_primary
        FROM {AUDIT_FULL_NAME}
        WHERE game_id = '2019_01_DEN_OAK'
        """
    ).fetchone()

    assert feature_row == (
        "OAK",
        103.0,
        99.0,
        True,
        4.0,
    )

    assert audit_row == (
        "RAIDERS_QB",
        True,
        True,
    )


def test_created_tables_pass_validation(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Accept internally consistent feature and audit tables."""

    build_target_tables(connection)

    validate_game_qb_features_table(connection)
    validate_game_qb_audit_table(connection)