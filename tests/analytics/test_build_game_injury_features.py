"""Tests for game-level injury features."""

import duckdb
import pytest

from src.analytics.build_game_injury_features import (
    TARGET_FULL_NAME,
    create_game_injury_features,
    validate_source_table,
    validate_target_table,
)


def create_source_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create deterministic team-game injury burdens."""

    connection.execute(
        """
        CREATE SCHEMA analytics;

        CREATE TABLE analytics.team_game_injury_burden (
            game_id VARCHAR,
            season INTEGER,
            game_type VARCHAR,
            week INTEGER,
            gameday DATE,
            team VARCHAR,
            opponent VARCHAR,
            is_home BOOLEAN,
            has_injury_report_data BOOLEAN,
            injury_report_player_count INTEGER,
            out_player_count INTEGER,
            doubtful_player_count INTEGER,
            questionable_player_count INTEGER,
            starter_out_count INTEGER,
            qb_out_count INTEGER,
            total_injury_burden DOUBLE,
            qb_injury_burden DOUBLE,
            non_qb_injury_burden DOUBLE,
            offense_injury_burden DOUBLE,
            defense_injury_burden DOUBLE,
            special_teams_injury_burden DOUBLE
        );

        INSERT INTO analytics.team_game_injury_burden
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
                TRUE,
                12,
                2,
                1,
                3,
                1,
                0,
                1.80,
                0.00,
                1.80,
                1.10,
                0.60,
                0.10
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
                TRUE,
                10,
                1,
                0,
                2,
                0,
                0,
                0.90,
                0.00,
                0.90,
                0.40,
                0.40,
                0.10
            ),
            (
                '2025_03_MIA_NYJ',
                2025,
                'REG',
                3,
                DATE '2025-09-21',
                'MIA',
                'NYJ',
                TRUE,
                TRUE,
                8,
                1,
                0,
                1,
                0,
                0,
                0.70,
                0.00,
                0.70,
                0.30,
                0.30,
                0.10
            ),
            (
                '2025_03_MIA_NYJ',
                2025,
                'REG',
                3,
                DATE '2025-09-21',
                'NYJ',
                'MIA',
                FALSE,
                FALSE,
                0,
                0,
                0,
                0,
                0,
                0,
                0.00,
                0.00,
                0.00,
                0.00,
                0.00,
                0.00
            );
        """
    )


def test_validate_source_table_accepts_valid_source(
) -> None:
    """Accept a valid team-game injury source."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )

        validate_source_table(
            connection
        )


def test_create_features_builds_one_row_per_game(
) -> None:
    """Build one row with home and away injury data."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )

        create_game_injury_features(
            connection
        )

        result = connection.execute(
            f"""
            SELECT
                game_id,
                home_team,
                away_team,
                home_has_injury_report_data,
                away_has_injury_report_data,
                has_complete_injury_data
            FROM {TARGET_FULL_NAME}
            WHERE game_id = '2025_03_NE_BUF'
            """
        ).fetchone()

    assert result == (
        "2025_03_NE_BUF",
        "NE",
        "BUF",
        True,
        True,
        True,
    )


def test_create_features_calculates_home_away_differences(
) -> None:
    """Calculate home-minus-away injury differences."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )

        create_game_injury_features(
            connection
        )

        result = connection.execute(
            f"""
            SELECT
                total_injury_burden_difference,
                non_qb_injury_burden_difference,
                offense_injury_burden_difference,
                defense_injury_burden_difference,
                special_teams_injury_burden_difference,
                out_player_count_difference,
                starter_out_count_difference
            FROM {TARGET_FULL_NAME}
            WHERE game_id = '2025_03_NE_BUF'
            """
        ).fetchone()

    assert result[0] == pytest.approx(
        0.90
    )
    assert result[1] == pytest.approx(
        0.90
    )
    assert result[2] == pytest.approx(
        0.70
    )
    assert result[3] == pytest.approx(
        0.20
    )
    assert result[4] == pytest.approx(
        0.00
    )
    assert result[5:] == (
        1,
        1,
    )


def test_create_features_preserves_missing_data(
) -> None:
    """Mark a game incomplete when one team lacks data."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )

        create_game_injury_features(
            connection
        )

        result = connection.execute(
            f"""
            SELECT
                home_has_injury_report_data,
                away_has_injury_report_data,
                has_complete_injury_data,
                home_non_qb_injury_burden,
                away_non_qb_injury_burden,
                non_qb_injury_burden_difference
            FROM {TARGET_FULL_NAME}
            WHERE game_id = '2025_03_MIA_NYJ'
            """
        ).fetchone()

    assert result[:3] == (
        True,
        False,
        False,
    )
    assert result[3] == pytest.approx(
        0.70
    )
    assert result[4] == pytest.approx(
        0.00
    )
    assert result[5] == pytest.approx(
        0.70
    )


def test_validate_target_table_accepts_valid_features(
) -> None:
    """Accept valid game-level injury features."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_source_table(
            connection
        )

        create_game_injury_features(
            connection
        )

        result = validate_target_table(
            connection=connection,
            expected_game_count=2,
        )

    assert result == (
        2,
        1,
    )