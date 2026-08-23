"""Tests for current production data access."""

import duckdb
import pandas as pd
import pytest

from src.modeling.current_production_data import (
    load_current_production_inputs,
    load_production_training_data,
    validate_current_production_sources,
)
from src.modeling.production_probability_model import (
    EXTERNAL_ELO_FEATURE,
    EXTERNAL_QB_FEATURE,
)


@pytest.fixture
def connection() -> duckdb.DuckDBPyConnection:
    """Create complete in-memory production sources."""

    database = duckdb.connect(":memory:")

    database.execute(
        """
        CREATE SCHEMA processed;
        CREATE SCHEMA analytics;

        CREATE TABLE processed.schedule (
            game_id VARCHAR,
            season INTEGER,
            game_type VARCHAR,
            week INTEGER,
            gameday DATE,
            gametime VARCHAR,
            home_team VARCHAR,
            away_team VARCHAR,
            location VARCHAR,
            home_qb_id VARCHAR,
            home_qb_name VARCHAR,
            away_qb_id VARCHAR,
            away_qb_name VARCHAR,
            is_completed BOOLEAN
        );

        CREATE TABLE analytics.current_elo_ratings (
            team VARCHAR,
            elo_rating DOUBLE,
            as_of_gameday DATE,
            last_completed_season INTEGER
        );

        CREATE TABLE analytics.current_qb_ratings (
            qb_id VARCHAR,
            qb_name VARCHAR,
            current_team VARCHAR,
            qb_rating DOUBLE,
            as_of_date DATE,
            rating_standard_error DOUBLE
        );

        CREATE TABLE analytics.game_injury_features (
            game_id VARCHAR,
            has_complete_injury_data BOOLEAN,
            offense_injury_burden_difference DOUBLE,
            defense_injury_burden_difference DOUBLE,
            special_teams_injury_burden_difference DOUBLE
        );

        CREATE TABLE analytics.game_modeling_dataset (
            game_id VARCHAR,
            season INTEGER,
            game_date DATE,
            target_home_win INTEGER,
            has_complete_injury_data BOOLEAN,
            listed_qb_rating_difference DOUBLE,
            offense_injury_burden_difference DOUBLE,
            defense_injury_burden_difference DOUBLE,
            special_teams_injury_burden_difference DOUBLE
        );

        CREATE TABLE processed.external_nfelo_game_ratings (
            normalized_game_id VARCHAR,
            source_season INTEGER,
            source_week INTEGER,
            away_team VARCHAR,
            home_team VARCHAR,
            starting_nfelo_home DOUBLE,
            starting_nfelo_away DOUBLE,
            home_538_qb_adj DOUBLE,
            away_538_qb_adj DOUBLE,
            nfelo_home_probability_open DOUBLE
        );

        INSERT INTO processed.schedule
        VALUES
            (
                '2026_01_NE_NYJ',
                2026,
                'REG',
                1,
                DATE '2026-09-10',
                '20:20',
                'NE',
                'NYJ',
                'Home',
                'QB_NE',
                'New England QB',
                'QB_NYJ',
                'New York QB',
                FALSE
            ),
            (
                '2026_01_BUF_KC',
                2026,
                'REG',
                1,
                DATE '2026-09-13',
                '13:00',
                'BUF',
                'KC',
                'Home',
                'QB_BUF',
                'Buffalo QB',
                NULL,
                NULL,
                FALSE
            ),
            (
                '2026_PRE_MIA_TB',
                2026,
                'PRE',
                1,
                DATE '2026-08-10',
                '18:00',
                'MIA',
                'TB',
                'Home',
                NULL,
                NULL,
                NULL,
                NULL,
                FALSE
            ),
            (
                '2025_18_NE_BUF',
                2025,
                'REG',
                18,
                DATE '2026-01-04',
                '13:00',
                'NE',
                'BUF',
                'Home',
                'QB_NE',
                'New England QB',
                'QB_BUF',
                'Buffalo QB',
                TRUE
            );

        INSERT INTO analytics.current_elo_ratings
        VALUES
            (
                'NE',
                1510.0,
                DATE '2026-01-04',
                2025
            ),
            (
                'NYJ',
                1475.0,
                DATE '2026-01-04',
                2025
            ),
            (
                'BUF',
                1570.0,
                DATE '2026-01-04',
                2025
            ),
            (
                'KC',
                1560.0,
                DATE '2026-01-04',
                2025
            ),
            (
                'MIA',
                1500.0,
                DATE '2026-01-04',
                2025
            ),
            (
                'TB',
                1500.0,
                DATE '2026-01-04',
                2025
            );

        INSERT INTO analytics.current_qb_ratings
        VALUES
            (
                'QB_NE',
                'New England QB',
                'NE',
                5.5,
                DATE '2026-01-04',
                1.1
            ),
            (
                'QB_NYJ',
                'New York QB',
                'NYJ',
                2.0,
                DATE '2026-01-04',
                1.3
            ),
            (
                'QB_BUF',
                'Buffalo QB',
                'BUF',
                7.0,
                DATE '2026-01-04',
                0.9
            );

        INSERT INTO analytics.game_injury_features
        VALUES
            (
                '2026_01_NE_NYJ',
                TRUE,
                -0.20,
                0.10,
                -0.05
            );

        INSERT INTO analytics.game_modeling_dataset
        VALUES
            (
                '2024_01_A_B',
                2024,
                DATE '2024-09-01',
                1,
                TRUE,
                3.0,
                -0.10,
                -0.05,
                0.00
            ),
            (
                '2025_01_C_D',
                2025,
                DATE '2025-09-01',
                0,
                TRUE,
                -2.0,
                0.10,
                0.05,
                0.00
            );

        INSERT INTO processed.external_nfelo_game_ratings
        VALUES
            (
                '2026_01_NE_NYJ',
                2026,
                1,
                'NYJ',
                'NE',
                1610.0,
                1420.0,
                12.0,
                -3.0,
                0.72
            ),
            (
                '2026_01_BUF_KC',
                2026,
                1,
                'KC',
                'BUF',
                1650.0,
                1575.0,
                8.0,
                4.0,
                0.61
            ),
            (
                '2024_01_A_B',
                2024,
                1,
                'A',
                'B',
                1580.0,
                1500.0,
                6.0,
                1.0,
                0.64
            ),
            (
                '2025_01_C_D',
                2025,
                1,
                'C',
                'D',
                1470.0,
                1540.0,
                -2.0,
                5.0,
                0.39
            );
        """
    )

    yield database

    database.close()


def test_validate_current_production_sources(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Accept complete current production sources."""

    validate_current_production_sources(
        connection
    )


def test_load_current_production_inputs(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Load upcoming regular-season games."""

    inputs = load_current_production_inputs(
        connection
    )

    assert list(inputs["game_id"]) == [
        "2026_01_NE_NYJ",
        "2026_01_BUF_KC",
    ]

    complete_row = inputs.loc[
        inputs["game_id"]
        == "2026_01_NE_NYJ"
    ].iloc[0]

    assert (
        complete_row[
            "home_listed_qb_rating"
        ]
        == pytest.approx(5.5)
    )

    assert (
        complete_row[
            "away_listed_qb_rating"
        ]
        == pytest.approx(2.0)
    )

    assert bool(
        complete_row[
            "has_complete_injury_data"
        ]
    )

    assert (
        complete_row[
            "offense_injury_burden_difference"
        ]
        == pytest.approx(-0.20)
    )


def test_load_inputs_contains_external_nfelo(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Load external rating and probability inputs."""

    inputs = load_current_production_inputs(
        connection
    )

    row = inputs.loc[
        inputs["game_id"]
        == "2026_01_NE_NYJ"
    ].iloc[0]

    assert (
        row["home_external_nfelo_rating"]
        == pytest.approx(1610.0)
    )

    assert (
        row["away_external_nfelo_rating"]
        == pytest.approx(1420.0)
    )

    assert (
        row[EXTERNAL_ELO_FEATURE]
        == pytest.approx(190.0)
    )

    assert (
        row[
            "home_external_nfelo_qb_adjustment"
        ]
        == pytest.approx(12.0)
    )

    assert (
        row[
            "away_external_nfelo_qb_adjustment"
        ]
        == pytest.approx(-3.0)
    )

    assert (
        row[EXTERNAL_QB_FEATURE]
        == pytest.approx(15.0)
    )

    assert (
        row[
            "published_nfelo_home_probability"
        ]
        == pytest.approx(0.72)
    )


def test_load_inputs_preserves_fallback_game(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Preserve games with missing QB and injury data."""

    inputs = load_current_production_inputs(
        connection
    )

    fallback_row = inputs.loc[
        inputs["game_id"]
        == "2026_01_BUF_KC"
    ].iloc[0]

    assert pd.isna(
        fallback_row[
            "away_listed_qb_rating"
        ]
    )

    assert pd.isna(
        fallback_row[
            "has_complete_injury_data"
        ]
    )

    assert pd.isna(
        fallback_row[
            "defense_injury_burden_difference"
        ]
    )

    assert (
        fallback_row[EXTERNAL_ELO_FEATURE]
        == pytest.approx(75.0)
    )

    assert (
        fallback_row[EXTERNAL_QB_FEATURE]
        == pytest.approx(4.0)
    )

    assert (
        fallback_row[
            "published_nfelo_home_probability"
        ]
        == pytest.approx(0.61)
    )


def test_load_production_training_data(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Load historical external-model training rows."""

    historical_data = (
        load_production_training_data(
            connection
        )
    )

    assert list(
        historical_data["game_id"]
    ) == [
        "2024_01_A_B",
        "2025_01_C_D",
    ]

    assert (
        historical_data["season"].max()
        < 2026
    )

    assert {
        EXTERNAL_ELO_FEATURE,
        "listed_qb_rating_difference",
        EXTERNAL_QB_FEATURE,
        "offense_injury_burden_difference",
        "defense_injury_burden_difference",
        "special_teams_injury_burden_difference",
        "published_nfelo_home_probability",
    }.issubset(
        historical_data.columns
    )

    first_row = historical_data.iloc[0]

    assert (
        first_row[EXTERNAL_ELO_FEATURE]
        == pytest.approx(80.0)
    )

    assert (
        first_row[EXTERNAL_QB_FEATURE]
        == pytest.approx(5.0)
    )

    assert (
        first_row[
            "published_nfelo_home_probability"
        ]
        == pytest.approx(0.64)
    )


def test_source_validation_rejects_missing_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Reject a missing current source table."""

    connection.execute(
        """
        DROP TABLE analytics.current_qb_ratings
        """
    )

    with pytest.raises(
        RuntimeError,
        match="does not exist",
    ):
        validate_current_production_sources(
            connection
        )


def test_source_validation_requires_external_nfelo(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Reject a missing external nfelo source."""

    connection.execute(
        """
        DROP TABLE processed.external_nfelo_game_ratings
        """
    )

    with pytest.raises(
        RuntimeError,
        match="external_nfelo_game_ratings",
    ):
        validate_current_production_sources(
            connection
        )


def test_input_load_rejects_missing_internal_elo(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Reject games without transitional internal Elo."""

    connection.execute(
        """
        DELETE FROM analytics.current_elo_ratings
        WHERE team = 'KC'
        """
    )

    with pytest.raises(
        RuntimeError,
        match="missing current internal Elo ratings",
    ):
        load_current_production_inputs(
            connection
        )


def test_input_load_rejects_missing_external_nfelo(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Reject an upcoming game without external inputs."""

    connection.execute(
        """
        DELETE FROM processed.external_nfelo_game_ratings
        WHERE normalized_game_id = '2026_01_BUF_KC'
        """
    )

    with pytest.raises(
        RuntimeError,
        match="missing external nfelo team snapshots",
    ):
        load_current_production_inputs(
            connection
        )


def test_training_load_rejects_missing_external_nfelo(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Reject incomplete historical external coverage."""

    connection.execute(
        """
        DELETE FROM processed.external_nfelo_game_ratings
        WHERE normalized_game_id = '2024_01_A_B'
        """
    )

    with pytest.raises(
        RuntimeError,
        match="missing external nfelo inputs",
    ):
        load_production_training_data(
            connection
        )