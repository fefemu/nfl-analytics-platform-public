"""Tests for the current totals prediction builder."""

from datetime import datetime

import duckdb
import pandas as pd
import pytest

from src.modeling.build_current_totals_predictions import (
    TARGET_FULL_NAME,
    create_current_totals_predictions_table,
    load_current_totals_inputs,
    load_production_totals_training_data,
    validate_current_totals_predictions_table,
)
from src.modeling.current_totals_predictions import (
    CURRENT_TOTALS_PREDICTION_COLUMNS,
)
from src.modeling.evaluate_totals_model_candidates import (
    RAW_TOTALS_FEATURE_COLUMNS,
)


def create_current_source_tables(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create minimal upcoming-game source tables."""

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
            gametime VARCHAR,
            home_team VARCHAR,
            away_team VARCHAR,
            location VARCHAR,
            home_qb_id VARCHAR,
            away_qb_id VARCHAR,
            is_completed BOOLEAN
        );

        INSERT INTO processed.schedule
        VALUES (
            '2026_01_BUF_NYJ',
            2026,
            'REG',
            1,
            DATE '2026-09-13',
            '13:00',
            'BUF',
            'NYJ',
            'Home',
            NULL,
            NULL,
            FALSE
        );

        CREATE TABLE
            processed.team_game_efficiency (
                game_id VARCHAR,
                season INTEGER,
                game_date DATE,
                team VARCHAR,
                offensive_epa_per_play DOUBLE,
                defensive_epa_allowed_per_play DOUBLE
            );

        CREATE TABLE analytics.current_elo_ratings (
            team VARCHAR,
            elo_rating DOUBLE
        );

        INSERT INTO analytics.current_elo_ratings
        VALUES
            ('BUF', 1550.0),
            ('NYJ', 1450.0);

        CREATE TABLE analytics.current_qb_ratings (
            qb_id VARCHAR,
            qb_rating DOUBLE
        );

        CREATE TABLE analytics.game_weather_features (
            game_id VARCHAR,
            is_indoor BOOLEAN,
            has_game_weather BOOLEAN,
            cold_degrees_below_50 DOUBLE,
            heat_degrees_above_80 DOUBLE,
            wind_mph_above_10 DOUBLE
        );

        INSERT INTO analytics.game_weather_features
        VALUES (
            '2026_01_BUF_NYJ',
            FALSE,
            FALSE,
            0.0,
            0.0,
            0.0
        );

        CREATE TABLE
            analytics.game_scoring_environment_features (
                game_id VARCHAR,
                league_average_total_last_64 DOUBLE
            );

        INSERT INTO
            analytics.game_scoring_environment_features
        VALUES (
            '2026_01_BUF_NYJ',
            45.5
        );
        """
    )


def create_historical_source_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create one complete modeling dataset row."""

    row: dict[str, object] = {
        column_name: 1.0
        for column_name
        in RAW_TOTALS_FEATURE_COLUMNS
    }

    row.update(
        {
            "game_id": "2024_01_BUF_NYJ",
            "season": 2024,
            "game_date": pd.Timestamp(
                "2024-09-08"
            ),
            "both_short_windows_complete": True,
            "target_total_points": 44.0,
            "home_elo_rating": 1510.0,
            "away_elo_rating": 1490.0,
            "is_indoor": False,
            "has_game_weather": True,
            "is_freezing": False,
            "is_high_wind": False,
            "is_extreme_heat": False,
        }
    )

    source = pd.DataFrame([row])

    connection.register(
        "_historical_source",
        source,
    )

    connection.execute(
        """
        CREATE TABLE analytics.game_modeling_dataset AS
        SELECT *
        FROM _historical_source
        """
    )

    connection.unregister(
        "_historical_source"
    )


def create_prediction_frame() -> pd.DataFrame:
    """Create valid primary and fallback predictions."""

    generated_at = datetime(
        2026,
        8,
        7,
        12,
        0,
        0,
    )

    rows = [
        {
            "game_id": "primary_game",
            "season": 2026,
            "game_type": "REG",
            "week": 5,
            "gameday": pd.Timestamp(
                "2026-10-11"
            ),
            "gametime": "13:00",
            "home_team": "BUF",
            "away_team": "NYJ",
            "is_neutral": False,
            "model_name": (
                "ridge_epa_weather_qb_"
                "league_64_totals"
            ),
            "model_version": "0.1.0",
            "prediction_mode": (
                "RIDGE_TOTALS_PRIMARY"
            ),
            "prediction_mode_reason": (
                "complete_locked_totals_features"
            ),
            "ridge_alpha": 100.0,
            "primary_training_game_count": 100,
            "fallback_training_game_count": 200,
            "home_prior_season_games": 4,
            "away_prior_season_games": 4,
            "both_short_windows_complete": True,
            (
                "both_listed_qb_ratings_available"
            ): True,
            "has_complete_primary_features": True,
            "offensive_epa_sum_last_4": 0.10,
            (
                "defensive_epa_allowed_sum_last_4"
            ): 0.05,
            "listed_qb_rating_sum": 4.0,
            "elo_rating_sum": 3000.0,
            "is_indoor": False,
            "has_game_weather": True,
            "cold_degrees_below_50": 2.0,
            "heat_degrees_above_80": 0.0,
            "wind_mph_above_10": 1.0,
            "league_average_total_last_64": 45.5,
            "predicted_total_points": 47.2,
            "prediction_generated_at": generated_at,
        },
        {
            "game_id": "fallback_game",
            "season": 2026,
            "game_type": "REG",
            "week": 1,
            "gameday": pd.Timestamp(
                "2026-09-13"
            ),
            "gametime": "16:25",
            "home_team": "LV",
            "away_team": "DEN",
            "is_neutral": False,
            "model_name": (
                "ridge_league_64_indoor_elo_totals"
            ),
            "model_version": "0.1.0",
            "prediction_mode": (
                "RIDGE_TOTALS_FALLBACK"
            ),
            "prediction_mode_reason": (
                "missing_primary_rolling_or_qb_features"
            ),
            "ridge_alpha": 1.0,
            "primary_training_game_count": 100,
            "fallback_training_game_count": 200,
            "home_prior_season_games": 0,
            "away_prior_season_games": 0,
            "both_short_windows_complete": False,
            (
                "both_listed_qb_ratings_available"
            ): False,
            "has_complete_primary_features": False,
            "offensive_epa_sum_last_4": None,
            (
                "defensive_epa_allowed_sum_last_4"
            ): None,
            "listed_qb_rating_sum": None,
            "elo_rating_sum": 2990.0,
            "is_indoor": True,
            "has_game_weather": False,
            "cold_degrees_below_50": 0.0,
            "heat_degrees_above_80": 0.0,
            "wind_mph_above_10": 0.0,
            "league_average_total_last_64": 45.5,
            "predicted_total_points": 45.8,
            "prediction_generated_at": generated_at,
        },
    ]

    return pd.DataFrame(
        rows,
        columns=(
            CURRENT_TOTALS_PREDICTION_COLUMNS
        ),
    )


def test_load_current_totals_inputs() -> None:
    """Load universal fallback inputs for upcoming games."""

    with duckdb.connect(":memory:") as connection:
        create_current_source_tables(connection)

        inputs = load_current_totals_inputs(
            connection
        )

    assert len(inputs) == 1

    row = inputs.iloc[0]

    assert row["home_prior_season_games"] == 0
    assert row["away_prior_season_games"] == 0
    assert row["home_elo_rating"] == 1550.0
    assert row["away_elo_rating"] == 1450.0
    assert row[
        "league_average_total_last_64"
    ] == pytest.approx(45.5)


def test_load_production_training_data() -> None:
    """Create primary and fallback aggregate features."""

    with duckdb.connect(":memory:") as connection:
        connection.execute(
            "CREATE SCHEMA analytics"
        )

        create_historical_source_table(
            connection
        )

        historical = (
            load_production_totals_training_data(
                connection
            )
        )

    assert len(historical) == 1

    assert historical.iloc[0][
        "offensive_epa_sum_last_4"
    ] == pytest.approx(2.0)

    assert historical.iloc[0][
        "elo_rating_sum"
    ] == pytest.approx(3000.0)


def test_create_and_validate_prediction_table(
) -> None:
    """Persist and validate both routing modes."""

    predictions = create_prediction_frame()

    with duckdb.connect(":memory:") as connection:
        create_current_totals_predictions_table(
            connection=connection,
            predictions=predictions,
        )

        validate_current_totals_predictions_table(
            connection=connection,
            expected_row_count=2,
        )

        row_count = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM {TARGET_FULL_NAME}
            """
        ).fetchone()[0]

    assert row_count == 2


def test_invalid_routing_is_rejected() -> None:
    """Reject inconsistent persisted routing metadata."""

    predictions = create_prediction_frame()

    with duckdb.connect(":memory:") as connection:
        create_current_totals_predictions_table(
            connection=connection,
            predictions=predictions,
        )

        connection.execute(
            f"""
            UPDATE {TARGET_FULL_NAME}
            SET ridge_alpha = 10.0
            WHERE game_id = 'fallback_game'
            """
        )

        with pytest.raises(
            RuntimeError,
            match="routing",
        ):
            validate_current_totals_predictions_table(
                connection=connection,
                expected_row_count=2,
            )


def test_missing_prediction_column_is_rejected(
) -> None:
    """Reject an incomplete prediction frame."""

    predictions = create_prediction_frame().drop(
        columns=[
            "predicted_total_points",
        ]
    )

    with duckdb.connect(":memory:") as connection:
        with pytest.raises(
            ValueError,
            match="missing columns",
        ):
            create_current_totals_predictions_table(
                connection=connection,
                predictions=predictions,
            )