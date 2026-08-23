"""Tests for persisted current production predictions."""

from datetime import datetime

import duckdb
import pandas as pd
import pytest

from src.modeling.build_current_game_predictions import (
    TARGET_FULL_NAME,
    create_current_production_predictions_table,
    validate_current_production_predictions_table,
)


@pytest.fixture
def connection() -> duckdb.DuckDBPyConnection:
    """Create an in-memory production database."""

    database = duckdb.connect(":memory:")

    yield database

    database.close()


def create_predictions() -> pd.DataFrame:
    """Create valid blend and fallback predictions."""

    generated_at = datetime(
        2026,
        8,
        6,
        8,
        0,
        0,
    )

    return pd.DataFrame(
        [
            {
                "game_id": "2026_01_NE_NYJ",
                "season": 2026,
                "game_type": "REG",
                "week": 1,
                "gameday": "2026-09-10",
                "gametime": "20:20",
                "home_team": "NE",
                "away_team": "NYJ",
                "is_neutral": False,
                "model_name": (
                    "external_nfelo_probability_routing"
                ),
                "model_version": "0.3.0",
                "home_rating_current": 1510.0,
                "away_rating_current": 1475.0,
                "home_rating_pregame": 1508.0,
                "away_rating_pregame": 1478.0,
                "applied_home_advantage": 48.0,
                "home_win_probability": 0.67,
                "away_win_probability": 0.33,
                "predicted_winner": "NE",
                "home_rating_as_of": "2026-01-04",
                "away_rating_as_of": "2026-01-04",
                "prediction_generated_at": (
                    generated_at
                ),
                "prediction_mode": "EXTERNAL_NFELO_BLEND",
                "prediction_mode_reason": (
                    "complete_external_primary_features"
                ),
                "published_nfelo_home_probability": 0.60,
                "primary_logistic_home_win_probability": 0.70,
                "fallback_logistic_home_win_probability": None,
                "applied_primary_logistic_weight": 0.70,
                "applied_published_nfelo_weight": 0.30,
                "elo_home_win_probability": 0.60,
                "logistic_home_win_probability": 0.70,
                "applied_logistic_weight": 0.70,
                "applied_elo_weight": 0.30,
                "has_complete_injury_data": True,
                "both_listed_qb_ratings_available": True,
                "has_complete_production_features": True,
                "has_complete_fallback_features": True,
                "external_nfelo_rating_difference": 30.0,
                "listed_qb_rating_difference": 3.5,
                "external_nfelo_qb_adjustment_difference": 2.0,
                "offense_injury_burden_difference": -0.20,
                "defense_injury_burden_difference": 0.10,
                "special_teams_injury_burden_difference": -0.05,
            },
            {
                "game_id": "2026_01_BUF_KC",
                "season": 2026,
                "game_type": "REG",
                "week": 1,
                "gameday": "2026-09-13",
                "gametime": "13:00",
                "home_team": "BUF",
                "away_team": "KC",
                "is_neutral": False,
                "model_name": (
                    "external_nfelo_probability_routing"
                ),
                "model_version": "0.3.0",
                "home_rating_current": 1570.0,
                "away_rating_current": 1560.0,
                "home_rating_pregame": 1565.0,
                "away_rating_pregame": 1555.0,
                "applied_home_advantage": 48.0,
                "home_win_probability": 0.55,
                "away_win_probability": 0.45,
                "predicted_winner": "BUF",
                "home_rating_as_of": "2026-01-04",
                "away_rating_as_of": "2026-01-04",
                "prediction_generated_at": (
                    generated_at
                ),
                "prediction_mode": "EXTERNAL_ELO_QB_FALLBACK",
                "prediction_mode_reason": (
                    "incomplete_external_primary_features"
                ),
                "published_nfelo_home_probability": 0.55,
                "primary_logistic_home_win_probability": None,
                "fallback_logistic_home_win_probability": 0.55,
                "applied_primary_logistic_weight": 0.0,
                "applied_published_nfelo_weight": 0.0,
                "elo_home_win_probability": 0.55,
                "logistic_home_win_probability": None,
                "applied_logistic_weight": 0.0,
                "applied_elo_weight": 0.0,
                "has_complete_injury_data": False,
                "both_listed_qb_ratings_available": True,
                "has_complete_production_features": False,
                "has_complete_fallback_features": True,
                "external_nfelo_rating_difference": 10.0,
                "listed_qb_rating_difference": 1.0,
                "external_nfelo_qb_adjustment_difference": 1.5,
                "offense_injury_burden_difference": None,
                "defense_injury_burden_difference": None,
                "special_teams_injury_burden_difference": None,
            },
        ]
    )


def test_create_and_validate_production_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Persist valid blend and fallback predictions."""

    predictions = create_predictions()

    create_current_production_predictions_table(
        connection=connection,
        predictions=predictions,
    )

    validate_current_production_predictions_table(
        connection=connection,
        expected_row_count=2,
    )

    stored = connection.execute(
        f"""
        SELECT
            game_id,
            model_name,
            model_version,
            prediction_mode,
            home_win_probability
        FROM {TARGET_FULL_NAME}
        ORDER BY game_id
        """
    ).fetchdf()

    assert len(stored) == 2

    assert set(
        stored["model_name"]
    ) == {
        "external_nfelo_probability_routing",
    }

    assert set(
        stored["model_version"]
    ) == {
        "0.3.0",
    }

    assert set(
        stored["prediction_mode"]
    ) == {
        "EXTERNAL_NFELO_BLEND",
        "EXTERNAL_ELO_QB_FALLBACK",
    }


def test_create_empty_production_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create a stable empty production table."""

    predictions = (
        create_predictions().iloc[0:0]
    )

    create_current_production_predictions_table(
        connection=connection,
        predictions=predictions,
    )

    validate_current_production_predictions_table(
        connection=connection,
        expected_row_count=0,
    )

    row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    assert row_count == 0


def test_writer_rejects_missing_audit_column(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Reject predictions without routing metadata."""

    predictions = (
        create_predictions().drop(
            columns=[
                "prediction_mode",
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        create_current_production_predictions_table(
            connection=connection,
            predictions=predictions,
        )


def test_validator_rejects_invalid_blend_math(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Reject a blend inconsistent with its components."""

    create_current_production_predictions_table(
        connection=connection,
        predictions=create_predictions(),
    )

    connection.execute(
        f"""
        UPDATE {TARGET_FULL_NAME}
        SET home_win_probability = 0.90,
            away_win_probability = 0.10
        WHERE prediction_mode = 'EXTERNAL_NFELO_BLEND'
        """
    )

    with pytest.raises(
        RuntimeError,
        match="Invalid current production primary routing",
    ):
        validate_current_production_predictions_table(
            connection=connection,
            expected_row_count=2,
        )


def test_validator_rejects_invalid_fallback(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Reject logistic output on an Elo fallback row."""

    create_current_production_predictions_table(
        connection=connection,
        predictions=create_predictions(),
    )

    connection.execute(
        f"""
        UPDATE {TARGET_FULL_NAME}
        SET fallback_logistic_home_win_probability = NULL
        WHERE prediction_mode = 'EXTERNAL_ELO_QB_FALLBACK'
        """
    )

    with pytest.raises(
        RuntimeError,
        match="Invalid current production fallback routing",
    ):
        validate_current_production_predictions_table(
            connection=connection,
            expected_row_count=2,
        )


def test_validator_rejects_invalid_feature_coverage(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Reject a complete row with a missing feature."""

    create_current_production_predictions_table(
        connection=connection,
        predictions=create_predictions(),
    )

    connection.execute(
        f"""
        UPDATE {TARGET_FULL_NAME}
        SET listed_qb_rating_difference = NULL
        WHERE prediction_mode = 'EXTERNAL_NFELO_BLEND'
        """
    )

    with pytest.raises(
        RuntimeError,
        match="Invalid current production features",
    ):
        validate_current_production_predictions_table(
            connection=connection,
            expected_row_count=2,
        )
