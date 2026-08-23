"""Tests for the current prediction Data Science view."""

import duckdb
import pandas as pd
import pytest

from src.modeling.build_current_prediction_data_science_view import (
    TARGET_FULL_NAME,
    create_current_prediction_data_science_view,
    validate_current_prediction_data_science_view,
)


@pytest.fixture
def connection() -> duckdb.DuckDBPyConnection:
    """Create an in-memory technical-view database."""

    database = duckdb.connect(":memory:")

    predictions = pd.DataFrame(
        [
            {
                "game_id": "blend_game",
                "season": 2026,
                "game_type": "REG",
                "week": 1,
                "gameday": "2026-09-10",
                "gametime": "20:20",
                "home_team": "PHI",
                "away_team": "DAL",
                "model_name": "blend",
                "model_version": "0.2.0",
                "prediction_generated_at": (
                    "2026-08-06 14:00:00"
                ),
                "prediction_mode": "EXTERNAL_NFELO_BLEND",
                "prediction_mode_reason": (
                    "complete_model_features"
                ),
                "predicted_winner": "PHI",
                "home_win_probability": 0.67,
                "away_win_probability": 0.33,
                "published_nfelo_home_probability": 0.60,
                "primary_logistic_home_win_probability": 0.70,
                "fallback_logistic_home_win_probability": None,
                "applied_primary_logistic_weight": 0.70,
                "applied_published_nfelo_weight": 0.30,
                "has_complete_injury_data": True,
                "both_listed_qb_ratings_available": True,
                "has_complete_production_features": True,
            },
            {
                "game_id": "fallback_game",
                "season": 2026,
                "game_type": "REG",
                "week": 1,
                "gameday": "2026-09-13",
                "gametime": "13:00",
                "home_team": "NE",
                "away_team": "MIA",
                "model_name": "blend",
                "model_version": "0.2.0",
                "prediction_generated_at": (
                    "2026-08-06 14:00:00"
                ),
                "prediction_mode": "EXTERNAL_ELO_QB_FALLBACK",
                "prediction_mode_reason": (
                    "incomplete_model_features"
                ),
                "predicted_winner": "NE",
                "home_win_probability": 0.55,
                "away_win_probability": 0.45,
                "published_nfelo_home_probability": None,
                "primary_logistic_home_win_probability": None,
                "fallback_logistic_home_win_probability": 0.55,
                "applied_primary_logistic_weight": 0.0,
                "applied_published_nfelo_weight": 0.0,
                "has_complete_injury_data": False,
                "both_listed_qb_ratings_available": True,
                "has_complete_production_features": False,
            },
        ]
    )

    intercept = -0.10
    contributions = pd.DataFrame(
        [
            {
                "game_id": "blend_game",
                "feature_name": "external_nfelo_rating_difference",
                "raw_feature_value": 50.0,
                "standardized_feature_value": 0.50,
                "coefficient": 0.80,
                "log_odds_contribution": 0.40,
                "absolute_log_odds_contribution": 0.40,
                "contribution_rank": 1,
            },
            {
                "game_id": "blend_game",
                "feature_name": (
                    "listed_qb_rating_difference"
                ),
                "raw_feature_value": 4.0,
                "standardized_feature_value": 0.40,
                "coefficient": 0.50,
                "log_odds_contribution": 0.20,
                "absolute_log_odds_contribution": 0.20,
                "contribution_rank": 2,
            },
        ]
    )

    total_log_odds = (
        intercept
        + contributions[
            "log_odds_contribution"
        ].sum()
    )

    reconstructed_probability = (
        1.0
        / (
            1.0
            + pow(
                2.718281828459045,
                -total_log_odds,
            )
        )
    )

    blend_probability = (
        0.70
        * reconstructed_probability
        + 0.30
        * 0.60
    )

    blend_mask = (
        predictions["game_id"]
        == "blend_game"
    )

    predictions.loc[
        blend_mask,
        "primary_logistic_home_win_probability",
    ] = reconstructed_probability

    predictions.loc[
        blend_mask,
        "home_win_probability",
    ] = blend_probability

    predictions.loc[
        blend_mask,
        "away_win_probability",
    ] = (
        1.0
        - blend_probability
    )

    contributions[
        "logistic_intercept"
    ] = intercept
    contributions[
        "logistic_total_log_odds"
    ] = total_log_odds
    contributions[
        "logistic_reconstructed_home_win_probability"
    ] = reconstructed_probability

    database.register(
        "prediction_source",
        predictions,
    )
    database.register(
        "contribution_source",
        contributions,
    )

    database.execute(
        """
        CREATE SCHEMA analytics;

        CREATE TABLE analytics.current_game_predictions
        AS SELECT * FROM prediction_source;

        CREATE TABLE
            analytics.current_game_logistic_feature_contributions
        AS SELECT * FROM contribution_source;
        """
    )

    yield database

    database.close()


def test_create_and_validate_data_science_view(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create valid blend and fallback technical rows."""

    create_current_prediction_data_science_view(
        connection
    )

    validate_current_prediction_data_science_view(
        connection=connection,
        expected_prediction_count=2,
        expected_feature_count=2,
    )

    rows = connection.execute(
        f"""
        SELECT *
        FROM {TARGET_FULL_NAME}
        """
    ).fetchdf()

    assert len(rows) == 3
    assert rows["game_id"].nunique() == 2


def test_view_exposes_exact_feature_math(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Expose raw, transformed and fitted values."""

    create_current_prediction_data_science_view(
        connection
    )

    blend_rows = connection.execute(
        f"""
        SELECT *
        FROM {TARGET_FULL_NAME}
        WHERE game_id = 'blend_game'
        ORDER BY contribution_rank
        """
    ).fetchdf()

    assert list(
        blend_rows["feature_name"]
    ) == [
        "external_nfelo_rating_difference",
        "listed_qb_rating_difference",
    ]

    assert (
        blend_rows.iloc[0][
            "log_odds_contribution"
        ]
        == pytest.approx(
            blend_rows.iloc[0][
                "standardized_feature_value"
            ]
            * blend_rows.iloc[0][
                "coefficient"
            ]
        )
    )


def test_view_preserves_fallback_without_features(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Keep one technical row for Elo fallback."""

    create_current_prediction_data_science_view(
        connection
    )

    fallback = connection.execute(
        f"""
        SELECT *
        FROM {TARGET_FULL_NAME}
        WHERE game_id = 'fallback_game'
        """
    ).fetchdf()

    assert len(fallback) == 1
    assert pd.isna(
        fallback.iloc[0]["feature_name"]
    )
    assert pd.isna(
        fallback.iloc[0][
            "log_odds_contribution"
        ]
    )


def test_validator_rejects_invalid_contribution(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Reject incorrect transformed contribution math."""

    create_current_prediction_data_science_view(
        connection
    )

    connection.execute(
        """
        UPDATE
            analytics.current_game_logistic_feature_contributions
        SET log_odds_contribution = 99.0
        WHERE contribution_rank = 1
        """
    )

    with pytest.raises(
        RuntimeError,
        match="contribution rows",
    ):
        validate_current_prediction_data_science_view(
            connection=connection,
            expected_prediction_count=2,
            expected_feature_count=2,
        )
