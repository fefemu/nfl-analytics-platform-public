"""Persistence tests for model-implied current scores."""

from datetime import datetime

import duckdb
import pandas as pd
import pytest

from src.modeling.build_current_game_score_predictions import (
    TARGET_FULL_NAME,
    create_current_game_score_predictions_table,
    validate_current_game_score_predictions_table,
)
from src.modeling.current_game_score_predictions import SCORE_PREDICTION_COLUMNS


def predictions() -> pd.DataFrame:
    row = {
        "game_id": "g", "season": 2026, "game_type": "REG", "week": 1,
        "gameday": "2026-09-10", "gametime": "20:20", "home_team": "BUF",
        "away_team": "NYJ", "spread_model_name": "spread",
        "spread_model_version": "1", "spread_prediction_mode": "spread_mode",
        "totals_model_name": "totals", "totals_model_version": "1",
        "totals_prediction_mode": "totals_mode", "predicted_home_margin": 6.0,
        "predicted_total_points": 46.0, "implied_home_score": 26.0,
        "implied_away_score": 20.0, "implied_score_winner": "BUF",
        "spread_prediction_generated_at": datetime(2026, 8, 9),
        "totals_prediction_generated_at": datetime(2026, 8, 9),
        "score_prediction_generated_at": datetime(2026, 8, 9),
    }
    return pd.DataFrame([row], columns=SCORE_PREDICTION_COLUMNS)


def test_create_and_validate_table() -> None:
    with duckdb.connect(":memory:") as connection:
        create_current_game_score_predictions_table(connection, predictions())
        validate_current_game_score_predictions_table(connection, 1)
        assert connection.execute(f"SELECT implied_home_score FROM {TARGET_FULL_NAME}").fetchone()[0] == 26.0


def test_invalid_identity_is_rejected() -> None:
    with duckdb.connect(":memory:") as connection:
        create_current_game_score_predictions_table(connection, predictions())
        connection.execute(f"UPDATE {TARGET_FULL_NAME} SET implied_home_score = 30.0")
        with pytest.raises(RuntimeError, match="Invalid"):
            validate_current_game_score_predictions_table(connection, 1)


def test_missing_column_is_rejected() -> None:
    with duckdb.connect(":memory:") as connection:
        with pytest.raises(ValueError, match="missing columns"):
            create_current_game_score_predictions_table(
                connection, predictions().drop(columns=["implied_away_score"])
            )
