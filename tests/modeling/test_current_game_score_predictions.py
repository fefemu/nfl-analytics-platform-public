"""Tests for model-implied team score derivation."""

from datetime import datetime, time

import pandas as pd
import pytest

from src.modeling.current_game_score_predictions import (
    SCORE_PREDICTION_COLUMNS,
    create_current_game_score_predictions,
)


def sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = {
        "game_id": "g", "season": 2026, "game_type": "REG", "week": 1,
        "gameday": "2026-09-10", "gametime": "20:20", "home_team": "BUF",
        "away_team": "NYJ", "model_name": "model", "model_version": "1",
        "prediction_mode": "mode", "prediction_generated_at": datetime(2026, 8, 9),
    }
    return (
        pd.DataFrame([{**metadata, "predicted_home_margin": 6.0}]),
        pd.DataFrame([{**metadata, "predicted_total_points": 46.0}]),
    )


def test_score_identity() -> None:
    spread, totals = sources()
    result = create_current_game_score_predictions(spread, totals)
    assert tuple(result.columns) == SCORE_PREDICTION_COLUMNS
    assert result.loc[0, "implied_home_score"] == pytest.approx(26.0)
    assert result.loc[0, "implied_away_score"] == pytest.approx(20.0)
    assert result.loc[0, "implied_score_winner"] == "BUF"


def test_away_favorite_score_identity() -> None:
    spread, totals = sources()
    spread["predicted_home_margin"] = -7.0
    totals["predicted_total_points"] = 43.0
    result = create_current_game_score_predictions(spread, totals)
    assert result.loc[0, "implied_home_score"] == pytest.approx(18.0)
    assert result.loc[0, "implied_away_score"] == pytest.approx(25.0)
    assert result.loc[0, "implied_score_winner"] == "NYJ"


def test_source_game_mismatch_is_rejected() -> None:
    spread, totals = sources()
    totals["game_id"] = "other"
    with pytest.raises(RuntimeError, match="do not match"):
        create_current_game_score_predictions(spread, totals)


def test_equivalent_gametime_types_are_accepted() -> None:
    spread, totals = sources()
    totals["gametime"] = time(20, 20)
    result = create_current_game_score_predictions(spread, totals)
    assert len(result) == 1


def test_source_metadata_mismatch_is_rejected() -> None:
    spread, totals = sources()
    totals["home_team"] = "MIA"
    with pytest.raises(RuntimeError, match="metadata differ: home_team"):
        create_current_game_score_predictions(spread, totals)


def test_negative_implied_score_is_rejected() -> None:
    spread, totals = sources()
    spread["predicted_home_margin"] = 60.0
    with pytest.raises(RuntimeError, match="must not be negative"):
        create_current_game_score_predictions(spread, totals)


def test_duplicate_source_is_rejected() -> None:
    spread, totals = sources()
    spread = pd.concat([spread, spread], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        create_current_game_score_predictions(spread, totals)
