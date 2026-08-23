"""Tests for current game prediction explanations."""

from datetime import datetime

import pandas as pd
import pytest

from src.modeling.current_game_prediction_explanations import (
    EXPLANATION_COLUMNS,
    create_prediction_explanation_frame,
)


def create_predictions() -> pd.DataFrame:
    """Create deterministic current predictions."""

    return pd.DataFrame(
        [
            {
                "game_id": "2026_01_NE_NYJ",
                "season": 2026,
                "game_type": "REG",
                "week": 1,
                "gameday": pd.Timestamp(
                    "2026-09-10"
                ),
                "gametime": "20:20",
                "home_team": "NE",
                "away_team": "NYJ",
                "is_neutral": False,
                "model_name": "elo",
                "model_version": "1.0.0",
                "home_rating_pregame": 1550.0,
                "away_rating_pregame": 1450.0,
                "applied_home_advantage": 50.0,
                "home_win_probability": (
                    0.7033850034718286
                ),
                "away_win_probability": (
                    0.2966149965281714
                ),
                "prediction_generated_at": datetime(
                    2026,
                    8,
                    2,
                    10,
                    0,
                    0,
                ),
            }
        ]
    )


def test_create_prediction_explanation_frame() -> None:
    """Create one structured explanation per game."""

    explanations = (
        create_prediction_explanation_frame(
            create_predictions()
        )
    )

    assert len(explanations) == 1
    assert tuple(explanations.columns) == (
        EXPLANATION_COLUMNS
    )

    explanation = explanations.iloc[0]

    assert explanation["favorite"] == "NE"
    assert explanation["underdog"] == "NYJ"
    assert explanation["matchup_label"] == (
        "strong_edge"
    )
    assert explanation[
        "favorite_win_probability"
    ] == pytest.approx(
        explanation["home_win_probability"]
    )


def test_explanation_preserves_model_metadata() -> None:
    """Keep prediction model version and timestamp."""

    explanations = (
        create_prediction_explanation_frame(
            create_predictions()
        )
    )

    explanation = explanations.iloc[0]

    assert explanation["model_name"] == "elo"
    assert explanation["model_version"] == "1.0.0"
    assert explanation[
        "prediction_generated_at"
    ] == datetime(
        2026,
        8,
        2,
        10,
        0,
        0,
    )


def test_explanation_rejects_probability_mismatch() -> None:
    """Reject explanations inconsistent with prediction."""

    predictions = create_predictions()

    predictions.loc[
        0,
        "home_win_probability",
    ] = 0.55

    with pytest.raises(
        RuntimeError,
        match="does not match prediction",
    ):
        create_prediction_explanation_frame(
            predictions
        )


def test_explanation_supports_empty_predictions() -> None:
    """Return a stable empty explanation schema."""

    empty_predictions = (
        create_predictions().iloc[0:0]
    )

    explanations = (
        create_prediction_explanation_frame(
            empty_predictions
        )
    )

    assert explanations.empty
    assert tuple(explanations.columns) == (
        EXPLANATION_COLUMNS
    )