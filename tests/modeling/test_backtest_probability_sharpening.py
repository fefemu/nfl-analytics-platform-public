import numpy as np
import pandas as pd
import pytest

from src.modeling.backtest_probability_sharpening import (
    SUMMARY_COLUMNS,
    apply_logit_sharpening,
    create_sharpening_predictions,
    evaluate_sharpening_candidates,
)


def create_raw_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3", "g4"],
            "validation_season": [2023, 2023, 2024, 2024],
            "actual_home_win": [0, 0, 1, 1],
            "home_win_probability": [0.40, 0.45, 0.55, 0.60],
        }
    )


def test_logit_sharpening_preserves_neutral_probability() -> None:
    result = apply_logit_sharpening(
        probabilities=np.array([0.25, 0.50, 0.75]),
        sharpening_factor=1.5,
    )

    assert result[1] == pytest.approx(0.5)
    assert result[0] < 0.25
    assert result[2] > 0.75


def test_logit_sharpening_rejects_invalid_factor() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        apply_logit_sharpening(
            probabilities=np.array([0.5]),
            sharpening_factor=0.0,
        )


def test_create_predictions_aligns_candidates() -> None:
    result = create_sharpening_predictions(
        raw_predictions=create_raw_predictions(),
        sharpening_factors=(1.0, 1.2),
    )

    assert len(result) == 8
    assert set(result["candidate_name"]) == {
        "fallback_logit_scale_1.00",
        "fallback_logit_scale_1.20",
    }


def test_candidate_evaluation_uses_raw_baseline() -> None:
    predictions = create_sharpening_predictions(
        raw_predictions=create_raw_predictions(),
        sharpening_factors=(1.0, 1.2),
    )
    summary, season_results = evaluate_sharpening_candidates(
        predictions
    )

    assert tuple(summary.columns) == SUMMARY_COLUMNS
    raw = summary.loc[summary["sharpening_factor"] == 1.0].iloc[0]
    assert raw["brier_score_delta_vs_raw"] == pytest.approx(0.0)
    assert raw["log_loss_delta_vs_raw"] == pytest.approx(0.0)
    assert len(season_results) == 4


def test_candidate_evaluation_requires_raw_factor() -> None:
    predictions = create_sharpening_predictions(
        raw_predictions=create_raw_predictions(),
        sharpening_factors=(1.2,),
    )

    with pytest.raises(ValueError, match="factor 1.00"):
        evaluate_sharpening_candidates(predictions)
