"""Tests for current Totals probabilities and EV."""

import pandas as pd
import pytest

from src.betting.current_totals_value import TOTALS_VALUE_COLUMNS, create_current_totals_value


def market() -> pd.DataFrame:
    base = {
        "snapshot_id": "s", "fetched_at": "2026-08-09", "game_id": "g",
        "season": 2026, "game_type": "REG", "week": 1, "gameday": "2026-09-10",
        "commence_time": "2026-09-10", "home_team": "BUF", "away_team": "NYJ",
        "market_key": "totals", "market_name": "Totals", "point": 45.0,
        "market_line": 45.0, "best_bookmaker_key": "book", "best_bookmaker_title": "Book",
        "best_american_price": -110, "best_decimal_odds": 2.0,
        "best_implied_probability": 0.5, "consensus_no_vig_probability": 0.5,
        "bookmaker_count": 5,
    }
    return pd.DataFrame([
        {**base, "outcome_name": "Over", "outcome_type": "over"},
        {**base, "outcome_name": "Under", "outcome_type": "under"},
    ])


def predictions() -> pd.DataFrame:
    return pd.DataFrame([{
        "game_id": "g", "model_name": "totals", "model_version": "1",
        "prediction_mode": "RIDGE_TOTALS_FALLBACK", "predicted_total_points": 45.0,
        "prediction_generated_at": "2026-08-09",
    }])


def residuals() -> pd.DataFrame:
    return pd.DataFrame({
        "prediction_mode": ["RIDGE_TOTALS_FALLBACK"] * 5,
        "residual_total_points": [-2.0, -1.0, 0.0, 1.0, 2.0],
    })


def test_totals_value_schema_and_probability_math() -> None:
    value = create_current_totals_value(market(), predictions(), residuals())
    assert tuple(value.columns) == TOTALS_VALUE_COLUMNS
    assert len(value) == 2
    assert value["win_probability"].eq(0.4).all()
    assert value["push_probability"].eq(0.2).all()
    assert value["loss_probability"].eq(0.4).all()
    assert value["no_push_win_probability"].eq(0.5).all()
    assert value["expected_value_per_unit"].eq(0.0).all()


def test_over_under_are_symmetric() -> None:
    value = create_current_totals_value(market(), predictions(), residuals()).set_index("outcome_type")
    assert value.loc["over", "win_probability"] == value.loc["under", "loss_probability"]
    assert value.loc["under", "win_probability"] == value.loc["over", "loss_probability"]


def test_unpaired_line_is_removed() -> None:
    with pytest.raises(RuntimeError, match="no paired"):
        create_current_totals_value(market().iloc[[0]], predictions(), residuals())


def test_missing_calibration_mode_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="missing prediction mode"):
        create_current_totals_value(
            market(), predictions(), residuals().assign(prediction_mode="other")
        )
