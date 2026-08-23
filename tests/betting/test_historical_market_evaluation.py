"""Tests for historical market evaluation and flat-stake accounting."""

import pandas as pd
import pytest

from src.betting.historical_market_evaluation import (
    american_to_decimal,
    create_historical_betting_ledger,
    create_historical_betting_summary,
)


def inputs():
    market = pd.DataFrame([{
        "game_id": "g", "season": 2024, "actual_home_margin": 7.0,
        "actual_total": 47.0, "home_market_probability_close": 0.55,
        "home_line_close": -3.0, "home_line_close_price": -110.0,
        "away_line_close_price": -110.0, "total_line_close": 44.0,
        "over_price_close": -110.0, "under_price_close": -110.0,
        "home_market_probability_open": 0.53, "home_line_open": -2.5,
        "total_line_open": 43.5,
    }])
    probability = pd.DataFrame([{"game_id": "g", "season": 2024, "model_name": "p", "home_win_probability": 0.65}])
    spread = pd.DataFrame([{"game_id": "g", "season": 2024, "model_name": "s", "predicted_home_margin": 6.0}])
    totals = pd.DataFrame([{"game_id": "g", "season": 2024, "model_name": "t", "predicted_total": 48.0}])
    return probability, spread, totals, market


def test_american_to_decimal():
    assert american_to_decimal(-110) == pytest.approx(1.9090909)
    assert american_to_decimal(150) == pytest.approx(2.5)


def test_invalid_american_price():
    with pytest.raises(ValueError, match="non-zero"):
        american_to_decimal(0)


def test_winning_home_over_ledger():
    ledger = create_historical_betting_ledger(*inputs())
    assert set(ledger["market_key"]) == {"h2h", "spreads", "totals"}
    assert ledger["result"].eq("WIN").all()
    assert ledger.loc[ledger["market_key"].eq("spreads"), "selection"].item() == "HOME"
    assert ledger.loc[ledger["market_key"].eq("totals"), "selection"].item() == "OVER"


def test_push_settlement():
    probability, spread, totals, market = inputs()
    market["actual_home_margin"] = 3.0
    market["actual_total"] = 44.0
    ledger = create_historical_betting_ledger(probability, spread, totals, market)
    assert ledger.loc[ledger["market_key"].isin(["spreads", "totals"]), "result"].eq("PUSH").all()


def test_summary_roi_and_counts():
    summary = create_historical_betting_summary(create_historical_betting_ledger(*inputs()))
    assert summary.loc[summary["edge_bucket"].eq("ALL"), "bet_count"].sum() == 3
    assert summary["roi_percent"].gt(0).all()
    assert summary["clv_bet_count"].gt(0).all()


def test_duplicate_market_game_is_rejected():
    probability, spread, totals, market = inputs()
    market = pd.concat([market, market], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        create_historical_betting_ledger(probability, spread, totals, market)
