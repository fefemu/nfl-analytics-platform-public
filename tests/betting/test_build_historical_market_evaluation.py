"""Persistence tests for historical market evaluation outputs."""

import duckdb
import pandas as pd
import pytest

from src.betting.build_historical_market_evaluation import (
    persist_historical_market_evaluation,
    validate_historical_market_evaluation,
)
from src.betting.historical_market_evaluation import (
    create_historical_betting_ledger,
    create_historical_betting_summary,
)


def outputs():
    market = pd.DataFrame([{
        "game_id": "g", "season": 2024, "actual_home_margin": 7.0,
        "actual_total": 47.0, "home_market_probability_open": 0.53,
        "home_market_probability_close": 0.55, "home_line_open": -2.5,
        "home_line_close": -3.0, "home_line_close_price": -110.0,
        "away_line_close_price": -110.0, "total_line_open": 43.5,
        "total_line_close": 44.0, "over_price_close": -110.0,
        "under_price_close": -110.0,
    }])
    probability = pd.DataFrame([{"game_id": "g", "season": 2024, "model_name": "p", "home_win_probability": 0.65}])
    spread = pd.DataFrame([{"game_id": "g", "season": 2024, "model_name": "s", "predicted_home_margin": 6.0}])
    totals = pd.DataFrame([{"game_id": "g", "season": 2024, "model_name": "t", "predicted_total": 48.0}])
    ledger = create_historical_betting_ledger(probability, spread, totals, market)
    return ledger, create_historical_betting_summary(ledger)


def test_persist_and_validate_outputs():
    ledger, summary = outputs()
    with duckdb.connect(":memory:") as connection:
        persist_historical_market_evaluation(connection, ledger, summary)
        validate_historical_market_evaluation(connection)
        assert connection.execute("SELECT COUNT(*) FROM analytics.historical_betting_ledger").fetchone()[0] == 3


def test_invalid_persisted_settlement_is_rejected():
    ledger, summary = outputs()
    ledger.loc[ledger.index[0], "profit_per_unit"] = 0.0
    with duckdb.connect(":memory:") as connection:
        persist_historical_market_evaluation(connection, ledger, summary)
        with pytest.raises(RuntimeError, match="validation failed"):
            validate_historical_market_evaluation(connection)
