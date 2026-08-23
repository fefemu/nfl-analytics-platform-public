"""Tests for current betting value audit helpers."""

import pandas as pd
import pytest

from src.betting.audit_current_betting_value import (
    calculate_market_shrinkage_sensitivity,
    summarize_extreme_offers,
    summarize_value_concentration,
    select_publishable_next_week_candidates,
    validate_betting_value_math,
    validate_market_pairing,
)


def board() -> pd.DataFrame:
    return pd.DataFrame({
        "game_id": ["a", "b"],
        "week": [1, 1],
        "commence_time": ["2099-09-01T18:00:00Z", "2099-09-01T20:00:00Z"],
        "outcome_name": ["A", "B"],
        "market_name": ["Moneyline", "Spread"],
        "pair_key": ["MONEYLINE", "-3.5"],
        "best_decimal_odds": [2.0, 2.0],
        "bookmaker_count": [1, 5],
        "prediction_mode": ["A", "B"],
        "model_probability": [0.60, 0.55],
        "push_probability": [0.0, 0.10],
        "loss_probability": [0.40, 0.35],
        "consensus_no_vig_probability": [0.50, 0.50],
        "probability_edge": [0.10, 0.55 / 0.90 - 0.50],
        "expected_value_per_unit": [0.20, 0.20],
        "expected_value_percent": [20.0, 20.0],
        "positive_expected_value": [True, True],
    })


def test_valid_formulas_pass() -> None:
    result = validate_betting_value_math(board())
    assert result["status"].eq("PASS").all()


def test_formula_error_is_detected() -> None:
    data = board()
    data.loc[0, "expected_value_per_unit"] = 0.5
    result = validate_betting_value_math(data)
    assert result.loc[result["check_name"].eq("ev_formula"), "issue_count"].item() == 1


def test_valid_market_pairs_pass() -> None:
    data = pd.concat([board().iloc[[0]], board().iloc[[0]]], ignore_index=True)
    data.loc[1, "outcome_name"] = "Opponent"
    data.loc[:, "consensus_no_vig_probability"] = [0.45, 0.55]
    result = validate_market_pairing(data)
    assert result["status"].eq("PASS").all()


def test_market_pair_error_is_detected() -> None:
    result = validate_market_pairing(board().iloc[[0]])
    assert result["status"].eq("FAIL").all()


def test_concentration_uses_bookmaker_buckets() -> None:
    result = summarize_value_concentration(board())
    assert set(result["bookmaker_bucket"].astype(str)) == {"1 bookmaker", "5+ bookmakers"}


def test_shrinkage_reduces_extreme_value() -> None:
    result = calculate_market_shrinkage_sensitivity(board(), model_weights=(1.0, 0.5))
    maximum = result.groupby("model_weight")["maximum_ev_percent"].max()
    assert maximum.loc[0.5] < maximum.loc[1.0]


def test_invalid_weight_is_rejected() -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        calculate_market_shrinkage_sensitivity(board(), model_weights=(1.1,))


def test_extreme_offers_use_edge_or_ev_threshold() -> None:
    data = board()
    data.loc[0, ["probability_edge", "expected_value_percent"]] = [0.16, 5.0]
    data.loc[1, ["probability_edge", "expected_value_percent"]] = [0.02, 21.0]
    result = summarize_extreme_offers(data)
    assert result["game_id"].tolist() == ["b", "a"]


def test_publishable_candidates_require_probability_and_market_depth() -> None:
    result = select_publishable_next_week_candidates(board(), maximum_edge=0.12)
    assert result["game_id"].tolist() == ["b"]


def test_publishable_candidates_exclude_extreme_research_signals() -> None:
    data = board().iloc[[1]].copy()
    data.loc[:, "probability_edge"] = 0.11
    assert select_publishable_next_week_candidates(data).empty

    data.loc[:, "probability_edge"] = 0.05
    data.loc[:, "expected_value_percent"] = 21.0
    assert select_publishable_next_week_candidates(data).empty
