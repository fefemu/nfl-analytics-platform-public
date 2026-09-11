"""Tests for the bilingual live-forward results page helpers."""

import pandas as pd
import pytest

from src.dashboard.pages.forward_performance import (
    _filter_settlement,
    _number,
    _result_label,
    _select_summary,
    betting_empty_message,
    has_live_results,
    has_settled_results,
    prepare_model_result_rows,
    results_updated_at,
    _regression_metrics,
)


def test_select_summary_respects_season_week_and_market():
    summary = pd.DataFrame([
        {"season": 2026, "week": None, "market_key": "ALL", "sample_scope": "ALL_TRACKED", "tracked_count": 3},
        {"season": 2026, "week": 1, "market_key": "h2h", "sample_scope": "ALL_TRACKED", "tracked_count": 1},
    ])
    assert _select_summary(summary, 2026, None, "ALL")["tracked_count"] == 3
    assert _select_summary(summary, 2026, 1, "h2h")["tracked_count"] == 1
    assert _select_summary(summary, 2026, 2, "h2h") is None


def test_filter_settlement_supports_market_week_and_settled_scope():
    rows = pd.DataFrame([
        {"season": 2026, "week": 1, "market_key": "h2h", "settlement_status": "SETTLED", "commence_time": "2026-09-01", "game_id": "a"},
        {"season": 2026, "week": 1, "market_key": "totals", "settlement_status": "PENDING", "commence_time": "2026-09-01", "game_id": "a"},
        {"season": 2026, "week": 2, "market_key": "h2h", "settlement_status": "PENDING", "commence_time": "2026-09-08", "game_id": "b"},
    ])
    result = _filter_settlement(rows, 2026, 1, "h2h", True)
    assert list(result["game_id"]) == ["a"]


def test_labels_and_numbers_are_bilingual():
    assert _result_label("WIN", "EN") == "Win"
    assert _result_label("WIN", "HU") == "Nyert"
    assert _result_label(None, "HU") == "Közelgő"
    assert _number(12.34, "HU", 1, "%") == "12,3%"
    assert _number(float("nan"), "EN") == "—"


def test_live_results_visibility_requires_a_settled_row():
    pending = pd.DataFrame({"settlement_status": ["PENDING", "PENDING"]})
    settled = pd.DataFrame({"settlement_status": ["PENDING", "SETTLED"]})

    assert has_settled_results(pd.DataFrame()) is False
    assert has_settled_results(pending) is False
    assert has_settled_results(settled) is True


def test_live_results_visibility_accepts_model_results_without_bets():
    assert has_live_results(pd.DataFrame({"game_id": ["g1"]}), pd.DataFrame())


def test_ne_sea_model_result_is_seattle_and_correct():
    result = prepare_model_result_rows(pd.DataFrame([{
        "game_id": "2026_01_NE_SEA", "season": 2026, "week": 1,
        "commence_time": "2026-09-10T00:20:00Z", "away_team": "NE",
        "home_team": "SEA", "away_win_probability": 0.397,
        "home_win_probability": 0.603, "predicted_winner": "SEA",
        "predicted_home_margin": 4.1, "predicted_total_points": 49.3,
        "away_score": 10, "home_score": 13,
        "result_evaluated_at": "2026-09-10T17:04:54Z",
    }])).iloc[0]
    assert result["predicted_winner"] == "SEA"
    assert result["actual_winner"] == "SEA"
    assert bool(result["moneyline_winner_correct"])
    assert result["predicted_win_probability"] == 0.603
    assert result["actual_home_margin"] == 3
    assert result["spread_absolute_error"] == pytest.approx(1.1)
    assert result["actual_total_points"] == 23
    assert result["total_absolute_error"] == pytest.approx(26.3)


def test_regression_metrics_use_prediction_minus_actual_bias():
    rows = pd.DataFrame({"error": [2.0, -1.0]})
    mae, rmse, bias = _regression_metrics(rows, "error")
    assert mae == pytest.approx(1.5)
    assert rmse == pytest.approx(2.5 ** 0.5)
    assert bias == pytest.approx(0.5)


def test_results_timestamp_comes_from_final_evaluation_dataset():
    rows = pd.DataFrame({"result_evaluated_at": [
        "2026-09-10T15:00:00Z", "2026-09-10T17:04:54Z",
    ]})
    assert results_updated_at(rows) == pd.Timestamp("2026-09-10T17:04:54Z")


def test_verified_betting_empty_state_is_bilingual():
    assert "Még nincs lezárt Javasolt tipp" in betting_empty_message("HU")
    assert "No Recommended Picks have been settled yet" in betting_empty_message("EN")


def test_legacy_opportunity_scope_is_not_user_facing():
    source = __import__("pathlib").Path(
        "src/dashboard/pages/forward_performance.py"
    ).read_text(encoding="utf-8")
    assert "All tracked" not in source
    assert "Minden követett" not in source
    assert "Locked selections" not in source
