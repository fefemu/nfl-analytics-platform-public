"""Tests for the bilingual live-forward results page helpers."""

import pandas as pd

from src.dashboard.pages.forward_performance import (
    _filter_settlement,
    _number,
    _result_label,
    _select_summary,
    has_settled_results,
)


def test_select_summary_respects_season_week_and_market():
    summary = pd.DataFrame([
        {"season": 2026, "week": None, "market_key": "ALL", "sample_scope": "ALL_TRACKED", "tracked_count": 3},
        {"season": 2026, "week": 1, "market_key": "h2h", "sample_scope": "ALL_TRACKED", "tracked_count": 1},
    ])
    assert _select_summary(summary, 2026, None, "ALL")["tracked_count"] == 3
    assert _select_summary(summary, 2026, 1, "h2h")["tracked_count"] == 1
    assert _select_summary(summary, 2026, 2, "h2h") is None


def test_select_summary_accepts_pre_scope_snapshot_for_all_tracked():
    legacy = pd.DataFrame([
        {"season": 2026, "week": None, "market_key": "ALL", "tracked_count": 2},
    ])
    assert _select_summary(legacy, 2026, None, "ALL")["tracked_count"] == 2
    assert _select_summary(legacy, 2026, None, "ALL", "SETTLED_ONLY") is None


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
