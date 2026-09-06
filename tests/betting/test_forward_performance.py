"""Tests for locked forward settlement and performance summaries."""

import duckdb
import pandas as pd
import pytest

from src.betting.forward_performance import (
    create_forward_performance_summary,
    create_forward_settlement,
    persist_forward_performance,
    validate_forward_performance,
)


def entries() -> pd.DataFrame:
    common = {
        "season": 2026, "week": 1, "commence_time": "2026-09-10T00:00:00Z",
        "home_team": "H", "away_team": "A", "outcome_name": "selection",
        "entry_fetched_at": "2026-09-01T00:00:00Z", "entry_bookmaker": "Book",
        "entry_expected_value_percent": 5.0, "entry_model_name": "model",
        "entry_model_version": "v1", "is_clv": True,
        "market_movement_direction": "POSITIVE",
        "latest_fetched_at": "2026-09-09T23:00:00Z", "latest_decimal_odds": 1.9,
        "latest_point": None, "price_movement_implied_probability_pp": 2.0,
        "entry_line_advantage_points": None,
    }
    return pd.DataFrame([
        {**common, "entry_archive_key": "ml", "game_id": "g1", "market_key": "h2h",
         "outcome_type": "home", "entry_point": None, "entry_decimal_odds": 2.0,
         "entry_model_probability": 0.6},
        {**common, "entry_archive_key": "spread", "game_id": "g1", "market_key": "spreads",
         "outcome_type": "away", "entry_point": 3.0, "entry_decimal_odds": 1.91,
         "entry_model_probability": 0.55, "latest_point": 2.5,
         "entry_line_advantage_points": 0.5},
        {**common, "entry_archive_key": "total", "game_id": "g1", "market_key": "totals",
         "outcome_type": "over", "entry_point": 47.0, "entry_decimal_odds": 1.95,
         "entry_model_probability": 0.54},
        {**common, "entry_archive_key": "pending", "game_id": "g2", "market_key": "h2h",
         "outcome_type": "away", "entry_point": None, "entry_decimal_odds": 2.2,
         "entry_model_probability": 0.52, "is_clv": False,
         "market_movement_direction": "NO_LATER_SNAPSHOT",
         "latest_fetched_at": None, "latest_decimal_odds": None,
         "price_movement_implied_probability_pp": None},
    ])


def schedule() -> pd.DataFrame:
    return pd.DataFrame([
        {"game_id": "g1", "home_score": 24, "away_score": 21, "is_completed": True},
        {"game_id": "g2", "home_score": None, "away_score": None, "is_completed": False},
    ])


def test_settlement_handles_win_push_loss_and_pending():
    result = create_forward_settlement(entries(), schedule()).set_index("entry_archive_key")
    assert (result.loc["ml", "result"], result.loc["ml", "profit_units"]) == ("WIN", 1.0)
    assert (result.loc["spread", "result"], result.loc["spread", "profit_units"]) == ("PUSH", 0.0)
    assert (result.loc["total", "result"], result.loc["total", "profit_units"]) == ("LOSS", -1.0)
    assert result.loc["pending", "settlement_status"] == "PENDING"
    assert pd.isna(result.loc["pending", "profit_units"])


def test_moneyline_probability_scores_use_locked_selection_probability():
    row = create_forward_settlement(entries(), schedule()).set_index("entry_archive_key").loc["ml"]
    assert row["brier_loss"] == pytest.approx(0.16)
    assert row["log_loss"] == pytest.approx(-__import__("math").log(0.6))


def test_summary_reports_roi_drawdown_and_missing_market_scores():
    ledger = create_forward_settlement(entries(), schedule())
    summary = create_forward_performance_summary(ledger)
    overall = summary.loc[(summary["selection_scope"] == "SEASON") & (summary["market_key"] == "ALL") & (summary["sample_scope"] == "ALL_TRACKED")].iloc[0]
    assert overall["tracked_count"] == 4
    assert overall["settled_count"] == 3
    assert overall["pending_count"] == 1
    assert overall["total_profit_units"] == pytest.approx(0.0)
    assert overall["roi_percent"] == pytest.approx(0.0)
    assert overall["maximum_drawdown_units"] == pytest.approx(1.0)
    assert overall["average_price_movement_probability_pp"] == pytest.approx(2.0)
    spread = summary.loc[(summary["selection_scope"] == "SEASON_MARKET") & (summary["market_key"] == "spreads") & (summary["sample_scope"] == "ALL_TRACKED")].iloc[0]
    assert pd.isna(spread["brier_score"])
    settled = summary.loc[(summary["selection_scope"] == "SEASON") & (summary["market_key"] == "ALL") & (summary["sample_scope"] == "SETTLED_ONLY")].iloc[0]
    assert settled["tracked_count"] == 3
    assert settled["pending_count"] == 0


def test_persistence_validation_rejects_duplicate_or_invalid_arithmetic():
    ledger = create_forward_settlement(entries(), schedule())
    summary = create_forward_performance_summary(ledger)
    with duckdb.connect(":memory:") as connection:
        persist_forward_performance(connection, ledger, summary)
        validate_forward_performance(connection)
        connection.execute('UPDATE analytics.forward_tip_settlement SET profit_units=99 WHERE "result"=\'WIN\'')
        with pytest.raises(RuntimeError, match="validation failed"):
            validate_forward_performance(connection)


def test_all_pending_results_are_persisted_as_varchar_and_validate():
    """An all-null preseason result column must not be inferred as INTEGER."""
    pending_schedule = schedule().assign(
        home_score=None,
        away_score=None,
        is_completed=False,
    )
    ledger = create_forward_settlement(entries(), pending_schedule)
    assert ledger["result"].isna().all()
    summary = create_forward_performance_summary(ledger)
    with duckdb.connect(":memory:") as connection:
        persist_forward_performance(connection, ledger, summary)
        assert connection.execute(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_schema='analytics' AND table_name='forward_tip_settlement' "
            "AND column_name='result'"
        ).fetchone()[0] == "VARCHAR"
        validate_forward_performance(connection)


def test_forward_performance_quality_sql_accepts_valid_tables():
    ledger = create_forward_settlement(entries(), schedule())
    summary = create_forward_performance_summary(ledger)
    with duckdb.connect(":memory:") as connection:
        persist_forward_performance(connection, ledger, summary)
        sql = __import__("pathlib").Path(
            "sql/037_forward_performance_quality_checks.sql"
        ).read_text(encoding="utf-8")
        assert connection.execute(sql).fetchall() == []


def test_duplicate_schedule_game_is_rejected():
    duplicate = pd.concat([schedule(), schedule().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate games"):
        create_forward_settlement(entries(), duplicate)
