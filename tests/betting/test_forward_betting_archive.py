"""Tests for immutable forward betting snapshots and CLV."""

from datetime import datetime, timezone

import duckdb
import pandas as pd
import pytest

from src.betting.build_current_betting_board import BOARD_COLUMNS
from src.betting.forward_betting_archive import (
    prepare_forward_archive_rows,
    persist_forward_archive,
    validate_forward_archive,
)


def board(
    snapshot="s1",
    fetched="2026-09-01T12:00:00Z",
    odds=2.0,
    positive=True,
    model_probability=0.6,
):
    row = {column: None for column in BOARD_COLUMNS}
    row.update({
        "snapshot_id": snapshot, "fetched_at": fetched, "game_id": "g",
        "season": 2026, "game_type": "REG", "week": 1,
        "gameday": "2026-09-10", "commence_time": "2026-09-10T20:00:00Z",
        "home_team": "BUF", "away_team": "NYJ", "market_key": "h2h",
        "market_name": "Moneyline", "outcome_name": "Buffalo Bills",
        "outcome_type": "home", "best_bookmaker_key": "book",
        "best_bookmaker_title": "Book", "best_american_price": 100,
        "best_decimal_odds": odds, "bookmaker_count": 3, "model_name": "m",
        "model_version": "1", "prediction_mode": "mode",
        "model_probability": model_probability,
        "push_probability": 0.0, "loss_probability": 0.4, "probability_edge": 0.1,
        "probability_edge_percentage_points": 10.0, "fair_decimal_odds": 1.67,
        "expected_value_per_unit": 0.2, "expected_value_percent": 20.0,
        "full_kelly_fraction": 0.2, "positive_expected_value": positive,
        "prediction_generated_at": "2026-08-31", "betting_board_generated_at": fetched,
    })
    return pd.DataFrame([row], columns=BOARD_COLUMNS)


def test_prepare_keeps_only_future_rows():
    rows = prepare_forward_archive_rows(board(), "run", datetime(2026, 9, 1, tzinfo=timezone.utc))
    assert len(rows) == 1
    assert rows.loc[0, "is_tip_candidate"]
    assert len(rows.loc[0, "archive_key"]) == 64


def test_prepare_rejects_only_started_games():
    started = board(fetched="2026-09-11T12:00:00Z")
    with pytest.raises(RuntimeError, match="No future"):
        prepare_forward_archive_rows(
            started,
            "run",
            datetime(2026, 9, 11, 12, 1, tzinfo=timezone.utc),
        )


def test_prepare_rejects_post_kickoff_lock_of_pregame_snapshot():
    with pytest.raises(RuntimeError, match="No future"):
        prepare_forward_archive_rows(
            board(fetched="2026-09-10T19:00:00Z"),
            "run",
            datetime(2026, 9, 10, 20, 1, tzinfo=timezone.utc),
        )


def test_persistence_is_idempotent_and_clv_uses_later_snapshot():
    first = prepare_forward_archive_rows(
        board(), "run1", datetime(2026, 9, 1, 12, 1, tzinfo=timezone.utc)
    )
    second = prepare_forward_archive_rows(
        board("s2", "2026-09-05T12:00:00Z", 1.8, False),
        "run2",
        datetime(2026, 9, 5, 12, 1, tzinfo=timezone.utc),
    )
    with duckdb.connect(":memory:") as connection:
        persist_forward_archive(connection, first)
        persist_forward_archive(connection, first)
        persist_forward_archive(connection, second)
        validate_forward_archive(connection)
        assert connection.execute("SELECT COUNT(*) FROM analytics.forward_betting_board_archive").fetchone()[0] == 2
        clv = connection.execute("SELECT clv_probability_percentage_points FROM analytics.forward_tip_clv").fetchone()[0]
        assert clv == pytest.approx(100 * (1 / 1.8 - 1 / 2.0))


def test_same_archive_identity_cannot_silently_change_locked_values():
    archived_at = datetime(2026, 9, 1, 12, 1, tzinfo=timezone.utc)
    first = prepare_forward_archive_rows(board(), "run1", archived_at)
    changed = prepare_forward_archive_rows(
        board(model_probability=0.61), "run2", archived_at
    )
    with duckdb.connect(":memory:") as connection:
        persist_forward_archive(connection, first)
        with pytest.raises(RuntimeError, match="different locked values"):
            persist_forward_archive(connection, changed)
