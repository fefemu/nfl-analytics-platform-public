"""Tests for immutable forward betting snapshots and market movement."""

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
    market_key="h2h",
    outcome_type="home",
    point=None,
):
    row = {column: None for column in BOARD_COLUMNS}
    row.update({
        "snapshot_id": snapshot, "fetched_at": fetched, "game_id": "g",
        "season": 2026, "game_type": "REG", "week": 1,
        "gameday": "2026-09-10", "commence_time": "2026-09-10T20:00:00Z",
        "home_team": "BUF", "away_team": "NYJ", "market_key": market_key,
        "market_name": {"h2h": "Moneyline", "spreads": "Spread", "totals": "Total"}[market_key],
        "outcome_name": "Over" if outcome_type == "over" else "Buffalo Bills",
        "outcome_type": outcome_type, "point": point, "market_line": point,
        "best_bookmaker_key": "book",
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


def test_persistence_is_idempotent_and_market_movement_uses_first_entry():
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
        movement = connection.execute(
            """SELECT entry_snapshot_id, latest_snapshot_id,
                      price_movement_implied_probability_pp,
                      market_movement_direction, is_clv
               FROM analytics.forward_tip_market_movement"""
        ).fetchone()
        assert movement[0:2] == ("s1", "s2")
        assert movement[2] == pytest.approx(100 * (1 / 1.8 - 1 / 2.0))
        assert movement[3:] == ("POSITIVE", False)


@pytest.mark.parametrize(
    ("market_key", "outcome_type", "entry_point", "latest_point", "expected_advantage"),
    [
        ("spreads", "home", 3.5, 2.5, 1.0),
        ("totals", "over", 46.5, 48.0, 1.5),
        ("totals", "under", 48.0, 46.5, 1.5),
    ],
)
def test_line_movement_is_oriented_from_entry_selection_perspective(
    market_key, outcome_type, entry_point, latest_point, expected_advantage
):
    first = prepare_forward_archive_rows(
        board(market_key=market_key, outcome_type=outcome_type, point=entry_point),
        "run1", datetime(2026, 9, 1, 12, 1, tzinfo=timezone.utc),
    )
    latest = prepare_forward_archive_rows(
        board("s2", "2026-09-05T12:00:00Z", 2.0, False,
              market_key=market_key, outcome_type=outcome_type, point=latest_point),
        "run2", datetime(2026, 9, 5, 12, 1, tzinfo=timezone.utc),
    )
    with duckdb.connect(":memory:") as connection:
        persist_forward_archive(connection, first)
        persist_forward_archive(connection, latest)
        result = connection.execute(
            """SELECT entry_point, latest_point, entry_line_advantage_points,
                      market_movement_direction, comparison_type, is_closing_snapshot
               FROM analytics.forward_tip_market_movement"""
        ).fetchone()
    assert result == (
        entry_point, latest_point, expected_advantage,
        "POSITIVE", "LATEST_PRE_KICKOFF", False,
    )


def test_signal_without_later_snapshot_is_not_reported_as_clv():
    rows = prepare_forward_archive_rows(
        board(), "run1", datetime(2026, 9, 1, 12, 1, tzinfo=timezone.utc)
    )
    with duckdb.connect(":memory:") as connection:
        persist_forward_archive(connection, rows)
        result = connection.execute(
            """SELECT has_latest_pregame_comparison, market_movement_direction,
                      is_closing_snapshot, is_clv
               FROM analytics.forward_tip_market_movement"""
        ).fetchone()
    assert result == (False, "NO_LATER_SNAPSHOT", False, False)


def test_later_positive_signal_does_not_replace_first_entry():
    first = prepare_forward_archive_rows(
        board(), "run1", datetime(2026, 9, 1, 12, 1, tzinfo=timezone.utc)
    )
    later_positive = prepare_forward_archive_rows(
        board("s2", "2026-09-05T12:00:00Z", 1.9, True),
        "run2", datetime(2026, 9, 5, 12, 1, tzinfo=timezone.utc),
    )
    with duckdb.connect(":memory:") as connection:
        persist_forward_archive(connection, first)
        persist_forward_archive(connection, later_positive)
        validate_forward_archive(connection)
        result = connection.execute(
            """SELECT COUNT(*), MIN(entry_snapshot_id), MIN(latest_snapshot_id)
               FROM analytics.forward_tip_market_movement"""
        ).fetchone()
    assert result == (1, "s1", "s2")


def test_forward_quality_checks_accept_market_movement_schema():
    first = prepare_forward_archive_rows(
        board(), "run1", datetime(2026, 9, 1, 12, 1, tzinfo=timezone.utc)
    )
    later = prepare_forward_archive_rows(
        board("s2", "2026-09-05T12:00:00Z", 1.9, False),
        "run2", datetime(2026, 9, 5, 12, 1, tzinfo=timezone.utc),
    )
    quality_sql = __import__("pathlib").Path(
        "sql/036_forward_refresh_quality_checks.sql"
    ).read_text(encoding="utf-8")
    with duckdb.connect(":memory:") as connection:
        connection.execute("CREATE SCHEMA analytics")
        connection.execute(
            """CREATE TABLE analytics.refresh_run_history (
                   status VARCHAR, started_at TIMESTAMP, completed_at TIMESTAMP,
                   archived_market_row_count BIGINT, error_message VARCHAR
               )"""
        )
        persist_forward_archive(connection, first)
        persist_forward_archive(connection, later)
        checks = connection.execute(quality_sql).fetchdf()
    assert checks["status"].eq("PASS").all()


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
