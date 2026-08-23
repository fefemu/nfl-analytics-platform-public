"""Tests for audited in-season refresh orchestration."""

from pathlib import Path

import duckdb
import pytest

from src.pipeline.run_in_season_refresh import run_in_season_refresh


def test_offline_refresh_runs_in_order_and_records_success(monkeypatch, tmp_path: Path):
    database = tmp_path / "test.duckdb"
    duckdb.connect(str(database)).close()
    snapshot = tmp_path / "odds.json"
    snapshot.write_text("{}", encoding="utf-8")
    calls = []
    monkeypatch.setattr("src.pipeline.run_in_season_refresh.run_modeling_pipeline", lambda database_file: calls.append("modeling"))
    monkeypatch.setattr("src.pipeline.run_in_season_refresh.run_odds_snapshot_pipeline", lambda snapshot_file: calls.append("offline_odds"))
    monkeypatch.setattr("src.pipeline.run_in_season_refresh.build_forward_betting_archive", lambda refresh_run_id, database_file: calls.append("archive") or 12)
    run_id = run_in_season_refresh(database, snapshot)
    assert calls == ["modeling", "offline_odds", "archive"]
    with duckdb.connect(str(database), read_only=True) as connection:
        row = connection.execute("SELECT status, refresh_mode, archived_market_row_count FROM analytics.refresh_run_history WHERE refresh_run_id=?", [run_id]).fetchone()
    assert row == ("SUCCESS", "OFFLINE_SNAPSHOT", 12)


def test_online_refresh_is_explicit_path(monkeypatch, tmp_path: Path):
    database = tmp_path / "test.duckdb"
    duckdb.connect(str(database)).close()
    calls = []
    monkeypatch.setattr("src.pipeline.run_in_season_refresh.run_modeling_pipeline", lambda database_file: calls.append("modeling"))
    monkeypatch.setattr("src.pipeline.run_in_season_refresh.run_odds_pipeline", lambda: calls.append("online_odds"))
    monkeypatch.setattr("src.pipeline.run_in_season_refresh.build_forward_betting_archive", lambda refresh_run_id, database_file: 1)
    run_in_season_refresh(database)
    assert calls == ["modeling", "online_odds"]


def test_failed_refresh_is_audited(monkeypatch, tmp_path: Path):
    database = tmp_path / "test.duckdb"
    duckdb.connect(str(database)).close()
    snapshot = tmp_path / "odds.json"
    snapshot.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("src.pipeline.run_in_season_refresh.run_modeling_pipeline", lambda database_file: (_ for _ in ()).throw(RuntimeError("model failed")))
    with pytest.raises(RuntimeError, match="model failed"):
        run_in_season_refresh(database, snapshot)
    with duckdb.connect(str(database), read_only=True) as connection:
        row = connection.execute("SELECT status, error_message FROM analytics.refresh_run_history").fetchone()
    assert row[0] == "FAILED"
    assert "model failed" in row[1]


def test_missing_snapshot_is_rejected_before_audit(tmp_path: Path):
    database = tmp_path / "test.duckdb"
    duckdb.connect(str(database)).close()
    with pytest.raises(FileNotFoundError, match="does not exist"):
        run_in_season_refresh(database, tmp_path / "missing.json")
