"""Tests for the lightweight kickoff-near odds capture."""

from pathlib import Path

import pytest

from src.pipeline.run_kickoff_odds_capture import run_kickoff_odds_capture


def test_capture_skips_models_and_publishes_successful_operational_state(
    monkeypatch, tmp_path: Path
) -> None:
    calls = []
    database = tmp_path / "operational.duckdb"
    output = tmp_path / "dashboard.duckdb"
    monkeypatch.setattr("src.pipeline.run_kickoff_odds_capture.validate_database_file", lambda path: calls.append("validate"))
    monkeypatch.setattr("src.pipeline.run_kickoff_odds_capture.create_refresh_run_id", lambda started: "refresh_test")
    monkeypatch.setattr("src.pipeline.run_kickoff_odds_capture.record_refresh_start", lambda *args: calls.append(("start", args[2])))
    monkeypatch.setattr("src.pipeline.run_kickoff_odds_capture.run_odds_pipeline", lambda database_file: calls.append(("odds", database_file)))
    monkeypatch.setattr("src.pipeline.run_kickoff_odds_capture.build_forward_betting_archive", lambda run_id, path: calls.append(("archive", run_id)) or 42)
    monkeypatch.setattr("src.pipeline.run_kickoff_odds_capture.record_refresh_completion", lambda *args, **kwargs: calls.append(("complete", args[2])))
    monkeypatch.setattr("src.pipeline.run_kickoff_odds_capture.build_dashboard_snapshot", lambda **kwargs: calls.append("snapshot") or {"table": 1})
    monkeypatch.setattr("src.pipeline.run_kickoff_odds_capture.publish_dashboard_release", lambda *args, **kwargs: calls.append(("publish", kwargs["operational_file"])))

    run_id = run_kickoff_odds_capture(database, output_file=output, publish=True)

    assert run_id == "kickoff_capture_test"
    assert calls == [
        "validate", ("start", "KICKOFF_ODDS_CAPTURE"), ("odds", database),
        ("archive", "kickoff_capture_test"), ("complete", "SUCCESS"),
        "snapshot", ("publish", database),
    ]


def test_capture_fails_before_work_when_private_publisher_is_unavailable(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "src.pipeline.run_kickoff_odds_capture.publish_dashboard_release", None
    )
    monkeypatch.setattr(
        "src.pipeline.run_kickoff_odds_capture.validate_database_file",
        lambda path: pytest.fail("validation must not start"),
    )

    with pytest.raises(RuntimeError, match="private release adapter"):
        run_kickoff_odds_capture(tmp_path / "operational.duckdb", publish=True)
