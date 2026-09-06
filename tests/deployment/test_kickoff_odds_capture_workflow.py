"""Contract tests for the hosted kickoff-near market capture."""

from pathlib import Path


WORKFLOW = Path(".github/workflows/kickoff-odds-capture.yml")


def test_workflow_is_dispatch_only_and_runs_lightweight_capture() -> None:
    content = WORKFLOW.read_text(encoding="utf-8")
    assert "types: [kickoff-odds-capture]" in content
    assert "src.pipeline.run_kickoff_odds_capture --publish" in content
    assert "run_production_refresh" not in content
    assert "--refresh-sources" not in content
    assert "secrets.ODDS_API_KEY" in content
    assert "src.deployment.operational_database" in content
