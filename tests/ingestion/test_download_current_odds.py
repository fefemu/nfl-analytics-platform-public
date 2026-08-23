"""Tests for the current NFL odds downloader."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.ingestion import download_current_odds
from src.ingestion.odds_api_client import OddsAPIResult


def test_build_snapshot_payload_includes_metadata() -> None:
    """Include request metadata and events in the snapshot."""

    fetched_at = datetime(
        2026,
        7,
        19,
        12,
        30,
        tzinfo=timezone.utc,
    )
    result = OddsAPIResult(
        events=[{"id": "test-event"}],
        requests_remaining=497,
        requests_used=3,
        requests_last=3,
    )

    payload = download_current_odds.build_snapshot_payload(
        result=result,
        fetched_at=fetched_at,
    )

    assert payload["metadata"] == {
        "fetched_at": "2026-07-19T12:30:00+00:00",
        "sport_key": "americanfootball_nfl",
        "regions": "us",
        "markets": ["h2h", "spreads", "totals"],
        "odds_format": "american",
        "event_count": 1,
        "requests_remaining": 497,
        "requests_used": 3,
        "requests_last": 3,
    }
    assert payload["events"] == [{"id": "test-event"}]


def test_save_current_nfl_odds_snapshot_writes_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Write a raw odds snapshot without calling the real API."""

    result = OddsAPIResult(
        events=[
            {
                "id": "test-event",
                "home_team": "Home Team",
                "away_team": "Away Team",
            }
        ],
        requests_remaining=497,
        requests_used=3,
        requests_last=3,
    )

    monkeypatch.setattr(
        download_current_odds,
        "SNAPSHOT_DIRECTORY",
        tmp_path,
    )
    monkeypatch.setattr(
        download_current_odds,
        "fetch_current_nfl_odds",
        lambda: result,
    )

    snapshot_file = (
        download_current_odds.save_current_nfl_odds_snapshot()
    )

    assert snapshot_file.exists()
    assert snapshot_file.parent == tmp_path
    assert snapshot_file.name.startswith("nfl_odds_")
    assert snapshot_file.suffix == ".json"

    snapshot = json.loads(
        snapshot_file.read_text(encoding="utf-8")
    )

    assert snapshot["metadata"]["event_count"] == 1
    assert snapshot["metadata"]["requests_remaining"] == 497
    assert snapshot["events"] == result.events