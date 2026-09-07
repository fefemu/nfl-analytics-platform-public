from pathlib import Path

import polars as pl
import pytest

from src.ingestion.download_current_espn_injuries import (
    _athlete_espn_id,
    _normalize_team_abbreviation,
    build_current_espn_injuries,
)


def _write_sources(tmp_path: Path) -> tuple[Path, Path]:
    schedule_file = tmp_path / "schedule.parquet"
    player_file = tmp_path / "players.parquet"
    pl.DataFrame({
        "season": [2026],
        "game_type": ["REG"],
        "week": [1],
        "gameday": ["2026-09-13"],
        "gametime": ["20:20"],
        "home_team": ["LA"],
        "away_team": ["WAS"],
        "home_score": [None],
    }).write_parquet(schedule_file)
    pl.DataFrame({
        "espn_id": ["1001", "1002", "1003"],
        "gsis_id": ["00-001", "00-002", "00-003"],
    }).write_parquet(player_file)
    return schedule_file, player_file


def _injury(espn_id: str, team: str, status: str) -> dict:
    return {
        "status": status,
        "date": "2026-09-07T10:00Z",
        "details": {"type": "Knee"},
        "athlete": {
            "displayName": f"Player {espn_id}",
            "firstName": "Player",
            "lastName": espn_id,
            "team": {"abbreviation": team},
            "position": {"abbreviation": "WR"},
            "links": [{"href": f"https://www.espn.com/nfl/player/_/id/{espn_id}/name"}],
        },
    }


def test_athlete_id_and_team_alias_normalization() -> None:
    athlete = {"links": [{"href": "https://www.espn.com/nfl/player/_/id/12345/name"}]}

    assert _athlete_espn_id(athlete) == "12345"
    assert _normalize_team_abbreviation("LAR") == "LA"
    assert _normalize_team_abbreviation("WSH") == "WAS"
    assert _normalize_team_abbreviation("buf") == "BUF"


def test_build_current_injuries_maps_aliases_and_excludes_active(tmp_path: Path) -> None:
    schedule_file, player_file = _write_sources(tmp_path)
    payload = {
        "season": {"year": 2026},
        "timestamp": "2026-09-07T09:00:00Z",
        "injuries": [
            {"injuries": [_injury("1001", "LAR", "Questionable"),
                           _injury("1003", "LAR", "Active")]},
            {"injuries": [_injury("1002", "WSH", "Injured Reserve")]},
        ],
    }

    result = build_current_espn_injuries(
        payload,
        schedule_file=schedule_file,
        player_directory_file=player_file,
    ).sort("team")

    assert result.select("team").to_series().to_list() == ["LA", "WAS"]
    assert result.select("week").to_series().to_list() == [1, 1]
    assert result.select("report_status").to_series().to_list() == ["Questionable", "Out"]
    assert "00-003" not in result.select("gsis_id").to_series().to_list()


def test_build_current_injuries_rejects_incomplete_team_coverage(tmp_path: Path) -> None:
    schedule_file, player_file = _write_sources(tmp_path)
    payload = {
        "season": {"year": 2026},
        "timestamp": "2026-09-07T09:00:00Z",
        "injuries": [{"injuries": [_injury("1001", "LAR", "Questionable")]}],
    }

    with pytest.raises(RuntimeError, match="WAS"):
        build_current_espn_injuries(
            payload,
            schedule_file=schedule_file,
            player_directory_file=player_file,
        )
