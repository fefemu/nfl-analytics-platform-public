"""Normalize ESPN's current NFL availability feed into the injury schema."""

from datetime import datetime, timezone
from pathlib import Path
import re

import polars as pl
import requests

from src.ingestion.download_historical_data import SCHEDULE_FILE
from src.ingestion.download_player_directory import PLAYER_DIRECTORY_FILE


ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
ESPN_TEAM_ALIASES = {
    "LAR": "LA",
    "WSH": "WAS",
}


def _normalize_team_abbreviation(team: object) -> str:
    """Translate ESPN team codes to the project's canonical abbreviations."""

    abbreviation = str(team or "").upper()
    return ESPN_TEAM_ALIASES.get(abbreviation, abbreviation)


def _athlete_espn_id(athlete: dict) -> str | None:
    """Extract ESPN's athlete id from the canonical player-card URL."""

    for link in athlete.get("links", []):
        match = re.search(r"/id/(\d+)(?:/|$)", str(link.get("href", "")))
        if match:
            return match.group(1)
    return None


def build_current_espn_injuries(
    payload: dict,
    *,
    schedule_file: Path = SCHEDULE_FILE,
    player_directory_file: Path = PLAYER_DIRECTORY_FILE,
) -> pl.DataFrame:
    """Build one current, next-game injury row per mapped unavailable player."""

    season = int(payload["season"]["year"])
    captured_at = datetime.fromisoformat(payload["timestamp"].replace("Z", "+00:00"))
    schedule = pl.read_parquet(schedule_file)
    gameday = (
        pl.col("gameday").str.to_date(strict=False)
        if schedule.schema["gameday"] == pl.String
        else pl.col("gameday").cast(pl.Date)
    )
    schedule = schedule.filter(
        (pl.col("season") == season)
        & pl.col("game_type").is_in(["REG", "POST"])
        & pl.col("home_score").is_null()
        & (gameday >= captured_at.date())
    ).sort(["gameday", "gametime"])
    team_games: dict[str, tuple[int, str]] = {}
    for game in schedule.iter_rows(named=True):
        for team in (game["home_team"], game["away_team"]):
            team_games.setdefault(str(team), (int(game["week"]), str(game["game_type"])))

    directory = pl.read_parquet(player_directory_file).select("espn_id", "gsis_id")
    id_map = {
        str(row[0]): str(row[1])
        for row in directory.drop_nulls().iter_rows()
    }
    status_map = {
        "Questionable": "Questionable",
        "Doubtful": "Doubtful",
        "Out": "Out",
        "Injured Reserve": "Out",
        "Suspension": "Out",
    }
    rows: list[dict[str, object]] = []
    covered_teams: set[str] = set()
    for team_block in payload.get("injuries", []):
        for item in team_block.get("injuries", []):
            report_status = status_map.get(item.get("status"))
            athlete = item.get("athlete") or {}
            team = _normalize_team_abbreviation(
                (athlete.get("team") or {}).get("abbreviation")
            )
            gsis_id = id_map.get(str(_athlete_espn_id(athlete)))
            if not report_status or not gsis_id or team not in team_games:
                continue
            week, game_type = team_games[team]
            details = item.get("details") or {}
            rows.append({
                "season": season, "season_type": game_type, "game_type": game_type,
                "team": team, "week": week, "gsis_id": gsis_id,
                "position": (athlete.get("position") or {}).get("abbreviation"),
                "full_name": athlete.get("displayName"), "first_name": athlete.get("firstName"),
                "last_name": athlete.get("lastName"),
                "report_primary_injury": details.get("type"),
                "report_secondary_injury": None, "report_status": report_status,
                "practice_primary_injury": None, "practice_secondary_injury": None,
                "practice_status": None, "date_modified": item.get("date") or captured_at.isoformat(),
            })
            covered_teams.add(team)
    missing = sorted(set(team_games) - covered_teams)
    if missing:
        raise RuntimeError("ESPN current injury coverage is incomplete for: " + ", ".join(missing))
    return pl.DataFrame(rows).with_columns(
        pl.col("date_modified").str.to_datetime(time_zone="UTC", strict=False)
    )


def download_current_espn_injuries(output_file: Path) -> Path:
    """Fetch, validate and atomically persist the current ESPN feed."""

    response = requests.get(ESPN_INJURIES_URL, timeout=60)
    response.raise_for_status()
    data = build_current_espn_injuries(response.json())
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_suffix(".tmp.parquet")
    data.write_parquet(temporary)
    temporary.replace(output_file)
    return output_file
