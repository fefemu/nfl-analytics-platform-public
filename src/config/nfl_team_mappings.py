"""
NFL Analytics Platform
NFL Team Name Mappings

Purpose:
    Map full Odds API team names to the abbreviated
    team codes used by nflverse schedule data.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""


ODDS_TEAM_TO_NFLVERSE = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}

HISTORICAL_TEAM_TO_CURRENT = {
    "LAR": "LA",
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LA",
}


def map_odds_team_name(team_name: str) -> str:
    """Return the nflverse code for an Odds API team."""

    normalized_name = team_name.strip()

    try:
        return ODDS_TEAM_TO_NFLVERSE[normalized_name]
    except KeyError:
        raise ValueError(
            f"Unknown Odds API NFL team: {team_name}"
        ) from None


def normalize_franchise_code(team_code: str) -> str:
    """Return the current franchise code for a historical team code."""

    normalized_code = team_code.strip().upper()

    return HISTORICAL_TEAM_TO_CURRENT.get(
        normalized_code,
        normalized_code,
    )
