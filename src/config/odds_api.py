"""
NFL Analytics Platform
The Odds API Configuration

Purpose:
    Store non-secret Odds API settings and securely
    load the API key from the local environment.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
NFL_SPORT_KEY = "americanfootball_nfl"

BOOKMAKER_REGIONS = "us"
BETTING_MARKETS = ("h2h", "spreads", "totals")
ODDS_FORMAT = "american"

load_dotenv(ENV_FILE)


def get_odds_api_key() -> str:
    """Return the configured Odds API key."""

    api_key = os.getenv("ODDS_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError(
            "ODDS_API_KEY is missing from the environment."
        )

    if api_key == "IDE_JÖN_A_SAJÁT_VALÓDI_API_KULCSOD":
        raise RuntimeError(
            "ODDS_API_KEY still contains the placeholder value."
        )

    return api_key