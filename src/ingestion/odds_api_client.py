"""
NFL Analytics Platform
The Odds API Client

Purpose:
    Retrieve current NFL Moneyline, Spread and Totals
    odds while tracking API quota usage.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging
from dataclasses import dataclass
from typing import Any

import requests

from src.config.odds_api import (
    BETTING_MARKETS,
    BOOKMAKER_REGIONS,
    NFL_SPORT_KEY,
    ODDS_API_BASE_URL,
    ODDS_FORMAT,
    get_odds_api_key,
)


logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class OddsAPIResult:
    """Store Odds API events and quota information."""

    events: list[dict[str, Any]]
    requests_remaining: int | None
    requests_used: int | None
    requests_last: int | None


def parse_optional_integer(value: str | None) -> int | None:
    """Convert an optional response header to an integer."""

    if value is None:
        return None

    try:
        return int(value)
    except ValueError:
        return None


def fetch_current_nfl_odds() -> OddsAPIResult:
    """Fetch current NFL odds from The Odds API."""

    endpoint = (
        f"{ODDS_API_BASE_URL}/sports/"
        f"{NFL_SPORT_KEY}/odds"
    )

    parameters = {
        "apiKey": get_odds_api_key(),
        "regions": BOOKMAKER_REGIONS,
        "markets": ",".join(BETTING_MARKETS),
        "oddsFormat": ODDS_FORMAT,
    }

    try:
        response = requests.get(
            endpoint,
            params=parameters,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException:
        raise RuntimeError(
            "The Odds API request failed."
        ) from None

    try:
        payload = response.json()
    except ValueError:
        raise RuntimeError(
            "The Odds API returned invalid JSON."
        ) from None

    if not isinstance(payload, list):
        raise RuntimeError(
            "The Odds API response must contain a list of events."
        )

    result = OddsAPIResult(
        events=payload,
        requests_remaining=parse_optional_integer(
            response.headers.get("x-requests-remaining")
        ),
        requests_used=parse_optional_integer(
            response.headers.get("x-requests-used")
        ),
        requests_last=parse_optional_integer(
            response.headers.get("x-requests-last")
        ),
    )

    logger.info(
        "Odds API request completed: %s events, "
        "%s credits remaining.",
        len(result.events),
        result.requests_remaining,
    )

    return result