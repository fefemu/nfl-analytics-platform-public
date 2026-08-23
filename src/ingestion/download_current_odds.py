"""
NFL Analytics Platform
Current NFL Odds Downloader

Purpose:
    Download current NFL betting odds and save a
    timestamped raw JSON snapshot for later processing.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config.odds_api import (
    BETTING_MARKETS,
    BOOKMAKER_REGIONS,
    NFL_SPORT_KEY,
    ODDS_FORMAT,
)
from src.ingestion.odds_api_client import (
    OddsAPIResult,
    fetch_current_nfl_odds,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "odds"


def build_snapshot_payload(
    result: OddsAPIResult,
    fetched_at: datetime,
) -> dict[str, Any]:
    """Build a raw snapshot with metadata and API events."""

    return {
        "metadata": {
            "fetched_at": fetched_at.isoformat(),
            "sport_key": NFL_SPORT_KEY,
            "regions": BOOKMAKER_REGIONS,
            "markets": list(BETTING_MARKETS),
            "odds_format": ODDS_FORMAT,
            "event_count": len(result.events),
            "requests_remaining": result.requests_remaining,
            "requests_used": result.requests_used,
            "requests_last": result.requests_last,
        },
        "events": result.events,
    }


def save_current_nfl_odds_snapshot() -> Path:
    """Download and save a timestamped NFL odds snapshot."""

    logger.info("Starting current NFL odds download...")

    result = fetch_current_nfl_odds()
    fetched_at = datetime.now(timezone.utc)

    snapshot_payload = build_snapshot_payload(
        result=result,
        fetched_at=fetched_at,
    )

    SNAPSHOT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = fetched_at.strftime("%Y%m%dT%H%M%SZ")
    snapshot_file = (
        SNAPSHOT_DIRECTORY
        / f"nfl_odds_{timestamp}.json"
    )

    snapshot_file.write_text(
        json.dumps(
            snapshot_payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    logger.info(
        "Odds snapshot saved: %s events to %s",
        len(result.events),
        snapshot_file,
    )

    return snapshot_file


def main() -> None:
    """Run the current NFL odds download workflow."""

    try:
        save_current_nfl_odds_snapshot()
    except Exception:
        logger.exception("Current NFL odds download failed.")
        raise


if __name__ == "__main__":
    main()