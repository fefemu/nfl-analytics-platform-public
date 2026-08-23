"""
NFL Analytics Platform
Current NFL Odds Pipeline Runner

Purpose:
    Download current NFL odds, load the raw snapshot
    and build analytics-ready processed odds data.

Author:
    Ferenc Kaizer

Version:
    0.5.0
"""

import logging

from src.ingestion.download_current_odds import (
    save_current_nfl_odds_snapshot,
)
from src.processing.build_processed_odds import (
    build_processed_odds,
)
from src.processing.load_odds_snapshot_to_duckdb import (
    load_odds_snapshot_to_duckdb,
)
from src.analytics.build_best_odds import build_best_odds
from src.analytics.build_odds_event_bridge import (
    build_odds_event_bridge,
)
from src.analytics.build_current_market_board import (
    build_current_market_board,
)
from src.betting.build_current_moneyline_value import build_current_moneyline_value
from src.betting.build_current_spread_value import build_current_spread_value
from src.betting.build_current_totals_value import build_current_totals_value
from src.betting.build_current_betting_board import build_current_betting_board


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def run_odds_pipeline() -> None:
    """Run the current NFL odds ingestion pipeline."""

    logger.info("Starting current NFL odds pipeline...")

    logger.info("Step 1/10: Downloading current NFL odds.")
    snapshot_file = save_current_nfl_odds_snapshot()

    logger.info("Step 2/10: Loading odds snapshot into DuckDB.")
    load_odds_snapshot_to_duckdb(
        snapshot_file=snapshot_file,
    )

    logger.info("Step 3/10: Building processed odds data.")
    build_processed_odds()

    logger.info("Step 4/10: Building best available odds.")
    build_best_odds()

    logger.info(
        "Step 5/10: Matching odds events to schedule games."
    )
    build_odds_event_bridge()

    logger.info(
        "Step 6/10: Building current NFL market board."
    )
    build_current_market_board()

    logger.info("Step 7/10: Building Moneyline expected value.")
    build_current_moneyline_value()

    logger.info("Step 8/10: Building Spread expected value.")
    build_current_spread_value()

    logger.info("Step 9/10: Building Totals expected value.")
    build_current_totals_value()

    logger.info("Step 10/10: Building combined betting board.")
    build_current_betting_board()

    logger.info(
        "Current NFL odds pipeline completed successfully."
    )


def main() -> None:
    """Run the current NFL odds pipeline workflow."""

    try:
        run_odds_pipeline()
    except Exception:
        logger.exception(
            "Current NFL odds pipeline failed."
        )
        raise


if __name__ == "__main__":
    main()
