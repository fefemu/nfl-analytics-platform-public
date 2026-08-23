"""
NFL Analytics Platform
Offline Odds Snapshot Pipeline Runner

Purpose:
    Rebuild all Odds API DuckDB and analytics layers
    from an existing local JSON snapshot without
    making a new API request.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import argparse
import logging
from pathlib import Path

from src.analytics.build_best_odds import build_best_odds
from src.analytics.build_current_market_board import (
    build_current_market_board,
)
from src.analytics.build_odds_event_bridge import (
    build_odds_event_bridge,
)
from src.processing.build_processed_odds import (
    build_processed_odds,
)
from src.processing.load_odds_snapshot_to_duckdb import (
    load_odds_snapshot_to_duckdb,
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


def run_odds_snapshot_pipeline(
    snapshot_file: Path,
) -> None:
    """Rebuild all odds layers from a local snapshot."""

    logger.info("Starting offline odds snapshot pipeline...")

    logger.info("Step 1/9: Loading snapshot into DuckDB.")
    load_odds_snapshot_to_duckdb(
        snapshot_file=snapshot_file,
    )

    logger.info("Step 2/9: Building processed odds data.")
    build_processed_odds()

    logger.info("Step 3/9: Building best available odds.")
    build_best_odds()

    logger.info(
        "Step 4/9: Matching odds events to schedule games."
    )
    build_odds_event_bridge()

    logger.info(
        "Step 5/9: Building current NFL market board."
    )
    build_current_market_board()

    logger.info("Step 6/9: Building Moneyline expected value.")
    build_current_moneyline_value()

    logger.info("Step 7/9: Building Spread expected value.")
    build_current_spread_value()

    logger.info("Step 8/9: Building Totals expected value.")
    build_current_totals_value()

    logger.info("Step 9/9: Building combined betting board.")
    build_current_betting_board()

    logger.info(
        "Offline odds snapshot pipeline completed successfully."
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Rebuild Odds API layers from a local JSON snapshot."
        )
    )

    parser.add_argument(
        "snapshot_file",
        type=Path,
        help=(
            "Path to an Odds API snapshot relative "
            "to the project root."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run the offline odds snapshot workflow."""

    args = parse_arguments()

    try:
        run_odds_snapshot_pipeline(
            snapshot_file=args.snapshot_file,
        )
    except Exception:
        logger.exception(
            "Offline odds snapshot pipeline failed."
        )
        raise


if __name__ == "__main__":
    main()
