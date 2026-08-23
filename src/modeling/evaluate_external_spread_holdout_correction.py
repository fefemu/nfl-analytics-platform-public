"""
NFL Analytics Platform
External Spread Holdout Protocol Correction

Purpose:
    Re-run only the locked 2025 Spread comparison after
    removing an unintended core-model eligibility filter.

    No model, feature or hyperparameter changes are made.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging
from pathlib import Path

import duckdb
import pandas as pd

from src.modeling.evaluate_external_model_upgrade_holdout import (
    create_layer_paired_summary,
)
from src.modeling.external_spread_holdout_component import (
    evaluate_locked_spread_holdout,
    load_spread_holdout_data,
)
from src.modeling.train_logistic_baseline import (
    DATABASE_FILE,
    validate_database_file,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def run_external_spread_holdout_correction(
    database_file: Path = DATABASE_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the documented Spread-only correction."""

    validate_database_file(database_file)

    logger.info(
        "Starting locked 2025 Spread holdout "
        "protocol correction..."
    )

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        spread_data = load_spread_holdout_data(
            connection
        )

    (
        spread_summary,
        spread_predictions,
    ) = evaluate_locked_spread_holdout(
        spread_data
    )

    paired_summary = pd.DataFrame(
        [
            create_layer_paired_summary(
                model_layer="SPREAD",
                loss_metric="ABSOLUTE_ERROR",
                current_losses=(
                    spread_predictions[
                        "current_absolute_error"
                    ]
                ),
                external_losses=(
                    spread_predictions[
                        "external_absolute_error"
                    ]
                ),
                bootstrap_iterations=10_000,
                random_seed=52,
            )
        ]
    )

    logger.info(
        "Locked 2025 Spread holdout protocol "
        "correction completed."
    )

    return spread_summary, paired_summary


def main() -> None:
    """Run and print corrected Spread results."""

    (
        spread_summary,
        paired_summary,
    ) = run_external_spread_holdout_correction()

    print(
        "\nCORRECTED FINAL EXTERNAL SPREAD HOLDOUT\n"
    )

    print(
        spread_summary.to_string(
            index=False
        )
    )

    print(
        "\nCORRECTED PAIRED SPREAD HOLDOUT SUMMARY\n"
    )

    print(
        paired_summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()