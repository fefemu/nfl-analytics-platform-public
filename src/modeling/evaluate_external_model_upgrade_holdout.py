"""
NFL Analytics Platform
External Model Upgrade Holdout Evaluation

Purpose:
    Open the protected 2025 holdout once and evaluate
    the development-locked external upgrades for:

    - win probability;
    - Spread;
    - Totals.

    All candidate selection and hyperparameter decisions
    were completed before this module is executed.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.modeling.diagnose_external_elo_totals_value import (
    DEFAULT_BOOTSTRAP_ITERATIONS,
    DEFAULT_RANDOM_SEED,
    bootstrap_paired_mean_delta,
)
from src.modeling.external_probability_holdout_component import (
    evaluate_locked_probability_holdout,
    load_probability_holdout_data,
)
from src.modeling.external_spread_holdout_component import (
    evaluate_locked_spread_holdout,
    load_spread_holdout_data,
)
from src.modeling.external_totals_holdout_component import (
    evaluate_locked_totals_routing_holdout,
    load_totals_holdout_data,
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

PAIRED_SUMMARY_COLUMNS = (
    "model_layer",
    "loss_metric",
    "holdout_game_count",
    "current_mean_loss",
    "external_mean_loss",
    "external_mean_loss_delta",
    "external_win_rate",
    "external_loss_rate",
    "equal_loss_rate",
    "bootstrap_mean_delta",
    "bootstrap_95_percent_lower",
    "bootstrap_95_percent_upper",
)


def create_layer_paired_summary(
    model_layer: str,
    loss_metric: str,
    current_losses: pd.Series,
    external_losses: pd.Series,
    bootstrap_iterations: int,
    random_seed: int,
) -> dict[str, object]:
    """Summarize one paired holdout comparison."""

    current_values = np.asarray(
        current_losses,
        dtype=float,
    )

    external_values = np.asarray(
        external_losses,
        dtype=float,
    )

    if current_values.ndim != 1:
        raise ValueError(
            "Current holdout losses must be "
            "one-dimensional."
        )

    if external_values.ndim != 1:
        raise ValueError(
            "External holdout losses must be "
            "one-dimensional."
        )

    if (
        current_values.size == 0
        or external_values.size == 0
    ):
        raise ValueError(
            "Holdout loss arrays must not be empty."
        )

    if current_values.size != external_values.size:
        raise ValueError(
            "Current and external holdout losses must "
            "have identical lengths."
        )

    if (
        not np.isfinite(
            current_values
        ).all()
        or not np.isfinite(
            external_values
        ).all()
    ):
        raise ValueError(
            "Holdout losses must be finite."
        )

    paired_deltas = (
        external_values
        - current_values
    )

    bootstrap_results = (
        bootstrap_paired_mean_delta(
            paired_deltas=paired_deltas,
            iteration_count=bootstrap_iterations,
            random_seed=random_seed,
        )
    )

    wins = int(
        np.sum(paired_deltas < 0.0)
    )

    losses = int(
        np.sum(paired_deltas > 0.0)
    )

    equal_losses = int(
        np.sum(paired_deltas == 0.0)
    )

    game_count = int(
        paired_deltas.size
    )

    return {
        "model_layer": model_layer,
        "loss_metric": loss_metric,
        "holdout_game_count": game_count,
        "current_mean_loss": float(
            current_values.mean()
        ),
        "external_mean_loss": float(
            external_values.mean()
        ),
        "external_mean_loss_delta": float(
            paired_deltas.mean()
        ),
        "external_win_rate": (
            wins / game_count
        ),
        "external_loss_rate": (
            losses / game_count
        ),
        "equal_loss_rate": (
            equal_losses / game_count
        ),
        **bootstrap_results,
    }


def create_holdout_paired_summary(
    probability_predictions: pd.DataFrame,
    spread_predictions: pd.DataFrame,
    totals_predictions: pd.DataFrame,
    bootstrap_iterations: int = (
        DEFAULT_BOOTSTRAP_ITERATIONS
    ),
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Create paired summaries for all model layers."""

    required_probability_columns = {
        "current_brier_loss",
        "external_brier_loss",
    }

    required_spread_columns = {
        "current_absolute_error",
        "external_absolute_error",
    }

    required_totals_columns = {
        "current_absolute_error",
        "external_absolute_error",
    }

    missing_probability_columns = sorted(
        required_probability_columns
        - set(probability_predictions.columns)
    )

    missing_spread_columns = sorted(
        required_spread_columns
        - set(spread_predictions.columns)
    )

    missing_totals_columns = sorted(
        required_totals_columns
        - set(totals_predictions.columns)
    )

    if missing_probability_columns:
        raise ValueError(
            "Probability predictions are missing "
            "columns: "
            + ", ".join(
                missing_probability_columns
            )
        )

    if missing_spread_columns:
        raise ValueError(
            "Spread predictions are missing columns: "
            + ", ".join(
                missing_spread_columns
            )
        )

    if missing_totals_columns:
        raise ValueError(
            "Totals predictions are missing columns: "
            + ", ".join(
                missing_totals_columns
            )
        )

    summary_rows = [
        create_layer_paired_summary(
            model_layer="PROBABILITY",
            loss_metric="BRIER_SCORE",
            current_losses=(
                probability_predictions[
                    "current_brier_loss"
                ]
            ),
            external_losses=(
                probability_predictions[
                    "external_brier_loss"
                ]
            ),
            bootstrap_iterations=(
                bootstrap_iterations
            ),
            random_seed=random_seed,
        ),
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
            bootstrap_iterations=(
                bootstrap_iterations
            ),
            random_seed=random_seed + 10,
        ),
        create_layer_paired_summary(
            model_layer="TOTALS",
            loss_metric="ABSOLUTE_ERROR",
            current_losses=(
                totals_predictions[
                    "current_absolute_error"
                ]
            ),
            external_losses=(
                totals_predictions[
                    "external_absolute_error"
                ]
            ),
            bootstrap_iterations=(
                bootstrap_iterations
            ),
            random_seed=random_seed + 20,
        ),
    ]

    return pd.DataFrame(
        summary_rows,
        columns=PAIRED_SUMMARY_COLUMNS,
    )


def run_external_model_upgrade_holdout(
    database_file: Path = DATABASE_FILE,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Open and evaluate the protected holdout once."""

    validate_database_file(database_file)

    logger.info(
        "Opening the locked 2025 external-model "
        "holdout evaluation..."
    )

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        probability_data = (
            load_probability_holdout_data(
                connection
            )
        )

        spread_data = load_spread_holdout_data(
            connection
        )

        totals_data = load_totals_holdout_data(
            connection
        )

    (
        probability_summary,
        probability_predictions,
    ) = evaluate_locked_probability_holdout(
        probability_data
    )

    (
        spread_summary,
        spread_predictions,
    ) = evaluate_locked_spread_holdout(
        spread_data
    )

    (
        totals_summary,
        totals_predictions,
    ) = evaluate_locked_totals_routing_holdout(
        totals_data
    )

    paired_summary = (
        create_holdout_paired_summary(
            probability_predictions=(
                probability_predictions
            ),
            spread_predictions=(
                spread_predictions
            ),
            totals_predictions=(
                totals_predictions
            ),
        )
    )

    logger.info(
        "Locked 2025 external-model holdout "
        "evaluation completed."
    )

    return (
        probability_summary,
        spread_summary,
        totals_summary,
        paired_summary,
    )


def main() -> None:
    """Run and print the one-time holdout evaluation."""

    (
        probability_summary,
        spread_summary,
        totals_summary,
        paired_summary,
    ) = run_external_model_upgrade_holdout()

    print(
        "\nFINAL EXTERNAL PROBABILITY HOLDOUT\n"
    )

    print(
        probability_summary.to_string(
            index=False
        )
    )

    print(
        "\nFINAL EXTERNAL SPREAD HOLDOUT\n"
    )

    print(
        spread_summary.to_string(
            index=False
        )
    )

    print(
        "\nFINAL EXTERNAL TOTALS HOLDOUT\n"
    )

    print(
        totals_summary.to_string(
            index=False
        )
    )

    print(
        "\nPAIRED EXTERNAL MODEL HOLDOUT SUMMARY\n"
    )

    print(
        paired_summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()