"""Paired uncertainty diagnostics for external unit ratings."""

import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.modeling.backtest_elo_rating_sources import (
    BACKTEST_VALIDATION_SEASONS,
)
from src.modeling.backtest_external_team_strength_candidates import (
    CANDIDATE_FEATURES,
    load_team_strength_backtest_data,
)
from src.modeling.train_logistic_baseline import (
    DATABASE_FILE,
    TARGET_COLUMN,
    create_logistic_pipeline,
    validate_database_file,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

BASE_CANDIDATE = "external_nfelo_qb"
UNIT_CANDIDATE = "external_nfelo_qb_units"
DEFAULT_BOOTSTRAP_ITERATIONS = 10_000
DEFAULT_RANDOM_SEED = 42

SUMMARY_COLUMNS = (
    "comparison_name",
    "fold_count",
    "validation_game_count",
    "base_brier_score",
    "unit_brier_score",
    "brier_score_delta",
    "unit_model_win_rate",
    "unit_model_loss_rate",
    "bootstrap_mean_delta",
    "bootstrap_95_percent_lower",
    "bootstrap_95_percent_upper",
)


def create_paired_oof_predictions(
    source_data: pd.DataFrame,
    validation_seasons: tuple[int, ...] = BACKTEST_VALIDATION_SEASONS,
) -> pd.DataFrame:
    """Create paired expanding-window OOF probabilities."""

    rows: list[pd.DataFrame] = []
    for validation_season in validation_seasons:
        train = source_data.loc[source_data["season"] < validation_season]
        validation = source_data.loc[
            source_data["season"] == validation_season
        ].copy()
        if train.empty or validation.empty:
            raise RuntimeError(
                f"Missing fold data for validation season {validation_season}."
            )

        fold = validation.loc[
            :, ["game_id", "season", TARGET_COLUMN]
        ].rename(columns={"season": "validation_season"})
        for candidate_name in (BASE_CANDIDATE, UNIT_CANDIDATE):
            features = CANDIDATE_FEATURES[candidate_name]
            model = create_logistic_pipeline(
                feature_columns=features,
                regularization_c=1.0,
            )
            model.fit(train.loc[:, features], train[TARGET_COLUMN])
            probabilities = model.predict_proba(
                validation.loc[:, features]
            )[:, 1]
            fold[f"{candidate_name}_probability"] = probabilities
            fold[f"{candidate_name}_brier_loss"] = (
                probabilities - validation[TARGET_COLUMN].to_numpy()
            ) ** 2
        rows.append(fold)

    predictions = pd.concat(rows, ignore_index=True)
    if predictions["game_id"].duplicated().any():
        raise RuntimeError("Paired OOF predictions contain duplicates.")
    return predictions


def summarize_paired_unit_value(
    predictions: pd.DataFrame,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Summarize paired Brier-loss deltas with bootstrap uncertainty."""

    if bootstrap_iterations <= 0:
        raise ValueError("Bootstrap iterations must be positive.")

    base_loss = predictions[f"{BASE_CANDIDATE}_brier_loss"].to_numpy()
    unit_loss = predictions[f"{UNIT_CANDIDATE}_brier_loss"].to_numpy()
    if len(base_loss) == 0:
        raise ValueError("Paired diagnostics require predictions.")

    deltas = unit_loss - base_loss
    random_generator = np.random.default_rng(random_seed)
    bootstrap_means = np.empty(bootstrap_iterations, dtype=float)
    for iteration in range(bootstrap_iterations):
        indices = random_generator.integers(0, len(deltas), len(deltas))
        bootstrap_means[iteration] = float(deltas[indices].mean())

    row = {
        "comparison_name": "units_vs_external_nfelo_qb",
        "fold_count": int(predictions["validation_season"].nunique()),
        "validation_game_count": len(predictions),
        "base_brier_score": float(base_loss.mean()),
        "unit_brier_score": float(unit_loss.mean()),
        "brier_score_delta": float(deltas.mean()),
        "unit_model_win_rate": float((deltas < 0.0).mean()),
        "unit_model_loss_rate": float((deltas > 0.0).mean()),
        "bootstrap_mean_delta": float(bootstrap_means.mean()),
        "bootstrap_95_percent_lower": float(
            np.quantile(bootstrap_means, 0.025)
        ),
        "bootstrap_95_percent_upper": float(
            np.quantile(bootstrap_means, 0.975)
        ),
    }
    return pd.DataFrame([row], columns=SUMMARY_COLUMNS)


def run_paired_unit_diagnostics(
    database_file: Path = DATABASE_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run paired diagnostics without accessing the protected holdout."""

    validate_database_file(database_file)
    with duckdb.connect(str(database_file), read_only=True) as connection:
        source_data = load_team_strength_backtest_data(connection)
    predictions = create_paired_oof_predictions(source_data)
    summary = summarize_paired_unit_value(predictions)
    logger.info(
        "Paired external unit diagnostics completed on %s OOF games "
        "without opening holdout.",
        len(predictions),
    )
    return summary, predictions


def main() -> None:
    """Run and print paired unit-rating diagnostics."""

    summary, _ = run_paired_unit_diagnostics()
    print("\nPAIRED EXTERNAL UNIT-RATING SUMMARY\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
