"""
NFL Analytics Platform
Selected Model Final Holdout Evaluation

Purpose:
    Perform the one-time final evaluation of the frozen
    selected model on the 2025 holdout season.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging
from pathlib import Path

import duckdb
import pandas as pd
from sklearn.pipeline import Pipeline

from src.modeling.selected_model import (
    SELECTED_MODEL,
)
from src.modeling.train_logistic_baseline import (
    DATABASE_FILE,
    DATASET_FULL_NAME,
    HOLDOUT_SPLIT,
    SPLIT_FULL_NAME,
    TARGET_COLUMN,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    ModelEvaluation,
    create_logistic_pipeline,
    evaluate_probabilities,
    validate_database_file,
    validate_source_tables,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


FINAL_TRAINING_SPLITS = (
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
)

EXPECTED_EVALUATION_SPLITS = {
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    HOLDOUT_SPLIT,
}


def load_final_evaluation_data(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Load eligible train, validation, and holdout games."""

    selected_columns = ",\n                ".join(
        f"dataset.{column_name}"
        for column_name in SELECTED_MODEL.feature_columns
    )

    data = connection.execute(
        f"""
        SELECT
            dataset.game_id,
            dataset.season,
            dataset.game_type,
            dataset.week,
            dataset.game_date,
            splits.split_name,
            dataset.{TARGET_COLUMN},
            dataset.elo_home_win_probability,
            {selected_columns}
        FROM {DATASET_FULL_NAME} AS dataset
        INNER JOIN {SPLIT_FULL_NAME} AS splits
            USING (game_id)
        WHERE splits.is_core_model_eligible = TRUE
          AND splits.split_name IN (
              '{TRAIN_SPLIT}',
              '{VALIDATION_SPLIT}',
              '{HOLDOUT_SPLIT}'
          )
        ORDER BY
            dataset.game_date,
            dataset.game_id
        """
    ).fetchdf()

    if data.empty:
        raise RuntimeError(
            "No final evaluation games were loaded."
        )

    loaded_splits = set(
        data["split_name"].unique()
    )

    if loaded_splits != EXPECTED_EVALUATION_SPLITS:
        raise RuntimeError(
            "Final evaluation data must contain train, "
            "validation, and holdout splits."
        )

    if data[TARGET_COLUMN].isna().any():
        raise RuntimeError(
            "Final evaluation data contains missing targets."
        )

    logger.info(
        "Final evaluation data loaded: %s development "
        "games and %s holdout games.",
        int(
            data["split_name"].isin(
                FINAL_TRAINING_SPLITS
            ).sum()
        ),
        int(
            (
                data["split_name"] == HOLDOUT_SPLIT
            ).sum()
        ),
    )

    return data


def train_frozen_selected_model(
    evaluation_data: pd.DataFrame,
) -> Pipeline:
    """Train the frozen model on train and validation games."""

    training_data = evaluation_data.loc[
        evaluation_data["split_name"].isin(
            FINAL_TRAINING_SPLITS
        )
    ].copy()

    if training_data.empty:
        raise RuntimeError(
            "No final model training games are available."
        )

    missing_features = sorted(
        set(SELECTED_MODEL.feature_columns)
        - set(training_data.columns)
    )

    if missing_features:
        raise RuntimeError(
            "Final training data is missing features: "
            + ", ".join(missing_features)
        )

    training_target = training_data[TARGET_COLUMN]

    if training_target.nunique() != 2:
        raise RuntimeError(
            "Final training data must contain both "
            "target classes."
        )

    model = create_logistic_pipeline(
        feature_columns=(
            SELECTED_MODEL.feature_columns
        ),
        regularization_c=(
            SELECTED_MODEL.regularization_c
        ),
    )

    model.fit(
        training_data.loc[
            :,
            SELECTED_MODEL.feature_columns,
        ],
        training_target,
    )

    logger.info(
        "Frozen selected model trained on %s games: "
        "%s | version=%s | features=%s | C=%s.",
        len(training_data),
        SELECTED_MODEL.model_name,
        SELECTED_MODEL.model_version,
        len(SELECTED_MODEL.feature_columns),
        SELECTED_MODEL.regularization_c,
    )

    return model


def evaluate_frozen_model_on_holdout(
    model: Pipeline,
    evaluation_data: pd.DataFrame,
) -> dict[str, ModelEvaluation]:
    """Evaluate the frozen model and Elo on holdout only."""

    holdout_data = evaluation_data.loc[
        evaluation_data["split_name"] == HOLDOUT_SPLIT
    ].copy()

    if holdout_data.empty:
        raise RuntimeError(
            "No holdout games are available."
        )

    model_probabilities = model.predict_proba(
        holdout_data.loc[
            :,
            SELECTED_MODEL.feature_columns,
        ]
    )[:, 1]

    elo_probabilities = holdout_data[
        "elo_home_win_probability"
    ].to_numpy(dtype=float)

    return {
        SELECTED_MODEL.model_name: (
            evaluate_probabilities(
                holdout_data[TARGET_COLUMN],
                model_probabilities,
            )
        ),
        "elo": evaluate_probabilities(
            holdout_data[TARGET_COLUMN],
            elo_probabilities,
        ),
    }


def log_holdout_results(
    evaluations: dict[str, ModelEvaluation],
) -> None:
    """Log the final frozen holdout results."""

    logger.info(
        "Final 2025 holdout results:"
    )

    for model_name, evaluation in evaluations.items():
        logger.info(
            "%s | Games=%s | Accuracy=%.2f%% | "
            "Brier=%.6f | Log loss=%.6f",
            model_name,
            evaluation.game_count,
            evaluation.accuracy * 100.0,
            evaluation.brier_score,
            evaluation.log_loss,
        )


def run_selected_model_holdout_evaluation(
    database_file: Path = DATABASE_FILE,
) -> dict[str, ModelEvaluation]:
    """Run the one-time final holdout evaluation."""

    validate_database_file(database_file)

    logger.info(
        "Starting frozen selected-model holdout "
        "evaluation..."
    )

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        validate_source_tables(connection)

        evaluation_data = load_final_evaluation_data(
            connection
        )

    model = train_frozen_selected_model(
        evaluation_data
    )

    evaluations = evaluate_frozen_model_on_holdout(
        model=model,
        evaluation_data=evaluation_data,
    )

    log_holdout_results(evaluations)

    logger.info(
        "Frozen selected-model holdout evaluation "
        "completed successfully."
    )

    return evaluations


def main() -> None:
    """Run the final holdout evaluation entry point."""

    try:
        run_selected_model_holdout_evaluation()
    except Exception:
        logger.exception(
            "Frozen selected-model holdout evaluation "
            "failed."
        )
        raise


if __name__ == "__main__":
    main()