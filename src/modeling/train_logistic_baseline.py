"""
NFL Analytics Platform
Logistic Regression Baseline Trainer

Purpose:
    Train and evaluate an interpretable leakage-safe
    NFL home-win probability baseline.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_FILE = PROJECT_ROOT / "data" / "nfl_analytics.duckdb"

DATASET_SCHEMA = "analytics"
DATASET_TABLE = "game_modeling_dataset"
DATASET_FULL_NAME = (
    f"{DATASET_SCHEMA}.{DATASET_TABLE}"
)

SPLIT_SCHEMA = "analytics"
SPLIT_TABLE = "modeling_game_splits"
SPLIT_FULL_NAME = (
    f"{SPLIT_SCHEMA}.{SPLIT_TABLE}"
)

TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
HOLDOUT_SPLIT = "holdout"

TARGET_COLUMN = "target_home_win"

ROLLING_DIFFERENCE_METRICS = (
    "offensive_epa_per_play",
    "competitive_epa_per_play",
    "dropback_epa_per_play",
    "designed_rush_epa_per_play",
    "early_down_epa_per_play",
    "success_rate",
    "explosive_play_rate",
    "sack_rate",
    "turnover_rate",
    "defensive_epa_allowed_per_play",
    "competitive_defensive_epa_allowed_per_play",
    "defensive_success_rate_allowed",
    "explosive_play_rate_allowed",
    "sack_rate_generated",
    "turnover_rate_generated",
)

MODEL_FEATURE_COLUMNS = (
    "elo_rating_difference",
    "listed_qb_rating_difference",
    *(
        f"{metric}_difference_last_4"
        for metric in ROLLING_DIFFERENCE_METRICS
    ),
)

SCHEDULE_CONTEXT_FEATURE_COLUMNS = (
    "rest_days_difference",
    "short_week_difference",
    "extended_rest_difference",
    "post_bye_difference",
)

DEVELOPMENT_FEATURE_COLUMNS = (
    *MODEL_FEATURE_COLUMNS,
    *SCHEDULE_CONTEXT_FEATURE_COLUMNS,
)

MODEL_RANDOM_STATE = 42
MODEL_MAX_ITERATIONS = 2000
PROBABILITY_EPSILON = 0.000001


@dataclass(frozen=True)
class ModelEvaluation:
    """Store probability-model evaluation metrics."""

    game_count: int
    accuracy: float
    brier_score: float
    log_loss: float


REQUIRED_DATASET_COLUMNS = {
    "game_id",
    "season",
    "game_date",
    TARGET_COLUMN,
    "elo_home_win_probability",
    *DEVELOPMENT_FEATURE_COLUMNS,
}

REQUIRED_SPLIT_COLUMNS = {
    "game_id",
    "split_name",
    "is_core_model_eligible",
}


def validate_database_file(
    database_file: Path = DATABASE_FILE,
) -> None:
    """Validate that the DuckDB database file exists."""

    if not database_file.exists():
        raise FileNotFoundError(
            f"Database file does not exist: {database_file}"
        )

    if not database_file.is_file():
        raise RuntimeError(
            f"Database path is not a file: {database_file}"
        )

    logger.info(
        "Database file validated: %s",
        database_file,
    )


def validate_source_tables(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate modeling dataset and split sources."""

    required_tables = {
        (
            DATASET_SCHEMA,
            DATASET_TABLE,
        ): REQUIRED_DATASET_COLUMNS,
        (
            SPLIT_SCHEMA,
            SPLIT_TABLE,
        ): REQUIRED_SPLIT_COLUMNS,
    }

    for (
        schema_name,
        table_name,
    ), required_columns in required_tables.items():
        table_exists = connection.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = ?
              AND table_name = ?
            """,
            [schema_name, table_name],
        ).fetchone()[0]

        full_name = f"{schema_name}.{table_name}"

        if table_exists == 0:
            raise RuntimeError(
                f"Source table does not exist: {full_name}"
            )

        available_columns = {
            row[0]
            for row in connection.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = ?
                  AND table_name = ?
                """,
                [schema_name, table_name],
            ).fetchall()
        }

        missing_columns = sorted(
            required_columns - available_columns
        )

        if missing_columns:
            missing_names = ", ".join(missing_columns)

            raise RuntimeError(
                f"Missing columns in {full_name}: "
                f"{missing_names}"
            )

    logger.info(
        "Logistic baseline sources validated: %s and %s.",
        DATASET_FULL_NAME,
        SPLIT_FULL_NAME,
    )


def load_development_data(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Load train and validation games without holdout access."""

    feature_select = ",\n            ".join(
        f"dataset.{column_name}"
        for column_name in DEVELOPMENT_FEATURE_COLUMNS
    )

    data = connection.execute(
        f"""
        SELECT
            dataset.game_id,
            dataset.season,
            dataset.game_date,
            splits.split_name,

            CAST(
                dataset.{TARGET_COLUMN}
                AS INTEGER
            ) AS {TARGET_COLUMN},

            dataset.elo_home_win_probability,

            {feature_select}

        FROM {DATASET_FULL_NAME} AS dataset

        INNER JOIN {SPLIT_FULL_NAME} AS splits
            ON dataset.game_id = splits.game_id

        WHERE splits.split_name IN (
                '{TRAIN_SPLIT}',
                '{VALIDATION_SPLIT}'
              )

          AND splits.is_core_model_eligible = TRUE

        ORDER BY
            dataset.game_date,
            dataset.game_id
        """
    ).fetchdf()

    if data.empty:
        raise RuntimeError(
            "No development games were loaded."
        )

    loaded_splits = set(
        data["split_name"].unique()
    )

    expected_splits = {
        TRAIN_SPLIT,
        VALIDATION_SPLIT,
    }

    if loaded_splits != expected_splits:
        raise RuntimeError(
            "Development data does not contain both "
            "train and validation splits."
        )

    if HOLDOUT_SPLIT in loaded_splits:
        raise RuntimeError(
            "Holdout data leaked into development data."
        )

    if data[TARGET_COLUMN].isna().any():
        raise RuntimeError(
            "Development data contains missing targets."
        )

    logger.info(
        "Development data loaded: %s train games and "
        "%s validation games.",
        int(
            (
                data["split_name"] == TRAIN_SPLIT
            ).sum()
        ),
        int(
            (
                data["split_name"] == VALIDATION_SPLIT
            ).sum()
        ),
    )

    return data


def create_logistic_pipeline(
    feature_columns: tuple[str, ...] = MODEL_FEATURE_COLUMNS,
    regularization_c: float = 1.0,
) -> Pipeline:
    """Create a configurable logistic regression pipeline."""

    if not feature_columns:
        raise ValueError(
            "At least one model feature is required."
        )

    if regularization_c <= 0.0:
        raise ValueError(
            "Regularization C must be greater than zero."
        )

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                list(feature_columns),
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                LogisticRegression(
                    C=regularization_c,
                    max_iter=MODEL_MAX_ITERATIONS,
                    random_state=MODEL_RANDOM_STATE,
                ),
            ),
        ]
    )


def train_logistic_model(
    development_data: pd.DataFrame,
    feature_columns: tuple[str, ...] = MODEL_FEATURE_COLUMNS,
    regularization_c: float = 1.0,
) -> Pipeline:
    """Train a configurable logistic model on train games only."""

    train_data = development_data.loc[
        development_data["split_name"] == TRAIN_SPLIT
    ].copy()

    if train_data.empty:
        raise RuntimeError(
            "The logistic model has no training games."
        )

    missing_features = sorted(
        set(feature_columns) - set(train_data.columns)
    )

    if missing_features:
        raise RuntimeError(
            "Training data is missing model features: "
            + ", ".join(missing_features)
        )

    training_target = train_data[TARGET_COLUMN]

    if training_target.nunique() != 2:
        raise RuntimeError(
            "Training data must contain both target classes."
        )

    model = create_logistic_pipeline(
        feature_columns=feature_columns,
        regularization_c=regularization_c,
    )

    model.fit(
        train_data.loc[:, feature_columns],
        training_target,
    )

    logger.info(
        "Logistic model trained on %s games with "
        "%s features and C=%s.",
        len(train_data),
        len(feature_columns),
        regularization_c,
    )

    return model


def evaluate_probabilities(
    actual_values: pd.Series,
    probabilities: np.ndarray | pd.Series,
) -> ModelEvaluation:
    """Evaluate binary outcome probabilities."""

    actual = np.asarray(
        actual_values,
        dtype=int,
    )

    predicted_probability = np.asarray(
        probabilities,
        dtype=float,
    )

    if len(actual) == 0:
        raise ValueError(
            "Probability evaluation requires games."
        )

    if len(actual) != len(predicted_probability):
        raise ValueError(
            "Actual values and probabilities "
            "must have equal length."
        )

    predicted_probability = np.clip(
        predicted_probability,
        PROBABILITY_EPSILON,
        1.0 - PROBABILITY_EPSILON,
    )

    predicted_class = (
        predicted_probability >= 0.5
    ).astype(int)

    return ModelEvaluation(
        game_count=len(actual),
        accuracy=float(
            accuracy_score(
                actual,
                predicted_class,
            )
        ),
        brier_score=float(
            brier_score_loss(
                actual,
                predicted_probability,
            )
        ),
        log_loss=float(
            log_loss(
                actual,
                predicted_probability,
                labels=[0, 1],
            )
        ),
    )


def evaluate_validation_models(
    model: Pipeline,
    development_data: pd.DataFrame,
    feature_columns: tuple[str, ...] = MODEL_FEATURE_COLUMNS,
) -> dict[str, ModelEvaluation]:
    """Evaluate baselines on validation games only."""

    train_data = development_data.loc[
        development_data["split_name"] == TRAIN_SPLIT
    ].copy()

    validation_data = development_data.loc[
        development_data["split_name"]
        == VALIDATION_SPLIT
    ].copy()

    if validation_data.empty:
        raise RuntimeError(
            "No validation games are available."
        )

    validation_target = validation_data[TARGET_COLUMN]

    logistic_probabilities = model.predict_proba(
        validation_data.loc[:, feature_columns]
    )[:, 1]

    elo_probabilities = validation_data[
        "elo_home_win_probability"
    ].to_numpy(dtype=float)

    train_home_win_rate = float(
        train_data[TARGET_COLUMN].mean()
    )

    constant_probabilities = np.full(
        shape=len(validation_data),
        fill_value=train_home_win_rate,
        dtype=float,
    )

    return {
        "constant_home_rate": evaluate_probabilities(
            validation_target,
            constant_probabilities,
        ),
        "elo": evaluate_probabilities(
            validation_target,
            elo_probabilities,
        ),
        "logistic": evaluate_probabilities(
            validation_target,
            logistic_probabilities,
        ),
    }


def log_model_evaluation(
    model_name: str,
    evaluation: ModelEvaluation,
) -> None:
    """Log one probability model evaluation."""

    logger.info(
        "%s | Games=%s | Accuracy=%.2f%% | "
        "Brier=%.6f | Log loss=%.6f",
        model_name,
        evaluation.game_count,
        100.0 * evaluation.accuracy,
        evaluation.brier_score,
        evaluation.log_loss,
    )


def log_validation_comparison(
    evaluations: dict[str, ModelEvaluation],
) -> None:
    """Log validation performance and Elo comparison."""

    logger.info(
        "Validation model comparison:"
    )

    for model_name in (
        "constant_home_rate",
        "elo",
        "logistic",
    ):
        log_model_evaluation(
            model_name=model_name,
            evaluation=evaluations[model_name],
        )

    elo_evaluation = evaluations["elo"]
    logistic_evaluation = evaluations["logistic"]

    logger.info(
        "Logistic improvement versus Elo | "
        "Brier=%+.6f | Log loss=%+.6f | Accuracy=%+.2f pp",
        (
            elo_evaluation.brier_score
            - logistic_evaluation.brier_score
        ),
        (
            elo_evaluation.log_loss
            - logistic_evaluation.log_loss
        ),
        100.0
        * (
            logistic_evaluation.accuracy
            - elo_evaluation.accuracy
        ),
    )


def log_feature_coefficients(
    model: Pipeline,
    feature_columns: tuple[str, ...] = MODEL_FEATURE_COLUMNS,
) -> None:
    """Log standardized logistic feature coefficients."""

    logistic_model = model.named_steps["model"]
    coefficients = logistic_model.coef_[0]

    coefficient_rows = sorted(
        zip(
            feature_columns,
            coefficients,
            strict=True,
        ),
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    logger.info(
        "Standardized logistic feature coefficients:"
    )

    for feature_name, coefficient in coefficient_rows:
        logger.info(
            "%+.6f | %s",
            coefficient,
            feature_name,
        )


def train_and_evaluate_logistic_baseline(
    database_file: Path = DATABASE_FILE,
) -> dict[str, ModelEvaluation]:
    """Train the baseline and evaluate validation performance."""

    validate_database_file(database_file)

    logger.info(
        "Starting logistic regression baseline training..."
    )

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        validate_source_tables(connection)

        development_data = load_development_data(
            connection
        )

    model = train_logistic_model(
        development_data
    )

    evaluations = evaluate_validation_models(
        model=model,
        development_data=development_data,
    )

    log_validation_comparison(evaluations)
    log_feature_coefficients(model)

    logger.info(
        "Logistic regression baseline training "
        "completed successfully."
    )

    return evaluations


def main() -> None:
    """Run the logistic regression baseline trainer."""

    try:
        train_and_evaluate_logistic_baseline()

    except Exception:
        logger.exception(
            "Logistic regression baseline trainer failed."
        )
        raise


if __name__ == "__main__":
    main()