"""
NFL Analytics Platform
Histogram Gradient Boosting Trainer

Purpose:
    Train a controlled nonlinear NFL home-win
    probability model without holdout access.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.modeling.run_logistic_ablation import (
    ELO_QB_FEATURES,
)
from src.modeling.train_logistic_baseline import (
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    TRAIN_SPLIT,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


TREE_FEATURE_GROUPS = {
    "elo_plus_qb": ELO_QB_FEATURES,
    "full_core": MODEL_FEATURE_COLUMNS,
}


@dataclass(frozen=True)
class HistGradientBoostingConfig:
    """Store gradient-boosting hyperparameters."""

    learning_rate: float = 0.05
    max_iterations: int = 200
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 20
    l2_regularization: float = 1.0


DEFAULT_CONFIG = HistGradientBoostingConfig()

VERY_CONSERVATIVE_CONFIG = (
    HistGradientBoostingConfig(
        learning_rate=0.03,
        max_iterations=100,
        max_leaf_nodes=3,
        min_samples_leaf=50,
        l2_regularization=10.0,
    )
)

CONSERVATIVE_CONFIG = (
    HistGradientBoostingConfig(
        learning_rate=0.03,
        max_iterations=150,
        max_leaf_nodes=7,
        min_samples_leaf=40,
        l2_regularization=5.0,
    )
)

MODERATE_CONFIG = (
    HistGradientBoostingConfig(
        learning_rate=0.05,
        max_iterations=100,
        max_leaf_nodes=7,
        min_samples_leaf=30,
        l2_regularization=5.0,
    )
)

BOOSTING_CONFIGURATIONS = {
    "very_conservative": VERY_CONSERVATIVE_CONFIG,
    "conservative": CONSERVATIVE_CONFIG,
    "moderate": MODERATE_CONFIG,
    "original_baseline": DEFAULT_CONFIG,
}

def validate_config(
    config: HistGradientBoostingConfig,
) -> None:
    """Validate gradient-boosting hyperparameters."""

    if config.learning_rate <= 0.0:
        raise ValueError(
            "Learning rate must be greater than zero."
        )

    if config.max_iterations <= 0:
        raise ValueError(
            "Maximum iterations must be greater than zero."
        )

    if config.max_leaf_nodes < 2:
        raise ValueError(
            "Maximum leaf nodes must be at least two."
        )

    if config.min_samples_leaf <= 0:
        raise ValueError(
            "Minimum samples per leaf must be positive."
        )

    if config.l2_regularization < 0.0:
        raise ValueError(
            "L2 regularization must not be negative."
        )


def create_hist_gradient_boosting_pipeline(
    feature_columns: tuple[str, ...],
    config: HistGradientBoostingConfig = DEFAULT_CONFIG,
) -> Pipeline:
    """Create a preprocessing and boosting pipeline."""

    if not feature_columns:
        raise ValueError(
            "At least one model feature is required."
        )

    validate_config(config)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(
                    strategy="median",
                ),
                list(feature_columns),
            ),
        ],
        remainder="drop",
    )

    classifier = HistGradientBoostingClassifier(
        learning_rate=config.learning_rate,
        max_iter=config.max_iterations,
        max_leaf_nodes=config.max_leaf_nodes,
        min_samples_leaf=config.min_samples_leaf,
        l2_regularization=(
            config.l2_regularization
        ),
        early_stopping=False,
        random_state=42,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                classifier,
            ),
        ]
    )


def train_hist_gradient_boosting_model(
    development_data: pd.DataFrame,
    feature_columns: tuple[str, ...],
    config: HistGradientBoostingConfig = DEFAULT_CONFIG,
) -> Pipeline:
    """Train boosting using train-split games only."""

    train_data = development_data.loc[
        development_data["split_name"] == TRAIN_SPLIT
    ].copy()

    if train_data.empty:
        raise RuntimeError(
            "The boosting model has no training games."
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

    model = create_hist_gradient_boosting_pipeline(
        feature_columns=feature_columns,
        config=config,
    )

    model.fit(
        train_data.loc[:, feature_columns],
        training_target,
    )

    logger.info(
        "Histogram gradient boosting trained on %s games "
        "with %s features.",
        len(train_data),
        len(feature_columns),
    )

    return model