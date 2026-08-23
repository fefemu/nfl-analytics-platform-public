"""
NFL Analytics Platform
XGBoost Model Trainer

Purpose:
    Train a controlled XGBoost NFL home-win
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
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.modeling.train_hist_gradient_boosting import (
    TREE_FEATURE_GROUPS,
)
from src.modeling.train_logistic_baseline import (
    TARGET_COLUMN,
    TRAIN_SPLIT,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class XGBoostConfig:
    """Store controlled XGBoost hyperparameters."""

    learning_rate: float = 0.03
    n_estimators: int = 100
    max_depth: int = 2
    min_child_weight: float = 10.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 1.0
    reg_lambda: float = 10.0


DEFAULT_XGBOOST_CONFIG = XGBoostConfig()
XGBOOST_FEATURE_GROUPS = TREE_FEATURE_GROUPS

VERY_CONSERVATIVE_XGBOOST_CONFIG = (
    XGBoostConfig(
        learning_rate=0.03,
        n_estimators=75,
        max_depth=1,
        min_child_weight=15.0,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_alpha=2.0,
        reg_lambda=20.0,
    )
)

LONG_SHALLOW_XGBOOST_CONFIG = (
    XGBoostConfig(
        learning_rate=0.02,
        n_estimators=150,
        max_depth=2,
        min_child_weight=10.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=10.0,
    )
)

MODERATE_XGBOOST_CONFIG = (
    XGBoostConfig(
        learning_rate=0.03,
        n_estimators=150,
        max_depth=2,
        min_child_weight=5.0,
        subsample=0.8,
        colsample_bytree=1.0,
        reg_alpha=0.5,
        reg_lambda=5.0,
    )
)

XGBOOST_CONFIGURATIONS = {
    "very_conservative": (
        VERY_CONSERVATIVE_XGBOOST_CONFIG
    ),
    "conservative_baseline": (
        DEFAULT_XGBOOST_CONFIG
    ),
    "long_shallow": (
        LONG_SHALLOW_XGBOOST_CONFIG
    ),
    "moderate": (
        MODERATE_XGBOOST_CONFIG
    ),
}


def validate_xgboost_config(
    config: XGBoostConfig,
) -> None:
    """Validate XGBoost hyperparameters."""

    if config.learning_rate <= 0.0:
        raise ValueError(
            "Learning rate must be greater than zero."
        )

    if config.n_estimators <= 0:
        raise ValueError(
            "Number of estimators must be positive."
        )

    if config.max_depth <= 0:
        raise ValueError(
            "Maximum depth must be positive."
        )

    if config.min_child_weight < 0.0:
        raise ValueError(
            "Minimum child weight must not be negative."
        )

    if not 0.0 < config.subsample <= 1.0:
        raise ValueError(
            "Subsample must be in the interval (0, 1]."
        )

    if not 0.0 < config.colsample_bytree <= 1.0:
        raise ValueError(
            "Column sample must be in the interval (0, 1]."
        )

    if config.reg_alpha < 0.0:
        raise ValueError(
            "L1 regularization must not be negative."
        )

    if config.reg_lambda < 0.0:
        raise ValueError(
            "L2 regularization must not be negative."
        )


def create_xgboost_pipeline(
    feature_columns: tuple[str, ...],
    config: XGBoostConfig = DEFAULT_XGBOOST_CONFIG,
) -> Pipeline:
    """Create preprocessing and XGBoost pipeline."""

    if not feature_columns:
        raise ValueError(
            "At least one model feature is required."
        )

    validate_xgboost_config(config)

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

    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        learning_rate=config.learning_rate,
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_child_weight=config.min_child_weight,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        random_state=42,
        n_jobs=1,
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


def train_xgboost_model(
    development_data: pd.DataFrame,
    feature_columns: tuple[str, ...],
    config: XGBoostConfig = DEFAULT_XGBOOST_CONFIG,
) -> Pipeline:
    """Train XGBoost using train-split games only."""

    train_data = development_data.loc[
        development_data["split_name"] == TRAIN_SPLIT
    ].copy()

    if train_data.empty:
        raise RuntimeError(
            "The XGBoost model has no training games."
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

    model = create_xgboost_pipeline(
        feature_columns=feature_columns,
        config=config,
    )

    model.fit(
        train_data.loc[:, feature_columns],
        training_target,
    )

    logger.info(
        "XGBoost trained on %s games with %s features.",
        len(train_data),
        len(feature_columns),
    )

    return model