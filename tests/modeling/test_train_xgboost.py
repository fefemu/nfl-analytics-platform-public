"""
Tests for the XGBoost model trainer.
"""

import numpy as np
import pandas as pd
import pytest

from src.modeling.train_logistic_baseline import (
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from src.modeling.train_xgboost import (
    DEFAULT_XGBOOST_CONFIG,
    XGBOOST_CONFIGURATIONS,
    XGBoostConfig,
    create_xgboost_pipeline,
    train_xgboost_model,
    validate_xgboost_config,
)


def create_development_frame() -> pd.DataFrame:
    """Create deterministic train and validation data."""

    rows = []

    for index in range(40):
        target = index % 2

        row = {
            "game_id": f"game_{index}",
            "season": 2020,
            "game_date": pd.Timestamp(
                "2020-01-01"
            )
            + pd.Timedelta(days=index),
            "split_name": (
                "train"
                if index < 30
                else "validation"
            ),
            TARGET_COLUMN: target,
            "elo_home_win_probability": (
                0.65 if target else 0.35
            ),
        }

        direction = (
            1.0 if target else -1.0
        )

        for feature_index, feature_name in enumerate(
            MODEL_FEATURE_COLUMNS
        ):
            row[feature_name] = (
                direction
                * (
                    1.0
                    + 0.01 * feature_index
                )
            )

        rows.append(row)

    return pd.DataFrame(rows)


def test_create_xgboost_pipeline_uses_controlled_model() -> None:
    """Create a deterministic XGBoost pipeline."""

    pipeline = create_xgboost_pipeline(
        feature_columns=(
            "elo_rating_difference",
            "listed_qb_rating_difference",
        )
    )

    classifier = pipeline.named_steps["model"]

    assert classifier.max_depth == 2
    assert classifier.n_jobs == 1
    assert classifier.tree_method == "hist"

    assert classifier.learning_rate == pytest.approx(
        DEFAULT_XGBOOST_CONFIG.learning_rate
    )


def test_train_xgboost_predicts_probabilities() -> None:
    """Train and predict valid home-win probabilities."""

    data = create_development_frame()

    feature_columns = (
        "elo_rating_difference",
        "listed_qb_rating_difference",
    )

    model = train_xgboost_model(
        development_data=data,
        feature_columns=feature_columns,
    )

    validation_data = data.loc[
        data["split_name"] == "validation"
    ]

    probabilities = model.predict_proba(
        validation_data.loc[:, feature_columns]
    )[:, 1]

    assert probabilities.shape == (10,)
    assert np.all(probabilities > 0.0)
    assert np.all(probabilities < 1.0)


def test_train_xgboost_supports_full_core_features() -> None:
    """Train using the complete core feature set."""

    data = create_development_frame()

    model = train_xgboost_model(
        development_data=data,
        feature_columns=MODEL_FEATURE_COLUMNS,
    )

    assert model.named_steps[
        "model"
    ].n_features_in_ == len(
        MODEL_FEATURE_COLUMNS
    )


@pytest.mark.parametrize(
    "config",
    [
        XGBoostConfig(
            learning_rate=0.0
        ),
        XGBoostConfig(
            n_estimators=0
        ),
        XGBoostConfig(
            max_depth=0
        ),
        XGBoostConfig(
            min_child_weight=-1.0
        ),
        XGBoostConfig(
            subsample=0.0
        ),
        XGBoostConfig(
            subsample=1.1
        ),
        XGBoostConfig(
            colsample_bytree=0.0
        ),
        XGBoostConfig(
            colsample_bytree=1.1
        ),
        XGBoostConfig(
            reg_alpha=-1.0
        ),
        XGBoostConfig(
            reg_lambda=-1.0
        ),
    ],
)
def test_validate_xgboost_config_rejects_invalid_values(
    config: XGBoostConfig,
) -> None:
    """Reject invalid XGBoost hyperparameters."""

    with pytest.raises(ValueError):
        validate_xgboost_config(config)


def test_train_xgboost_requires_both_target_classes() -> None:
    """Reject single-class training data."""

    data = create_development_frame()

    data.loc[
        data["split_name"] == "train",
        TARGET_COLUMN,
    ] = 1

    with pytest.raises(
        RuntimeError,
        match="both target classes",
    ):
        train_xgboost_model(
            development_data=data,
            feature_columns=(
                "elo_rating_difference",
            ),
        )


def test_xgboost_configurations_are_valid() -> None:
    """Validate every configured XGBoost candidate."""

    assert set(XGBOOST_CONFIGURATIONS) == {
        "very_conservative",
        "conservative_baseline",
        "long_shallow",
        "moderate",
    }

    for config in XGBOOST_CONFIGURATIONS.values():
        validate_xgboost_config(config)