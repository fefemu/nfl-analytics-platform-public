"""
Tests for histogram gradient boosting training.
"""

import numpy as np
import pandas as pd
import pytest

from src.modeling.train_hist_gradient_boosting import (
    DEFAULT_CONFIG,
    BOOSTING_CONFIGURATIONS,
    HistGradientBoostingConfig,
    create_hist_gradient_boosting_pipeline,
    train_hist_gradient_boosting_model,
    validate_config,
)
from src.modeling.train_logistic_baseline import (
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
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


def test_create_pipeline_uses_controlled_classifier() -> None:
    """Create a deterministic nonlinear pipeline."""

    feature_columns = (
        "elo_rating_difference",
        "listed_qb_rating_difference",
    )

    pipeline = (
        create_hist_gradient_boosting_pipeline(
            feature_columns=feature_columns
        )
    )

    classifier = pipeline.named_steps["model"]

    assert classifier.early_stopping is False
    assert classifier.learning_rate == pytest.approx(
        DEFAULT_CONFIG.learning_rate
    )


def test_train_model_predicts_probabilities() -> None:
    """Train and predict valid home-win probabilities."""

    data = create_development_frame()

    feature_columns = (
        "elo_rating_difference",
        "listed_qb_rating_difference",
    )

    model = train_hist_gradient_boosting_model(
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


def test_train_model_supports_full_core_features() -> None:
    """Train using the complete core feature set."""

    data = create_development_frame()

    model = train_hist_gradient_boosting_model(
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
        HistGradientBoostingConfig(
            learning_rate=0.0
        ),
        HistGradientBoostingConfig(
            max_iterations=0
        ),
        HistGradientBoostingConfig(
            max_leaf_nodes=1
        ),
        HistGradientBoostingConfig(
            min_samples_leaf=0
        ),
        HistGradientBoostingConfig(
            l2_regularization=-1.0
        ),
    ],
)
def test_validate_config_rejects_invalid_values(
    config: HistGradientBoostingConfig,
) -> None:
    """Reject invalid boosting hyperparameters."""

    with pytest.raises(ValueError):
        validate_config(config)


def test_train_model_requires_both_target_classes() -> None:
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
        train_hist_gradient_boosting_model(
            development_data=data,
            feature_columns=(
                "elo_rating_difference",
            ),
        )


def test_boosting_configurations_are_valid() -> None:
    """Validate every configured boosting candidate."""

    assert set(BOOSTING_CONFIGURATIONS) == {
        "very_conservative",
        "conservative",
        "moderate",
        "original_baseline",
    }

    for config in BOOSTING_CONFIGURATIONS.values():
        validate_config(config)