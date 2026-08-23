"""Tests for the production probability fallback."""

import numpy as np
import pandas as pd
import pytest

from src.modeling.production_probability_fallback_component import (
    FALLBACK_FEATURE_COVERAGE_COLUMN,
    FALLBACK_PROBABILITY_COLUMN,
    prepare_fallback_training_data,
    score_probability_fallback_component,
    train_probability_fallback_component,
)
from src.modeling.production_probability_model import (
    EXTERNAL_ELO_FEATURE,
    EXTERNAL_FALLBACK_FEATURES,
    EXTERNAL_QB_FEATURE,
)


def create_historical_data() -> pd.DataFrame:
    """Create binary external fallback training data."""

    return pd.DataFrame(
        [
            {
                "game_id": "game_1",
                "target_home_win": 1,
                EXTERNAL_ELO_FEATURE: 150.0,
                EXTERNAL_QB_FEATURE: 12.0,
            },
            {
                "game_id": "game_2",
                "target_home_win": 1,
                EXTERNAL_ELO_FEATURE: 90.0,
                EXTERNAL_QB_FEATURE: 5.0,
            },
            {
                "game_id": "game_3",
                "target_home_win": 1,
                EXTERNAL_ELO_FEATURE: 40.0,
                EXTERNAL_QB_FEATURE: 2.0,
            },
            {
                "game_id": "game_4",
                "target_home_win": 0,
                EXTERNAL_ELO_FEATURE: -30.0,
                EXTERNAL_QB_FEATURE: -1.0,
            },
            {
                "game_id": "game_5",
                "target_home_win": 0,
                EXTERNAL_ELO_FEATURE: -85.0,
                EXTERNAL_QB_FEATURE: -6.0,
            },
            {
                "game_id": "game_6",
                "target_home_win": 0,
                EXTERNAL_ELO_FEATURE: -140.0,
                EXTERNAL_QB_FEATURE: -10.0,
            },
            {
                "game_id": "tie_game",
                "target_home_win": np.nan,
                EXTERNAL_ELO_FEATURE: 0.0,
                EXTERNAL_QB_FEATURE: 0.0,
            },
        ]
    )


def create_current_features() -> pd.DataFrame:
    """Create complete current fallback features."""

    return pd.DataFrame(
        [
            {
                "game_id": "current_home",
                EXTERNAL_ELO_FEATURE: 110.0,
                EXTERNAL_QB_FEATURE: 7.0,
                FALLBACK_FEATURE_COVERAGE_COLUMN: True,
            },
            {
                "game_id": "current_away",
                EXTERNAL_ELO_FEATURE: -70.0,
                EXTERNAL_QB_FEATURE: -4.0,
                FALLBACK_FEATURE_COVERAGE_COLUMN: True,
            },
        ]
    )


def test_prepare_fallback_training_data(
) -> None:
    """Keep complete binary training rows."""

    training_data = (
        prepare_fallback_training_data(
            create_historical_data()
        )
    )

    assert len(training_data) == 6

    assert (
        training_data[
            "target_home_win"
        ].nunique()
        == 2
    )

    assert (
        "tie_game"
        not in set(
            training_data["game_id"]
        )
    )


def test_train_probability_fallback_component(
) -> None:
    """Fit the frozen external fallback model."""

    trained = (
        train_probability_fallback_component(
            create_historical_data()
        )
    )

    assert (
        trained.feature_columns
        == EXTERNAL_FALLBACK_FEATURES
    )

    assert trained.regularization_c == 0.1
    assert trained.training_game_count == 6


def test_score_probability_fallback_component(
) -> None:
    """Score every current game."""

    trained = (
        train_probability_fallback_component(
            create_historical_data()
        )
    )

    scored = (
        score_probability_fallback_component(
            current_features=(
                create_current_features()
            ),
            trained_fallback=trained,
        )
    )

    assert list(
        scored["game_id"]
    ) == [
        "current_home",
        "current_away",
    ]

    assert scored[
        FALLBACK_PROBABILITY_COLUMN
    ].between(
        0.0,
        1.0,
        inclusive="both",
    ).all()

    home_probability = scored.loc[
        scored["game_id"]
        == "current_home",
        FALLBACK_PROBABILITY_COLUMN,
    ].iloc[0]

    away_probability = scored.loc[
        scored["game_id"]
        == "current_away",
        FALLBACK_PROBABILITY_COLUMN,
    ].iloc[0]

    assert home_probability > away_probability


def test_score_preserves_original_columns(
) -> None:
    """Append fallback output without losing features."""

    current_features = create_current_features()

    trained = (
        train_probability_fallback_component(
            create_historical_data()
        )
    )

    scored = (
        score_probability_fallback_component(
            current_features=current_features,
            trained_fallback=trained,
        )
    )

    assert set(
        current_features.columns
    ).issubset(
        scored.columns
    )

    assert (
        FALLBACK_PROBABILITY_COLUMN
        in scored.columns
    )


def test_prepare_rejects_single_target_class(
) -> None:
    """Reject training without both outcomes."""

    historical_data = create_historical_data()

    historical_data = historical_data.loc[
        historical_data[
            "target_home_win"
        ].eq(1)
    ].copy()

    with pytest.raises(
        RuntimeError,
        match="both target classes",
    ):
        prepare_fallback_training_data(
            historical_data
        )


def test_score_rejects_missing_coverage(
) -> None:
    """Reject a current game without fallback inputs."""

    current_features = create_current_features()

    current_features.loc[
        current_features["game_id"]
        == "current_away",
        FALLBACK_FEATURE_COVERAGE_COLUMN,
    ] = False

    trained = (
        train_probability_fallback_component(
            create_historical_data()
        )
    )

    with pytest.raises(
        RuntimeError,
        match="missing probability fallback features",
    ):
        score_probability_fallback_component(
            current_features=current_features,
            trained_fallback=trained,
        )


def test_score_rejects_duplicate_games(
) -> None:
    """Reject duplicate current game identifiers."""

    current_features = create_current_features()

    duplicated = pd.concat(
        [
            current_features,
            current_features.iloc[[0]],
        ],
        ignore_index=True,
    )

    trained = (
        train_probability_fallback_component(
            create_historical_data()
        )
    )

    with pytest.raises(
        ValueError,
        match="duplicate game identifiers",
    ):
        score_probability_fallback_component(
            current_features=duplicated,
            trained_fallback=trained,
        )


def test_score_empty_frame(
) -> None:
    """Return an empty frame with output schema."""

    trained = (
        train_probability_fallback_component(
            create_historical_data()
        )
    )

    empty_features = (
        create_current_features().iloc[0:0]
    )

    scored = (
        score_probability_fallback_component(
            current_features=empty_features,
            trained_fallback=trained,
        )
    )

    assert scored.empty

    assert (
        FALLBACK_PROBABILITY_COLUMN
        in scored.columns
    )