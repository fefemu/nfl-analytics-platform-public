"""Tests for the production totals component."""

import numpy as np
import pandas as pd
import pytest

from src.modeling.production_totals_component import (
    FALLBACK_PREDICTION_MODE,
    OUTPUT_COLUMNS,
    PRIMARY_PREDICTION_MODE,
    prepare_production_totals_training_data,
    score_current_totals_predictions,
    train_production_totals_models,
)
from src.modeling.production_totals_model import (
    PRODUCTION_TOTALS_MODEL,
)


def all_feature_columns() -> tuple[str, ...]:
    """Return the unique primary/fallback union."""

    return tuple(
        dict.fromkeys(
            (
                *PRODUCTION_TOTALS_MODEL
                .feature_columns,
                *PRODUCTION_TOTALS_MODEL
                .fallback_feature_columns,
            )
        )
    )


def create_historical_data() -> pd.DataFrame:
    """Create primary, fallback and future games."""

    rows: list[dict[str, object]] = []

    for index in range(80):
        features = {
            feature_name: float(
                1 + index % 9
            )
            for feature_name
            in all_feature_columns()
        }

        rows.append(
            {
                "game_id": f"historical_{index}",
                "season": 2022 + index // 30,
                "both_short_windows_complete": True,
                "target_total_points": (
                    35.0
                    + 0.4 * index
                    + 0.2 * sum(features.values())
                ),
                **features,
            }
        )

    fallback_only = {
        **rows[-1],
        "game_id": "fallback_only_history",
        "season": 2025,
        "both_short_windows_complete": False,
        "offensive_epa_sum_last_4": np.nan,
        "defensive_epa_allowed_sum_last_4": (
            np.nan
        ),
        "listed_qb_rating_sum": np.nan,
    }

    future = {
        **rows[-1],
        "game_id": "future_game",
        "season": 2026,
    }

    rows.extend(
        [
            fallback_only,
            future,
        ]
    )

    return pd.DataFrame(rows)


def create_current_features() -> pd.DataFrame:
    """Create primary and fallback current games."""

    primary_features = {
        feature_name: float(index + 1)
        for index, feature_name
        in enumerate(
            all_feature_columns()
        )
    }

    fallback_features = {
        **primary_features,
        "offensive_epa_sum_last_4": np.nan,
        "defensive_epa_allowed_sum_last_4": (
            np.nan
        ),
        "listed_qb_rating_sum": np.nan,
    }

    return pd.DataFrame(
        [
            {
                "game_id": "primary_game",
                "home_team": "BUF",
                "away_team": "NYJ",
                "both_short_windows_complete": True,
                **primary_features,
            },
            {
                "game_id": "fallback_game",
                "home_team": "LV",
                "away_team": "DEN",
                "both_short_windows_complete": False,
                **fallback_features,
            },
        ]
    )


def test_training_samples_use_expected_coverage(
) -> None:
    """Use more games for the fallback model."""

    primary, fallback = (
        prepare_production_totals_training_data(
            create_historical_data()
        )
    )

    assert len(primary) == 80
    assert len(fallback) == 81

    assert primary[
        "both_short_windows_complete"
    ].all()


def test_train_and_route_current_games() -> None:
    """Route complete and incomplete primary inputs."""

    trained_models = (
        train_production_totals_models(
            create_historical_data()
        )
    )

    predictions = score_current_totals_predictions(
        current_features=create_current_features(),
        trained_models=trained_models,
    ).set_index(
        "game_id"
    )

    assert (
        predictions.loc[
            "primary_game",
            "prediction_mode",
        ]
        == PRIMARY_PREDICTION_MODE
    )

    assert (
        predictions.loc[
            "fallback_game",
            "prediction_mode",
        ]
        == FALLBACK_PREDICTION_MODE
    )

    assert (
        predictions.loc[
            "primary_game",
            "ridge_alpha",
        ]
        == 100.0
    )

    assert (
        predictions.loc[
            "fallback_game",
            "ridge_alpha",
        ]
        == 1.0
    )

    assert predictions.loc[
        "primary_game",
        "has_complete_primary_features",
    ]

    assert not predictions.loc[
        "fallback_game",
        "has_complete_primary_features",
    ]


def test_predictions_are_finite() -> None:
    """Create valid predictions for both routes."""

    trained_models = (
        train_production_totals_models(
            create_historical_data()
        )
    )

    predictions = score_current_totals_predictions(
        current_features=create_current_features(),
        trained_models=trained_models,
    )

    assert tuple(
        predictions.columns
    ) == OUTPUT_COLUMNS

    assert np.isfinite(
        predictions["predicted_total_points"]
    ).all()

    assert set(
        predictions["primary_training_game_count"]
    ) == {
        80,
    }

    assert set(
        predictions["fallback_training_game_count"]
    ) == {
        81,
    }


def test_missing_fallback_feature_is_rejected(
) -> None:
    """Require the production-safe fallback inputs."""

    current_features = create_current_features()

    current_features.loc[
        current_features["game_id"]
        == "fallback_game",
        "elo_rating_sum",
    ] = np.nan

    trained_models = (
        train_production_totals_models(
            create_historical_data()
        )
    )

    with pytest.raises(
        RuntimeError,
        match="missing fallback features",
    ):
        score_current_totals_predictions(
            current_features=current_features,
            trained_models=trained_models,
        )


def test_duplicate_historical_game_is_rejected(
) -> None:
    """Reject duplicate training identifiers."""

    historical_data = create_historical_data()

    duplicate = pd.concat(
        [
            historical_data,
            historical_data.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        prepare_production_totals_training_data(
            duplicate
        )