"""Tests for the production spread component."""

import numpy as np
import pandas as pd
import pytest

from src.modeling.production_spread_component import (
    FALLBACK_PREDICTION_MODE,
    OUTPUT_COLUMNS,
    PRIMARY_PREDICTION_MODE,
    prepare_production_spread_training_data,
    score_current_spread_predictions,
    train_production_spread_models,
)
from src.modeling.production_spread_model import (
    PRODUCTION_SPREAD_MODEL,
    ProductionSpreadModel,
)


def create_historical_data() -> pd.DataFrame:
    """Create synthetic historical spread games."""

    rows: list[dict[str, object]] = []

    for index in range(80):
        elo_difference = float(
            -160 + index * 4
        )

        qb_difference = float(
            -8 + index * 0.2
        )

        rows.append(
            {
                "game_id": f"historical_{index}",
                "season": 2022 + index // 30,
                "target_point_differential": (
                    2.0
                    + 0.04 * elo_difference
                    + 0.8 * qb_difference
                ),
                "external_nfelo_rating_difference": elo_difference,
                "external_nfelo_qb_adjustment_difference": qb_difference,
                "listed_qb_rating_difference": (
                    qb_difference
                ),
            }
        )

    rows.append(
        {
            "game_id": "missing_qb_history",
            "season": 2025,
            "target_point_differential": 3.0,
            "external_nfelo_rating_difference": 25.0,
            "external_nfelo_qb_adjustment_difference": 0.0,
            "listed_qb_rating_difference": np.nan,
        }
    )

    rows.append(
        {
            "game_id": "future_game",
            "season": 2026,
            "target_point_differential": 7.0,
            "external_nfelo_rating_difference": 40.0,
            "external_nfelo_qb_adjustment_difference": 2.0,
            "listed_qb_rating_difference": 2.0,
        }
    )

    return pd.DataFrame(rows)


def create_current_features() -> pd.DataFrame:
    """Create complete and missing-QB current games."""

    return pd.DataFrame(
        [
            {
                "game_id": "complete_game",
                "home_team": "BUF",
                "away_team": "NYJ",
                "external_nfelo_rating_difference": 80.0,
                "external_nfelo_qb_adjustment_difference": 4.0,
                "listed_qb_rating_difference": 4.0,
            },
            {
                "game_id": "fallback_game",
                "home_team": "LV",
                "away_team": "DEN",
                "external_nfelo_rating_difference": -60.0,
                "external_nfelo_qb_adjustment_difference": -2.0,
                "listed_qb_rating_difference": np.nan,
            },
        ]
    )


def test_production_specification_is_frozen() -> None:
    """Expose the selected production settings."""

    assert PRODUCTION_SPREAD_MODEL.feature_columns == (
        "external_nfelo_rating_difference",
        "external_nfelo_qb_adjustment_difference",
    )

    assert PRODUCTION_SPREAD_MODEL.ridge_alpha == 10.0
    assert (
        PRODUCTION_SPREAD_MODEL
        .fallback_ridge_alpha
        == 10.0
    )

    assert (
        PRODUCTION_SPREAD_MODEL.forward_test_season
        == 2026
    )


def test_training_samples_use_external_features() -> None:
    """Use every complete external Elo/QB training row."""

    primary, fallback = (
        prepare_production_spread_training_data(
            create_historical_data()
        )
    )

    assert len(primary) == 81
    assert len(fallback) == 81

    assert primary["external_nfelo_rating_difference"].notna().all()


def test_current_games_use_external_model() -> None:
    """Listed-QB availability does not change external routing."""

    models = train_production_spread_models(
        create_historical_data()
    )

    predictions = score_current_spread_predictions(
        current_features=create_current_features(),
        trained_models=models,
    ).set_index(
        "game_id"
    )

    assert (
        predictions.loc[
            "complete_game",
            "prediction_mode",
        ]
        == PRIMARY_PREDICTION_MODE
    )

    assert (
        predictions.loc[
            "fallback_game",
            "prediction_mode",
        ]
        == PRIMARY_PREDICTION_MODE
    )

    assert (
        predictions.loc[
            "complete_game",
            "model_name",
        ]
        == "external_nfelo_external_qb_spread"
    )

    assert (
        predictions.loc[
            "fallback_game",
            "model_name",
        ]
        == "external_nfelo_external_qb_spread"
    )


def test_predictions_are_finite_and_symmetric() -> None:
    """Create valid home- and away-perspective margins."""

    models = train_production_spread_models(
        create_historical_data()
    )

    predictions = score_current_spread_predictions(
        current_features=create_current_features(),
        trained_models=models,
    )

    assert tuple(predictions.columns) == OUTPUT_COLUMNS

    assert np.isfinite(
        predictions["predicted_home_margin"]
    ).all()

    assert np.allclose(
        predictions["predicted_home_margin"],
        -predictions["predicted_away_margin"],
    )

    assert (
        predictions["predicted_winner"].isin(
            [
                "BUF",
                "NYJ",
                "LV",
                "DEN",
            ]
        )
    ).all()


def test_missing_current_elo_is_rejected() -> None:
    """Elo is mandatory for every routing mode."""

    features = create_current_features()

    features.loc[
        features["game_id"] == "fallback_game",
        "external_nfelo_rating_difference",
    ] = np.nan

    models = train_production_spread_models(
        create_historical_data()
    )

    with pytest.raises(
        RuntimeError,
        match="missing Elo features",
    ):
        score_current_spread_predictions(
            current_features=features,
            trained_models=models,
        )


def test_invalid_production_spec_is_rejected() -> None:
    """Reject invalid frozen model settings."""

    with pytest.raises(
        ValueError,
        match="must not be negative",
    ):
        ProductionSpreadModel(
            model_name="spread",
            model_version="0.1.0",
            feature_columns=(
                "elo_rating_difference",
            ),
            ridge_alpha=-1.0,
            fallback_model_name="elo",
            fallback_feature_columns=(
                "elo_rating_difference",
            ),
            fallback_ridge_alpha=10.0,
            target_column=(
                "target_point_differential"
            ),
            forward_test_season=2026,
        )
