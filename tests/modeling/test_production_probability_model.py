"""Tests for the production probability model."""

import pytest

from src.modeling.production_probability_model import (
    EXTERNAL_ELO_FEATURE,
    EXTERNAL_FALLBACK_FEATURES,
    EXTERNAL_PRIMARY_FEATURES,
    EXTERNAL_QB_FEATURE,
    LISTED_QB_FEATURE,
    PRODUCTION_MODEL_DEPLOYMENT_STATUS,
    PRODUCTION_PROBABILITY_MODEL,
    ProductionProbabilityModel,
)


def test_production_model_has_expected_identity(
) -> None:
    """Freeze the selected routing identity."""

    assert isinstance(
        PRODUCTION_PROBABILITY_MODEL,
        ProductionProbabilityModel,
    )

    assert (
        PRODUCTION_PROBABILITY_MODEL.model_name
        == "external_nfelo_probability_routing"
    )

    assert (
        PRODUCTION_PROBABILITY_MODEL.model_version
        == "0.3.0"
    )

    assert (
        PRODUCTION_MODEL_DEPLOYMENT_STATUS
        == "selected_for_2026_forward_test"
    )


def test_production_model_has_expected_primary(
) -> None:
    """Freeze the external primary model."""

    assert (
        PRODUCTION_PROBABILITY_MODEL
        .logistic_component_name
        == "external_elo_qb_injury_logistic"
    )

    assert (
        PRODUCTION_PROBABILITY_MODEL
        .logistic_feature_columns
        == EXTERNAL_PRIMARY_FEATURES
    )

    assert EXTERNAL_PRIMARY_FEATURES == (
        EXTERNAL_ELO_FEATURE,
        LISTED_QB_FEATURE,
        EXTERNAL_QB_FEATURE,
        "offense_injury_burden_difference",
        "defense_injury_burden_difference",
        "special_teams_injury_burden_difference",
    )

    assert (
        PRODUCTION_PROBABILITY_MODEL
        .logistic_regularization_c
        == 0.1
    )


def test_production_model_has_expected_blend_policy(
) -> None:
    """Freeze the primary blend policy."""

    assert (
        PRODUCTION_PROBABILITY_MODEL.logistic_weight
        == pytest.approx(0.70)
    )

    assert (
        PRODUCTION_PROBABILITY_MODEL.elo_weight
        == pytest.approx(0.30)
    )

    assert (
        PRODUCTION_PROBABILITY_MODEL
        .blend_probability_source
        == "published_nfelo_home_probability"
    )

    assert (
        PRODUCTION_PROBABILITY_MODEL
        .requires_complete_injury_data
        is True
    )


def test_production_model_has_expected_fallback(
) -> None:
    """Freeze the external logistic fallback."""

    assert (
        PRODUCTION_PROBABILITY_MODEL
        .incomplete_injury_fallback_model
        == "external_elo_qb_logistic"
    )

    assert (
        PRODUCTION_PROBABILITY_MODEL
        .fallback_feature_columns
        == EXTERNAL_FALLBACK_FEATURES
    )

    assert EXTERNAL_FALLBACK_FEATURES == (
        EXTERNAL_ELO_FEATURE,
        EXTERNAL_QB_FEATURE,
    )

    assert (
        PRODUCTION_PROBABILITY_MODEL
        .fallback_regularization_c
        == 0.1
    )

    assert (
        PRODUCTION_PROBABILITY_MODEL
        .forward_test_season
        == 2026
    )


def test_production_model_rejects_invalid_weights(
) -> None:
    """Reject blend weights that do not sum to one."""

    with pytest.raises(
        ValueError,
        match="must sum to one",
    ):
        ProductionProbabilityModel(
            model_name="invalid",
            model_version="0.0.0",
            logistic_component_name="logistic",
            logistic_feature_columns=(
                "feature",
            ),
            logistic_regularization_c=1.0,
            logistic_weight=0.80,
            elo_weight=0.30,
            blend_probability_source=(
                "published_probability"
            ),
            classification_threshold=0.5,
            requires_complete_injury_data=True,
            incomplete_injury_fallback_model=(
                "fallback"
            ),
            fallback_feature_columns=(
                "fallback_feature",
            ),
            fallback_regularization_c=1.0,
            forward_test_season=2026,
        )


def test_production_model_rejects_empty_fallback(
) -> None:
    """Reject a missing fallback feature set."""

    with pytest.raises(
        ValueError,
        match="fallback features must not be empty",
    ):
        ProductionProbabilityModel(
            model_name="invalid",
            model_version="0.0.0",
            logistic_component_name="logistic",
            logistic_feature_columns=(
                "feature",
            ),
            logistic_regularization_c=1.0,
            logistic_weight=0.70,
            elo_weight=0.30,
            blend_probability_source=(
                "published_probability"
            ),
            classification_threshold=0.5,
            requires_complete_injury_data=True,
            incomplete_injury_fallback_model=(
                "fallback"
            ),
            fallback_feature_columns=(),
            fallback_regularization_c=1.0,
            forward_test_season=2026,
        )