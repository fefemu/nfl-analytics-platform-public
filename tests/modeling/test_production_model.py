"""Tests for the production model specification."""

from src.modeling.production_model import (
    PRODUCTION_MODEL,
    ProductionModelSpecification,
)


def test_production_model_has_expected_identity() -> None:
    """Identify Elo as the deployed probability model."""

    assert isinstance(
        PRODUCTION_MODEL,
        ProductionModelSpecification,
    )
    assert PRODUCTION_MODEL.model_name == "elo"
    assert PRODUCTION_MODEL.model_version == "1.0.0"
    assert PRODUCTION_MODEL.model_family == "elo"
    assert (
        PRODUCTION_MODEL.deployment_status
        == "production"
    )


def test_production_model_has_expected_parameters() -> None:
    """Keep deployed Elo parameters reproducible."""

    assert PRODUCTION_MODEL.k_factor == 45.0
    assert PRODUCTION_MODEL.home_advantage == 50.0
    assert PRODUCTION_MODEL.season_retention == 0.60
    assert (
        PRODUCTION_MODEL.classification_threshold
        == 0.5
    )


def test_rejected_candidate_is_not_production_model() -> None:
    """Keep the rejected logistic candidate separate."""

    assert (
        PRODUCTION_MODEL.model_name
        != "logistic_elo_qb_post_bye"
    )