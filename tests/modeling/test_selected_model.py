"""Tests for the selected pregame model specification."""

from src.modeling.selected_model import (
    MODEL_DEPLOYMENT_STATUS,
    SELECTED_MODEL,
    SelectedModelSpecification,
)


def test_selected_model_has_expected_identity() -> None:
    """Store a stable name and version for the selected model."""

    assert isinstance(
        SELECTED_MODEL,
        SelectedModelSpecification,
    )
    assert (
        SELECTED_MODEL.model_name
        == "logistic_elo_qb_post_bye"
    )
    assert SELECTED_MODEL.model_version == "0.1.0"


def test_selected_model_has_expected_features() -> None:
    """Use the validated Elo, QB, and post-bye features."""

    assert SELECTED_MODEL.feature_columns == (
        "elo_rating_difference",
        "listed_qb_rating_difference",
        "post_bye_difference",
    )


def test_selected_model_has_expected_parameters() -> None:
    """Use the externally validated logistic parameters."""

    assert SELECTED_MODEL.regularization_c == 1.0
    assert SELECTED_MODEL.classification_threshold == 0.5


def test_selected_model_is_rejected_for_deployment() -> None:
    """Record the final 2025 holdout decision."""

    assert (
        MODEL_DEPLOYMENT_STATUS
        == "rejected_on_2025_holdout"
    )