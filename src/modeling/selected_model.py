"""
NFL Analytics Platform
Selected Pregame Model Specification

Purpose:
    Define the single production-candidate model configuration
    used by prediction, calibration, simulation, and betting
    workflows.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SelectedModelSpecification:
    """Describe the selected pregame probability model."""

    model_name: str
    model_version: str
    feature_columns: tuple[str, ...]
    regularization_c: float
    classification_threshold: float


MODEL_DEPLOYMENT_STATUS = (
    "rejected_on_2025_holdout"
)


SELECTED_MODEL = SelectedModelSpecification(
    model_name="logistic_elo_qb_post_bye",
    model_version="0.1.0",
    feature_columns=(
        "elo_rating_difference",
        "listed_qb_rating_difference",
        "post_bye_difference",
    ),
    regularization_c=1.0,
    classification_threshold=0.5,
)