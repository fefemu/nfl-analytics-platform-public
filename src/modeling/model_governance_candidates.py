"""
NFL Analytics Platform
Model Governance Candidate Specifications

Purpose:
    Freeze champion and challenger configurations used
    by the 2020-2025 expanding-window scorecard.

Governance policy:
    Candidate definitions must not change in response
    to one evaluated season. A new specification requires
    a new model name or version.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

from dataclasses import dataclass

from src.modeling.run_logistic_ablation import (
    ELO_QB_FEATURES,
    ELO_QB_POST_BYE_FEATURES,
)
from src.modeling.run_logistic_injury_time_cv import (
    UNIT_BURDEN_FEATURES,
)
from src.modeling.train_logistic_baseline import (
    MODEL_FEATURE_COLUMNS,
)


@dataclass(frozen=True)
class GovernanceCandidate:
    """Describe one frozen logistic challenger."""

    model_name: str
    model_version: str
    feature_columns: tuple[str, ...]
    regularization_c: float


GOVERNANCE_CANDIDATES = (
    GovernanceCandidate(
        model_name="logistic_elo_plus_qb",
        model_version="0.1.0",
        feature_columns=ELO_QB_FEATURES,
        regularization_c=1.0,
    ),
    GovernanceCandidate(
        model_name="logistic_elo_qb_post_bye",
        model_version="0.1.0",
        feature_columns=(
            ELO_QB_POST_BYE_FEATURES
        ),
        regularization_c=1.0,
    ),
    GovernanceCandidate(
        model_name=(
            "logistic_elo_qb_unit_burdens"
        ),
        model_version="0.1.0",
        feature_columns=(
            *ELO_QB_FEATURES,
            *UNIT_BURDEN_FEATURES,
        ),
        regularization_c=0.1,
    ),
    GovernanceCandidate(
        model_name="logistic_full_core",
        model_version="0.1.0",
        feature_columns=MODEL_FEATURE_COLUMNS,
        regularization_c=0.01,
    ),
)