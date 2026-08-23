"""
NFL Analytics Platform
Production Model Specification

Purpose:
    Define the deployed pregame probability model used by
    prediction, simulation, and betting workflows.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

from dataclasses import dataclass

from src.models.elo import (
    DEFAULT_K_FACTOR,
    DEFAULT_SEASON_RETENTION,
)


@dataclass(frozen=True)
class ProductionModelSpecification:
    """Describe the deployed Elo probability model."""

    model_name: str
    model_version: str
    model_family: str
    k_factor: float
    home_advantage: float
    season_retention: float
    classification_threshold: float
    deployment_status: str


PRODUCTION_MODEL = ProductionModelSpecification(
    model_name="elo",
    model_version="1.0.0",
    model_family="elo",
    k_factor=DEFAULT_K_FACTOR,
    home_advantage=50.0,
    season_retention=DEFAULT_SEASON_RETENTION,
    classification_threshold=0.5,
    deployment_status="production",
)