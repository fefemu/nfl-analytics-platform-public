"""Tests for bilingual game prediction narratives."""

from datetime import datetime

import pandas as pd
import pytest

from src.modeling.current_game_prediction_narratives import (
    NARRATIVE_COLUMNS,
    create_current_game_prediction_narratives,
    format_percentage,
)


def create_explanations() -> pd.DataFrame:
    """Create one blend and one fallback explanation."""

    generated_at = datetime(
        2026,
        8,
        6,
        14,
        0,
        0,
    )

    return pd.DataFrame(
        [
            {
                "game_id": "blend_game",
                "home_team": "PHI",
                "away_team": "DAL",
                "favorite": "PHI",
                "underdog": "DAL",
                "favorite_win_probability": 0.67,
                "home_win_probability": 0.67,
                "away_win_probability": 0.33,
                "published_nfelo_home_probability": 0.60,
                "primary_logistic_home_win_probability": 0.70,
                "fallback_logistic_home_win_probability": None,
                "applied_primary_logistic_weight": 0.70,
                "applied_published_nfelo_weight": 0.30,
                "prediction_mode": "EXTERNAL_NFELO_BLEND",
                "has_complete_injury_data": True,
                "both_listed_qb_ratings_available": True,
                "matchup_label": "strong_edge",
                "model_name": (
                    "external_nfelo_probability_routing"
                ),
                "model_version": "0.3.0",
                "prediction_generated_at": (
                    generated_at
                ),
            },
            {
                "game_id": "fallback_game",
                "home_team": "NE",
                "away_team": "MIA",
                "favorite": "NE",
                "underdog": "MIA",
                "favorite_win_probability": 0.55,
                "home_win_probability": 0.55,
                "away_win_probability": 0.45,
                "published_nfelo_home_probability": 0.55,
                "primary_logistic_home_win_probability": None,
                "fallback_logistic_home_win_probability": 0.55,
                "applied_primary_logistic_weight": 0.0,
                "applied_published_nfelo_weight": 1.0,
                "prediction_mode": "EXTERNAL_ELO_QB_FALLBACK",
                "has_complete_injury_data": False,
                "both_listed_qb_ratings_available": True,
                "matchup_label": "slight_edge",
                "model_name": (
                    "external_nfelo_probability_routing"
                ),
                "model_version": "0.3.0",
                "prediction_generated_at": (
                    generated_at
                ),
            },
        ]
    )


def create_contributions() -> pd.DataFrame:
    """Create ranked logistic contributions."""

    return pd.DataFrame(
        [
            {
                "game_id": "blend_game",
                "feature_name": (
                    "external_nfelo_rating_difference"
                ),
                "log_odds_contribution": 0.40,
                "contribution_rank": 1,
            },
            {
                "game_id": "blend_game",
                "feature_name": (
                    "listed_qb_rating_difference"
                ),
                "log_odds_contribution": -0.10,
                "contribution_rank": 2,
            },
        ]
    )


def test_create_blend_narrative(
) -> None:
    """Explain a blended favorite in both languages."""

    narratives = (
        create_current_game_prediction_narratives(
            explanations=create_explanations(),
            feature_contributions=(
                create_contributions()
            ),
        )
    )

    blend = narratives.loc[
        narratives["game_id"]
        == "blend_game"
    ].iloc[0]

    assert "PHI win probability: 67.0%" in (
        blend["headline_en"]
    )

    assert "67,0%" in blend["headline_hu"]

    assert "70% logistic" in blend[
        "model_context_en"
    ]

    assert "30% published nfelo" in blend[
        "model_context_hu"
    ]

    assert (
        blend["top_factor_feature"]
        == "external_nfelo_rating_difference"
    )

    assert (
        blend["top_factor_direction"]
        == "supports_favorite"
    )
    assert (
        blend["model_name"]
        == "external_nfelo_probability_routing"
    )

    assert (
        blend["model_version"]
        == "0.3.0"
    )

    assert pd.notna(
        blend["prediction_generated_at"]
    )


def test_create_fallback_narrative(
) -> None:
    """Explain why Elo fallback is active."""

    narratives = (
        create_current_game_prediction_narratives(
            explanations=create_explanations(),
            feature_contributions=(
                create_contributions()
            ),
        )
    )

    fallback = narratives.loc[
        narratives["game_id"]
        == "fallback_game"
    ].iloc[0]

    assert "External Elo-QB fallback" in fallback[
        "model_context_en"
    ]

    assert "injury data" in fallback[
        "model_context_en"
    ]

    assert "External Elo-QB fallback" in fallback[
        "model_context_hu"
    ]

    assert pd.isna(
        fallback["top_factor_feature"]
    )

    assert pd.isna(
        fallback["top_factor_en"]
    )


def test_away_favorite_interprets_direction(
) -> None:
    """Interpret home-oriented contribution for away favorite."""

    explanations = create_explanations().iloc[
        [0]
    ].copy()

    explanations[
        "home_team"
    ] = "DAL"
    explanations[
        "away_team"
    ] = "PHI"
    explanations[
        "favorite"
    ] = "PHI"
    explanations[
        "underdog"
    ] = "DAL"
    explanations[
        "home_win_probability"
    ] = 0.33
    explanations[
        "away_win_probability"
    ] = 0.67
    explanations[
        "published_nfelo_home_probability"
    ] = 0.40
    explanations[
        "primary_logistic_home_win_probability"
    ] = 0.30

    narratives = (
        create_current_game_prediction_narratives(
            explanations=explanations,
            feature_contributions=(
                create_contributions()
            ),
        )
    )

    assert narratives.iloc[0][
        "top_factor_direction"
    ] == "opposes_favorite"


def test_fallback_rejects_contributions(
) -> None:
    """Reject logistic factors attached to fallback."""

    explanations = create_explanations().loc[
        lambda data: (
            data["game_id"]
            == "fallback_game"
        )
    ]

    contributions = (
        create_contributions().copy()
    )
    contributions[
        "game_id"
    ] = "fallback_game"

    with pytest.raises(
        RuntimeError,
        match="must not have",
    ):
        create_current_game_prediction_narratives(
            explanations=explanations,
            feature_contributions=contributions,
        )


def test_blend_requires_contributions(
) -> None:
    """Require factors for every blended prediction."""

    explanations = create_explanations().loc[
        lambda data: (
            data["game_id"]
            == "blend_game"
        )
    ]

    empty_contributions = (
        create_contributions().iloc[0:0]
    )

    with pytest.raises(
        RuntimeError,
        match="missing logistic",
    ):
        create_current_game_prediction_narratives(
            explanations=explanations,
            feature_contributions=(
                empty_contributions
            ),
        )


def test_narratives_have_stable_schema(
) -> None:
    """Return stable columns for empty inputs."""

    explanations = (
        create_explanations().iloc[0:0]
    )

    contributions = (
        create_contributions().iloc[0:0]
    )

    narratives = (
        create_current_game_prediction_narratives(
            explanations=explanations,
            feature_contributions=contributions,
        )
    )

    assert narratives.empty

    assert tuple(
        narratives.columns
    ) == NARRATIVE_COLUMNS


def test_percentage_formatting(
) -> None:
    """Use language-specific decimal separators."""

    assert (
        format_percentage(
            0.675,
            language="en",
        )
        == "67.5%"
    )

    assert (
        format_percentage(
            0.675,
            language="hu",
        )
        == "67,5%"
    )

    with pytest.raises(
        ValueError,
        match="Unsupported",
    ):
        format_percentage(
            0.675,
            language="de",
        )
