"""Tests for external production explanations."""

import pandas as pd
import pytest

from src.modeling.current_production_prediction_explanations import (
    PRODUCTION_EXPLANATION_COLUMNS,
    classify_production_matchup,
    create_production_prediction_explanation_frame,
)
from src.modeling.production_probability_predictions import (
    EXTERNAL_ELO_QB_FALLBACK_PREDICTION_MODE,
    EXTERNAL_NFELO_BLEND_PREDICTION_MODE,
)


def create_predictions() -> pd.DataFrame:
    """Create one primary and one fallback prediction."""

    return pd.DataFrame(
        [
            {
                "game_id": "primary_game",
                "season": 2026,
                "game_type": "REG",
                "week": 1,
                "gameday": "2026-09-10",
                "gametime": "20:20",
                "home_team": "PHI",
                "away_team": "DAL",
                "is_neutral": False,
                "model_name": (
                    "external_nfelo_probability_routing"
                ),
                "model_version": "0.3.0",
                "home_win_probability": 0.67,
                "away_win_probability": 0.33,
                "prediction_mode": (
                    EXTERNAL_NFELO_BLEND_PREDICTION_MODE
                ),
                "prediction_mode_reason": (
                    "complete_external_primary_features"
                ),
                "published_nfelo_home_probability": 0.60,
                "primary_logistic_home_win_probability": 0.70,
                "fallback_logistic_home_win_probability": None,
                "applied_primary_logistic_weight": 0.70,
                "applied_published_nfelo_weight": 0.30,
                "has_complete_injury_data": True,
                "both_listed_qb_ratings_available": True,
                "has_complete_production_features": True,
                "has_complete_fallback_features": True,
                "external_nfelo_rating_difference": 100.0,
                "listed_qb_rating_difference": 4.0,
                "external_nfelo_qb_adjustment_difference": 6.0,
                "offense_injury_burden_difference": -0.20,
                "defense_injury_burden_difference": -0.10,
                "special_teams_injury_burden_difference": -0.05,
                "prediction_generated_at": (
                    "2026-08-09 12:00:00"
                ),
            },
            {
                "game_id": "fallback_game",
                "season": 2026,
                "game_type": "REG",
                "week": 1,
                "gameday": "2026-09-13",
                "gametime": "13:00",
                "home_team": "NE",
                "away_team": "MIA",
                "is_neutral": False,
                "model_name": (
                    "external_nfelo_probability_routing"
                ),
                "model_version": "0.3.0",
                "home_win_probability": 0.43,
                "away_win_probability": 0.57,
                "prediction_mode": (
                    EXTERNAL_ELO_QB_FALLBACK_PREDICTION_MODE
                ),
                "prediction_mode_reason": (
                    "incomplete_external_primary_features"
                ),
                "published_nfelo_home_probability": 0.47,
                "primary_logistic_home_win_probability": None,
                "fallback_logistic_home_win_probability": 0.43,
                "applied_primary_logistic_weight": 0.0,
                "applied_published_nfelo_weight": 0.0,
                "has_complete_injury_data": False,
                "both_listed_qb_ratings_available": False,
                "has_complete_production_features": False,
                "has_complete_fallback_features": True,
                "external_nfelo_rating_difference": -45.0,
                "listed_qb_rating_difference": None,
                "external_nfelo_qb_adjustment_difference": -3.0,
                "offense_injury_burden_difference": None,
                "defense_injury_burden_difference": None,
                "special_teams_injury_burden_difference": None,
                "prediction_generated_at": (
                    "2026-08-09 12:00:00"
                ),
            },
        ]
    )


def test_create_primary_explanation(
) -> None:
    """Explain the selected external blend."""

    explanations = (
        create_production_prediction_explanation_frame(
            create_predictions()
        )
    )

    row = explanations.loc[
        explanations["game_id"]
        == "primary_game"
    ].iloc[0]

    assert (
        row["prediction_mode"]
        == EXTERNAL_NFELO_BLEND_PREDICTION_MODE
    )

    assert row["favorite"] == "PHI"
    assert row["underdog"] == "DAL"

    assert (
        row["favorite_win_probability"]
        == pytest.approx(0.67)
    )

    assert (
        row[
            "published_nfelo_away_probability"
        ]
        == pytest.approx(0.40)
    )

    assert (
        row[
            "primary_logistic_away_win_probability"
        ]
        == pytest.approx(0.30)
    )

    assert pd.isna(
        row[
            "fallback_logistic_home_win_probability"
        ]
    )

    assert (
        row[
            "production_probability_adjustment_from_published_nfelo"
        ]
        == pytest.approx(0.07)
    )


def test_create_fallback_explanation(
) -> None:
    """Explain the external logistic fallback."""

    explanations = (
        create_production_prediction_explanation_frame(
            create_predictions()
        )
    )

    row = explanations.loc[
        explanations["game_id"]
        == "fallback_game"
    ].iloc[0]

    assert (
        row["prediction_mode"]
        == (
            EXTERNAL_ELO_QB_FALLBACK_PREDICTION_MODE
        )
    )

    assert row["favorite"] == "MIA"
    assert row["underdog"] == "NE"

    assert (
        row["favorite_win_probability"]
        == pytest.approx(0.57)
    )

    assert pd.isna(
        row[
            "primary_logistic_home_win_probability"
        ]
    )

    assert (
        row[
            "fallback_logistic_home_win_probability"
        ]
        == pytest.approx(0.43)
    )

    assert (
        row[
            "fallback_logistic_away_win_probability"
        ]
        == pytest.approx(0.57)
    )


def test_explanations_have_stable_schema(
) -> None:
    """Return the documented explanation columns."""

    explanations = (
        create_production_prediction_explanation_frame(
            create_predictions()
        )
    )

    assert tuple(
        explanations.columns
    ) == PRODUCTION_EXPLANATION_COLUMNS

    assert len(explanations) == 2


@pytest.mark.parametrize(
    (
        "favorite_probability",
        "expected_label",
    ),
    [
        (0.50, "toss_up"),
        (0.524, "toss_up"),
        (0.55, "slight_edge"),
        (0.60, "clear_edge"),
        (0.70, "strong_edge"),
    ],
)
def test_classify_production_matchup(
    favorite_probability: float,
    expected_label: str,
) -> None:
    """Classify final probability strength."""

    assert (
        classify_production_matchup(
            favorite_probability
        )
        == expected_label
    )


def test_explanation_rejects_bad_probability_pair(
) -> None:
    """Reject final probabilities that do not sum to one."""

    predictions = create_predictions()

    predictions.loc[
        predictions["game_id"]
        == "primary_game",
        "away_win_probability",
    ] = 0.20

    with pytest.raises(
        ValueError,
        match="Production probabilities are invalid",
    ):
        create_production_prediction_explanation_frame(
            predictions
        )


def test_explanation_rejects_invalid_routing_components(
) -> None:
    """Reject a primary row with a fallback output."""

    predictions = create_predictions()

    predictions.loc[
        predictions["game_id"]
        == "primary_game",
        "fallback_logistic_home_win_probability",
    ] = 0.55

    with pytest.raises(
        ValueError,
        match="primary explanation",
    ):
        create_production_prediction_explanation_frame(
            predictions
        )


def test_explanation_rejects_unknown_mode(
) -> None:
    """Reject an undocumented routing mode."""

    predictions = create_predictions()

    predictions.loc[
        predictions["game_id"]
        == "primary_game",
        "prediction_mode",
    ] = "UNKNOWN"

    with pytest.raises(
        ValueError,
        match="Unknown production prediction mode",
    ):
        create_production_prediction_explanation_frame(
            predictions
        )


def test_explanation_supports_empty_predictions(
) -> None:
    """Return a stable empty explanation frame."""

    explanations = (
        create_production_prediction_explanation_frame(
            create_predictions().iloc[0:0]
        )
    )

    assert explanations.empty

    assert tuple(
        explanations.columns
    ) == PRODUCTION_EXPLANATION_COLUMNS