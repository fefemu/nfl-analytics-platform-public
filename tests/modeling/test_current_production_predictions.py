"""Tests for current production predictions."""

import pandas as pd
import pytest

from src.modeling.current_production_predictions import (
    CURRENT_LOGISTIC_CONTRIBUTION_COLUMNS,
    PRODUCTION_AUDIT_COLUMNS,
    create_current_production_predictions,
)
from src.modeling.production_probability_model import (
    EXTERNAL_ELO_FEATURE,
    EXTERNAL_PRIMARY_FEATURES,
    EXTERNAL_QB_FEATURE,
)
from src.modeling.production_probability_predictions import (
    EXTERNAL_ELO_QB_FALLBACK_PREDICTION_MODE,
    EXTERNAL_NFELO_BLEND_PREDICTION_MODE,
)
from src.modeling.train_logistic_baseline import (
    TARGET_COLUMN,
)


def create_historical_data() -> pd.DataFrame:
    """Create primary and fallback training games."""

    rows: list[dict[str, object]] = []

    for index in range(20):
        target = index % 2

        direction = (
            1.0
            if target == 1
            else -1.0
        )

        rows.append(
            {
                "game_id": f"history_{index}",
                "season": 2020 + index // 4,
                TARGET_COLUMN: target,
                "has_complete_injury_data": True,
                EXTERNAL_ELO_FEATURE: (
                    direction * 90.0
                ),
                "listed_qb_rating_difference": (
                    direction * 4.0
                ),
                EXTERNAL_QB_FEATURE: (
                    direction * 6.0
                ),
                "offense_injury_burden_difference": (
                    direction * -0.20
                ),
                "defense_injury_burden_difference": (
                    direction * -0.15
                ),
                "special_teams_injury_burden_difference": (
                    direction * -0.05
                ),
                "published_nfelo_home_probability": (
                    0.68
                    if target == 1
                    else 0.36
                ),
            }
        )

    rows.extend(
        [
            {
                "game_id": "fallback_history_home",
                "season": 2024,
                TARGET_COLUMN: 1,
                "has_complete_injury_data": False,
                EXTERNAL_ELO_FEATURE: 55.0,
                "listed_qb_rating_difference": None,
                EXTERNAL_QB_FEATURE: 3.0,
                "offense_injury_burden_difference": None,
                "defense_injury_burden_difference": None,
                "special_teams_injury_burden_difference": None,
                "published_nfelo_home_probability": 0.60,
            },
            {
                "game_id": "fallback_history_away",
                "season": 2024,
                TARGET_COLUMN: 0,
                "has_complete_injury_data": False,
                EXTERNAL_ELO_FEATURE: -65.0,
                "listed_qb_rating_difference": None,
                EXTERNAL_QB_FEATURE: -4.0,
                "offense_injury_burden_difference": None,
                "defense_injury_burden_difference": None,
                "special_teams_injury_burden_difference": None,
                "published_nfelo_home_probability": 0.40,
            },
        ]
    )

    return pd.DataFrame(rows)


def create_upcoming_games() -> pd.DataFrame:
    """Create current external probability inputs."""

    return pd.DataFrame(
        [
            {
                "game_id": "primary_game",
                "home_listed_qb_rating": 7.0,
                "away_listed_qb_rating": 2.0,
                "has_complete_injury_data": True,
                EXTERNAL_ELO_FEATURE: 105.0,
                EXTERNAL_QB_FEATURE: 7.0,
                "published_nfelo_home_probability": 0.66,
                "external_game_available": True,
                "offense_injury_burden_difference": -0.20,
                "defense_injury_burden_difference": -0.10,
                "special_teams_injury_burden_difference": -0.05,
            },
            {
                "game_id": "fallback_game",
                "home_listed_qb_rating": 4.0,
                "away_listed_qb_rating": None,
                "has_complete_injury_data": False,
                EXTERNAL_ELO_FEATURE: -45.0,
                EXTERNAL_QB_FEATURE: -3.0,
                "published_nfelo_home_probability": None,
                "external_game_available": False,
                "offense_injury_burden_difference": None,
                "defense_injury_burden_difference": None,
                "special_teams_injury_burden_difference": None,
            },
        ]
    )


def create_elo_predictions() -> pd.DataFrame:
    """Create the transitional prediction spine."""

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
                "model_name": "elo",
                "model_version": "1.0.0",
                "home_rating_current": 1550.0,
                "away_rating_current": 1500.0,
                "home_rating_pregame": 1540.0,
                "away_rating_pregame": 1500.0,
                "applied_home_advantage": 48.0,
                "home_win_probability": 0.62,
                "away_win_probability": 0.38,
                "predicted_winner": "PHI",
                "home_rating_as_of": "2026-02-01",
                "away_rating_as_of": "2026-01-15",
                "prediction_generated_at": (
                    "2026-08-09 10:00:00"
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
                "model_name": "elo",
                "model_version": "1.0.0",
                "home_rating_current": 1490.0,
                "away_rating_current": 1510.0,
                "home_rating_pregame": 1495.0,
                "away_rating_pregame": 1505.0,
                "applied_home_advantage": 48.0,
                "home_win_probability": 0.55,
                "away_win_probability": 0.45,
                "predicted_winner": "NE",
                "home_rating_as_of": "2026-01-10",
                "away_rating_as_of": "2026-01-20",
                "prediction_generated_at": (
                    "2026-08-09 10:00:00"
                ),
            },
        ]
    )


def create_predictions(
    return_feature_contributions: bool = False,
):
    """Create the standard routed output."""

    return create_current_production_predictions(
        upcoming_games=create_upcoming_games(),
        elo_predictions=create_elo_predictions(),
        historical_data=create_historical_data(),
        return_feature_contributions=(
            return_feature_contributions
        ),
    )


def test_complete_game_uses_external_blend(
) -> None:
    """Apply the selected external primary blend."""

    predictions = create_predictions()

    row = predictions.loc[
        predictions["game_id"]
        == "primary_game"
    ].iloc[0]

    assert (
        row["model_name"]
        == "external_nfelo_probability_routing"
    )

    assert row["model_version"] == "0.3.0"

    assert (
        row["prediction_mode"]
        == EXTERNAL_NFELO_BLEND_PREDICTION_MODE
    )

    assert bool(
        row["has_complete_production_features"]
    )

    assert bool(
        row["has_complete_fallback_features"]
    )

    assert (
        row[
            "primary_logistic_home_win_probability"
        ]
        is not None
    )

    expected_probability = (
        0.70
        * row[
            "primary_logistic_home_win_probability"
        ]
        + 0.30
        * row[
            "published_nfelo_home_probability"
        ]
    )

    assert (
        row["home_win_probability"]
        == pytest.approx(expected_probability)
    )

    assert (
        row["away_win_probability"]
        == pytest.approx(
            1.0 - expected_probability
        )
    )


def test_incomplete_game_uses_external_fallback(
) -> None:
    """Route incomplete primary inputs to fallback."""

    predictions = create_predictions()

    row = predictions.loc[
        predictions["game_id"]
        == "fallback_game"
    ].iloc[0]

    assert (
        row["prediction_mode"]
        == (
            EXTERNAL_ELO_QB_FALLBACK_PREDICTION_MODE
        )
    )

    assert not bool(
        row["has_complete_production_features"]
    )

    assert bool(
        row["has_complete_fallback_features"]
    )

    assert pd.isna(
        row[
            "primary_logistic_home_win_probability"
        ]
    )

    assert (
        row["home_win_probability"]
        == pytest.approx(
            row[
                "fallback_logistic_home_win_probability"
            ]
        )
    )

    assert (
        row["applied_primary_logistic_weight"]
        == pytest.approx(0.0)
    )

    assert (
        row["applied_published_nfelo_weight"]
        == pytest.approx(0.0)
    )


def test_transitional_audit_aliases_are_consistent(
) -> None:
    """Keep old audit aliases internally consistent."""

    predictions = create_predictions()

    primary_row = predictions.loc[
        predictions["game_id"]
        == "primary_game"
    ].iloc[0]

    assert (
        primary_row["elo_home_win_probability"]
        == pytest.approx(
            primary_row[
                "published_nfelo_home_probability"
            ]
        )
    )

    assert (
        primary_row[
            "logistic_home_win_probability"
        ]
        == pytest.approx(
            primary_row[
                "primary_logistic_home_win_probability"
            ]
        )
    )

    assert (
        primary_row["applied_logistic_weight"]
        == pytest.approx(
            primary_row[
                "applied_primary_logistic_weight"
            ]
        )
    )

    assert (
        primary_row["applied_elo_weight"]
        == pytest.approx(
            primary_row[
                "applied_published_nfelo_weight"
            ]
        )
    )


def test_predictions_preserve_spine_and_audit_schema(
) -> None:
    """Preserve schedule metadata and append audit data."""

    predictions = create_predictions()

    assert list(
        predictions["game_id"]
    ) == [
        "primary_game",
        "fallback_game",
    ]

    assert set(
        PRODUCTION_AUDIT_COLUMNS
    ).issubset(
        predictions.columns
    )

    assert list(
        predictions["home_team"]
    ) == [
        "PHI",
        "NE",
    ]


def test_predicted_winner_matches_probability(
) -> None:
    """Derive winner from the final routed probability."""

    predictions = create_predictions()

    expected_winners = (
        predictions["home_team"].where(
            predictions[
                "home_win_probability"
            ] >= 0.5,
            predictions["away_team"],
        )
    )

    assert list(
        predictions["predicted_winner"]
    ) == list(expected_winners)


def test_external_features_are_auditable(
) -> None:
    """Expose the external production signals."""

    predictions = create_predictions()

    primary_row = predictions.loc[
        predictions["game_id"]
        == "primary_game"
    ].iloc[0]

    assert (
        primary_row[EXTERNAL_ELO_FEATURE]
        == pytest.approx(105.0)
    )

    assert (
        primary_row[EXTERNAL_QB_FEATURE]
        == pytest.approx(7.0)
    )

    assert (
        primary_row[
            "listed_qb_rating_difference"
        ]
        == pytest.approx(5.0)
    )


def test_feature_contributions_use_primary_features(
) -> None:
    """Return explainability for the external primary."""

    (
        predictions,
        contributions,
    ) = create_predictions(
        return_feature_contributions=True
    )

    assert not predictions.empty

    assert tuple(
        contributions.columns
    ) == CURRENT_LOGISTIC_CONTRIBUTION_COLUMNS

    feature_names = set(
        contributions["feature_name"]
    )

    assert feature_names == set(
        EXTERNAL_PRIMARY_FEATURES
    )

    assert set(
        contributions["model_name"]
    ) == {
        "external_nfelo_probability_routing",
    }


def test_empty_current_output(
) -> None:
    """Return empty prediction and contribution schemas."""

    empty_upcoming = (
        create_upcoming_games().iloc[0:0]
    )

    empty_spine = (
        create_elo_predictions().iloc[0:0]
    )

    (
        predictions,
        contributions,
    ) = create_current_production_predictions(
        upcoming_games=empty_upcoming,
        elo_predictions=empty_spine,
        historical_data=create_historical_data(),
        return_feature_contributions=True,
    )

    assert predictions.empty
    assert contributions.empty

    assert set(
        PRODUCTION_AUDIT_COLUMNS
    ).issubset(
        predictions.columns
    )

    assert tuple(
        contributions.columns
    ) == CURRENT_LOGISTIC_CONTRIBUTION_COLUMNS
