"""Tests for current production feature creation."""

import numpy as np
import pandas as pd
import pytest

from src.modeling.current_production_features import (
    create_current_production_feature_frame,
)
from src.modeling.production_probability_model import (
    EXTERNAL_ELO_FEATURE,
    EXTERNAL_QB_FEATURE,
)


def create_upcoming_games() -> pd.DataFrame:
    """Create current external, QB and injury inputs."""

    return pd.DataFrame(
        [
            {
                "game_id": "complete_game",
                "home_listed_qb_rating": 6.0,
                "away_listed_qb_rating": 2.0,
                "has_complete_injury_data": True,
                EXTERNAL_ELO_FEATURE: 120.0,
                EXTERNAL_QB_FEATURE: 8.0,
                "published_nfelo_home_probability": 0.72,
                "external_game_available": True,
                "offense_injury_burden_difference": -0.20,
                "defense_injury_burden_difference": 0.10,
                "special_teams_injury_burden_difference": -0.05,
            },
            {
                "game_id": "missing_qb_game",
                "home_listed_qb_rating": np.nan,
                "away_listed_qb_rating": 1.0,
                "has_complete_injury_data": True,
                EXTERNAL_ELO_FEATURE: -25.0,
                EXTERNAL_QB_FEATURE: -4.0,
                "published_nfelo_home_probability": 0.48,
                "external_game_available": True,
                "offense_injury_burden_difference": 0.00,
                "defense_injury_burden_difference": 0.00,
                "special_teams_injury_burden_difference": 0.00,
            },
            {
                "game_id": "missing_injury_game",
                "home_listed_qb_rating": 3.0,
                "away_listed_qb_rating": 2.0,
                "has_complete_injury_data": False,
                EXTERNAL_ELO_FEATURE: 45.0,
                EXTERNAL_QB_FEATURE: 2.0,
                "published_nfelo_home_probability": 0.59,
                "external_game_available": True,
                "offense_injury_burden_difference": np.nan,
                "defense_injury_burden_difference": np.nan,
                "special_teams_injury_burden_difference": np.nan,
            },
        ]
    )


def create_elo_predictions() -> pd.DataFrame:
    """Create transitional internal Elo predictions."""

    return pd.DataFrame(
        [
            {
                "game_id": "complete_game",
                "home_rating_pregame": 1550.0,
                "away_rating_pregame": 1500.0,
                "home_win_probability": 0.64,
            },
            {
                "game_id": "missing_qb_game",
                "home_rating_pregame": 1490.0,
                "away_rating_pregame": 1510.0,
                "home_win_probability": 0.52,
            },
            {
                "game_id": "missing_injury_game",
                "home_rating_pregame": 1525.0,
                "away_rating_pregame": 1500.0,
                "home_win_probability": 0.60,
            },
        ]
    )


def create_features() -> pd.DataFrame:
    """Create the standard current feature frame."""

    return create_current_production_feature_frame(
        upcoming_games=create_upcoming_games(),
        elo_predictions=create_elo_predictions(),
    )


def test_create_current_features_derives_differences(
) -> None:
    """Derive listed-QB and transitional Elo features."""

    features = create_features()

    complete_row = features.loc[
        features["game_id"]
        == "complete_game"
    ].iloc[0]

    assert (
        complete_row["elo_rating_difference"]
        == pytest.approx(50.0)
    )

    assert (
        complete_row[
            "listed_qb_rating_difference"
        ]
        == pytest.approx(4.0)
    )

    assert (
        complete_row[
            "internal_elo_home_win_probability"
        ]
        == pytest.approx(0.64)
    )

    assert (
        complete_row[EXTERNAL_ELO_FEATURE]
        == pytest.approx(120.0)
    )

    assert (
        complete_row[EXTERNAL_QB_FEATURE]
        == pytest.approx(8.0)
    )

    assert (
        complete_row[
            "published_nfelo_home_probability"
        ]
        == pytest.approx(0.72)
    )


def test_complete_game_is_primary_eligible(
) -> None:
    """Activate the external injury-enhanced primary."""

    features = create_features()

    complete_row = features.loc[
        features["game_id"]
        == "complete_game"
    ].iloc[0]

    assert bool(
        complete_row[
            "both_listed_qb_ratings_available"
        ]
    )

    assert bool(
        complete_row[
            "has_complete_production_features"
        ]
    )

    assert bool(
        complete_row[
            "has_complete_fallback_features"
        ]
    )


def test_missing_qb_routes_to_fallback(
) -> None:
    """Preserve fallback eligibility without listed QB."""

    features = create_features()

    row = features.loc[
        features["game_id"]
        == "missing_qb_game"
    ].iloc[0]

    assert not bool(
        row[
            "both_listed_qb_ratings_available"
        ]
    )

    assert pd.isna(
        row[
            "listed_qb_rating_difference"
        ]
    )

    assert not bool(
        row[
            "has_complete_production_features"
        ]
    )

    assert bool(
        row[
            "has_complete_fallback_features"
        ]
    )


def test_missing_injury_routes_to_fallback(
) -> None:
    """Preserve fallback eligibility without injury data."""

    features = create_features()

    row = features.loc[
        features["game_id"]
        == "missing_injury_game"
    ].iloc[0]

    assert not bool(
        row[
            "has_complete_injury_data"
        ]
    )

    assert not bool(
        row[
            "has_complete_production_features"
        ]
    )

    assert bool(
        row[
            "has_complete_fallback_features"
        ]
    )


def test_missing_injury_flag_becomes_false(
) -> None:
    """Normalize a null injury coverage flag."""

    upcoming_games = create_upcoming_games()

    upcoming_games[
        "has_complete_injury_data"
    ] = upcoming_games[
        "has_complete_injury_data"
    ].astype("boolean")

    upcoming_games.loc[
        upcoming_games["game_id"]
        == "missing_injury_game",
        "has_complete_injury_data",
    ] = pd.NA

    features = (
        create_current_production_feature_frame(
            upcoming_games=upcoming_games,
            elo_predictions=create_elo_predictions(),
        )
    )

    row = features.loc[
        features["game_id"]
        == "missing_injury_game"
    ].iloc[0]

    assert not bool(
        row["has_complete_injury_data"]
    )


def test_feature_creation_preserves_game_order(
) -> None:
    """Preserve schedule ordering."""

    features = create_features()

    assert list(
        features["game_id"]
    ) == [
        "complete_game",
        "missing_qb_game",
        "missing_injury_game",
    ]


def test_feature_creation_rejects_missing_columns(
) -> None:
    """Reject an incomplete current input schema."""

    upcoming_games = (
        create_upcoming_games().drop(
            columns=[
                EXTERNAL_ELO_FEATURE,
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match="is missing columns",
    ):
        create_current_production_feature_frame(
            upcoming_games=upcoming_games,
            elo_predictions=create_elo_predictions(),
        )


def test_feature_creation_rejects_duplicate_games(
) -> None:
    """Reject duplicate upcoming game identifiers."""

    upcoming_games = create_upcoming_games()

    duplicated_games = pd.concat(
        [
            upcoming_games,
            upcoming_games.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate game identifiers",
    ):
        create_current_production_feature_frame(
            upcoming_games=duplicated_games,
            elo_predictions=create_elo_predictions(),
        )


def test_feature_creation_rejects_missing_internal_game(
) -> None:
    """Reject a game without transitional Elo output."""

    elo_predictions = (
        create_elo_predictions().loc[
            lambda data: (
                data["game_id"]
                != "complete_game"
            )
        ].copy()
    )

    with pytest.raises(
        RuntimeError,
        match="missing current internal Elo predictions",
    ):
        create_current_production_feature_frame(
            upcoming_games=create_upcoming_games(),
            elo_predictions=elo_predictions,
        )


def test_feature_creation_rejects_missing_fallback(
) -> None:
    """Reject a game without external fallback features."""

    upcoming_games = create_upcoming_games()

    upcoming_games.loc[
        upcoming_games["game_id"]
        == "missing_qb_game",
        EXTERNAL_QB_FEATURE,
    ] = np.nan

    with pytest.raises(
        RuntimeError,
        match="missing external probability fallback",
    ):
        create_current_production_feature_frame(
            upcoming_games=upcoming_games,
            elo_predictions=create_elo_predictions(),
        )


@pytest.mark.parametrize(
    "invalid_probability",
    [
        np.nan,
        0.0,
        1.0,
        np.inf,
    ],
)
def test_feature_creation_rejects_invalid_probability(
    invalid_probability: float,
) -> None:
    """Reject invalid published nfelo probabilities."""

    upcoming_games = create_upcoming_games()

    upcoming_games.loc[
        upcoming_games["game_id"]
        == "complete_game",
        "published_nfelo_home_probability",
    ] = invalid_probability

    with pytest.raises(
        ValueError,
        match="published nfelo probabilities",
    ):
        create_current_production_feature_frame(
            upcoming_games=upcoming_games,
            elo_predictions=create_elo_predictions(),
        )
