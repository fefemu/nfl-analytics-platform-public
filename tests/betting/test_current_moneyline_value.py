"""
Tests for current Moneyline value calculation.
"""

import pandas as pd
import pytest

from src.betting.current_moneyline_value import (
    MONEYLINE_VALUE_COLUMNS,
    create_current_moneyline_value,
)


def create_market_board() -> pd.DataFrame:
    """Create a valid synthetic market board."""

    common_values = {
        "snapshot_id": "snapshot_001",
        "fetched_at": pd.Timestamp(
            "2026-08-09 08:00:00",
            tz="Europe/Budapest",
        ),
        "season": 2026,
        "game_type": "REG",
        "week": 1,
        "gameday": pd.Timestamp("2026-09-10"),
        "commence_time": pd.Timestamp(
            "2026-09-11 02:20:00",
            tz="Europe/Budapest",
        ),
        "home_team": "PHI",
        "away_team": "DAL",
        "market_key": "h2h",
        "market_name": "Moneyline",
        "best_bookmaker_key": "test_book",
        "best_bookmaker_title": "Test Book",
        "bookmaker_count": 5,
    }

    return pd.DataFrame(
        [
            {
                **common_values,
                "game_id": "game_001",
                "outcome_name": "PHI",
                "outcome_type": "home",
                "best_american_price": 120,
                "best_decimal_odds": 2.20,
                "best_implied_probability": (
                    1.0 / 2.20
                ),
                "consensus_no_vig_probability": 0.45,
            },
            {
                **common_values,
                "game_id": "game_001",
                "outcome_name": "DAL",
                "outcome_type": "away",
                "best_american_price": -125,
                "best_decimal_odds": 1.80,
                "best_implied_probability": (
                    1.0 / 1.80
                ),
                "consensus_no_vig_probability": 0.55,
            },
            {
                **common_values,
                "game_id": "game_002",
                "home_team": "KC",
                "away_team": "BUF",
                "outcome_name": "KC",
                "outcome_type": "home",
                "best_american_price": -150,
                "best_decimal_odds": 1.67,
                "best_implied_probability": (
                    1.0 / 1.67
                ),
                "consensus_no_vig_probability": 0.60,
            },
            {
                **common_values,
                "game_id": "game_002",
                "home_team": "KC",
                "away_team": "BUF",
                "outcome_name": "BUF",
                "outcome_type": "away",
                "best_american_price": 150,
                "best_decimal_odds": 2.50,
                "best_implied_probability": 0.40,
                "consensus_no_vig_probability": 0.40,
            },
        ]
    )


def create_predictions() -> pd.DataFrame:
    """Create valid synthetic production predictions."""

    return pd.DataFrame(
        [
            {
                "game_id": "game_001",
                "model_name": "production_logistic",
                "model_version": "0.1.0",
                "prediction_mode": "PRIMARY",
                "home_win_probability": 0.50,
                "away_win_probability": 0.50,
                "prediction_generated_at": pd.Timestamp(
                    "2026-08-09 08:30:00",
                    tz="Europe/Budapest",
                ),
            },
            {
                "game_id": "game_002",
                "model_name": "production_logistic",
                "model_version": "0.1.0",
                "prediction_mode": "PRIMARY",
                "home_win_probability": 0.55,
                "away_win_probability": 0.45,
                "prediction_generated_at": pd.Timestamp(
                    "2026-08-09 08:30:00",
                    tz="Europe/Budapest",
                ),
            },
        ]
    )


def test_create_current_moneyline_value_schema() -> None:
    """The result contains the documented columns."""

    result = create_current_moneyline_value(
        market_board=create_market_board(),
        predictions=create_predictions(),
    )

    assert tuple(result.columns) == MONEYLINE_VALUE_COLUMNS
    assert len(result) == 4


def test_model_probability_uses_outcome_type() -> None:
    """Home and away offers use the correct model probability."""

    result = create_current_moneyline_value(
        market_board=create_market_board(),
        predictions=create_predictions(),
    )

    game_one = result.loc[
        result["game_id"] == "game_001"
    ].set_index("outcome_type")

    assert game_one.loc[
        "home",
        "model_probability",
    ] == pytest.approx(0.50)

    assert game_one.loc[
        "away",
        "model_probability",
    ] == pytest.approx(0.50)


def test_probability_edge_and_expected_value() -> None:
    """Probability edge, fair odds, EV and Kelly are correct."""

    result = create_current_moneyline_value(
        market_board=create_market_board(),
        predictions=create_predictions(),
    )

    home_offer = result.loc[
        (result["game_id"] == "game_001")
        & (result["outcome_type"] == "home")
    ].iloc[0]

    assert home_offer[
        "probability_edge"
    ] == pytest.approx(0.05)

    assert home_offer[
        "probability_edge_percentage_points"
    ] == pytest.approx(5.0)

    assert home_offer[
        "fair_decimal_odds"
    ] == pytest.approx(2.0)

    assert home_offer[
        "expected_value_per_unit"
    ] == pytest.approx(0.10)

    assert home_offer[
        "expected_value_percent"
    ] == pytest.approx(10.0)

    assert home_offer[
        "full_kelly_fraction"
    ] == pytest.approx(
        0.10 / 1.20
    )

    assert bool(
        home_offer["positive_expected_value"]
    )


def test_negative_expected_value_has_zero_kelly() -> None:
    """Negative EV offers receive no positive Kelly stake."""

    result = create_current_moneyline_value(
        market_board=create_market_board(),
        predictions=create_predictions(),
    )

    away_offer = result.loc[
        (result["game_id"] == "game_001")
        & (result["outcome_type"] == "away")
    ].iloc[0]

    assert away_offer[
        "expected_value_per_unit"
    ] == pytest.approx(-0.10)

    assert away_offer[
        "full_kelly_fraction"
    ] == pytest.approx(0.0)

    assert not bool(
        away_offer["positive_expected_value"]
    )


def test_results_are_sorted_by_expected_value() -> None:
    """The strongest EV offer appears first."""

    result = create_current_moneyline_value(
        market_board=create_market_board(),
        predictions=create_predictions(),
    )

    expected_values = result[
        "expected_value_per_unit"
    ].tolist()

    assert expected_values == sorted(
        expected_values,
        reverse=True,
    )

    assert result.iloc[0]["game_id"] == "game_002"
    assert result.iloc[0]["outcome_type"] == "away"


def test_duplicate_moneyline_outcome_is_rejected() -> None:
    """Duplicate game-outcome market rows are invalid."""

    market_board = create_market_board()

    duplicate_row = market_board.iloc[[0]].copy()

    market_board = pd.concat(
        [
            market_board,
            duplicate_row,
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate game-outcome",
    ):
        create_current_moneyline_value(
            market_board=market_board,
            predictions=create_predictions(),
        )


def test_incomplete_moneyline_market_is_rejected() -> None:
    """Every game must contain home and away offers."""

    market_board = create_market_board()

    market_board = market_board.loc[
        ~(
            (
                market_board["game_id"]
                == "game_001"
            )
            & (
                market_board["outcome_type"]
                == "away"
            )
        )
    ].copy()

    with pytest.raises(
        ValueError,
        match="exactly one home and one away",
    ):
        create_current_moneyline_value(
            market_board=market_board,
            predictions=create_predictions(),
        )


def test_invalid_consensus_probability_sum_is_rejected() -> None:
    """No-vig probabilities must sum to one by game."""

    market_board = create_market_board()

    market_board.loc[
        market_board["game_id"] == "game_001",
        "consensus_no_vig_probability",
    ] = [
        0.60,
        0.50,
    ]

    with pytest.raises(
        ValueError,
        match="must sum to one",
    ):
        create_current_moneyline_value(
            market_board=market_board,
            predictions=create_predictions(),
        )


def test_missing_prediction_is_rejected() -> None:
    """Every Moneyline game requires a production prediction."""

    predictions = create_predictions()

    predictions = predictions.loc[
        predictions["game_id"] != "game_002"
    ].copy()

    with pytest.raises(
        RuntimeError,
        match="missing production predictions",
    ):
        create_current_moneyline_value(
            market_board=create_market_board(),
            predictions=predictions,
        )


def test_invalid_model_probability_sum_is_rejected() -> None:
    """Home and away model probabilities must sum to one."""

    predictions = create_predictions()

    predictions.loc[
        predictions["game_id"] == "game_001",
        "away_win_probability",
    ] = 0.40

    with pytest.raises(
        ValueError,
        match="must sum to one",
    ):
        create_current_moneyline_value(
            market_board=create_market_board(),
            predictions=predictions,
        )