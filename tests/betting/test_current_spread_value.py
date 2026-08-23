"""
Tests for current Spread value calculation.
"""

import pandas as pd
import pytest

from src.betting.current_spread_value import (
    SPREAD_VALUE_COLUMNS,
    create_current_spread_value,
)


def create_market_board() -> pd.DataFrame:
    """Create one valid paired Spread market line."""

    common_values = {
        "snapshot_id": "snapshot_001",
        "fetched_at": pd.Timestamp(
            "2026-08-09 09:00:00",
            tz="Europe/Budapest",
        ),
        "game_id": "game_001",
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
        "market_key": "spreads",
        "market_name": "Spread",
        "market_line": 3.0,
    }

    return pd.DataFrame(
        [
            {
                **common_values,
                "outcome_name": "PHI",
                "outcome_type": "home",
                "point": -3.0,
                "best_bookmaker_key": "book_a",
                "best_bookmaker_title": "Book A",
                "best_american_price": 120,
                "best_decimal_odds": 2.20,
                "best_implied_probability": (
                    1.0 / 2.20
                ),
                "bookmaker_count": 5,
                "consensus_no_vig_probability": 0.48,
            },
            {
                **common_values,
                "outcome_name": "DAL",
                "outcome_type": "away",
                "point": 3.0,
                "best_bookmaker_key": "book_b",
                "best_bookmaker_title": "Book B",
                "best_american_price": -125,
                "best_decimal_odds": 1.80,
                "best_implied_probability": (
                    1.0 / 1.80
                ),
                "bookmaker_count": 4,
                "consensus_no_vig_probability": 0.52,
            },
        ]
    )


def create_predictions() -> pd.DataFrame:
    """Create one valid current Spread prediction."""

    return pd.DataFrame(
        [
            {
                "game_id": "game_001",
                "model_name": "ridge_elo_qb_spread",
                "model_version": "0.1.0",
                "prediction_mode": "RIDGE_ELO_QB",
                "predicted_home_margin": 3.2,
                "predicted_away_margin": -3.2,
                "prediction_generated_at": pd.Timestamp(
                    "2026-08-09 09:05:00"
                ),
            },
        ]
    )


def create_residuals() -> pd.DataFrame:
    """Create empirical residuals for both model modes."""

    rows = []

    for prediction_mode in [
        "RIDGE_ELO_QB",
        "RIDGE_ELO_FALLBACK",
    ]:
        for residual in [
            -3.0,
            -2.0,
            -1.0,
            0.0,
            1.0,
            2.0,
            3.0,
        ]:
            rows.append(
                {
                    "prediction_mode": (
                        prediction_mode
                    ),
                    "residual_home_margin": (
                        residual
                    ),
                }
            )

    return pd.DataFrame(rows)


def test_create_current_spread_value_schema(
) -> None:
    """The result contains the documented columns."""

    result = create_current_spread_value(
        market_board=create_market_board(),
        predictions=create_predictions(),
        residuals=create_residuals(),
    )

    assert tuple(result.columns) == (
        SPREAD_VALUE_COLUMNS
    )

    assert len(result) == 2


def test_home_spread_probabilities_and_value(
) -> None:
    """Calculate home cover, push, loss and EV."""

    result = create_current_spread_value(
        market_board=create_market_board(),
        predictions=create_predictions(),
        residuals=create_residuals(),
    )

    home_row = result.loc[
        result["outcome_type"] == "home"
    ].iloc[0]

    assert home_row[
        "predicted_outcome_margin"
    ] == pytest.approx(3.2)

    assert home_row[
        "calibration_sample_count"
    ] == 7

    assert home_row[
        "cover_probability"
    ] == pytest.approx(3.0 / 7.0)

    assert home_row[
        "push_probability"
    ] == pytest.approx(1.0 / 7.0)

    assert home_row[
        "loss_probability"
    ] == pytest.approx(3.0 / 7.0)

    assert home_row[
        "no_push_cover_probability"
    ] == pytest.approx(0.50)

    assert home_row[
        "probability_edge"
    ] == pytest.approx(0.02)

    assert home_row[
        "fair_decimal_odds"
    ] == pytest.approx(2.0)

    assert home_row[
        "expected_value_per_unit"
    ] == pytest.approx(
        0.6 / 7.0
    )

    assert home_row[
        "expected_value_percent"
    ] == pytest.approx(
        100.0 * 0.6 / 7.0
    )

    assert bool(
        home_row["positive_expected_value"]
    )


def test_away_margin_and_negative_value() -> None:
    """Away offers use the opposite predicted margin."""

    result = create_current_spread_value(
        market_board=create_market_board(),
        predictions=create_predictions(),
        residuals=create_residuals(),
    )

    away_row = result.loc[
        result["outcome_type"] == "away"
    ].iloc[0]

    assert away_row[
        "predicted_outcome_margin"
    ] == pytest.approx(-3.2)

    assert away_row[
        "cover_probability"
    ] == pytest.approx(3.0 / 7.0)

    assert away_row[
        "push_probability"
    ] == pytest.approx(1.0 / 7.0)

    assert away_row[
        "expected_value_per_unit"
    ] == pytest.approx(
        -0.6 / 7.0
    )

    assert away_row[
        "full_kelly_fraction"
    ] == pytest.approx(0.0)

    assert not bool(
        away_row["positive_expected_value"]
    )


def test_positive_value_is_sorted_first() -> None:
    """The strongest Spread EV appears first."""

    result = create_current_spread_value(
        market_board=create_market_board(),
        predictions=create_predictions(),
        residuals=create_residuals(),
    )

    assert result.iloc[0][
        "outcome_type"
    ] == "home"

    assert result.iloc[0][
        "expected_value_per_unit"
    ] > result.iloc[1][
        "expected_value_per_unit"
    ]


def test_incomplete_market_line_is_rejected() -> None:
    """Every Spread line requires home and away."""

    market_board = (
        create_market_board().iloc[[0]].copy()
    )

    with pytest.raises(
        ValueError,
        match="paired home and away",
    ):
        create_current_spread_value(
            market_board=market_board,
            predictions=create_predictions(),
            residuals=create_residuals(),
        )


def test_non_opposite_points_are_rejected() -> None:
    """Paired Spread points must sum to zero."""

    market_board = create_market_board()

    market_board.loc[
        market_board["outcome_type"] == "away",
        "point",
    ] = 2.5

    with pytest.raises(
        ValueError,
        match="opposite points",
    ):
        create_current_spread_value(
            market_board=market_board,
            predictions=create_predictions(),
            residuals=create_residuals(),
        )


def test_missing_prediction_is_rejected() -> None:
    """Every Spread game requires a prediction."""

    predictions = create_predictions()

    predictions["game_id"] = "different_game"

    with pytest.raises(
        RuntimeError,
        match="missing production predictions",
    ):
        create_current_spread_value(
            market_board=create_market_board(),
            predictions=predictions,
            residuals=create_residuals(),
        )


def test_missing_calibration_mode_is_rejected(
) -> None:
    """Every routed model mode requires residuals."""

    residuals = create_residuals()

    residuals = residuals.loc[
        residuals["prediction_mode"]
        == "RIDGE_ELO_FALLBACK"
    ].copy()

    with pytest.raises(
        RuntimeError,
        match="missing prediction modes",
    ):
        create_current_spread_value(
            market_board=create_market_board(),
            predictions=create_predictions(),
            residuals=residuals,
        )


def test_invalid_prediction_margin_sum_is_rejected(
) -> None:
    """Home and away predicted margins must be opposite."""

    predictions = create_predictions()

    predictions.loc[
        0,
        "predicted_away_margin",
    ] = -2.0

    with pytest.raises(
        ValueError,
        match="must sum to zero",
    ):
        create_current_spread_value(
            market_board=create_market_board(),
            predictions=predictions,
            residuals=create_residuals(),
        )