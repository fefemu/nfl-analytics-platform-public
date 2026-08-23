"""
Tests for empirical Spread cover probabilities.
"""

import numpy as np
import pytest

from src.betting.spread_cover_probability import (
    calculate_spread_expected_value,
    estimate_spread_cover_probabilities,
)


def create_residuals() -> np.ndarray:
    """Create a symmetric empirical residual sample."""

    return np.asarray(
        [
            -3.0,
            -2.0,
            -1.0,
            0.0,
            1.0,
            2.0,
            3.0,
        ],
        dtype=float,
    )


def test_integer_home_spread_supports_push() -> None:
    """An integer home line produces win/push/loss."""

    result = estimate_spread_cover_probabilities(
        predicted_home_margin=3.2,
        outcome_type="home",
        spread_point=-3.0,
        residual_home_margins=create_residuals(),
    )

    assert result[
        "cover_probability"
    ] == pytest.approx(3.0 / 7.0)

    assert result[
        "push_probability"
    ] == pytest.approx(1.0 / 7.0)

    assert result[
        "loss_probability"
    ] == pytest.approx(3.0 / 7.0)

    assert result[
        "simulation_count"
    ] == 7


def test_away_spread_reverses_home_margin() -> None:
    """Away cover probability uses the opposite margin."""

    result = estimate_spread_cover_probabilities(
        predicted_home_margin=3.2,
        outcome_type="away",
        spread_point=3.0,
        residual_home_margins=create_residuals(),
    )

    assert result[
        "cover_probability"
    ] == pytest.approx(3.0 / 7.0)

    assert result[
        "push_probability"
    ] == pytest.approx(1.0 / 7.0)

    assert result[
        "loss_probability"
    ] == pytest.approx(3.0 / 7.0)


def test_half_point_spread_has_no_push() -> None:
    """A half-point line cannot produce a push."""

    result = estimate_spread_cover_probabilities(
        predicted_home_margin=3.2,
        outcome_type="home",
        spread_point=-2.5,
        residual_home_margins=create_residuals(),
    )

    assert result[
        "cover_probability"
    ] == pytest.approx(4.0 / 7.0)

    assert result[
        "push_probability"
    ] == pytest.approx(0.0)

    assert result[
        "loss_probability"
    ] == pytest.approx(3.0 / 7.0)


def test_probabilities_sum_to_one() -> None:
    """Every residual simulation has one outcome."""

    result = estimate_spread_cover_probabilities(
        predicted_home_margin=-1.7,
        outcome_type="away",
        spread_point=-1.5,
        residual_home_margins=create_residuals(),
    )

    probability_sum = (
        result["cover_probability"]
        + result["push_probability"]
        + result["loss_probability"]
    )

    assert probability_sum == pytest.approx(1.0)


def test_expected_value_includes_push_refund() -> None:
    """Spread EV handles wins, pushes and losses."""

    result = calculate_spread_expected_value(
        cover_probability=0.50,
        push_probability=0.10,
        loss_probability=0.40,
        decimal_odds=1.91,
    )

    assert result[
        "expected_value_per_unit"
    ] == pytest.approx(0.055)

    assert result[
        "expected_value_percent"
    ] == pytest.approx(5.5)

    assert result[
        "positive_expected_value"
    ]


def test_negative_expected_value_flag() -> None:
    """A losing price is not marked positive EV."""

    result = calculate_spread_expected_value(
        cover_probability=0.45,
        push_probability=0.05,
        loss_probability=0.50,
        decimal_odds=1.90,
    )

    assert result[
        "expected_value_per_unit"
    ] == pytest.approx(-0.095)

    assert result[
        "expected_value_percent"
    ] == pytest.approx(-9.5)

    assert not result[
        "positive_expected_value"
    ]


def test_unknown_outcome_type_is_rejected() -> None:
    """Only home and away Spread outcomes are valid."""

    with pytest.raises(
        ValueError,
        match="must be home or away",
    ):
        estimate_spread_cover_probabilities(
            predicted_home_margin=3.0,
            outcome_type="draw",
            spread_point=-3.0,
            residual_home_margins=(
                create_residuals()
            ),
        )


def test_invalid_residuals_are_rejected() -> None:
    """Residuals must be finite and non-empty."""

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        estimate_spread_cover_probabilities(
            predicted_home_margin=3.0,
            outcome_type="home",
            spread_point=-3.0,
            residual_home_margins=np.asarray(
                [],
                dtype=float,
            ),
        )

    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        estimate_spread_cover_probabilities(
            predicted_home_margin=3.0,
            outcome_type="home",
            spread_point=-3.0,
            residual_home_margins=np.asarray(
                [
                    0.0,
                    np.nan,
                ]
            ),
        )


def test_invalid_probability_sum_is_rejected() -> None:
    """EV inputs must form a probability distribution."""

    with pytest.raises(
        ValueError,
        match="must sum to one",
    ):
        calculate_spread_expected_value(
            cover_probability=0.50,
            push_probability=0.10,
            loss_probability=0.30,
            decimal_odds=1.91,
        )


def test_invalid_decimal_odds_are_rejected() -> None:
    """Decimal odds must exceed the returned stake."""

    with pytest.raises(
        ValueError,
        match="greater than one",
    ):
        calculate_spread_expected_value(
            cover_probability=0.50,
            push_probability=0.10,
            loss_probability=0.40,
            decimal_odds=1.0,
        )