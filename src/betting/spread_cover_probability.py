"""
NFL Analytics Platform
Spread Cover Probability

Purpose:
    Convert one predicted home margin and an empirical
    out-of-sample residual distribution into win, push
    and loss probabilities for a bookmaker spread line.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import numpy as np


HOME_OUTCOME = "home"
AWAY_OUTCOME = "away"

SPREAD_PROBABILITY_KEYS = (
    "cover_probability",
    "push_probability",
    "loss_probability",
    "simulation_count",
)


def validate_probability_inputs(
    predicted_home_margin: float,
    outcome_type: str,
    spread_point: float,
    residual_home_margins: np.ndarray,
) -> np.ndarray:
    """Validate and normalize probability inputs."""

    if not np.isfinite(
        predicted_home_margin
    ):
        raise ValueError(
            "Predicted home margin must be finite."
        )

    if outcome_type not in {
        HOME_OUTCOME,
        AWAY_OUTCOME,
    }:
        raise ValueError(
            "Spread outcome type must be home or away."
        )

    if not np.isfinite(spread_point):
        raise ValueError(
            "Spread point must be finite."
        )

    residuals = np.asarray(
        residual_home_margins,
        dtype=float,
    )

    if residuals.ndim != 1:
        raise ValueError(
            "Spread residuals must be one-dimensional."
        )

    if residuals.size == 0:
        raise ValueError(
            "Spread residuals must not be empty."
        )

    if not np.isfinite(residuals).all():
        raise ValueError(
            "Spread residuals must be finite."
        )

    return residuals


def estimate_spread_cover_probabilities(
    predicted_home_margin: float,
    outcome_type: str,
    spread_point: float,
    residual_home_margins: np.ndarray,
) -> dict[str, float | int]:
    """
    Estimate empirical win, push and loss probabilities.

    Each historical out-of-sample residual is added to
    the current predicted home margin. The resulting
    simulated final margin is rounded to an integer
    because NFL final score margins are discrete.

    A spread outcome wins when:

        simulated outcome margin + spread point > 0
    """

    residuals = validate_probability_inputs(
        predicted_home_margin=(
            predicted_home_margin
        ),
        outcome_type=outcome_type,
        spread_point=spread_point,
        residual_home_margins=(
            residual_home_margins
        ),
    )

    simulated_home_margins = np.rint(
        predicted_home_margin + residuals
    )

    if outcome_type == HOME_OUTCOME:
        simulated_outcome_margins = (
            simulated_home_margins
        )
    else:
        simulated_outcome_margins = (
            -simulated_home_margins
        )

    spread_results = (
        simulated_outcome_margins
        + spread_point
    )

    win_count = int(
        np.count_nonzero(
            spread_results > 0.0
        )
    )

    push_count = int(
        np.count_nonzero(
            spread_results == 0.0
        )
    )

    loss_count = int(
        np.count_nonzero(
            spread_results < 0.0
        )
    )

    simulation_count = int(
        residuals.size
    )

    if (
        win_count
        + push_count
        + loss_count
        != simulation_count
    ):
        raise RuntimeError(
            "Spread simulation outcome counts do "
            "not match the residual count."
        )

    cover_probability = (
        win_count / simulation_count
    )

    push_probability = (
        push_count / simulation_count
    )

    loss_probability = (
        loss_count / simulation_count
    )

    probability_sum = (
        cover_probability
        + push_probability
        + loss_probability
    )

    if not np.isclose(
        probability_sum,
        1.0,
        atol=0.000000001,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Spread probabilities must sum to one."
        )

    return {
        "cover_probability": float(
            cover_probability
        ),
        "push_probability": float(
            push_probability
        ),
        "loss_probability": float(
            loss_probability
        ),
        "simulation_count": (
            simulation_count
        ),
    }


def calculate_spread_expected_value(
    cover_probability: float,
    push_probability: float,
    loss_probability: float,
    decimal_odds: float,
) -> dict[str, float | bool]:
    """
    Calculate one-unit Spread expected value.

    A win returns decimal odds, a push refunds the stake,
    and a loss loses the full stake.
    """

    probabilities = np.asarray(
        [
            cover_probability,
            push_probability,
            loss_probability,
        ],
        dtype=float,
    )

    if not np.isfinite(
        probabilities
    ).all():
        raise ValueError(
            "Spread probabilities must be finite."
        )

    if (
        (probabilities < 0.0).any()
        or (probabilities > 1.0).any()
    ):
        raise ValueError(
            "Spread probabilities must be between "
            "zero and one."
        )

    if not np.isclose(
        probabilities.sum(),
        1.0,
        atol=0.000001,
        rtol=0.0,
    ):
        raise ValueError(
            "Spread probabilities must sum to one."
        )

    if (
        not np.isfinite(decimal_odds)
        or decimal_odds <= 1.0
    ):
        raise ValueError(
            "Spread decimal odds must be greater "
            "than one."
        )

    expected_value_per_unit = (
        cover_probability
        * (decimal_odds - 1.0)
        - loss_probability
    )

    expected_value_percent = (
        100.0 * expected_value_per_unit
    )

    return {
        "expected_value_per_unit": float(
            expected_value_per_unit
        ),
        "expected_value_percent": float(
            expected_value_percent
        ),
        "positive_expected_value": bool(
            expected_value_per_unit > 0.0
        ),
    }