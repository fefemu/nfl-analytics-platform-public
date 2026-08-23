"""
NFL Analytics Platform
Current Spread Value Calculation

Purpose:
    Join current spread predictions to available market
    lines and estimate cover, push, loss probabilities
    from chronological out-of-sample residuals.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import numpy as np
import pandas as pd

from src.betting.spread_cover_probability import (
    calculate_spread_expected_value,
    estimate_spread_cover_probabilities,
)


SPREAD_MARKET_KEY = "spreads"

REQUIRED_MARKET_COLUMNS = {
    "snapshot_id",
    "fetched_at",
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "commence_time",
    "home_team",
    "away_team",
    "market_key",
    "market_name",
    "outcome_name",
    "outcome_type",
    "point",
    "market_line",
    "best_bookmaker_key",
    "best_bookmaker_title",
    "best_american_price",
    "best_decimal_odds",
    "best_implied_probability",
    "bookmaker_count",
    "consensus_no_vig_probability",
}

REQUIRED_PREDICTION_COLUMNS = {
    "game_id",
    "model_name",
    "model_version",
    "prediction_mode",
    "predicted_home_margin",
    "predicted_away_margin",
    "prediction_generated_at",
}

REQUIRED_RESIDUAL_COLUMNS = {
    "prediction_mode",
    "residual_home_margin",
}

SPREAD_VALUE_COLUMNS = (
    "snapshot_id",
    "fetched_at",
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "commence_time",
    "home_team",
    "away_team",
    "market_key",
    "market_name",
    "market_line",
    "home_spread_line",
    "outcome_name",
    "outcome_type",
    "point",
    "best_bookmaker_key",
    "best_bookmaker_title",
    "best_american_price",
    "best_decimal_odds",
    "best_implied_probability",
    "consensus_no_vig_probability",
    "bookmaker_count",
    "model_name",
    "model_version",
    "prediction_mode",
    "predicted_home_margin",
    "predicted_away_margin",
    "predicted_outcome_margin",
    "calibration_sample_count",
    "cover_probability",
    "push_probability",
    "loss_probability",
    "no_push_cover_probability",
    "probability_edge",
    "probability_edge_percentage_points",
    "fair_decimal_odds",
    "expected_value_per_unit",
    "expected_value_percent",
    "full_kelly_fraction",
    "positive_expected_value",
    "prediction_generated_at",
)


def validate_required_columns(
    data: pd.DataFrame,
    required_columns: set[str],
    data_name: str,
) -> None:
    """Validate required DataFrame columns."""

    missing_columns = sorted(
        required_columns - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{data_name} is missing columns: "
            + ", ".join(missing_columns)
        )


def validate_spread_market(
    market_board: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and return paired Spread market lines."""

    validate_required_columns(
        data=market_board,
        required_columns=REQUIRED_MARKET_COLUMNS,
        data_name="Current market board",
    )

    spread_market = market_board.loc[
        market_board["market_key"]
        == SPREAD_MARKET_KEY
    ].copy()

    if spread_market.empty:
        raise RuntimeError(
            "Current market board contains no "
            "Spread offers."
        )

    invalid_outcome_mask = ~spread_market[
        "outcome_type"
    ].isin(
        [
            "home",
            "away",
        ]
    )

    if invalid_outcome_mask.any():
        raise ValueError(
            "Spread market contains unknown "
            "outcome types."
        )

    numeric_columns = [
        "point",
        "market_line",
        "best_decimal_odds",
        "best_implied_probability",
        "consensus_no_vig_probability",
    ]

    numeric_values = spread_market[
        numeric_columns
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Spread market contains non-finite "
            "line, price or probability values."
        )

    invalid_value_mask = (
        spread_market["market_line"].lt(0.0)
        | spread_market[
            "best_decimal_odds"
        ].le(1.0)
        | spread_market[
            "best_implied_probability"
        ].le(0.0)
        | spread_market[
            "best_implied_probability"
        ].ge(1.0)
        | spread_market[
            "consensus_no_vig_probability"
        ].le(0.0)
        | spread_market[
            "consensus_no_vig_probability"
        ].ge(1.0)
        | spread_market[
            "bookmaker_count"
        ].le(0)
    )

    if invalid_value_mask.any():
        raise ValueError(
            "Spread market contains invalid line, "
            "price or probability values."
        )

    if spread_market[
        [
            "game_id",
            "outcome_type",
            "point",
        ]
    ].duplicated().any():
        raise ValueError(
            "Spread market contains duplicate "
            "game-outcome-line offers."
        )

    spread_market[
        "home_spread_line"
    ] = np.where(
        spread_market["outcome_type"] == "home",
        spread_market["point"],
        -spread_market["point"],
    ).astype(float)

    line_groups = spread_market.groupby(
        [
            "game_id",
            "home_spread_line",
        ],
        sort=False,
    ).agg(
        outcome_count=(
            "outcome_type",
            "count",
        ),
        unique_outcome_count=(
            "outcome_type",
            "nunique",
        ),
        point_sum=(
            "point",
            "sum",
        ),
    )

    invalid_line_groups = line_groups.loc[
        (
            line_groups["outcome_count"] != 2
        )
        | (
            line_groups[
                "unique_outcome_count"
            ] != 2
        )
        | (
            line_groups[
                "point_sum"
            ].abs() > 0.000001
        )
    ]

    if not invalid_line_groups.empty:
        raise ValueError(
            "Every Spread line must contain paired "
            "home and away outcomes with opposite points."
        )

    return spread_market


def validate_spread_predictions(
    predictions: pd.DataFrame,
) -> None:
    """Validate current production spread predictions."""

    validate_required_columns(
        data=predictions,
        required_columns=(
            REQUIRED_PREDICTION_COLUMNS
        ),
        data_name="Current spread predictions",
    )

    if predictions["game_id"].duplicated().any():
        raise ValueError(
            "Current spread predictions contain "
            "duplicate game identifiers."
        )

    margin_values = predictions[
        [
            "predicted_home_margin",
            "predicted_away_margin",
        ]
    ].to_numpy(dtype=float)

    if not np.isfinite(margin_values).all():
        raise ValueError(
            "Current spread predictions contain "
            "non-finite margins."
        )

    margin_sum = (
        predictions["predicted_home_margin"]
        + predictions["predicted_away_margin"]
    )

    if not np.allclose(
        margin_sum.to_numpy(dtype=float),
        0.0,
        atol=0.000001,
        rtol=0.0,
    ):
        raise ValueError(
            "Predicted home and away margins must "
            "sum to zero."
        )


def validate_spread_residuals(
    residuals: pd.DataFrame,
) -> None:
    """Validate empirical residual distributions."""

    validate_required_columns(
        data=residuals,
        required_columns=(
            REQUIRED_RESIDUAL_COLUMNS
        ),
        data_name="Spread calibration residuals",
    )

    if residuals.empty:
        raise RuntimeError(
            "Spread calibration residuals are empty."
        )

    if residuals[
        "prediction_mode"
    ].isna().any():
        raise ValueError(
            "Spread calibration residuals contain "
            "missing prediction modes."
        )

    residual_values = residuals[
        "residual_home_margin"
    ].to_numpy(dtype=float)

    if not np.isfinite(
        residual_values
    ).all():
        raise ValueError(
            "Spread calibration residuals must "
            "be finite."
        )


def create_current_spread_value(
    market_board: pd.DataFrame,
    predictions: pd.DataFrame,
    residuals: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate current Spread probabilities and EV."""

    spread_market = validate_spread_market(
        market_board
    )

    validate_spread_predictions(
        predictions
    )

    validate_spread_residuals(
        residuals
    )

    prediction_source = predictions.loc[
        :,
        [
            "game_id",
            "model_name",
            "model_version",
            "prediction_mode",
            "predicted_home_margin",
            "predicted_away_margin",
            "prediction_generated_at",
        ],
    ]

    value = spread_market.merge(
        prediction_source,
        on="game_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )

    missing_prediction_mask = value[
        "model_name"
    ].isna()

    if missing_prediction_mask.any():
        missing_game_ids = ", ".join(
            value.loc[
                missing_prediction_mask,
                "game_id",
            ].astype(str).unique()
        )

        raise RuntimeError(
            "Spread games are missing production "
            f"predictions: {missing_game_ids}"
        )

    residuals_by_mode = {
        prediction_mode: group[
            "residual_home_margin"
        ].to_numpy(dtype=float)
        for prediction_mode, group
        in residuals.groupby(
            "prediction_mode",
            sort=False,
        )
    }

    missing_modes = sorted(
        set(
            value["prediction_mode"]
        )
        - set(residuals_by_mode)
    )

    if missing_modes:
        raise RuntimeError(
            "Spread calibration residuals are missing "
            "prediction modes: "
            + ", ".join(missing_modes)
        )

    result_rows: list[
        dict[str, object]
    ] = []

    for row in value.to_dict(
        orient="records"
    ):
        prediction_mode = row[
            "prediction_mode"
        ]

        mode_residuals = residuals_by_mode[
            prediction_mode
        ]

        predicted_outcome_margin = (
            row["predicted_home_margin"]
            if row["outcome_type"] == "home"
            else row["predicted_away_margin"]
        )

        probabilities = (
            estimate_spread_cover_probabilities(
                predicted_home_margin=float(
                    row["predicted_home_margin"]
                ),
                outcome_type=row["outcome_type"],
                spread_point=float(row["point"]),
                residual_home_margins=(
                    mode_residuals
                ),
            )
        )

        cover_probability = float(
            probabilities[
                "cover_probability"
            ]
        )

        push_probability = float(
            probabilities[
                "push_probability"
            ]
        )

        loss_probability = float(
            probabilities[
                "loss_probability"
            ]
        )

        no_push_denominator = (
            cover_probability
            + loss_probability
        )

        if no_push_denominator <= 0.0:
            raise RuntimeError(
                "Spread win and loss probabilities "
                "cannot both be zero."
            )

        no_push_cover_probability = (
            cover_probability
            / no_push_denominator
        )

        probability_edge = (
            no_push_cover_probability
            - row[
                "consensus_no_vig_probability"
            ]
        )

        if cover_probability <= 0.0:
            fair_decimal_odds = np.inf
        else:
            fair_decimal_odds = (
                1.0
                + loss_probability
                / cover_probability
            )

        expected_value = (
            calculate_spread_expected_value(
                cover_probability=(
                    cover_probability
                ),
                push_probability=(
                    push_probability
                ),
                loss_probability=(
                    loss_probability
                ),
                decimal_odds=float(
                    row["best_decimal_odds"]
                ),
            )
        )

        decimal_profit = (
            float(
                row["best_decimal_odds"]
            )
            - 1.0
        )

        non_push_probability = (
            cover_probability
            + loss_probability
        )

        full_kelly_fraction = max(
            0.0,
            (
                expected_value[
                    "expected_value_per_unit"
                ]
                / (
                    decimal_profit
                    * non_push_probability
                )
            ),
        )

        result_rows.append(
            {
                **row,
                "predicted_outcome_margin": float(
                    predicted_outcome_margin
                ),
                "calibration_sample_count": int(
                    probabilities[
                        "simulation_count"
                    ]
                ),
                "cover_probability": (
                    cover_probability
                ),
                "push_probability": (
                    push_probability
                ),
                "loss_probability": (
                    loss_probability
                ),
                "no_push_cover_probability": (
                    no_push_cover_probability
                ),
                "probability_edge": float(
                    probability_edge
                ),
                (
                    "probability_edge_"
                    "percentage_points"
                ): float(
                    100.0 * probability_edge
                ),
                "fair_decimal_odds": float(
                    fair_decimal_odds
                ),
                "expected_value_per_unit": (
                    expected_value[
                        "expected_value_per_unit"
                    ]
                ),
                "expected_value_percent": (
                    expected_value[
                        "expected_value_percent"
                    ]
                ),
                "full_kelly_fraction": float(
                    full_kelly_fraction
                ),
                "positive_expected_value": (
                    expected_value[
                        "positive_expected_value"
                    ]
                ),
            }
        )

    result = pd.DataFrame(
        result_rows
    )

    if not np.isfinite(
        result[
            [
                "cover_probability",
                "push_probability",
                "loss_probability",
                "no_push_cover_probability",
                "probability_edge",
                "fair_decimal_odds",
                "expected_value_per_unit",
                "expected_value_percent",
                "full_kelly_fraction",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            "Current Spread value calculations "
            "must be finite."
        )

    return result.loc[
        :,
        SPREAD_VALUE_COLUMNS,
    ].sort_values(
        by=[
            "expected_value_per_unit",
            "probability_edge",
            "game_id",
            "market_line",
            "outcome_type",
        ],
        ascending=[
            False,
            False,
            True,
            True,
            True,
        ],
        kind="stable",
    ).reset_index(drop=True)