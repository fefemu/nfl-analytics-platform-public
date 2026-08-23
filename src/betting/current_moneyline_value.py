"""
NFL Analytics Platform
Current Moneyline Value Calculation

Purpose:
    Join current production win probabilities to the
    latest schedule-linked Moneyline market and calculate
    auditable probability edge and expected value.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import numpy as np
import pandas as pd


HOME_OUTCOME = "home"
AWAY_OUTCOME = "away"
MONEYLINE_MARKET_KEY = "h2h"

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
    "home_win_probability",
    "away_win_probability",
    "prediction_generated_at",
}

MONEYLINE_VALUE_COLUMNS = (
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
    "model_probability",
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


def validate_moneyline_market(
    market_board: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and return two-way Moneyline offers."""

    validate_required_columns(
        data=market_board,
        required_columns=REQUIRED_MARKET_COLUMNS,
        data_name="Current market board",
    )

    moneyline = market_board.loc[
        market_board["market_key"]
        == MONEYLINE_MARKET_KEY
    ].copy()

    if moneyline.empty:
        raise RuntimeError(
            "Current market board contains no "
            "Moneyline offers."
        )

    invalid_outcome_mask = ~moneyline[
        "outcome_type"
    ].isin(
        [
            HOME_OUTCOME,
            AWAY_OUTCOME,
        ]
    )

    if invalid_outcome_mask.any():
        raise ValueError(
            "Moneyline market contains unknown "
            "outcome types."
        )

    if moneyline[
        [
            "game_id",
            "outcome_type",
        ]
    ].duplicated().any():
        raise ValueError(
            "Moneyline market contains duplicate "
            "game-outcome offers."
        )

    outcome_counts = moneyline.groupby(
        "game_id",
        sort=False,
    )["outcome_type"].agg(
        outcome_count="count",
        unique_outcome_count="nunique",
    )

    invalid_games = outcome_counts.loc[
        (
            outcome_counts["outcome_count"] != 2
        )
        | (
            outcome_counts[
                "unique_outcome_count"
            ] != 2
        )
    ]

    if not invalid_games.empty:
        raise ValueError(
            "Every Moneyline game must contain exactly "
            "one home and one away outcome."
        )

    invalid_price_mask = (
        moneyline["best_decimal_odds"].le(1.0)
        | moneyline[
            "best_decimal_odds"
        ].isna()
        | moneyline[
            "consensus_no_vig_probability"
        ].le(0.0)
        | moneyline[
            "consensus_no_vig_probability"
        ].ge(1.0)
    )

    if invalid_price_mask.any():
        raise ValueError(
            "Moneyline market contains invalid odds "
            "or no-vig probabilities."
        )

    probability_sums = moneyline.groupby(
        "game_id",
        sort=False,
    )[
        "consensus_no_vig_probability"
    ].sum()

    if not np.allclose(
        probability_sums.to_numpy(dtype=float),
        1.0,
        atol=0.000001,
        rtol=0.0,
    ):
        raise ValueError(
            "Moneyline consensus no-vig probabilities "
            "must sum to one per game."
        )

    return moneyline


def validate_current_predictions(
    predictions: pd.DataFrame,
) -> None:
    """Validate current production probabilities."""

    validate_required_columns(
        data=predictions,
        required_columns=(
            REQUIRED_PREDICTION_COLUMNS
        ),
        data_name="Current game predictions",
    )

    if predictions["game_id"].duplicated().any():
        raise ValueError(
            "Current game predictions contain duplicate "
            "game identifiers."
        )

    probability_columns = [
        "home_win_probability",
        "away_win_probability",
    ]

    invalid_probability_mask = (
        predictions[
            probability_columns
        ].isna().any(axis=1)
        | predictions[
            probability_columns
        ].le(0.0).any(axis=1)
        | predictions[
            probability_columns
        ].ge(1.0).any(axis=1)
    )

    if invalid_probability_mask.any():
        raise ValueError(
            "Current game predictions contain invalid "
            "win probabilities."
        )

    probability_sums = predictions[
        "home_win_probability"
    ] + predictions[
        "away_win_probability"
    ]

    if not np.allclose(
        probability_sums.to_numpy(dtype=float),
        1.0,
        atol=0.000001,
        rtol=0.0,
    ):
        raise ValueError(
            "Current model probabilities must sum "
            "to one per game."
        )


def create_current_moneyline_value(
    market_board: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate current Moneyline edge and EV."""

    moneyline = validate_moneyline_market(
        market_board
    )

    validate_current_predictions(
        predictions
    )

    prediction_source = predictions.loc[
        :,
        [
            "game_id",
            "model_name",
            "model_version",
            "prediction_mode",
            "home_win_probability",
            "away_win_probability",
            "prediction_generated_at",
        ],
    ]

    value = moneyline.merge(
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
            "Moneyline games are missing production "
            f"predictions: {missing_game_ids}"
        )

    value["model_probability"] = np.where(
        value["outcome_type"] == HOME_OUTCOME,
        value["home_win_probability"],
        value["away_win_probability"],
    ).astype(float)

    value["probability_edge"] = (
        value["model_probability"]
        - value[
            "consensus_no_vig_probability"
        ]
    )

    value[
        "probability_edge_percentage_points"
    ] = 100.0 * value["probability_edge"]

    value["fair_decimal_odds"] = (
        1.0 / value["model_probability"]
    )

    value["expected_value_per_unit"] = (
        value["model_probability"]
        * value["best_decimal_odds"]
        - 1.0
    )

    value["expected_value_percent"] = (
        100.0
        * value["expected_value_per_unit"]
    )

    value["full_kelly_fraction"] = (
        value["expected_value_per_unit"]
        / (
            value["best_decimal_odds"]
            - 1.0
        )
    ).clip(
        lower=0.0
    )

    value["positive_expected_value"] = (
        value["expected_value_per_unit"] > 0.0
    )

    return value.loc[
        :,
        MONEYLINE_VALUE_COLUMNS,
    ].sort_values(
        by=[
            "expected_value_per_unit",
            "probability_edge",
            "game_id",
            "outcome_type",
        ],
        ascending=[
            False,
            False,
            True,
            True,
        ],
        kind="stable",
    ).reset_index(drop=True)