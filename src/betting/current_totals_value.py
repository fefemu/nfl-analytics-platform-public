"""Calculate Over/Under probabilities and expected value from Totals residuals."""

import numpy as np
import pandas as pd

TOTALS_VALUE_COLUMNS = (
    "snapshot_id", "fetched_at", "game_id", "season", "game_type", "week",
    "gameday", "commence_time", "home_team", "away_team", "market_key",
    "market_name", "market_line", "outcome_name", "outcome_type", "point",
    "best_bookmaker_key", "best_bookmaker_title", "best_american_price",
    "best_decimal_odds", "best_implied_probability", "consensus_no_vig_probability",
    "bookmaker_count", "model_name", "model_version", "prediction_mode",
    "predicted_total_points", "calibration_sample_count", "win_probability",
    "push_probability", "loss_probability", "no_push_win_probability",
    "probability_edge", "probability_edge_percentage_points", "fair_decimal_odds",
    "expected_value_per_unit", "expected_value_percent", "full_kelly_fraction",
    "positive_expected_value", "prediction_generated_at",
)


def create_current_totals_value(
    market_board: pd.DataFrame,
    predictions: pd.DataFrame,
    residuals: pd.DataFrame,
) -> pd.DataFrame:
    """Join current Totals offers to empirical model uncertainty."""
    market = market_board.loc[market_board["market_key"] == "totals"].copy()
    if market.empty:
        raise RuntimeError("Current market board contains no Totals offers.")
    if not set(market["outcome_type"]).issubset({"over", "under"}):
        raise ValueError("Totals market contains unknown outcome types.")
    pair_counts = market.groupby(["game_id", "market_line"])["outcome_type"].nunique()
    valid_pairs = pair_counts.loc[pair_counts == 2].rename("outcome_count").reset_index()
    if valid_pairs.empty:
        raise RuntimeError("Current Totals market contains no paired Over/Under lines.")
    market = market.merge(valid_pairs[["game_id", "market_line"]],
                          on=["game_id", "market_line"], how="inner", validate="many_to_one")
    value = market.merge(predictions, on="game_id", how="inner", validate="many_to_one")
    residual_map = {
        mode: group["residual_total_points"].to_numpy(dtype=float)
        for mode, group in residuals.groupby("prediction_mode")
    }
    rows: list[dict[str, object]] = []
    for row in value.to_dict("records"):
        samples = residual_map.get(row["prediction_mode"])
        if samples is None or samples.size == 0:
            raise RuntimeError("Totals calibration is missing prediction mode: " + row["prediction_mode"])
        threshold = float(row["market_line"] - row["predicted_total_points"])
        push = np.isclose(samples, threshold, atol=1e-9)
        if row["outcome_type"] == "over":
            win = samples > threshold
            loss = samples < threshold
        else:
            win = samples < threshold
            loss = samples > threshold
        win_probability = float(np.mean(win))
        push_probability = float(np.mean(push))
        loss_probability = float(np.mean(loss))
        decisive = win_probability + loss_probability
        no_push = win_probability / decisive if decisive > 0 else 0.5
        odds = float(row["best_decimal_odds"])
        expected_value = win_probability * (odds - 1.0) - loss_probability
        edge = no_push - float(row["consensus_no_vig_probability"])
        fair_odds = 1.0 / no_push if no_push > 0 else np.inf
        b = odds - 1.0
        kelly = max(0.0, (b * no_push - (1.0 - no_push)) / b) if b > 0 else 0.0
        row.update({
            "calibration_sample_count": int(samples.size),
            "win_probability": win_probability,
            "push_probability": push_probability,
            "loss_probability": loss_probability,
            "no_push_win_probability": no_push,
            "probability_edge": edge,
            "probability_edge_percentage_points": 100.0 * edge,
            "fair_decimal_odds": fair_odds,
            "expected_value_per_unit": expected_value,
            "expected_value_percent": 100.0 * expected_value,
            "full_kelly_fraction": kelly,
            "positive_expected_value": expected_value > 0.0,
        })
        rows.append(row)
    return pd.DataFrame(rows).loc[:, TOTALS_VALUE_COLUMNS].sort_values(
        ["commence_time", "game_id", "market_line", "outcome_type"]
    ).reset_index(drop=True)
