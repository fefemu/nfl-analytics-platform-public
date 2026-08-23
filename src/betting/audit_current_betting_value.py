"""Read-only mathematical and concentration audit of the current betting board."""

from pathlib import Path
import logging

import duckdb
import numpy as np
import pandas as pd

from src.modeling.train_logistic_baseline import DATABASE_FILE, validate_database_file


logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "market_name", "best_decimal_odds", "bookmaker_count", "prediction_mode",
    "model_probability", "push_probability", "loss_probability",
    "probability_edge", "consensus_no_vig_probability",
    "expected_value_per_unit", "expected_value_percent", "positive_expected_value",
}


def select_publishable_next_week_candidates(
    board: pd.DataFrame,
    minimum_model_probability: float = 0.50,
    minimum_bookmaker_count: int = 5,
    maximum_edge: float = 0.10,
    maximum_ev_percent: float = 20.0,
) -> pd.DataFrame:
    """Apply transparent publication guardrails to the next available week."""

    required = REQUIRED_COLUMNS | {"game_id", "week", "commence_time", "outcome_name"}
    missing = sorted(required - set(board.columns))
    if missing:
        raise ValueError("Betting board is missing columns: " + ", ".join(missing))
    if not 0.0 < minimum_model_probability < 1.0:
        raise ValueError("Minimum model probability must be between zero and one.")
    if minimum_bookmaker_count <= 0:
        raise ValueError("Minimum bookmaker count must be positive.")
    if not 0.0 < maximum_edge < 1.0:
        raise ValueError("Maximum Edge must be between zero and one.")
    if maximum_ev_percent <= 0.0:
        raise ValueError("Maximum EV percent must be positive.")

    future = board.loc[pd.to_datetime(board["commence_time"], utc=True) >= pd.Timestamp.now(tz="UTC")].copy()
    if future.empty:
        return future
    next_time = pd.to_datetime(future["commence_time"], utc=True).min()
    next_week = int(future.loc[pd.to_datetime(future["commence_time"], utc=True).eq(next_time), "week"].iloc[0])
    return future.loc[
        future["week"].eq(next_week)
        & future["model_probability"].ge(minimum_model_probability)
        & future["bookmaker_count"].ge(minimum_bookmaker_count)
        & future["positive_expected_value"].astype(bool)
        & future["probability_edge"].le(maximum_edge)
        & future["expected_value_percent"].le(maximum_ev_percent)
    ].sort_values(["expected_value_percent", "probability_edge"], ascending=False, kind="stable").reset_index(drop=True)


def validate_betting_value_math(board: pd.DataFrame) -> pd.DataFrame:
    """Recalculate the core probability, Edge and EV identities."""

    missing = sorted(REQUIRED_COLUMNS - set(board.columns))
    if missing:
        raise ValueError("Betting board is missing columns: " + ", ".join(missing))
    if board.empty:
        raise ValueError("Betting board is empty.")

    probability_sum = (
        board["model_probability"].astype(float)
        + board["push_probability"].astype(float)
        + board["loss_probability"].astype(float)
    )
    decisive = 1.0 - board["push_probability"].astype(float)
    no_push_probability = board["model_probability"].astype(float) / decisive
    expected_edge = no_push_probability - board["consensus_no_vig_probability"].astype(float)
    expected_ev = (
        board["model_probability"].astype(float)
        * (board["best_decimal_odds"].astype(float) - 1.0)
        - board["loss_probability"].astype(float)
    )

    checks = {
        "probabilities_sum_to_one": ~np.isclose(probability_sum, 1.0, atol=1e-9),
        "edge_formula": ~np.isclose(board["probability_edge"], expected_edge, atol=1e-9),
        "ev_formula": ~np.isclose(board["expected_value_per_unit"], expected_ev, atol=1e-9),
        "ev_percent_formula": ~np.isclose(
            board["expected_value_percent"], 100.0 * expected_ev, atol=1e-8
        ),
        "positive_ev_flag": board["positive_expected_value"].astype(bool) != (expected_ev > 0.0),
    }
    return pd.DataFrame(
        [{"check_name": name, "issue_count": int(mask.sum()), "status": "PASS" if int(mask.sum()) == 0 else "FAIL"}
         for name, mask in checks.items()]
    )


def validate_market_pairing(board: pd.DataFrame) -> pd.DataFrame:
    """Validate paired outcomes and no-vig probability sums by market line."""

    required = {"market_name", "game_id", "pair_key", "consensus_no_vig_probability"}
    missing = sorted(required - set(board.columns))
    if missing:
        raise ValueError("Betting board is missing columns: " + ", ".join(missing))
    groups = (
        board.groupby(["market_name", "game_id", "pair_key"], dropna=False)
        .agg(
            outcome_count=("consensus_no_vig_probability", "size"),
            probability_sum=("consensus_no_vig_probability", "sum"),
        )
        .reset_index()
    )
    checks = {
        "paired_outcome_count": groups["outcome_count"].ne(2),
        "paired_no_vig_sum": ~np.isclose(groups["probability_sum"], 1.0, atol=1e-9),
    }
    return pd.DataFrame([
        {
            "check_name": name,
            "issue_count": int(mask.sum()),
            "status": "PASS" if int(mask.sum()) == 0 else "FAIL",
        }
        for name, mask in checks.items()
    ])


def summarize_extreme_offers(
    board: pd.DataFrame,
    minimum_edge: float = 0.15,
    minimum_ev_percent: float = 20.0,
) -> pd.DataFrame:
    """Return unusually large model-market gaps for manual review."""

    if not 0.0 <= minimum_edge < 1.0:
        raise ValueError("Minimum Edge must be between zero and one.")
    if minimum_ev_percent < 0.0:
        raise ValueError("Minimum EV percent must not be negative.")
    return board.loc[
        board["probability_edge"].ge(minimum_edge)
        | board["expected_value_percent"].ge(minimum_ev_percent)
    ].sort_values(
        ["expected_value_percent", "probability_edge"],
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)


def summarize_value_concentration(board: pd.DataFrame) -> pd.DataFrame:
    """Show where large displayed EV values are concentrated."""

    data = board.copy()
    data["bookmaker_bucket"] = pd.cut(
        data["bookmaker_count"], bins=[0, 1, 4, np.inf], labels=["1 bookmaker", "2-4 bookmakers", "5+ bookmakers"]
    )
    return (
        data.groupby(["market_name", "prediction_mode", "bookmaker_bucket"], observed=True)
        .agg(
            offer_count=("expected_value_percent", "size"),
            positive_ev_count=("positive_expected_value", "sum"),
            ev_10_plus_count=("expected_value_percent", lambda value: int((value >= 10.0).sum())),
            ev_20_plus_count=("expected_value_percent", lambda value: int((value >= 20.0).sum())),
            median_ev_percent=("expected_value_percent", "median"),
            maximum_ev_percent=("expected_value_percent", "max"),
        )
        .reset_index()
        .sort_values(["market_name", "prediction_mode", "bookmaker_bucket"], kind="stable")
    )


def calculate_market_shrinkage_sensitivity(
    board: pd.DataFrame,
    model_weights: tuple[float, ...] = (1.0, 0.75, 0.50),
) -> pd.DataFrame:
    """Measure how positive EV survives probability shrinkage toward market."""

    rows: list[dict[str, float | int | str]] = []
    decisive = 1.0 - board["push_probability"].astype(float)
    model_no_push = board["model_probability"].astype(float) / decisive
    market = board["consensus_no_vig_probability"].astype(float)
    odds = board["best_decimal_odds"].astype(float)

    for weight in model_weights:
        if not 0.0 <= weight <= 1.0:
            raise ValueError("Model weights must be between zero and one.")
        probability = weight * model_no_push + (1.0 - weight) * market
        win = decisive * probability
        loss = decisive * (1.0 - probability)
        ev = win * (odds - 1.0) - loss
        for market_name, indexes in board.groupby("market_name").groups.items():
            values = ev.loc[indexes]
            rows.append({
                "market_name": market_name,
                "model_weight": float(weight),
                "offer_count": len(values),
                "positive_ev_count": int((values > 0.0).sum()),
                "ev_10_plus_count": int((values >= 0.10).sum()),
                "ev_20_plus_count": int((values >= 0.20).sum()),
                "median_ev_percent": float(100.0 * values.median()),
                "maximum_ev_percent": float(100.0 * values.max()),
            })
    return pd.DataFrame(rows)


def load_current_board(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load all value products including their market probability input."""

    return connection.execute(
        """
        SELECT game_id, week, commence_time, outcome_name, market_name,
               'MONEYLINE' AS pair_key,
               best_decimal_odds, bookmaker_count, prediction_mode,
               model_probability, 0.0 AS push_probability,
               1.0 - model_probability AS loss_probability,
               probability_edge, consensus_no_vig_probability,
               expected_value_per_unit, expected_value_percent, positive_expected_value
        FROM analytics.current_moneyline_value
        UNION ALL
        SELECT game_id, week, commence_time, outcome_name, market_name,
               CASE
                   WHEN ABS(home_spread_line) < 0.000000001 THEN '0'
                   ELSE CAST(home_spread_line AS VARCHAR)
               END AS pair_key,
               best_decimal_odds, bookmaker_count, prediction_mode,
               cover_probability, push_probability, loss_probability,
               probability_edge, consensus_no_vig_probability,
               expected_value_per_unit, expected_value_percent, positive_expected_value
        FROM analytics.current_spread_value
        UNION ALL
        SELECT game_id, week, commence_time, outcome_name, market_name,
               CAST(market_line AS VARCHAR) AS pair_key,
               best_decimal_odds, bookmaker_count, prediction_mode,
               win_probability, push_probability, loss_probability,
               probability_edge, consensus_no_vig_probability,
               expected_value_per_unit, expected_value_percent, positive_expected_value
        FROM analytics.current_totals_value
        """
    ).fetchdf()


def run_current_betting_value_audit(
    database_file: Path = DATABASE_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the complete audit without changing database state."""

    validate_database_file(database_file)
    with duckdb.connect(str(database_file), read_only=True) as connection:
        board = load_current_board(connection)
    checks = pd.concat(
        [validate_betting_value_math(board), validate_market_pairing(board)],
        ignore_index=True,
    )
    concentration = summarize_value_concentration(board)
    sensitivity = calculate_market_shrinkage_sensitivity(board)
    publishable = select_publishable_next_week_candidates(board)
    logger.info("Current betting value audit completed on %s offers.", len(board))
    extremes = summarize_extreme_offers(board)
    return checks, concentration, sensitivity, publishable, extremes


def main() -> None:
    checks, concentration, sensitivity, publishable, extremes = run_current_betting_value_audit()
    print("\nBETTING VALUE FORMULA CHECKS\n", checks.to_string(index=False))
    print("\nVALUE CONCENTRATION\n", concentration.to_string(index=False))
    print("\nMARKET-SHRINKAGE SENSITIVITY\n", sensitivity.to_string(index=False))
    print("\nNEXT-WEEK PUBLISHABLE CANDIDATES\n", publishable[
        ["game_id", "market_name", "outcome_name", "model_probability", "probability_edge", "expected_value_percent", "best_decimal_odds", "bookmaker_count"]
    ].head(25).to_string(index=False))
    print("\nEXTREME OFFERS FOR MANUAL REVIEW\n", extremes[
        ["game_id", "market_name", "outcome_name", "model_probability", "consensus_no_vig_probability", "probability_edge", "expected_value_percent", "best_decimal_odds", "bookmaker_count"]
    ].head(25).to_string(index=False))


if __name__ == "__main__":
    main()
