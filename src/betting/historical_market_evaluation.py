"""Create a leakage-safe historical betting benchmark from OOF predictions."""

import numpy as np
import pandas as pd

LEDGER_COLUMNS = (
    "game_id", "season", "market_key", "model_name", "selection",
    "pricing_basis", "market_line", "market_price", "decimal_odds", "model_value",
    "market_value", "edge", "absolute_edge", "edge_bucket", "result",
    "profit_per_unit", "clv_value", "clv_available", "actual_home_margin",
    "actual_total",
)

SUMMARY_COLUMNS = (
    "market_key", "edge_bucket", "bet_count", "win_count", "loss_count",
    "push_count", "win_rate_excluding_pushes", "total_profit", "roi_percent",
    "mean_edge", "clv_bet_count", "mean_clv", "maximum_drawdown_units",
)


def american_to_decimal(price: float) -> float:
    """Convert a non-zero American price to decimal odds."""
    value = float(price)
    if not np.isfinite(value) or value == 0.0:
        raise ValueError("American odds must be finite and non-zero.")
    return 1.0 + (100.0 / abs(value) if value < 0.0 else value / 100.0)


def _edge_bucket(edge: float, market_key: str) -> str:
    thresholds = (0.025, 0.05, 0.10) if market_key == "h2h" else (1.0, 2.0, 3.0)
    labels = ("LOW", "MEDIUM", "HIGH", "VERY_HIGH")
    return labels[sum(float(edge) >= threshold for threshold in thresholds)]


def _settle(comparison: float, decimal_odds: float) -> tuple[str, float]:
    if comparison > 1e-9:
        return "WIN", decimal_odds - 1.0
    if comparison < -1e-9:
        return "LOSS", -1.0
    return "PUSH", 0.0


def create_historical_betting_ledger(
    probability_predictions: pd.DataFrame,
    spread_predictions: pd.DataFrame,
    totals_predictions: pd.DataFrame,
    market_data: pd.DataFrame,
) -> pd.DataFrame:
    """Select one model-preferred side per game and settle at closing prices."""
    market_required = {
        "game_id", "season", "actual_home_margin", "actual_total",
        "home_market_probability_close", "home_line_close",
        "home_line_close_price", "away_line_close_price", "total_line_close",
        "over_price_close", "under_price_close", "home_market_probability_open",
        "home_line_open", "total_line_open",
    }
    missing = sorted(market_required - set(market_data.columns))
    if missing:
        raise ValueError("Historical market data is missing columns: " + ", ".join(missing))
    if market_data["game_id"].duplicated().any():
        raise ValueError("Historical market data contains duplicate games.")

    rows: list[dict[str, object]] = []

    probability = probability_predictions.merge(market_data, on=["game_id", "season"], validate="one_to_one")
    for row in probability.itertuples(index=False):
        home_edge = float(row.home_win_probability) - float(row.home_market_probability_close)
        home = home_edge >= 0.0
        market_probability = float(row.home_market_probability_close if home else 1.0 - row.home_market_probability_close)
        decimal_odds = 1.0 / market_probability
        actual_win = float(row.actual_home_margin) > 0.0 if home else float(row.actual_home_margin) < 0.0
        result, profit = _settle(1.0 if actual_win else -1.0, decimal_odds)
        edge = abs(home_edge)
        opening_probability = float(row.home_market_probability_open if home else 1.0 - row.home_market_probability_open)
        rows.append({
            "game_id": row.game_id, "season": row.season, "market_key": "h2h",
            "model_name": row.model_name, "selection": "HOME" if home else "AWAY",
            "pricing_basis": "SYNTHETIC_CLOSE_FAIR",
            "market_line": np.nan, "market_price": np.nan, "decimal_odds": decimal_odds,
            "model_value": row.home_win_probability if home else 1.0 - row.home_win_probability,
            "market_value": market_probability, "edge": edge, "absolute_edge": edge,
            "edge_bucket": _edge_bucket(edge, "h2h"), "result": result,
            "profit_per_unit": profit,
            "clv_value": market_probability - opening_probability,
            "clv_available": True, "actual_home_margin": row.actual_home_margin,
            "actual_total": row.actual_total,
        })

    spread = spread_predictions.merge(market_data, on=["game_id", "season"], validate="one_to_one")
    for row in spread.itertuples(index=False):
        edge_signed = float(row.predicted_home_margin) + float(row.home_line_close)
        home = edge_signed >= 0.0
        price = float(row.home_line_close_price if home else row.away_line_close_price)
        decimal_odds = american_to_decimal(price)
        comparison = float(row.actual_home_margin) + float(row.home_line_close)
        result, profit = _settle(comparison if home else -comparison, decimal_odds)
        edge = abs(edge_signed)
        clv = np.nan if pd.isna(row.home_line_open) else (
            float(row.home_line_open) - float(row.home_line_close)
        ) * (1.0 if home else -1.0)
        rows.append({
            "game_id": row.game_id, "season": row.season, "market_key": "spreads",
            "model_name": row.model_name, "selection": "HOME" if home else "AWAY",
            "pricing_basis": "CLOSING_PRICE",
            "market_line": row.home_line_close if home else -float(row.home_line_close),
            "market_price": price, "decimal_odds": decimal_odds,
            "model_value": row.predicted_home_margin, "market_value": -float(row.home_line_close),
            "edge": edge, "absolute_edge": edge, "edge_bucket": _edge_bucket(edge, "spreads"),
            "result": result, "profit_per_unit": profit, "clv_value": clv,
            "clv_available": bool(pd.notna(clv)),
            "actual_home_margin": row.actual_home_margin, "actual_total": row.actual_total,
        })

    totals = totals_predictions.merge(market_data, on=["game_id", "season"], validate="one_to_one")
    for row in totals.itertuples(index=False):
        edge_signed = float(row.predicted_total) - float(row.total_line_close)
        over = edge_signed >= 0.0
        price = float(row.over_price_close if over else row.under_price_close)
        decimal_odds = american_to_decimal(price)
        comparison = float(row.actual_total) - float(row.total_line_close)
        result, profit = _settle(comparison if over else -comparison, decimal_odds)
        edge = abs(edge_signed)
        clv = np.nan if pd.isna(row.total_line_open) else (
            float(row.total_line_close) - float(row.total_line_open)
        ) * (1.0 if over else -1.0)
        rows.append({
            "game_id": row.game_id, "season": row.season, "market_key": "totals",
            "model_name": row.model_name, "selection": "OVER" if over else "UNDER",
            "pricing_basis": "CLOSING_PRICE",
            "market_line": row.total_line_close, "market_price": price,
            "decimal_odds": decimal_odds, "model_value": row.predicted_total,
            "market_value": row.total_line_close, "edge": edge, "absolute_edge": edge,
            "edge_bucket": _edge_bucket(edge, "totals"), "result": result,
            "profit_per_unit": profit, "clv_value": clv,
            "clv_available": bool(pd.notna(clv)), "actual_home_margin": row.actual_home_margin,
            "actual_total": row.actual_total,
        })

    ledger = pd.DataFrame(rows, columns=LEDGER_COLUMNS)
    if ledger.empty or not np.isfinite(ledger["profit_per_unit"]).all():
        raise RuntimeError("Historical betting ledger is empty or invalid.")
    return ledger.sort_values(["season", "game_id", "market_key"]).reset_index(drop=True)


def create_historical_betting_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    """Summarize flat-stake ROI and path-dependent drawdown by market/edge bucket."""
    missing = sorted(set(LEDGER_COLUMNS) - set(ledger.columns))
    if missing:
        raise ValueError("Historical betting ledger is missing columns: " + ", ".join(missing))
    rows: list[dict[str, object]] = []
    grouped = list(ledger.groupby(["market_key", "edge_bucket"], sort=True))
    grouped.extend(
        ((market_key, "ALL"), group)
        for market_key, group in ledger.groupby("market_key", sort=True)
    )
    for (market_key, edge_bucket), group in grouped:
        profits = group["profit_per_unit"].to_numpy(dtype=float)
        cumulative = np.cumsum(profits)
        peaks = np.maximum.accumulate(np.concatenate(([0.0], cumulative)))[:-1]
        drawdown = peaks - cumulative
        decisions = group["result"].ne("PUSH")
        wins = int(group["result"].eq("WIN").sum())
        rows.append({
            "market_key": market_key, "edge_bucket": edge_bucket, "bet_count": len(group),
            "win_count": wins, "loss_count": int(group["result"].eq("LOSS").sum()),
            "push_count": int(group["result"].eq("PUSH").sum()),
            "win_rate_excluding_pushes": wins / int(decisions.sum()) if decisions.any() else np.nan,
            "total_profit": float(profits.sum()), "roi_percent": float(profits.mean() * 100.0),
            "mean_edge": float(group["edge"].mean()),
            "clv_bet_count": int(group["clv_available"].sum()),
            "mean_clv": float(group.loc[group["clv_available"], "clv_value"].mean()),
            "maximum_drawdown_units": float(max(0.0, drawdown.max(initial=0.0))),
        })
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
