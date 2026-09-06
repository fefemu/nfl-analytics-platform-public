"""Settle locked forward selections and summarize live performance."""

from __future__ import annotations

import math

import duckdb
import numpy as np
import pandas as pd


SETTLEMENT_TABLE = "analytics.forward_tip_settlement"
SUMMARY_TABLE = "analytics.forward_performance_summary"
SETTLEMENT_COLUMNS = (
    "entry_archive_key", "game_id", "season", "week", "commence_time",
    "home_team", "away_team", "market_key", "outcome_name", "outcome_type",
    "entry_fetched_at", "entry_point", "entry_bookmaker", "entry_decimal_odds",
    "entry_model_probability", "entry_expected_value_percent",
    "entry_model_name", "entry_model_version", "home_score", "away_score",
    "actual_home_margin", "actual_total", "settlement_status", "result",
    "profit_units", "is_probability_score_eligible", "brier_loss", "log_loss",
    "latest_fetched_at", "latest_decimal_odds", "latest_point",
    "price_movement_implied_probability_pp", "entry_line_advantage_points",
    "is_clv", "market_movement_direction",
)
SUMMARY_COLUMNS = (
    "season", "week", "market_key", "selection_scope", "sample_scope", "tracked_count",
    "settled_count", "pending_count", "win_count", "loss_count", "push_count",
    "win_rate", "average_decimal_odds", "total_profit_units", "roi_percent",
    "maximum_drawdown_units", "brier_score", "log_loss", "clv_count",
    "positive_clv_count", "positive_clv_rate",
    "average_price_movement_probability_pp", "average_line_advantage_points",
)


def _settle(comparison: float, decimal_odds: float) -> tuple[str, float]:
    if comparison > 1e-9:
        return "WIN", decimal_odds - 1.0
    if comparison < -1e-9:
        return "LOSS", -1.0
    return "PUSH", 0.0


def create_forward_settlement(
    entries: pd.DataFrame,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    """Join locked entry selections to completed games and settle one-unit bets."""
    entry_required = {
        "entry_archive_key", "game_id", "season", "week", "commence_time",
        "home_team", "away_team", "market_key", "outcome_name", "outcome_type",
        "entry_fetched_at", "entry_point", "entry_bookmaker", "entry_decimal_odds",
        "entry_model_probability", "entry_expected_value_percent",
        "entry_model_name", "entry_model_version", "is_clv",
        "market_movement_direction", "latest_fetched_at", "latest_decimal_odds",
        "latest_point", "price_movement_implied_probability_pp",
        "entry_line_advantage_points",
    }
    schedule_required = {"game_id", "home_score", "away_score", "is_completed"}
    missing = sorted(entry_required - set(entries.columns))
    if missing:
        raise ValueError("Forward entries are missing columns: " + ", ".join(missing))
    missing = sorted(schedule_required - set(schedule.columns))
    if missing:
        raise ValueError("Schedule is missing columns: " + ", ".join(missing))
    if entries["entry_archive_key"].duplicated().any():
        raise ValueError("Forward entries contain duplicate archive keys.")
    if schedule["game_id"].duplicated().any():
        raise ValueError("Schedule contains duplicate games.")

    games = schedule.loc[:, list(schedule_required)].copy()
    rows = entries.merge(games, on="game_id", how="left", validate="many_to_one")
    output: list[dict[str, object]] = []
    for row in rows.itertuples(index=False):
        completed = bool(row.is_completed) and pd.notna(row.home_score) and pd.notna(row.away_score)
        result = None
        profit = np.nan
        brier = np.nan
        log_loss = np.nan
        score_eligible = False
        home_score = int(row.home_score) if completed else None
        away_score = int(row.away_score) if completed else None
        margin = home_score - away_score if completed else None
        total = home_score + away_score if completed else None
        if completed:
            side = str(row.outcome_type).lower()
            if row.market_key == "h2h":
                comparison = margin if side == "home" else -margin
            elif row.market_key == "spreads":
                comparison = (margin if side == "home" else -margin) + float(row.entry_point)
            elif row.market_key == "totals":
                comparison = total - float(row.entry_point) if side == "over" else float(row.entry_point) - total
            else:
                raise ValueError(f"Unsupported forward market: {row.market_key}")
            result, profit = _settle(comparison, float(row.entry_decimal_odds))
            if row.market_key == "h2h" and result != "PUSH":
                probability = min(max(float(row.entry_model_probability), 1e-15), 1.0 - 1e-15)
                outcome = 1.0 if result == "WIN" else 0.0
                brier = (probability - outcome) ** 2
                log_loss = -(outcome * math.log(probability) + (1.0 - outcome) * math.log(1.0 - probability))
                score_eligible = True
        output.append({
            **{column: getattr(row, column) for column in entry_required},
            "home_score": home_score, "away_score": away_score,
            "actual_home_margin": margin, "actual_total": total,
            "settlement_status": "SETTLED" if completed else "PENDING",
            "result": result, "profit_units": profit,
            "is_probability_score_eligible": score_eligible,
            "brier_loss": brier, "log_loss": log_loss,
        })
    return pd.DataFrame(output, columns=SETTLEMENT_COLUMNS).sort_values(
        ["commence_time", "game_id", "market_key", "outcome_type"]
    ).reset_index(drop=True)


def _maximum_drawdown(profits: pd.Series) -> float:
    values = profits.dropna().to_numpy(dtype=float)
    if not len(values):
        return 0.0
    cumulative = np.cumsum(values)
    peaks = np.maximum.accumulate(np.concatenate(([0.0], cumulative)))[:-1]
    return float(max(0.0, (peaks - cumulative).max(initial=0.0)))


def create_forward_performance_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    """Create overall, market and week summaries from the settlement ledger."""
    missing = sorted(set(SETTLEMENT_COLUMNS) - set(ledger.columns))
    if missing:
        raise ValueError("Forward settlement is missing columns: " + ", ".join(missing))
    scopes: list[tuple[int | None, int | None, str, str, pd.DataFrame]] = []
    for season, season_rows in ledger.groupby("season", sort=True):
        scopes.append((int(season), None, "ALL", "SEASON", season_rows))
        for market, market_rows in season_rows.groupby("market_key", sort=True):
            scopes.append((int(season), None, str(market), "SEASON_MARKET", market_rows))
        for week, week_rows in season_rows.groupby("week", sort=True):
            scopes.append((int(season), int(week), "ALL", "WEEK", week_rows))
            for market, market_rows in week_rows.groupby("market_key", sort=True):
                scopes.append((int(season), int(week), str(market), "WEEK_MARKET", market_rows))
    summaries: list[dict[str, object]] = []
    scoped_groups = []
    for season, week, market, scope, group in scopes:
        scoped_groups.extend((
            (season, week, market, scope, "ALL_TRACKED", group),
            (season, week, market, scope, "SETTLED_ONLY", group.loc[group["settlement_status"].eq("SETTLED")]),
            (season, week, market, scope, "CLV_ELIGIBLE", group.loc[group["is_clv"]]),
        ))
    for season, week, market, scope, sample_scope, group in scoped_groups:
        settled = group.loc[group["settlement_status"].eq("SETTLED")]
        decisions = settled.loc[settled["result"].isin(["WIN", "LOSS"])]
        scored = settled.loc[settled["is_probability_score_eligible"]]
        clv = group.loc[group["is_clv"]]
        summaries.append({
            "season": season, "week": week, "market_key": market,
            "selection_scope": scope, "sample_scope": sample_scope,
            "tracked_count": len(group),
            "settled_count": len(settled), "pending_count": len(group) - len(settled),
            "win_count": int(settled["result"].eq("WIN").sum()),
            "loss_count": int(settled["result"].eq("LOSS").sum()),
            "push_count": int(settled["result"].eq("PUSH").sum()),
            "win_rate": float(decisions["result"].eq("WIN").mean()) if len(decisions) else np.nan,
            "average_decimal_odds": float(settled["entry_decimal_odds"].mean()) if len(settled) else np.nan,
            "total_profit_units": float(settled["profit_units"].sum()) if len(settled) else 0.0,
            "roi_percent": float(settled["profit_units"].mean() * 100.0) if len(settled) else np.nan,
            "maximum_drawdown_units": _maximum_drawdown(settled["profit_units"]),
            "brier_score": float(scored["brier_loss"].mean()) if len(scored) else np.nan,
            "log_loss": float(scored["log_loss"].mean()) if len(scored) else np.nan,
            "clv_count": len(clv),
            "positive_clv_count": int(clv["market_movement_direction"].eq("POSITIVE").sum()),
            "positive_clv_rate": float(clv["market_movement_direction"].eq("POSITIVE").mean()) if len(clv) else np.nan,
            "average_price_movement_probability_pp": float(clv["price_movement_implied_probability_pp"].mean()) if len(clv) else np.nan,
            "average_line_advantage_points": float(clv["entry_line_advantage_points"].mean()) if clv["entry_line_advantage_points"].notna().any() else np.nan,
        })
    return pd.DataFrame(summaries, columns=SUMMARY_COLUMNS)


def persist_forward_performance(
    connection: duckdb.DuckDBPyConnection,
    ledger: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Atomically replace derived settlement and summary tables."""
    connection.execute("CREATE SCHEMA IF NOT EXISTS analytics")
    connection.register("_forward_settlement", ledger)
    connection.register("_forward_summary", summary)
    try:
        connection.execute(f"""
            CREATE OR REPLACE TABLE {SETTLEMENT_TABLE} AS
            SELECT * REPLACE (CAST("result" AS VARCHAR) AS "result")
            FROM _forward_settlement
        """)
        connection.execute(f"CREATE OR REPLACE TABLE {SUMMARY_TABLE} AS SELECT * FROM _forward_summary")
    finally:
        connection.unregister("_forward_settlement")
        connection.unregister("_forward_summary")


def validate_forward_performance(connection: duckdb.DuckDBPyConnection) -> None:
    """Reject impossible settlement or aggregation output."""
    invalid = connection.execute(f"""
        SELECT COUNT(*) FROM {SETTLEMENT_TABLE}
        WHERE settlement_status NOT IN ('PENDING', 'SETTLED')
           OR (settlement_status = 'PENDING' AND ("result" IS NOT NULL OR profit_units IS NOT NULL))
           OR (settlement_status = 'SETTLED' AND "result" NOT IN ('WIN', 'LOSS', 'PUSH'))
           OR ("result" = 'LOSS' AND profit_units <> -1.0)
           OR ("result" = 'PUSH' AND profit_units <> 0.0)
           OR ("result" = 'WIN' AND ABS(profit_units - (entry_decimal_odds - 1.0)) > 0.000001)
           OR (is_probability_score_eligible AND (market_key <> 'h2h' OR "result" = 'PUSH'))
           OR (NOT is_probability_score_eligible AND (brier_loss IS NOT NULL OR log_loss IS NOT NULL))
    """).fetchone()[0]
    duplicates = connection.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT entry_archive_key FROM {SETTLEMENT_TABLE} GROUP BY 1 HAVING COUNT(*) > 1
        )
    """).fetchone()[0]
    if invalid or duplicates:
        raise RuntimeError("Forward performance validation failed.")
