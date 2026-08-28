"""Pure, tested transformations from analytics outputs to public UI rows."""

from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


NFL_SCHEDULE_TIMEZONE = ZoneInfo("America/New_York")
HUNGARIAN_TIMEZONE = ZoneInfo("Europe/Budapest")


@dataclass(frozen=True)
class TopPickCriteria:
    """Single source of truth for public Top pick selection."""

    minimum_edge_percentage_points: float = 3.0
    minimum_expected_value_percent: float = 0.0
    minimum_model_probability: float = 0.50
    minimum_bookmakers: int = 5
    maximum_edge_percentage_points: float = 10.0
    maximum_expected_value_percent: float = 20.0


TOP_PICK_CRITERIA = TopPickCriteria()


def hungarian_kickoff_timestamp(
    gameday: object,
    gametime: object,
) -> pd.Timestamp:
    """Convert nflverse's US-Eastern schedule time to Budapest time."""

    source = pd.Timestamp(f"{pd.Timestamp(gameday).date()} {str(gametime)[:5]}")
    return source.tz_localize(NFL_SCHEDULE_TIMEZONE).tz_convert(HUNGARIAN_TIMEZONE)


def format_hungarian_kickoff(
    gameday: object,
    gametime: object,
    language: str = "EN",
    include_timezone: bool = True,
) -> str:
    """Return an explicit Hungarian local kickoff label."""

    timestamp = hungarian_kickoff_timestamp(gameday, gametime)
    if language == "HU":
        label = timestamp.strftime("%Y.%m.%d. · %H:%M")
    else:
        label = timestamp.strftime("%Y-%m-%d · %H:%M")
    return f"{label} {timestamp.strftime('%Z')}" if include_timezone else label


def format_utc_timestamp_in_hungary(value: object) -> str:
    """Format an absolute market timestamp in Budapest time."""

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.tz_convert(HUNGARIAN_TIMEZONE).strftime(
        "%Y-%m-%d · %H:%M %Z"
    )


def format_refresh_timestamp(value: object, language: str) -> str:
    """Format one global data-refresh timestamp in Budapest local time."""

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    local = timestamp.tz_convert(HUNGARIAN_TIMEZONE)
    if language == "HU":
        return f"Adatok frissítve: {local.strftime('%Y.%m.%d. %H:%M')}"
    return f"Data updated: {local.strftime('%Y-%m-%d %H:%M %Z')}"


def format_decimal_odds(value: object, language: str) -> str:
    """Format decimal odds consistently for the public UI."""

    odds = float(value)
    formatted = f"{odds:.2f}"
    return formatted.replace(".", ",") if language == "HU" else formatted


FORWARD_BOARD_REQUIRED = {
    "game_id", "week", "commence_time", "fetched_at", "home_team",
    "away_team", "market_key", "market_name", "market_line", "outcome_name",
    "outcome_type", "point", "best_bookmaker_title", "best_american_price",
    "best_decimal_odds", "bookmaker_count", "prediction_mode",
    "model_probability", "probability_edge_percentage_points",
    "expected_value_percent", "positive_expected_value",
}


def prepare_forward_candidates(
    board: pd.DataFrame,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Keep only market rows that were captured and viewed before kickoff."""

    missing = sorted(FORWARD_BOARD_REQUIRED - set(board.columns))
    if missing:
        raise ValueError("Betting board is missing columns: " + ", ".join(missing))
    if board.empty:
        return board.copy()

    viewed_at = pd.Timestamp(now or datetime.now(timezone.utc))
    if viewed_at.tzinfo is None:
        viewed_at = viewed_at.tz_localize("UTC")
    else:
        viewed_at = viewed_at.tz_convert("UTC")

    result = board.copy()
    result["commence_time"] = pd.to_datetime(result["commence_time"], utc=True)
    result["fetched_at"] = pd.to_datetime(result["fetched_at"], utc=True)
    result = result.loc[
        (result["commence_time"] > viewed_at)
        & (result["commence_time"] > result["fetched_at"])
    ].copy()
    result["positive_expected_value"] = result["positive_expected_value"].astype(bool)
    return result.sort_values(
        ["expected_value_percent", "bookmaker_count", "commence_time"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)


def select_best_candidates(
    board: pd.DataFrame,
    positive_only: bool = True,
) -> pd.DataFrame:
    """Return one strongest line per game and market for the public card view."""

    result = board.copy()
    if positive_only:
        result = result.loc[result["positive_expected_value"]]
    if result.empty:
        return result
    result = result.sort_values(
        ["expected_value_percent", "bookmaker_count"],
        ascending=[False, False],
        kind="stable",
    )
    return result.drop_duplicates(["game_id", "market_key"], keep="first").reset_index(drop=True)


def classify_publication_candidates(
    board: pd.DataFrame,
    criteria: TopPickCriteria = TOP_PICK_CRITERIA,
) -> pd.DataFrame:
    """Flag candidates that satisfy the complete public Top pick rule."""

    required = {
        "bookmaker_count", "model_probability",
        "probability_edge_percentage_points", "expected_value_percent",
        "positive_expected_value",
    }
    missing = sorted(required - set(board.columns))
    if missing:
        raise ValueError("Betting candidates are missing columns: " + ", ".join(missing))

    result = board.copy()
    result["publication_eligible"] = (
        result["positive_expected_value"].astype(bool)
        & result["bookmaker_count"].ge(criteria.minimum_bookmakers)
        & result["model_probability"].ge(criteria.minimum_model_probability)
        & result["probability_edge_percentage_points"].ge(
            criteria.minimum_edge_percentage_points
        )
        & result["expected_value_percent"].ge(
            criteria.minimum_expected_value_percent
        )
        & result["probability_edge_percentage_points"].le(
            criteria.maximum_edge_percentage_points
        )
        & result["expected_value_percent"].le(
            criteria.maximum_expected_value_percent
        )
    )
    result["publication_status"] = np.where(
        result["publication_eligible"], "TOP_PICK", "NOT_SELECTED"
    )
    return result


def top_pick_criteria_text(language: str) -> str:
    """Describe the live criteria from the same object used for selection."""

    criteria = TOP_PICK_CRITERIA
    probability = criteria.minimum_model_probability * 100.0
    if language == "HU":
        return (
            f"Aktuális feltételek: legalább {probability:.0f}% modellvalószínűség, "
            f"{criteria.minimum_edge_percentage_points:g}–"
            f"{criteria.maximum_edge_percentage_points:g} pp Edge, "
            f"{criteria.minimum_expected_value_percent:g}–"
            f"{criteria.maximum_expected_value_percent:g}% EV és legalább "
            f"{criteria.minimum_bookmakers} fogadóiroda."
        )
    return (
        f"Current criteria: at least {probability:.0f}% model probability, "
        f"{criteria.minimum_edge_percentage_points:g}–"
        f"{criteria.maximum_edge_percentage_points:g} pp Edge, "
        f"{criteria.minimum_expected_value_percent:g}–"
        f"{criteria.maximum_expected_value_percent:g}% EV and at least "
        f"{criteria.minimum_bookmakers} bookmakers."
    )


def select_preferred_market_sides(board: pd.DataFrame) -> pd.DataFrame:
    """Select one model-preferred side for each market using a liquid line."""

    required = {
        "market_key", "point", "model_probability", "bookmaker_count",
        "probability_edge_percentage_points", "expected_value_percent",
    }
    missing = sorted(required - set(board.columns))
    if missing:
        raise ValueError("Market preferences are missing columns: " + ", ".join(missing))
    selected: list[pd.Series] = []
    for market_key in ("h2h", "spreads", "totals"):
        market = board.loc[board["market_key"] == market_key].copy()
        if market.empty:
            continue
        if market_key == "spreads":
            market["_line_group"] = pd.to_numeric(
                market["point"], errors="coerce"
            ).abs().round(6)
        elif market_key == "totals":
            market["_line_group"] = pd.to_numeric(
                market["point"], errors="coerce"
            ).round(6)
        else:
            market["_line_group"] = "moneyline"
        liquidity = (
            market.groupby("_line_group", dropna=False)["bookmaker_count"]
            .max()
            .sort_values(ascending=False, kind="stable")
        )
        canonical_line = liquidity.index[0]
        line = market.loc[market["_line_group"] == canonical_line].copy()
        line = line.sort_values(
            ["model_probability", "bookmaker_count"],
            ascending=[False, False],
            kind="stable",
        )
        preferred = line.iloc[0].copy()
        preferred["market_probability"] = (
            float(preferred["model_probability"])
            - float(preferred["probability_edge_percentage_points"]) / 100.0
        )
        selected.append(preferred)
    if not selected:
        return board.iloc[0:0].copy()
    return pd.DataFrame(selected).reset_index(drop=True)


def select_next_betting_week(board: pd.DataFrame) -> tuple[int | None, pd.DataFrame]:
    """Keep the week containing the earliest future kickoff on the board."""

    if board.empty:
        return None, board.copy()
    if "week" not in board or "commence_time" not in board:
        raise ValueError("Betting board requires week and commence_time columns.")
    ordered = board.copy()
    ordered["week"] = pd.to_numeric(ordered["week"], errors="coerce")
    ordered["commence_time"] = pd.to_datetime(ordered["commence_time"], utc=True)
    ordered = ordered.dropna(subset=["week", "commence_time"])
    if ordered.empty:
        return None, ordered
    next_week = int(ordered.sort_values("commence_time", kind="stable").iloc[0]["week"])
    return next_week, ordered.loc[ordered["week"] == next_week].reset_index(drop=True)


def select_current_week(games: pd.DataFrame) -> int | None:
    """Select the earliest week represented by current production predictions."""

    if games.empty or "week" not in games:
        return None
    weeks = pd.to_numeric(games["week"], errors="coerce").dropna()
    return int(weeks.min()) if not weeks.empty else None


def select_weekly_highlights(
    week_games: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Select closest matchup, strongest favorite and highest projected Total."""

    required = {
        "away_win_probability", "home_win_probability",
        "predicted_total_points",
    }
    missing = sorted(required - set(week_games.columns))
    if missing:
        raise ValueError("Weekly games are missing columns: " + ", ".join(missing))
    if week_games.empty:
        raise ValueError("Weekly highlights require at least one game.")
    closest = week_games.loc[
        (week_games["home_win_probability"] - 0.5).abs().idxmin()
    ]
    favorite_strength = week_games[
        ["away_win_probability", "home_win_probability"]
    ].max(axis=1)
    favorite = week_games.loc[favorite_strength.idxmax()]
    highest_total = week_games.loc[week_games["predicted_total_points"].idxmax()]
    return closest, favorite, highest_total


def create_matchup_labels(games: pd.DataFrame) -> dict[str, str]:
    """Return stable display labels mapped to game identifiers."""

    required = {
        "game_id",
        "week",
        "away_team",
        "home_team",
        "gameday",
        "gametime",
    }
    missing = sorted(required - set(games.columns))
    if missing:
        raise ValueError("Game Center data is missing columns: " + ", ".join(missing))
    labels: dict[str, str] = {}
    for row in games.itertuples(index=False):
        kickoff_label = format_hungarian_kickoff(row.gameday, row.gametime)
        label = f"Week {int(row.week)} · {row.away_team} @ {row.home_team} · {kickoff_label}"
        labels[label] = str(row.game_id)
    return labels


SIMULATION_SUMMARY_REQUIRED = {
    "team", "games", "expected_wins", "expected_losses", "median_wins",
    "p10_wins", "p90_wins", "most_likely_wins", "expected_final_elo",
}


def prepare_simulation_standings(summary: pd.DataFrame) -> pd.DataFrame:
    """Create ranked dashboard standings from Monte Carlo team summaries."""

    missing = sorted(SIMULATION_SUMMARY_REQUIRED - set(summary.columns))
    if missing:
        raise ValueError("Simulation summary is missing columns: " + ", ".join(missing))
    result = summary.copy().sort_values(
        ["expected_wins", "expected_final_elo", "team"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    result.insert(0, "rank", result.index + 1)
    result["p10_p90_range"] = result.apply(
        lambda row: f"{row['p10_wins']:.0f}–{row['p90_wins']:.0f}",
        axis=1,
    )
    return result


def calculate_win_thresholds(
    distribution: pd.DataFrame,
    team: str,
    thresholds: tuple[int, ...] = (8, 10, 12, 14),
) -> dict[int, float]:
    """Calculate selected at-least-win probabilities for one team."""

    required = {"team", "wins", "probability"}
    missing = sorted(required - set(distribution.columns))
    if missing:
        raise ValueError("Win distribution is missing columns: " + ", ".join(missing))
    team_rows = distribution.loc[distribution["team"] == team]
    if team_rows.empty:
        return {threshold: 0.0 for threshold in thresholds}
    return {
        threshold: float(
            team_rows.loc[team_rows["wins"] >= threshold, "probability"].sum()
        )
        for threshold in thresholds
    }


def market_display(row: pd.Series) -> str:
    """Return a concise market/outcome label for a candidate card."""

    market = str(row["market_key"])
    outcome = str(row["outcome_name"])
    point = row.get("point")
    if market == "h2h" or pd.isna(point):
        return f"Moneyline · {outcome}"
    signed = f"{float(point):+g}" if market == "spreads" else f"{float(point):g}"
    market_name = "Total" if market == "totals" else str(row["market_name"])
    return f"{market_name} · {outcome} {signed}"
