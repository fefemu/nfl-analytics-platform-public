from datetime import datetime, timezone

import pandas as pd
import pytest

from src.dashboard.view_models import (
    calculate_win_thresholds,
    classify_publication_candidates,
    create_matchup_labels,
    format_hungarian_kickoff,
    format_decimal_odds,
    format_refresh_timestamp,
    format_utc_timestamp_in_hungary,
    market_display,
    prepare_forward_candidates,
    prepare_simulation_standings,
    select_best_candidates,
    select_current_week,
    select_next_betting_week,
    select_weekly_highlights,
)


def create_board() -> pd.DataFrame:
    return pd.DataFrame({
        "game_id": ["future", "started", "future"],
        "week": [1, 1, 1],
        "commence_time": ["2026-09-10T20:00:00Z", "2026-08-01T20:00:00Z", "2026-09-10T20:00:00Z"],
        "fetched_at": ["2026-08-01T10:00:00Z"] * 3,
        "home_team": ["KC", "BUF", "KC"], "away_team": ["BUF", "KC", "BUF"],
        "market_key": ["h2h", "h2h", "h2h"], "market_name": ["Moneyline"] * 3,
        "market_line": [None] * 3, "outcome_name": ["KC", "BUF", "BUF"],
        "outcome_type": ["home", "home", "away"], "point": [None] * 3,
        "best_bookmaker_title": ["Book A", "Book B", "Book C"],
        "best_american_price": [-110, 120, 130], "best_decimal_odds": [1.91, 2.2, 2.3],
        "bookmaker_count": [4, 5, 3], "prediction_mode": ["PRIMARY", "PRIMARY", "FALLBACK"],
        "model_probability": [0.6, 0.5, 0.45],
        "probability_edge_percentage_points": [4.0, -1.0, 2.0],
        "expected_value_percent": [10.0, -2.0, 4.0],
        "positive_expected_value": [True, False, True],
    })


def test_prepare_forward_candidates_rejects_started_games() -> None:
    result = prepare_forward_candidates(create_board(), datetime(2026, 8, 14, tzinfo=timezone.utc))
    assert set(result["game_id"]) == {"future"}
    assert len(result) == 2


def test_prepare_forward_candidates_requires_schema() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        prepare_forward_candidates(pd.DataFrame({"game_id": []}))


def test_best_candidates_keeps_one_row_per_game_market() -> None:
    forward = prepare_forward_candidates(create_board(), datetime(2026, 8, 14, tzinfo=timezone.utc))
    result = select_best_candidates(forward)
    assert len(result) == 1
    assert result.iloc[0]["expected_value_percent"] == 10.0


def test_publication_guardrail_separates_extreme_signal() -> None:
    data = create_board().loc[lambda frame: frame["positive_expected_value"]].copy()
    data.loc[data.index[0], ["bookmaker_count", "model_probability",
        "probability_edge_percentage_points", "expected_value_percent"]] = [6, 0.60, 8.0, 15.0]
    data.loc[data.index[1], ["bookmaker_count", "model_probability",
        "probability_edge_percentage_points", "expected_value_percent"]] = [6, 0.60, 16.0, 30.0]

    result = classify_publication_candidates(data)

    assert result["publication_status"].tolist() == ["TOP_PICK", "RESEARCH_SIGNAL"]


def test_select_next_betting_week_uses_earliest_future_kickoff() -> None:
    board = pd.DataFrame({
        "week": [2, 1, 1, 3],
        "commence_time": [
            "2026-09-20T17:00:00Z",
            "2026-09-13T17:00:00Z",
            "2026-09-11T00:20:00Z",
            "2026-09-27T17:00:00Z",
        ],
    })

    week, result = select_next_betting_week(board)

    assert week == 1
    assert result["week"].tolist() == [1, 1]


def test_current_week_is_earliest_available() -> None:
    assert select_current_week(pd.DataFrame({"week": [3, 1, 2]})) == 1
    assert select_current_week(pd.DataFrame()) is None


def test_market_display_formats_spread_and_moneyline() -> None:
    moneyline = create_board().iloc[0]
    assert market_display(moneyline) == "Moneyline · KC"
    spread = moneyline.copy()
    spread["market_key"] = "spreads"
    spread["market_name"] = "Spread"
    spread["point"] = -3.5
    assert market_display(spread) == "Spread · KC -3.5"
    total = moneyline.copy()
    total["market_key"] = "totals"
    total["market_name"] = "Totals"
    total["outcome_name"] = "Under"
    total["point"] = 47.5
    assert market_display(total) == "Total · Under 47.5"


def test_create_matchup_labels_is_stable_and_readable() -> None:
    games = pd.DataFrame({
        "game_id": ["2026_01_BUF_KC"],
        "week": [1],
        "away_team": ["BUF"],
        "home_team": ["KC"],
        "gameday": ["2026-09-10"],
        "gametime": ["20:00"],
    })

    assert create_matchup_labels(games) == {
        "Week 1 · BUF @ KC · 2026-09-11 · 02:00 CEST": "2026_01_BUF_KC"
    }


def test_nflverse_schedule_time_converts_to_hungarian_next_day() -> None:
    assert format_hungarian_kickoff("2026-09-10", "20:20") == (
        "2026-09-11 · 02:20 CEST"
    )


def test_hungarian_kickoff_uses_local_ui_format_without_repeated_timezone() -> None:
    assert format_hungarian_kickoff(
        "2026-09-10", "20:20", "HU", include_timezone=False,
    ) == "2026.09.11. · 02:20"


def test_weekly_highlights_identify_business_metrics() -> None:
    games = pd.DataFrame({
        "away_team": ["BUF", "DAL", "SF"],
        "home_team": ["KC", "NYG", "LA"],
        "away_win_probability": [0.35, 0.49, 0.72],
        "home_win_probability": [0.65, 0.51, 0.28],
        "predicted_total_points": [47.0, 43.0, 51.5],
    })

    closest, favorite, highest_total = select_weekly_highlights(games)

    assert closest["away_team"] == "DAL"
    assert favorite["away_team"] == "SF"
    assert highest_total["away_team"] == "SF"


def test_utc_market_time_converts_to_hungarian_time() -> None:
    assert format_utc_timestamp_in_hungary("2026-09-10T00:20:00Z") == (
        "2026-09-10 · 02:20 CEST"
    )


def test_global_refresh_uses_budapest_time_and_hungarian_format() -> None:
    assert format_refresh_timestamp("2026-08-21T12:32:00Z", "HU") == (
        "Adatok frissítve: 2026.08.21. 14:32"
    )


def test_decimal_odds_uses_language_specific_separator() -> None:
    assert format_decimal_odds(1.91, "HU") == "1,91"
    assert format_decimal_odds(1.91, "EN") == "1.91"


def test_prepare_simulation_standings_ranks_and_formats_range() -> None:
    summary = pd.DataFrame({
        "team": ["KC", "BUF"], "games": [17, 17],
        "expected_wins": [10.4, 11.2], "expected_losses": [6.6, 5.8],
        "median_wins": [10.0, 11.0], "p10_wins": [8.0, 9.0],
        "p90_wins": [13.0, 14.0], "most_likely_wins": [10, 11],
        "expected_final_elo": [1580.0, 1610.0],
    })

    result = prepare_simulation_standings(summary)

    assert list(result["team"]) == ["BUF", "KC"]
    assert list(result["rank"]) == [1, 2]
    assert result.iloc[0]["p10_p90_range"] == "9–14"


def test_calculate_win_thresholds_uses_distribution_probability() -> None:
    distribution = pd.DataFrame({
        "team": ["BUF", "BUF", "BUF", "KC"],
        "wins": [8, 10, 12, 10],
        "probability": [0.2, 0.5, 0.3, 1.0],
    })

    assert calculate_win_thresholds(distribution, "BUF", (10, 12, 14)) == {
        10: pytest.approx(0.8),
        12: pytest.approx(0.3),
        14: pytest.approx(0.0),
    }
