import pandas as pd

from src.dashboard.pages.betting import _candidate_card, _filter_candidates, _signed_number


def create_candidate() -> pd.Series:
    return pd.Series({
        "game_id": "2026_01_BUF_KC", "away_team": "BUF", "home_team": "KC",
        "commence_time": "2026-09-10T00:20:00Z", "market_key": "h2h",
        "market_name": "Moneyline", "outcome_name": "BUF", "point": None,
        "model_probability": 0.60, "probability_edge_percentage_points": 8.0,
        "expected_value_percent": 12.0, "best_decimal_odds": 2.10,
        "best_bookmaker_title": "Book A", "bookmaker_count": 4,
        "prediction_mode": "FALLBACK", "positive_expected_value": True,
    })


def test_hungarian_candidate_card_uses_decimal_odds_without_routing_label() -> None:
    markup = _candidate_card(create_candidate(), "HU")

    assert "Book A · 2,10" in markup
    assert "Modell esélye" in markup
    assert 'class="nap-tooltip-trigger"' in markup
    assert 'role="tooltip"' in markup
    assert "FALLBACK" not in markup
    assert "PRIMARY" not in markup
    assert "+8,0%" in markup
    assert "százalékpont-különbséget" in markup
    assert " pp" not in markup


def test_signed_number_normalizes_displayed_zero() -> None:
    assert _signed_number(0.04, "EN", "%") == "0.0%"
    assert _signed_number(-0.04, "EN", "%") == "0.0%"


def test_candidate_filters_apply_market_threshold_and_matchup() -> None:
    first = create_candidate()
    second = first.copy()
    second["away_team"] = "DAL"
    second["home_team"] = "NYG"
    second["probability_edge_percentage_points"] = 2.0
    first["bookmaker_count"] = 5
    board = pd.DataFrame([first, second])
    result = _filter_candidates(board, "h2h", "BUF @ KC")

    assert len(result) == 1
    assert result.iloc[0]["game_id"] == "2026_01_BUF_KC"
