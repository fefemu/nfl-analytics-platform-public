import pandas as pd

from src.dashboard.pages.game_center import _hero, _preferred_label


def test_game_center_hero_contains_predictions_and_team_logos() -> None:
    row = pd.Series({
        "away_team": "BUF",
        "home_team": "KC",
        "week": 1,
        "gameday": "2026-09-10",
        "gametime": "20:20",
        "away_win_probability": 0.443,
        "home_win_probability": 0.557,
        "implied_away_score": 24.1,
        "implied_home_score": 27.4,
        "predicted_home_margin": 3.3,
        "predicted_total_points": 51.5,
    })

    markup = _hero(row)

    assert "BUF 44.3 percent, KC 55.7 percent" in markup
    assert "24.1 – 27.4" in markup
    assert "HOME MARGIN" in markup
    assert markup.count("nap-team-logo") == 2


def test_preferred_market_label_formats_spread_and_total() -> None:
    assert _preferred_label(pd.Series({
        "market_key": "spreads", "outcome_name": "BUF", "point": 3.5,
    })) == "BUF +3.5"
    assert _preferred_label(pd.Series({
        "market_key": "totals", "outcome_name": "Under", "point": 47.5,
    })) == "Under 47.5"
