import pandas as pd

from src.dashboard.pages.game_center import _edge_status, _hero, _preferred_label


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
        "away_probability_trend": "UNCHANGED",
        "away_probability_change_pp": -0.2,
        "away_previous_win_probability": 0.445,
        "home_probability_trend": "UNCHANGED",
        "home_probability_change_pp": 0.2,
        "home_previous_win_probability": 0.555,
    })

    markup = _hero(row, "HU")

    assert "BUF 44.3 percent, KC 55.7 percent" in markup
    assert "24.1 – 27.4" in markup
    assert "VÁRHATÓ KÜLÖNBSÉG" in markup
    assert "nem két önálló csapatpontszám" in markup
    assert "1. HÉT" in markup
    assert markup.count("nap-team-logo") == 2
    assert markup.count("Lényegében változatlan") == 2
    assert markup.count("0,0%") >= 2
    assert "+0,0%" not in markup
    assert "-0,0%" not in markup


def test_preferred_market_label_formats_spread_and_total() -> None:
    assert _preferred_label(pd.Series({
        "market_key": "spreads", "outcome_name": "BUF", "point": 3.5,
    })) == "BUF +3.5"
    assert _preferred_label(pd.Series({
        "market_key": "totals", "outcome_name": "Under", "point": 47.5,
    })) == "Under 47.5"


def test_edge_status_distinguishes_positive_negative_and_displayed_zero() -> None:
    assert _edge_status(18.5, "EN") == (
        "Model above market", "nap-positive", "positive",
    )
    assert _edge_status(-19.5, "EN") == (
        "Model below market", "nap-negative", "negative",
    )
    assert _edge_status(-0.04, "HU") == (
        "Piaccal egyező becslés", "nap-neutral", "neutral",
    )


def test_game_center_source_uses_percent_for_visible_edge() -> None:
    from pathlib import Path

    source = Path("src/dashboard/pages/game_center.py").read_text(encoding="utf-8")
    assert 'edge_text = f"{edge:+.1f}%"' in source
    assert 'edge_text = f"{edge:+.1f} pp"' not in source
