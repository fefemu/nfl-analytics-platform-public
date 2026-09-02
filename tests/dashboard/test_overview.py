import inspect

import pandas as pd

from src.dashboard.pages.overview import _matchup_card, render_weekly_overview


def test_hungarian_matchup_card_is_localized_and_links_to_game_center() -> None:
    row = pd.Series({
        "game_id": "2026_01_BUF_KC",
        "week": 1,
        "gameday": "2026-09-10",
        "gametime": "20:20",
        "away_team": "BUF",
        "home_team": "KC",
        "away_win_probability": 0.42,
        "home_win_probability": 0.58,
        "implied_away_score": 22.6,
        "implied_home_score": 26.7,
        "predicted_home_margin": 4.1,
        "predicted_total_points": 49.3,
        "away_probability_trend": "DECREASE",
        "away_probability_change_pp": -1.2,
        "away_previous_win_probability": 0.432,
        "home_probability_trend": "INCREASE",
        "home_probability_change_pp": 1.2,
        "home_previous_win_probability": 0.568,
    })

    markup = _matchup_card(row, "HU")

    assert "href=" not in markup
    assert "1. HÉT · 2026.09.11. · 02:20" in markup
    assert "Várt eredmény" in markup
    assert "22,6 – 26,7" in markup
    assert "↓ -1,2%" in markup
    assert "↑ +1,2%" in markup
    assert "Csökkent" not in markup
    assert "Növekedett" not in markup
    assert markup.count("nap-tooltip-left") == 1
    assert markup.count("nap-tooltip-right") == 1


def test_weekly_summary_does_not_use_directional_metric_deltas() -> None:
    source = inspect.getsource(render_weekly_overview)

    assert "delta_color" not in source
    assert "summary[1].caption" in source
    assert "summary[2].caption" in source
    assert "summary[3].caption" in source
