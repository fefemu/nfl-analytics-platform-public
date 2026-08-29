import pandas as pd

from src.dashboard.pages.overview import _matchup_card


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
    })

    markup = _matchup_card(row, "HU")

    assert "?language=HU&amp;page=GAMES&amp;game_id=2026_01_BUF_KC" in markup
    assert 'target="_self"' in markup
    assert "1. HÉT · 2026.09.11. · 02:20" in markup
    assert "Várt eredmény" in markup
    assert "22,6 – 26,7" in markup
    assert "Meccs részletei" in markup
