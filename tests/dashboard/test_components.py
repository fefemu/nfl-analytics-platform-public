from src.dashboard.components import probability_bar, team_badge


def test_known_team_badge_uses_remote_logo_with_fallback() -> None:
    markup = team_badge("KC")

    assert "nap-team-logo" in markup
    assert "https://a.espncdn.com/" in markup
    assert "nap-team-fallback" in markup
    assert ">KC</span>" in markup


def test_unknown_team_badge_uses_local_badge_only() -> None:
    markup = team_badge("XYZ")

    assert "nap-team-logo" not in markup
    assert ">XYZ</span>" in markup


def test_probability_bar_contains_accessible_team_probabilities() -> None:
    markup = probability_bar("BUF", 0.443, "KC", 0.557)

    assert "BUF 44.3 percent, KC 55.7 percent" in markup
    assert "width:44.300%" in markup
    assert "width:55.700%" in markup
