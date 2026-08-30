from src.dashboard.components import probability_bar, team_badge, tooltip_icon
from src.dashboard.styles import APP_CSS


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


def test_mobile_sidebar_reopen_control_remains_visible_and_touch_sized() -> None:
    mobile_css = APP_CSS.split("@media (max-width: 760px)", maxsplit=1)[1]

    assert '[data-testid="collapsedControl"]' in mobile_css
    assert '[data-testid="stSidebarCollapsedControl"]' in mobile_css
    assert '[data-testid="stSidebarCollapseButton"]' in mobile_css
    assert '[data-testid="stToolbar"]' in mobile_css
    assert "display:flex !important" in mobile_css
    assert "position:fixed !important" in mobile_css
    assert "z-index:1000000 !important" in mobile_css
    assert "width:2.75rem !important" in mobile_css
    assert "height:2.75rem !important" in mobile_css


def test_tooltip_icon_is_touch_and_keyboard_accessible() -> None:
    markup = tooltip_icon(
        'Model probability < market probability',
        accessible_label='Explain "Model probability"',
        align="right",
    )

    assert 'class="nap-tooltip nap-tooltip-right"' in markup
    assert '<button class="nap-tooltip-trigger" type="button"' in markup
    assert 'aria-label="Explain &quot;Model probability&quot;"' in markup
    assert 'role="tooltip"' in markup
    assert "Model probability &lt; market probability" in markup


def test_candidate_grid_styles_do_not_leak_into_nested_tooltips() -> None:
    assert ".nap-candidate-grid > span {" in APP_CSS
    assert ".nap-candidate-grid > span > b {" in APP_CSS
    assert ".nap-candidate-grid span {" not in APP_CSS
    assert ".nap-candidate-card:hover,.nap-candidate-card:focus-within" in APP_CSS
    assert "z-index:10010" in APP_CSS
