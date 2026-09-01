from src.dashboard.components import (
    probability_bar,
    probability_trend_badge,
    team_badge,
    tooltip_icon,
)
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


def test_probability_trend_badge_localizes_direction_and_disclaimer() -> None:
    increase = probability_trend_badge(
        "INCREASE", 1.24, "HU",
        previous_probability=0.603,
        current_probability=0.6154,
    )
    new = probability_trend_badge("NEW", None, "EN")
    compact_new = probability_trend_badge("NEW", None, "HU", compact=True)

    assert "↑ Növekedett +1,2%" in increase
    assert "nem relatív százalékos változást" in increase
    assert "Előző: 60,3% · Aktuális: 61,5% · Változás: +1,2%" in increase
    assert "New prediction" in new
    assert "Új előrejelzés" in compact_new
    assert "0.0" not in new


def test_probability_trend_never_renders_signed_zero() -> None:
    positive_zero = probability_trend_badge("UNCHANGED", 0.04, "EN", compact=True)
    negative_zero = probability_trend_badge("UNCHANGED", -0.04, "EN")

    assert "→ 0.0%" in positive_zero
    assert "Essentially unchanged · 0.0%" in negative_zero
    assert "+0.0%" not in positive_zero + negative_zero
    assert "-0.0%" not in positive_zero + negative_zero
    assert "pp" not in positive_zero + negative_zero


def test_probability_trend_supports_inward_tooltip_alignment() -> None:
    left = probability_trend_badge(
        "UNCHANGED", 0.0, "EN", compact=True, tooltip_align="left"
    )
    right = probability_trend_badge(
        "UNCHANGED", 0.0, "EN", compact=True, tooltip_align="right"
    )

    assert "nap-tooltip-left" in left
    assert "nap-tooltip-right" in right


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
