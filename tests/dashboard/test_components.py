from src.dashboard.components import (
    metric_tile,
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
    assert "--nap-team-accent:#E31837" in markup
    assert "previousElementSibling.style.display='flex'" in markup


def test_loaded_team_logo_does_not_show_fallback_text_behind_it() -> None:
    markup = team_badge("NYG")

    assert 'class="nap-team-fallback" aria-hidden="true"' in markup
    assert "onerror=" in markup
    assert "onload=" not in markup


def test_unknown_team_badge_uses_local_badge_only() -> None:
    markup = team_badge("XYZ")

    assert "nap-team-logo" not in markup
    assert ">XYZ</span>" in markup


def test_team_logo_css_uses_neutral_surface_and_safe_inset() -> None:
    assert "background:linear-gradient(145deg,#f8fafc,#dce5ef)" in APP_CSS
    assert ".nap-team-fallback { display:none;" in APP_CSS
    assert ".nap-team-logo { position:absolute; inset:8%;" in APP_CSS
    assert "overflow:hidden" in APP_CSS


def test_roster_header_does_not_reveal_hidden_logo_fallback() -> None:
    assert ".nap-roster-hero span" not in APP_CSS
    assert ".nap-roster-identity span,.nap-roster-identity small" in APP_CSS


def test_depth_chart_group_anchor_icons_are_hidden() -> None:
    assert ".nap-depth-group h4 a { display:none !important; }" in APP_CSS


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
    assert '[data-testid="stSidebar"][aria-expanded="true"]' in mobile_css
    assert '[data-testid="stHeader"] button[kind="headerNoPadding"]' not in mobile_css
    assert '[data-testid="stToolbar"] {' not in mobile_css
    assert "display:flex !important" in mobile_css
    assert "position:fixed !important" in mobile_css
    assert "z-index:1000000 !important" in mobile_css
    assert "pointer-events:auto !important" in mobile_css
    assert "min-width:2.75rem !important" in mobile_css
    assert "min-height:2.75rem !important" in mobile_css


def test_metric_tile_uses_shared_info_tooltip(monkeypatch) -> None:
    rendered: list[str] = []
    monkeypatch.setattr(
        "src.dashboard.components.st.markdown",
        lambda markup, **_: rendered.append(markup),
    )

    metric_tile("Simulations", "10,000", help_text="Complete simulated seasons.")

    assert "nap-metric-label" in rendered[0]
    assert "ⓘ" in rendered[0]
    assert "Complete simulated seasons." in rendered[0]


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


def test_tooltip_typography_does_not_inherit_condensed_parent_styles() -> None:
    assert 'font:500 .74rem/1.55 "Segoe UI",Arial,sans-serif' in APP_CSS
    assert "letter-spacing:normal !important" in APP_CSS
    assert "word-spacing:normal !important" in APP_CSS
    assert "text-transform:none" in APP_CSS


def test_candidate_grid_styles_do_not_leak_into_nested_tooltips() -> None:
    assert ".nap-candidate-grid > span {" in APP_CSS
    assert ".nap-candidate-grid > span > b {" in APP_CSS
    assert ".nap-candidate-grid span {" not in APP_CSS
    assert ".nap-candidate-card:hover,.nap-candidate-card:focus-within" in APP_CSS
    assert "z-index:10010" in APP_CSS
