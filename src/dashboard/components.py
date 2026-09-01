"""Reusable Streamlit presentation components."""

from html import escape

import pandas as pd

import streamlit as st

from src.dashboard.team_branding import get_team_brand


def inject_styles(css: str) -> None:
    st.markdown(css, unsafe_allow_html=True)


def render_brand() -> None:
    st.markdown(
        '<div class="nap-brand"><span class="nap-mark">⌁</span>'
        '<span>NFL Analytics Platform</span></div>',
        unsafe_allow_html=True,
    )


def render_page_header(
    eyebrow: str,
    title: str,
    subtitle: str,
    refresh_label: str | None = None,
) -> None:
    refresh = (
        f'<div class="nap-refresh">{escape(refresh_label)}</div>'
        if refresh_label else ""
    )
    st.markdown(
        '<div class="nap-page-header"><div>'
        f'<div class="nap-eyebrow">{escape(eyebrow)}</div>'
        f'<div class="nap-title">{escape(title)}</div>'
        f'<div class="nap-subtitle">{escape(subtitle)}</div></div>'
        f'{refresh}</div>',
        unsafe_allow_html=True,
    )


def metric_tile(label: str, value: str, tone: str = "") -> None:
    safe_tone = tone if tone in {"green", "blue", "amber"} else ""
    st.markdown(
        '<div class="nap-card">'
        f'<div class="nap-metric-label">{escape(label)}</div>'
        f'<div class="nap-metric-value {safe_tone}">{escape(value)}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def status_pill(label: str, ready: bool) -> str:
    tone = "ready" if ready else "warning"
    return f'<span class="nap-pill {tone}">{escape(label)}</span>'


def tooltip_icon(
    help_text: str,
    *,
    accessible_label: str | None = None,
    align: str = "left",
) -> str:
    """Return a hover-, touch- and keyboard-accessible tooltip trigger."""

    safe_alignment = "right" if align == "right" else "left"
    label = accessible_label or help_text
    return (
        f'<span class="nap-tooltip nap-tooltip-{safe_alignment}">'
        '<button class="nap-tooltip-trigger" type="button" '
        f'aria-label="{escape(label)}">ⓘ</button>'
        f'<span class="nap-tooltip-content" role="tooltip">'
        f'{escape(help_text)}</span></span>'
    )


def team_badge(team_code: str, size: int = 42) -> str:
    brand = get_team_brand(team_code)
    if brand.logo_url:
        return (
            f'<span class="nap-team-identity" title="{escape(brand.display_name)}" '
            f'style="width:{size}px;height:{size}px;background:'
            f'linear-gradient(145deg,{brand.primary_color},{brand.secondary_color})">'
            f'<span class="nap-team-fallback" style="font-size:{max(11, size // 3)}px">'
            f'{escape(brand.code)}</span>'
            f'<img class="nap-team-logo" src="{escape(brand.logo_url)}" '
            f'alt="{escape(brand.display_name)} logo" loading="lazy" '
            f'onerror="this.style.display=\'none\'" '
            f'style="width:{size}px;height:{size}px" /></span>'
        )
    return (
        f'<span title="{escape(brand.display_name)}" style="display:inline-flex;'
        f'width:{size}px;height:{size}px;border-radius:50%;align-items:center;'
        f'justify-content:center;font-weight:850;font-size:{max(11, size // 3)}px;'
        f'color:#fff;background:linear-gradient(145deg,{brand.primary_color},'
        f'{brand.secondary_color});border:1px solid rgba(255,255,255,.2);'
        f'box-shadow:0 8px 20px rgba(0,0,0,.25)">{escape(brand.code)}</span>'
    )


def probability_bar(
    away_team: str,
    away_probability: float,
    home_team: str,
    home_probability: float,
) -> str:
    """Render a two-sided, accessible win-probability bar."""

    away_width = max(0.0, min(100.0, float(away_probability) * 100.0))
    home_width = max(0.0, min(100.0, float(home_probability) * 100.0))
    return (
        '<div class="nap-probability-labels">'
        f'<span>{escape(away_team)} <b>{away_width:.1f}%</b></span>'
        f'<span><b>{home_width:.1f}%</b> {escape(home_team)}</span></div>'
        '<div class="nap-probability-bar" role="img" '
        f'aria-label="{escape(away_team)} {away_width:.1f} percent, '
        f'{escape(home_team)} {home_width:.1f} percent">'
        f'<span class="away" style="width:{away_width:.3f}%"></span>'
        f'<span class="home" style="width:{home_width:.3f}%"></span></div>'
    )


def probability_trend_badge(
    direction: object,
    change_pp: object,
    language: str,
    *,
    compact: bool = False,
    previous_probability: object = None,
    current_probability: object = None,
    tooltip_align: str = "right",
) -> str:
    """Render an already classified probability trend with clear semantics."""

    normalized = str(direction).upper() if not pd.isna(direction) else "NEW"
    labels = {
        "HU": {
            "NEW": "Új előrejelzés",
            "UNCHANGED": "Lényegében változatlan",
            "INCREASE": "Növekedett",
            "DECREASE": "Csökkent",
        },
        "EN": {
            "NEW": "New prediction",
            "UNCHANGED": "Essentially unchanged",
            "INCREASE": "Increased",
            "DECREASE": "Decreased",
        },
    }
    symbols = {"NEW": "•", "UNCHANGED": "→", "INCREASE": "↑", "DECREASE": "↓"}
    tones = {"NEW": "new", "UNCHANGED": "neutral", "INCREASE": "increase", "DECREASE": "decrease"}
    tooltip = (
        "Modellváltozás — A modell győzelmi valószínűségének változása az előző "
        "sikeres, publikált modellfrissítéshez képest. A megjelenített % érték a "
        "valószínűség abszolút változását mutatja, nem relatív százalékos változást. "
        "A nyíl a modellbecslés változásának irányát jelzi, nem a fogadási értéket."
        if language == "HU" else
        "Model probability change — Change in model win probability compared with the previous "
        "successfully published model refresh. The displayed % value represents the absolute "
        "change in probability, not relative percentage growth. The arrow indicates the direction "
        "of the model estimate, not betting value."
    )
    numeric_change = None if pd.isna(change_pp) else float(change_pp)
    if normalized == "UNCHANGED" and numeric_change is not None:
        numeric_change = 0.0
    value = ""
    if normalized != "NEW" and numeric_change is not None:
        value = f"{numeric_change:+.1f}%" if normalized != "UNCHANGED" else "0.0%"
        if language == "HU":
            value = value.replace(".", ",")
    if (
        normalized != "NEW"
        and not pd.isna(previous_probability)
        and not pd.isna(current_probability)
        and numeric_change is not None
    ):
        previous = f"{float(previous_probability):.1%}"
        current = f"{float(current_probability):.1%}"
        change = value
        if language == "HU":
            previous = previous.replace(".", ",")
            current = current.replace(".", ",")
            tooltip += f" Előző: {previous} · Aktuális: {current} · Változás: {change}."
        else:
            tooltip += f" Previous: {previous} · Current: {current} · Change: {change}."
    label = labels.get(language, labels["EN"]).get(normalized, labels["EN"]["NEW"])
    if compact and normalized != "NEW":
        visible = f"{symbols.get(normalized, '•')} {value}"
    elif normalized == "NEW":
        visible = label
    elif normalized == "UNCHANGED":
        visible = f"{symbols[normalized]} {label} · {value}"
    else:
        visible = f"{symbols.get(normalized, '•')} {label} {value}"
    return (
        f'<span class="nap-probability-trend {tones.get(normalized, "new")}">'
        f'{escape(visible)}{tooltip_icon(tooltip, accessible_label=tooltip, align=tooltip_align)}</span>'
    )


def empty_state(title: str, message: str) -> None:
    st.markdown(
        '<div class="nap-card nap-empty">'
        f'<div class="nap-panel-title">{escape(title)}</div>'
        f'<div class="nap-muted">{escape(message)}</div></div>',
        unsafe_allow_html=True,
    )
