"""Selected-matchup Game Center page."""

from html import escape
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from src.dashboard.components import empty_state, probability_bar, status_pill, team_badge
from src.dashboard.i18n import DEFAULT_LANGUAGE, Language, tr
from src.dashboard.view_models import (
    create_matchup_labels,
    classify_publication_candidates,
    format_decimal_odds,
    format_hungarian_kickoff,
    prepare_forward_candidates,
    select_preferred_market_sides,
)


def _hero(row: pd.Series) -> str:
    away = str(row["away_team"])
    home = str(row["home_team"])
    kickoff = format_hungarian_kickoff(row["gameday"], row["gametime"])
    return f"""
    <div class="nap-card nap-game-hero">
      <div class="nap-game-teams">
        <div class="nap-game-team">{team_badge(away, 58)}<span>{away}</span></div>
        <div class="nap-game-context">WEEK {int(row['week'])}<br>{kickoff}</div>
        <div class="nap-game-team home"><span>{home}</span>{team_badge(home, 58)}</div>
      </div>
      {probability_bar(away, row['away_win_probability'], home, row['home_win_probability'])}
      <div class="nap-prediction-grid">
        <div class="nap-prediction-tile"><span>MODEL SCORE</span><b>{row['implied_away_score']:.1f} – {row['implied_home_score']:.1f}</b></div>
        <div class="nap-prediction-tile"><span>HOME MARGIN</span><b>{row['predicted_home_margin']:+.1f}</b></div>
        <div class="nap-prediction-tile"><span>MODEL TOTAL</span><b>{row['predicted_total_points']:.1f}</b></div>
      </div>
    </div>
    """


def _render_narrative(row: pd.Series, language: Language) -> None:
    if "summary_en" not in row.index or pd.isna(row.get("summary_en")):
        empty_state(
            "Model narrative is not available" if language == "EN" else "A model narrative nem elérhető",
            "The numerical prediction remains valid; rebuild narratives to add its plain-language explanation."
            if language == "EN" else
            "A számszerű prediction érvényes; építsd újra a narrative-okat a közérthető magyarázathoz.",
        )
        return
    suffix = "hu" if language == "HU" else "en"
    headline = escape(str(row.get(f"headline_{suffix}", "")))
    summary = escape(str(row.get(f"summary_{suffix}", "")))
    context = escape(str(row.get(f"model_context_{suffix}", "")))
    factor = row.get(f"top_factor_{suffix}")
    factor_html = ""
    if pd.notna(factor):
        factor_html = f'<div class="nap-muted"><b>Top factor:</b> {escape(str(factor))}</div>'
    st.markdown(
        f'<div class="nap-card"><div class="nap-panel-title">{headline}</div>'
        f'<div class="nap-narrative">{summary}</div><div class="nap-divider"></div>'
        f'<div class="nap-muted">{context}</div>{factor_html}</div>',
        unsafe_allow_html=True,
    )


def _render_market(
    game_id: str,
    market_board: pd.DataFrame,
    language: Language,
) -> None:
    if market_board.empty or "game_id" not in market_board:
        empty_state(
            "Market comparison is not available" if language == "EN" else "A market összehasonlítás nem elérhető",
            "Run the odds pipeline to add current prices and model-versus-market edges."
            if language == "EN" else
            "Futtasd az odds pipeline-t az aktuális árak és model-versus-market edge-ek betöltéséhez.",
        )
        return
    forward = prepare_forward_candidates(
        market_board,
        now=datetime.now(timezone.utc),
    )
    rows = forward.loc[forward["game_id"].astype(str) == game_id].copy()
    if rows.empty:
        empty_state(
            "No market for this matchup" if language == "EN" else "Nincs market ehhez a meccshez",
            "No matched current bookmaker offers exist for the selected game."
            if language == "EN" else "Nincs párosított aktuális bookmaker ajánlat a kiválasztott meccshez.",
        )
        return
    preferred = select_preferred_market_sides(rows)
    classified = classify_publication_candidates(preferred)
    title = "Piaci értékelés" if language == "HU" else "Market assessment"
    explanation = (
        "A modell által preferált oldal markettípusonként, összevetve az aktuális "
        "fogadóirodai árazással. A modell által valószínűbbnek tartott kimenetel "
        "nem feltétlenül jelent fogadási value-t."
        if language == "HU" else
        "The model-preferred side in each market, compared with current bookmaker pricing. "
        "The more likely model outcome is not necessarily a value bet."
    )
    st.markdown(f"#### {title}")
    st.caption(explanation)
    market_labels = {"h2h": "MONEYLINE", "spreads": "SPREAD", "totals": "TOTAL"}
    by_market = {row["market_key"]: row for _, row in classified.iterrows()}
    columns = st.columns(3)
    for column, market_key in zip(columns, ("h2h", "spreads", "totals"), strict=True):
        with column:
            if market_key not in by_market:
                missing_text = "Nincs elérhető piaci adat" if language == "HU" else "Market data unavailable"
                st.markdown(
                    f'<div class="nap-card"><div class="nap-panel-title">{market_labels[market_key]}</div>'
                    f'<div class="nap-muted">{missing_text}</div></div>',
                    unsafe_allow_html=True,
                )
                continue
            offer = by_market[market_key]
            edge = float(offer["probability_edge_percentage_points"])
            if bool(offer["publication_eligible"]):
                status = "VALUE"
                status_class = "nap-positive"
            elif edge > 0:
                status = "KIS ELŐNY" if language == "HU" else "SMALL EDGE"
                status_class = ""
            else:
                status = "NINCS VALUE" if language == "HU" else "NO VALUE"
                status_class = "nap-negative"
            model_help = (
                "A modell által becsült valószínűség az adott kimenetelre."
                if language == "HU" else "The model-estimated probability of this outcome."
            )
            market_help = (
                "A fogadóirodai margintól megtisztított piaci valószínűség."
                if language == "HU" else "The market probability after removing bookmaker margin."
            )
            edge_help = (
                "A modell és a no-vig piaci valószínűség különbsége százalékpontban."
                if language == "HU" else "Model probability minus no-vig market probability, in percentage points."
            )
            odds_help = (
                "Az adott kimenetelhez jelenleg elérhető legjobb ár a betöltött fogadóirodák között."
                if language == "HU" else "The best currently available price among loaded bookmakers."
            )
            odds = format_decimal_odds(offer["best_decimal_odds"], language)
            card = (
                '<div class="nap-card">'
                f'<div class="nap-panel-title">{market_labels[market_key]}</div>'
                f'<div class="nap-candidate-market">{escape(_preferred_label(offer))}</div>'
                '<div class="nap-divider"></div>'
                f'<div title="{escape(model_help)}">{"Modell" if language == "HU" else "Model"} ⓘ <b>{offer["model_probability"]:.1%}</b></div>'
                f'<div title="{escape(market_help)}">{"Piac" if language == "HU" else "Market"} ⓘ <b>{offer["market_probability"]:.1%}</b></div>'
                f'<div title="{escape(edge_help)}">Edge ⓘ <b>{edge:+.1f} pp</b></div>'
                f'<div title="{escape(odds_help)}">{"Legjobb odds" if language == "HU" else "Best odds"} ⓘ <b>{odds}</b></div>'
                f'<div class="nap-muted">{escape(str(offer["best_bookmaker_title"]))}</div>'
                f'<div class="nap-divider"></div><b class="{status_class}">{status}</b>'
                '</div>'
            )
            st.markdown(card, unsafe_allow_html=True)
    st.caption(
        "A pozitív Edge nem garantált profitot jelent."
        if language == "HU" else "A positive Edge does not guarantee profit."
    )


def _preferred_label(offer: pd.Series) -> str:
    """Format one already selected model-preferred market side."""

    market_key = str(offer["market_key"])
    outcome = str(offer["outcome_name"])
    point = offer.get("point")
    if market_key == "h2h" or pd.isna(point):
        return outcome
    if market_key == "spreads":
        return f"{outcome} {float(point):+g}"
    return f"{outcome} {float(point):g}"


def render_game_center(
    games: pd.DataFrame,
    market_board: pd.DataFrame,
    language: Language = DEFAULT_LANGUAGE,
    preferred_game_id: str | None = None,
) -> None:
    """Render one selected future matchup and its supporting context."""

    if games.empty:
        empty_state(
            "Game Center predictions are not available" if language == "EN" else "A Game Center predictionök nem elérhetők",
            "Build the modeling pipeline to populate matchup-level analysis."
            if language == "EN" else "Futtasd a modeling pipeline-t a meccsszintű elemzés felépítéséhez.",
        )
        return
    labels = create_matchup_labels(games)
    if preferred_game_id is not None and preferred_game_id in set(labels.values()):
        preferred_label = next(
            label for label, game_id in labels.items()
            if game_id == preferred_game_id
        )
        st.session_state["game_center_matchup"] = preferred_label
    selected_label = st.selectbox(
        tr(language, "select_matchup"), tuple(labels), key="game_center_matchup",
    )
    game_id = labels[selected_label]
    row = games.loc[games["game_id"].astype(str) == game_id].iloc[0]
    st.markdown(_hero(row), unsafe_allow_html=True)

    probability_fallback = "FALLBACK" in str(row.get("probability_prediction_mode", "")).upper()
    spread_fallback = "FALLBACK" in str(row.get("spread_prediction_mode", "")).upper()
    totals_fallback = "FALLBACK" in str(row.get("totals_prediction_mode", "")).upper()
    modes = st.columns(3)
    modes[0].markdown(status_pill("PROBABILITY FALLBACK" if probability_fallback else "PROBABILITY PRIMARY", not probability_fallback), unsafe_allow_html=True)
    modes[1].markdown(status_pill("SPREAD FALLBACK" if spread_fallback else "SPREAD PRIMARY", not spread_fallback), unsafe_allow_html=True)
    modes[2].markdown(status_pill("TOTALS FALLBACK" if totals_fallback else "TOTALS PRIMARY", not totals_fallback), unsafe_allow_html=True)

    left, right = st.columns([1, 1])
    with left:
        st.markdown(f"### {tr(language, 'why_model')}")
        _render_narrative(row, language)
    with right:
        _render_market(game_id, market_board, language)

    with st.expander(tr(language, "technical_routing")):
        st.json({
            "game_id": game_id,
            "probability_model": str(row.get("probability_model_name", "Unavailable")),
            "probability_mode": str(row.get("probability_prediction_mode", "Unavailable")),
            "spread_mode": str(row.get("spread_prediction_mode", "Unavailable")),
            "totals_mode": str(row.get("totals_prediction_mode", "Unavailable")),
        })
