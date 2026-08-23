"""Selected-matchup Game Center page."""

from html import escape
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from src.dashboard.components import empty_state, probability_bar, status_pill, team_badge
from src.dashboard.i18n import DEFAULT_LANGUAGE, Language, tr
from src.dashboard.view_models import (
    create_matchup_labels,
    format_hungarian_kickoff,
    market_display,
    prepare_forward_candidates,
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
    rows = rows.sort_values("expected_value_percent", ascending=False).head(9)
    rendered = ['<div class="nap-card"><div class="nap-panel-title">Best current market comparisons</div>']
    for _, offer in rows.iterrows():
        rendered.append(
            '<div class="nap-market-row">'
            f'<b>{escape(market_display(offer))}</b>'
            f'<span>Model {offer["model_probability"]:.1%}</span>'
            f'<span>Edge {offer["probability_edge_percentage_points"]:+.1f} pp</span>'
            f'<span>{escape(str(offer["best_bookmaker_title"]))} · {int(offer["best_american_price"]):+d}</span>'
            '</div>'
        )
    rendered.append('</div>')
    st.markdown("".join(rendered), unsafe_allow_html=True)


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
        st.markdown(f"### {tr(language, 'market_comparison')}")
        _render_market(game_id, market_board, language)

    with st.expander(tr(language, "technical_routing")):
        st.json({
            "game_id": game_id,
            "probability_model": str(row.get("probability_model_name", "Unavailable")),
            "probability_mode": str(row.get("probability_prediction_mode", "Unavailable")),
            "spread_mode": str(row.get("spread_prediction_mode", "Unavailable")),
            "totals_mode": str(row.get("totals_prediction_mode", "Unavailable")),
        })
