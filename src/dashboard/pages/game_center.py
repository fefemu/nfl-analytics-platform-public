"""Selected-matchup Game Center page."""

from html import escape
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from src.dashboard.components import (
    empty_state,
    probability_bar,
    probability_trend_badge,
    team_badge,
    tooltip_icon,
)
from src.dashboard.i18n import DEFAULT_LANGUAGE, Language, tr
from src.dashboard.view_models import (
    create_matchup_labels,
    classify_publication_candidates,
    format_decimal_odds,
    format_hungarian_kickoff,
    prepare_forward_candidates,
    select_preferred_market_sides,
)


def _hero(row: pd.Series, language: Language = DEFAULT_LANGUAGE) -> str:
    away = str(row["away_team"])
    home = str(row["home_team"])
    kickoff = format_hungarian_kickoff(row["gameday"], row["gametime"])
    week_label = f"{int(row['week'])}. HÉT" if language == "HU" else f"WEEK {int(row['week'])}"
    score_label = "VÁRHATÓ EREDMÉNY" if language == "HU" else "MODEL SCORE"
    margin_label = "VÁRHATÓ KÜLÖNBSÉG" if language == "HU" else "HOME MARGIN"
    total_label = "VÁRHATÓ ÖSSZPONTSZÁM" if language == "HU" else "MODEL TOTAL"
    away_trend = probability_trend_badge(
        row.get("away_probability_trend"), row.get("away_probability_change_pp"), language,
        previous_probability=row.get("away_previous_win_probability"),
        current_probability=row.get("away_win_probability"),
    )
    home_trend = probability_trend_badge(
        row.get("home_probability_trend"), row.get("home_probability_change_pp"), language,
        previous_probability=row.get("home_previous_win_probability"),
        current_probability=row.get("home_win_probability"),
    )
    return f"""
    <div class="nap-card nap-game-hero">
      <div class="nap-game-teams">
        <div class="nap-game-team">{team_badge(away, 58)}<span>{away}</span></div>
        <div class="nap-game-context">{week_label}<br>{kickoff}</div>
        <div class="nap-game-team home"><span>{home}</span>{team_badge(home, 58)}</div>
      </div>
      {probability_bar(away, row['away_win_probability'], home, row['home_win_probability'])}
      <div class="nap-game-trends"><span>{away_trend}</span><span>{home_trend}</span></div>
      <div class="nap-prediction-grid">
        <div class="nap-prediction-tile"><span>{score_label}</span><b>{row['implied_away_score']:.1f} – {row['implied_home_score']:.1f}</b></div>
        <div class="nap-prediction-tile"><span>{margin_label}</span><b>{row['predicted_home_margin']:+.1f}</b></div>
        <div class="nap-prediction-tile"><span>{total_label}</span><b>{row['predicted_total_points']:.1f}</b></div>
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
    favourite = str(row["home_team"] if row["home_win_probability"] >= 0.5 else row["away_team"])
    summary = (
        f"A jelenleg elérhető csapat-, irányító- és teljesítményadatok alapján "
        f"a modell {escape(favourite)} csapatát tartja esélyesebbnek."
        if language == "HU" else
        f"Based on the currently available team, quarterback and performance data, "
        f"the model considers {escape(favourite)} more likely to win."
    )
    is_fallback = "FALLBACK" in str(row.get("probability_prediction_mode", "")).upper()
    context = (
        "Egyes aktuális adatok még hiányosak, ezért a becslés a platform validált tartalékmodelljét használja."
        if language == "HU" else
        "Some current inputs are incomplete, so this estimate uses the platform's validated fallback model."
    ) if is_fallback else (
        "A becslés a platform aktuális, validált győzelmi modelljéből származik."
        if language == "HU" else
        "This estimate comes from the platform's current validated win-probability model."
    )
    st.markdown(
        f'<div class="nap-card"><div class="nap-panel-title">{headline}</div>'
        f'<div class="nap-narrative">{summary}</div><div class="nap-divider"></div>'
        f'<div class="nap-muted">{context}</div></div>',
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
        "A modell becslése és a fogadóirodák aktuális árazásának összehasonlítása. "
        "A pozitív Edge azt jelzi, hogy a modell nagyobb esélyt ad az adott "
        "kimenetelnek, mint a piac."
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
            status, status_class, edge_class = _edge_status(edge, language)
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
            model_probability = f'{offer["model_probability"]:.1%}'
            market_probability = f'{offer["market_probability"]:.1%}'
            edge_text = f"{edge:+.1f} pp"
            if language == "HU":
                model_probability = model_probability.replace(".", ",")
                market_probability = market_probability.replace(".", ",")
                edge_text = edge_text.replace(".", ",").replace("-", "−")
            card = (
                '<div class="nap-card nap-market-card">'
                f'<div class="nap-panel-title">{market_labels[market_key]}</div>'
                f'<div class="nap-candidate-market">{escape(_preferred_label(offer))}</div>'
                f'<div class="nap-market-edge {edge_class}">{edge_text}'
                f'{tooltip_icon(edge_help, accessible_label=edge_help, align="right")}</div>'
                f'<div class="nap-market-status {status_class}">{status}</div>'
                '<div class="nap-market-probabilities">'
                f'<span>{"Modell esélye" if language == "HU" else "Model probability"}'
                f'{tooltip_icon(model_help, accessible_label=model_help)}<b>{model_probability}</b></span>'
                f'<span>{"Piaci esély" if language == "HU" else "Market probability"}'
                f'{tooltip_icon(market_help, accessible_label=market_help, align="right")}<b>{market_probability}</b></span>'
                '</div>'
                f'<div class="nap-market-price">{"Legjobb odds" if language == "HU" else "Best odds"}'
                f'{tooltip_icon(odds_help, accessible_label=odds_help)} '
                f'<b>{odds}</b><span class="nap-market-book">{escape(str(offer["best_bookmaker_title"]))}</span></div>'
                '</div>'
            )
            st.markdown(card, unsafe_allow_html=True)
    st.caption(
        "A pozitív Edge nem jelent garantált nyereséget."
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


def _edge_status(edge: float, language: Language) -> tuple[str, str, str]:
    """Return a status aligned with the edge value shown to the user."""

    displayed_edge = round(float(edge), 1)
    if displayed_edge > 0:
        return (
            "Pozitív modell-előny" if language == "HU" else "Positive model edge",
            "nap-positive",
            "positive",
        )
    if displayed_edge < 0:
        return (
            "Negatív modell-előny" if language == "HU" else "Negative model edge",
            "nap-negative",
            "negative",
        )
    return (
        "Nincs modell-előny" if language == "HU" else "No model edge",
        "nap-neutral",
        "neutral",
    )


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
    st.markdown(_hero(row, language), unsafe_allow_html=True)

    left, right = st.columns([1, 1])
    with left:
        st.markdown(f"### {tr(language, 'why_model')}")
        _render_narrative(row, language)
    with right:
        _render_market(game_id, market_board, language)
