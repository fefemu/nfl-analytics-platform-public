"""Forward-only Betting Board page."""

from datetime import datetime, timezone
from html import escape

import pandas as pd
import streamlit as st

from src.dashboard.components import empty_state, team_badge, tooltip_icon
from src.dashboard.i18n import DEFAULT_LANGUAGE, Language, tr
from src.dashboard.view_models import (
    classify_publication_candidates,
    format_decimal_odds,
    format_utc_timestamp_in_hungary,
    market_display,
    prepare_forward_candidates,
    select_best_candidates,
    select_next_betting_week,
    top_pick_criteria_text,
)


MARKET_OPTIONS = {
    "Összes piac": None,
    "All markets": None,
    "Moneyline": "h2h",
    "Spread": "spreads",
    "Total": "totals",
}


def _number(value: float, language: Language, digits: int = 1) -> str:
    result = f"{float(value):.{digits}f}"
    return result.replace(".", ",") if language == "HU" else result


def _candidate_card(row: pd.Series, language: Language) -> str:
    model_label = "Modell esélye" if language == "HU" else "Model probability"
    model_help = (
        "A modell által becsült valószínűség, hogy az adott kimenetel bekövetkezik."
        if language == "HU" else
        "The model-estimated probability that the selected outcome occurs."
    )
    edge_help = (
        "A modell becsült valószínűsége és a margin nélküli piaci valószínűség "
        "közötti abszolút különbség. A megjelenített % érték százalékpont-különbséget "
        "jelent, nem relatív százalékos változást."
        if language == "HU" else
        "Absolute difference between the model probability and the no-vig market probability. "
        "The displayed % value represents a percentage-point difference, not relative percentage change."
    )
    ev_help = (
        "A modell becslése alapján számított várható érték az adott oddson."
        if language == "HU" else
        "Theoretical long-run return at the displayed odds; not guaranteed profit."
    )
    kickoff = format_utc_timestamp_in_hungary(row["commence_time"])
    if language == "HU":
        kickoff = kickoff.replace("-", ".").replace(" ·", ". ·")
    decimal_odds = format_decimal_odds(row["best_decimal_odds"], language)
    return f"""
    <div class="nap-card nap-candidate-card">
      <div class="nap-candidate-teams">{team_badge(str(row['away_team']), 34)}
      <b>{escape(str(row['away_team']))} @ {escape(str(row['home_team']))}</b>{team_badge(str(row['home_team']), 34)}</div>
      <div class="nap-matchup-meta">{kickoff}</div>
      <div class="nap-candidate-market">{escape(market_display(row))}</div>
      <div class="nap-candidate-grid">
        <span>{model_label}{tooltip_icon(model_help, accessible_label=model_help)} <b>{row['model_probability']:.1%}</b></span>
        <span>Edge{tooltip_icon(edge_help, accessible_label=edge_help)} <b class="nap-positive">{_signed_number(row['probability_edge_percentage_points'], language, '%')}</b></span>
        <span>EV{tooltip_icon(ev_help, accessible_label=ev_help, align="right")} <b class="nap-positive">{_signed_number(row['expected_value_percent'], language, '%')}</b></span>
      </div>
      <div class="nap-candidate-footer"><span>{escape(str(row['best_bookmaker_title']))} · {decimal_odds}</span></div>
    </div>
    """


def _signed_number(value: float, language: Language, suffix: str) -> str:
    numeric = float(value)
    if round(numeric, 1) == 0:
        numeric = 0.0
    sign = "+" if numeric > 0 else ""
    return f"{sign}{_number(numeric, language)}{suffix}"


def _filter_candidates(
    board: pd.DataFrame,
    market_key: str | None,
    matchup: str | None,
) -> pd.DataFrame:
    filtered = classify_publication_candidates(board)
    filtered = filtered.loc[filtered["publication_eligible"]].copy()
    if market_key:
        filtered = filtered.loc[filtered["market_key"] == market_key]
    if matchup:
        filtered = filtered.loc[
            (filtered["away_team"] + " @ " + filtered["home_team"]) == matchup
        ]
    return filtered


def _render_board_explanation(language: Language) -> None:
    if language == "HU":
        st.info(
            "A **Top tippek** a következő NFL-hét azon piacait mutatják, ahol a modell "
            "a legnagyobb eltérést látja a fogadóirodák árazásához képest. A rangsorolás "
            "a modell által becsült valószínűség, a margin nélküli piaci valószínűség "
            "és a várható érték alapján történik."
        )
    else:
        st.info(
            "**Top picks** show next week's NFL markets where the model differs most "
            "from current bookmaker pricing. Ranking uses model probability, no-vig "
            "market probability and expected value."
        )


def render_betting_board(
    board: pd.DataFrame,
    language: Language = DEFAULT_LANGUAGE,
) -> None:
    if board.empty:
        empty_state(
            "Current market candidates are not available" if language == "EN" else "Az aktuális piaci ajánlatok jelenleg nem érhetők el",
            "The current forward betting board has not been generated yet."
            if language == "EN" else
            "A következő mérkőzések fogadási összehasonlítása még nem készült el.",
        )
        return

    forward = prepare_forward_candidates(board, now=datetime.now(timezone.utc))
    if forward.empty:
        empty_state(
            "No future market rows" if language == "EN" else "Nincs megjeleníthető jövőbeli piac",
            "The current snapshot contains no games that have not started."
            if language == "EN" else
            "A jelenlegi adatpillanatképben nincs még el nem kezdődött mérkőzés.",
        )
        return

    next_week, forward = select_next_betting_week(forward)
    latest_snapshot = forward["fetched_at"].max()
    snapshot_age = pd.Timestamp.now(tz="UTC") - latest_snapshot
    if snapshot_age > pd.Timedelta(hours=24):
        st.warning(
            (
                "ELAVULT ODDSOK — a megjelenített oddsok több mint 24 órásak. "
                "Fogadási döntés előtt frissítsd az adatokat."
                if language == "HU" else
                "STALE ODDS — displayed prices are more than 24 hours old. Refresh "
                "the data before making a betting decision."
            )
        )

    _render_board_explanation(language)

    market_labels = (
        {"Összes piac": None, "Moneyline": "h2h", "Spread": "spreads", "Total": "totals"}
        if language == "HU" else
        {"All markets": None, "Moneyline": "h2h", "Spread": "spreads", "Total": "totals"}
    )
    matchups = sorted((forward["away_team"] + " @ " + forward["home_team"]).unique())
    all_matchups = "Összes mérkőzés" if language == "HU" else "All matchups"

    first_row = st.columns(2)
    with first_row[0]:
        market_label = st.selectbox("Piac" if language == "HU" else "Market", tuple(market_labels))
    with first_row[1]:
        matchup_label = st.selectbox("Mérkőzés" if language == "HU" else "Matchup", (all_matchups, *matchups))
    filtered = _filter_candidates(
        forward,
        market_labels[market_label],
        None if matchup_label == all_matchups else matchup_label,
    )
    classified = filtered
    cards = select_best_candidates(
        classified.loc[classified["publication_eligible"]],
        positive_only=True,
    )

    st.caption(
        (
            f"Következő hét: **{next_week}. hét** · {len(cards)} Top tipp. "
            "Csak a következő aktuális hét, kezdés előtt rögzített oddsai jelennek meg."
        ) if language == "HU" else (
            f"Week {next_week} · {len(cards)} Top picks. "
            "Only pre-kickoff odds for the next upcoming week are shown."
        )
    )
    if classified.empty:
        empty_state(
            "No selected signals match these filters" if language == "EN" else "Nincs a szűrőknek megfelelő kiválasztott jelzés",
            "Choose another market or matchup."
            if language == "EN" else "Válassz másik piacot vagy mérkőzést.",
        )
        return

    if cards.empty:
        st.info(
            "A jelenlegi piacon nincs a kiválasztási feltételeknek megfelelő jelzés."
            if language == "HU" else
            "The current market has no signals matching the selection criteria."
        )

    st.subheader("Top tippek" if language == "HU" else "Top picks")
    top = cards.head(6)
    for start in range(0, len(top), 3):
        columns = st.columns(3)
        for offset, column in enumerate(columns):
            index = start + offset
            if index < len(top):
                with column:
                    st.markdown(
                        _candidate_card(top.iloc[index], language),
                        unsafe_allow_html=True,
                    )

    st.markdown("### " + ("Kiválasztott piaci jelzések" if language == "HU" else "Selected market signals"))
    st.caption(
        "A táblázat csak azokat a piacokat mutatja, amelyek megfelelnek a modell kiválasztási feltételeinek."
        if language == "HU" else
        "The table only shows markets that satisfy the model's selection criteria."
    )
    with st.expander("ⓘ Aktuális kiválasztási feltételek" if language == "HU" else "ⓘ Current selection criteria"):
        st.write(top_pick_criteria_text(language))
    detail = select_best_candidates(
        classified.loc[classified["publication_eligible"]], positive_only=True
    )
    if detail.empty:
        st.info(
            "A jelenlegi piacon nincs a kiválasztási feltételeknek megfelelő jelzés."
            if language == "HU" else
            "The current market has no signals matching the selection criteria."
        )
        return
    detail["matchup"] = detail["away_team"] + " @ " + detail["home_team"]
    detail["market"] = detail.apply(market_display, axis=1)
    detail["decimal_odds"] = detail["best_decimal_odds"].map(lambda value: format_decimal_odds(value, language))
    detail["edge_display"] = detail["probability_edge_percentage_points"].map(
        lambda value: _signed_number(value, language, "%")
    )
    st.dataframe(
        detail[["matchup", "market", "model_probability", "edge_display",
                "expected_value_percent", "decimal_odds", "best_bookmaker_title", "bookmaker_count"]],
        width="stretch",
        hide_index=True,
        column_config={
            "matchup": "Mérkőzés" if language == "HU" else "Matchup",
            "market": st.column_config.TextColumn(
                "Piac" if language == "HU" else "Market",
                help=("A vizsgált fogadási piac és kimenetel." if language == "HU" else "The selected betting market and outcome."),
            ),
            "model_probability": st.column_config.NumberColumn(
                "Modell %" if language == "HU" else "Model %", format="percent",
                help=("A modell által becsült valószínűség az adott kimenetelre." if language == "HU" else "Model-estimated probability of the outcome."),
            ),
            "edge_display": st.column_config.TextColumn(
                "Edge",
                help=(
                    "A modell becsült valószínűsége és a margin nélküli piaci valószínűség "
                    "közötti abszolút különbség. A megjelenített % érték százalékpont-különbséget "
                    "jelent, nem relatív százalékos változást."
                    if language == "HU" else
                    "Absolute difference between the model probability and the no-vig market probability. "
                    "The displayed % value represents a percentage-point difference, not relative percentage change."
                ),
            ),
            "expected_value_percent": st.column_config.NumberColumn(
                "EV", format="%.1f",
                help=("Az odds és a modell valószínűsége alapján számított elméleti várható hozam; nem garantált profit." if language == "HU" else "Theoretical expected return from odds and model probability; not guaranteed profit."),
            ),
            "decimal_odds": st.column_config.TextColumn(
                "Legjobb odds" if language == "HU" else "Best odds",
                help=("Az adatforrásban elérhető legkedvezőbb aktuális decimális odds." if language == "HU" else "Best available current decimal odds in the data source."),
            ),
            "best_bookmaker_title": "Fogadóiroda" if language == "HU" else "Bookmaker",
            "bookmaker_count": st.column_config.NumberColumn(
                "Irodák száma" if language == "HU" else "Bookmakers", format="%d",
                help=("Az összehasonlításhoz felhasznált fogadóirodák száma." if language == "HU" else "Number of bookmakers used for the comparison."),
            ),
        },
    )
