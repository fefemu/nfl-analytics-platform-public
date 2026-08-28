"""Forward-only Betting Board page."""

from datetime import datetime, timezone
from html import escape

import pandas as pd
import streamlit as st

from src.dashboard.components import empty_state, team_badge
from src.dashboard.i18n import DEFAULT_LANGUAGE, Language, tr
from src.dashboard.view_models import (
    TOP_PICK_CRITERIA,
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
    model_label = "Modell" if language == "HU" else "Model"
    model_help = (
        "A modell által becsült valószínűség, hogy az adott kimenetel bekövetkezik."
        if language == "HU" else
        "The model-estimated probability that the selected outcome occurs."
    )
    edge_help = (
        "A modell valószínűsége és a fogadóirodai margin nélküli piaci valószínűség közötti eltérés."
        if language == "HU" else
        "Difference between model probability and the bookmaker-margin-free market probability."
    )
    ev_help = (
        "Az adott odds mellett számított elméleti várható hozam. Például +10% EV sok hasonló, egyenként 100 egységnyi fogadásra átlagosan +10 egység elméleti eredményt jelent; nem garantált profit."
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
        <span title="{escape(model_help)}">{model_label} ⓘ <b>{row['model_probability']:.1%}</b></span>
        <span title="{escape(edge_help)}">Edge ⓘ <b class="nap-positive">{_signed_number(row['probability_edge_percentage_points'], language, ' pp')}</b></span>
        <span title="{escape(ev_help)}">EV ⓘ <b class="nap-positive">{_signed_number(row['expected_value_percent'], language, '%')}</b></span>
      </div>
      <div class="nap-candidate-footer"><span>{escape(str(row['best_bookmaker_title']))} · {decimal_odds}</span></div>
    </div>
    """


def _signed_number(value: float, language: Language, suffix: str) -> str:
    sign = "+" if float(value) >= 0 else ""
    return f"{sign}{_number(value, language)}{suffix}"


def _filter_candidates(
    board: pd.DataFrame,
    market_key: str | None,
    matchup: str | None,
    minimum_edge: float,
    minimum_ev: float,
    minimum_model_probability: float,
    minimum_books: int,
) -> pd.DataFrame:
    filtered = board.loc[
        (board["probability_edge_percentage_points"] >= minimum_edge)
        & (board["expected_value_percent"] >= minimum_ev)
        & (board["model_probability"] >= minimum_model_probability / 100.0)
        & (board["bookmaker_count"] >= minimum_books)
        & board["positive_expected_value"]
    ].copy()
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

    first_row = st.columns([1.0, 1.25, 1, 1])
    with first_row[0]:
        market_label = st.selectbox("Piac" if language == "HU" else "Market", tuple(market_labels))
    with first_row[1]:
        matchup_label = st.selectbox("Mérkőzés" if language == "HU" else "Matchup", (all_matchups, *matchups))
    with first_row[2]:
        minimum_edge = st.number_input("Minimum Edge (pp)", min_value=0.0, value=TOP_PICK_CRITERIA.minimum_edge_percentage_points, step=1.0)
    with first_row[3]:
        minimum_ev = st.number_input("Minimum EV (%)", min_value=0.0, value=TOP_PICK_CRITERIA.minimum_expected_value_percent, step=1.0)

    second_row = st.columns([1, 1, 2])
    with second_row[0]:
        minimum_model_probability = st.number_input(
            "Minimum modellvalószínűség (%)" if language == "HU" else "Minimum model probability (%)",
            min_value=0.0, max_value=100.0, value=TOP_PICK_CRITERIA.minimum_model_probability * 100.0, step=5.0,
            help=(
                "Alapértelmezésben csak olyan kimenetelek jelennek meg, amelyeket a modell legalább 50%-ban valószínűnek tart."
                if language == "HU" else
                "By default, only outcomes the model considers at least 50% likely are shown."
            ),
        )
    with second_row[1]:
        minimum_books = st.number_input(
            "Irodák minimális száma" if language == "HU" else "Minimum bookmakers",
            min_value=1, value=TOP_PICK_CRITERIA.minimum_bookmakers, step=1,
            help=(
                "Az adott piac összehasonlításához felhasznált fogadóirodák minimális száma."
                if language == "HU" else
                "Minimum number of bookmakers used to compare the market."
            ),
        )

    filtered = _filter_candidates(
        forward,
        market_labels[market_label],
        None if matchup_label == all_matchups else matchup_label,
        minimum_edge,
        minimum_ev,
        minimum_model_probability,
        minimum_books,
    )
    classified = classify_publication_candidates(filtered)
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
            "No candidates match these filters" if language == "EN" else "Nincs a szűrőknek megfelelő tipp",
            "Lower one of the thresholds to inspect more offers."
            if language == "EN" else
            "Csökkents valamelyik küszöbértéken további ajánlatok megtekintéséhez.",
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

    st.markdown("### " + ("Részletes piaci adatok" if language == "HU" else "Detailed market data"))
    detail_help = (
        "Csak a platform kiválasztási feltételeinek megfelelő piaci jelzések jelennek meg. "
        "A szűrés a modell becsült valószínűsége, a piachoz képesti eltérés (Edge), "
        "a várható érték (EV), valamint az elérhető odds- és bookmakeradatok alapján történik. "
        if language == "HU" else
        "Only market signals satisfying the platform selection criteria are shown. Selection uses "
        "model probability, Edge, EV and available odds and bookmaker data. "
    ) + top_pick_criteria_text(language)
    st.caption("ⓘ " + detail_help)
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
    st.dataframe(
        detail[["matchup", "market", "model_probability", "probability_edge_percentage_points",
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
            "probability_edge_percentage_points": st.column_config.NumberColumn(
                "Edge", format="%.1f",
                help=("A modell és a margin nélküli piac valószínűségének különbsége, százalékpontban." if language == "HU" else "Model probability minus no-vig market probability, in percentage points."),
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
