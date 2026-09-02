"""Monte Carlo season outlook page."""

from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.components import empty_state, metric_tile, team_badge, tooltip_icon
from src.dashboard.i18n import DEFAULT_LANGUAGE, Language, tr
from src.dashboard.view_models import (
    calculate_win_thresholds,
    prepare_simulation_standings,
)


def _number(value: float, language: Language, digits: int = 1) -> str:
    formatted = f"{float(value):.{digits}f}"
    return formatted.replace(".", ",") if language == "HU" else formatted


def _leader_cards(standings: pd.DataFrame, language: Language = "EN") -> str:
    cards = ['<div class="nap-sim-leaders">']
    label = "várható győzelem" if language == "HU" else "expected wins"
    for row in standings.head(5).itertuples(index=False):
        cards.append(
            '<div class="nap-card nap-sim-leader">'
            f'{team_badge(str(row.team), 38)}<div><span class="nap-sim-rank">#{int(row.rank)}</span>'
            f'<b>{escape(str(row.team))}</b><strong>{_number(row.expected_wins, language)}</strong>'
            f'<span class="nap-muted"> {label}</span></div></div>'
        )
    cards.append('</div>')
    return "".join(cards)


def _distribution_chart(
    team: str,
    distribution: pd.DataFrame,
    expected: float,
    language: Language = "EN",
) -> go.Figure:
    team_distribution = distribution.loc[distribution["team"] == team].sort_values("wins")
    wins_word = "győzelem" if language == "HU" else "wins"
    expected_label = "Várható" if language == "HU" else "Expected"
    figure = go.Figure(go.Bar(
        x=team_distribution["wins"],
        y=team_distribution["probability"],
        marker={"color": "#42a5ff", "line": {"color": "#75c2ff", "width": 1}},
        hovertemplate=f"%{{x}} {wins_word}: %{{y:.1%}}<extra></extra>",
    ))
    figure.add_vline(
        x=expected,
        line_dash="dash",
        line_color="#36e39a",
        annotation_text=f"{expected_label}: {_number(expected, language)}",
        annotation_font_color="#36e39a",
    )
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        height=350,
        xaxis_title=("Alapszakasz-győzelmek" if language == "HU" else "Regular-season wins"),
        yaxis_title=("Valószínűség" if language == "HU" else "Probability"),
        yaxis_tickformat=".0%",
        showlegend=False,
    )
    return figure


def _methodology_copy(language: Language, simulation_count: int) -> str:
    count = f"{simulation_count:,}".replace(",", " ") if language == "HU" else f"{simulation_count:,}"
    if language == "HU":
        return (
            f"A rendszer minden alapszakasz-mérkőzéshez meghatározza a csapatok "
            f"győzelmi valószínűségét, majd ezek alapján **{count} alkalommal** "
            "leszimulálja a teljes szezont. A fő, Dinamikus Elo módban a csapatok "
            "Elo-értéke minden szimulált mérkőzés után frissül, ezért egy korábbi "
            "eredmény a későbbi meccsek esélyeit is módosíthatja. Az eredmények "
            "eloszlásából számítjuk a várható győzelmeket, a leggyakoribb kimenetelt "
            "és a bizonytalansági tartományokat."
        )
    return (
        f"The system assigns a win probability to every regular-season matchup and "
        f"then simulates the full season **{count} times**. In the primary Dynamic Elo "
        "mode, team Elo ratings update after every simulated game, so earlier results "
        "can change later matchup probabilities. The resulting distribution provides "
        "expected wins, the most common outcome and uncertainty ranges."
    )


def render_season_simulator(
    summary: pd.DataFrame,
    distribution: pd.DataFrame,
    benchmark: pd.DataFrame,
    language: Language = DEFAULT_LANGUAGE,
) -> None:
    """Render league standings and selected-team uncertainty."""

    if summary.empty or distribution.empty:
        empty_state(
            "Season simulation is not available" if language == "EN" else "A szezonszimuláció jelenleg nem érhető el",
            "The current season outlook has not been generated yet."
            if language == "EN" else
            "Az aktuális szezon-előrejelzés még nem készült el.",
        )
        return

    standings = prepare_simulation_standings(summary)
    simulation_count = int(summary["simulation_count"].max()) if "simulation_count" in summary else 0

    with st.expander(
        "How does the simulation work?" if language == "EN" else "Hogyan működik a szimuláció?",
    ):
        st.markdown(_methodology_copy(language, simulation_count))

    metrics = st.columns(3)
    with metrics[0]:
        metric_tile(tr(language, "teams"), str(len(standings)))
    with metrics[1]:
        metric_tile(
            tr(language, "simulations"),
            f"{simulation_count:,}".replace(",", " ") if language == "HU" else f"{simulation_count:,}",
            help_text=(
                "Ennyi teljes szezont futtat le a Monte Carlo-szimuláció."
                if language == "HU" else
                "Number of complete seasons generated by the Monte Carlo simulation."
            ),
        )
    leader = standings.iloc[0]
    with metrics[2]:
        metric_tile(
            "Highest expected wins" if language == "EN" else "Legmagasabb várható győzelemszám",
            f"{leader['team']} · {_number(leader['expected_wins'], language)}",
        )

    st.markdown("### " + ("Expected-wins leaders" if language == "EN" else "Várható győzelmek – élmezőny"))
    st.markdown(_leader_cards(standings, language), unsafe_allow_html=True)

    left, right = st.columns([1.05, 1.45])
    with left:
        table_help = (
            "Várható: a győzelmek átlaga. Medián: a szimulációk fele ez alatt, fele e fölött zárult. "
            "P10–P90: a kimenetek középső 80%-a. Módusz: a leggyakoribb győzelemszám."
            if language == "HU" else
            "Expected: average wins. Median: half of simulations finished below and half above. "
            "P10–P90: middle 80% of outcomes. Mode: most frequent win total."
        )
        st.markdown(
            '<div class="nap-heading-with-help"><h3>'
            + ("League outlook" if language == "EN" else "Teljes liga előrejelzése")
            + "</h3>"
            + tooltip_icon(table_help, align="left")
            + "</div>",
            unsafe_allow_html=True,
        )
        table = standings[["rank", "team", "expected_wins", "median_wins", "p10_p90_range", "most_likely_wins"]].copy()
        table["expected_wins"] = table["expected_wins"].round(1)
        table["median_wins"] = table["median_wins"].round(1)
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            column_config={
                "rank": st.column_config.NumberColumn("Helyezés" if language == "HU" else "Rank", format="%d"),
                "team": "Csapat" if language == "HU" else "Team",
                "expected_wins": st.column_config.NumberColumn(
                    "Várható" if language == "HU" else "Expected", format="%.1f",
                ),
                "median_wins": st.column_config.NumberColumn(
                    "Medián" if language == "HU" else "Median", format="%.1f",
                ),
                "p10_p90_range": st.column_config.TextColumn("P10–P90"),
                "most_likely_wins": st.column_config.NumberColumn(
                    "Módusz" if language == "HU" else "Mode", format="%d",
                ),
            },
        )
    with right:
        st.markdown("### " + ("Team win distribution" if language == "EN" else "Győzelmek várható eloszlása"))
        team = st.selectbox(tr(language, "team"), tuple(standings["team"]))
        selected = standings.loc[standings["team"] == team].iloc[0]
        outlook_label = "szezonkilátásai" if language == "HU" else "season outlook"
        expected_label = "várható győzelem" if language == "HU" else "expected wins"
        st.markdown(
            '<div class="nap-team-sim-header">'
            f'{team_badge(team, 54)}<div><h3>{escape(team)} {outlook_label}</h3>'
            f'<span>{_number(selected["expected_wins"], language)} {expected_label} · '
            f'P10–P90: {selected["p10_wins"]:.0f}–{selected["p90_wins"]:.0f}</span></div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            _distribution_chart(team, distribution, float(selected["expected_wins"]), language),
            width="stretch",
            config={"displayModeBar": False},
        )
        thresholds = calculate_win_thresholds(distribution, team)
        threshold_columns = st.columns(len(thresholds))
        for column, (wins, probability) in zip(threshold_columns, thresholds.items(), strict=True):
            label = f"Legalább {wins} győzelem" if language == "HU" else f"At least {wins} wins"
            column.metric(label, f"{probability:.1%}")

        if not benchmark.empty and team in set(benchmark["team"]):
            comparison = benchmark.loc[benchmark["team"] == team].iloc[0]
            st.markdown("#### " + ("Dynamic and frozen Elo comparison" if language == "EN" else "Dinamikus és rögzített Elo összehasonlítása"))
            compare = st.columns(3)
            dynamic_help = (
                "A csapatok Elo-értéke minden szimulált mérkőzés után frissül, így a későbbi meccsek esélyei a korábbi szimulált eredményektől is függenek."
                if language == "HU" else
                "Team Elo ratings update after every simulated game, so later probabilities depend on earlier simulated results."
            )
            frozen_help = (
                "A csapatok kiinduló Elo-értéke a teljes szimulált szezon során változatlan marad."
                if language == "HU" else
                "Starting Elo ratings remain unchanged throughout the simulated season."
            )
            difference_help = (
                "Megmutatja, mennyivel változik a várható győzelemszám, ha a csapaterősség a szimuláció során is frissül."
                if language == "HU" else
                "Change in expected wins when team strength updates during the simulated season."
            )
            with compare[0]:
                metric_tile(
                    "Dinamikus Elo" if language == "HU" else "Dynamic Elo",
                    _number(comparison["dynamic_expected_wins"], language, 2),
                    help_text=dynamic_help,
                )
            with compare[1]:
                metric_tile(
                    "Rögzített Elo" if language == "HU" else "Frozen Elo",
                    _number(comparison["frozen_expected_wins"], language, 2),
                    help_text=frozen_help,
                )
            with compare[2]:
                metric_tile(
                    "Eltérés" if language == "HU" else "Difference",
                    ("+" if comparison["expected_wins_delta"] >= 0 else "")
                    + _number(comparison["expected_wins_delta"], language, 2),
                    help_text=difference_help,
                )

    st.caption(
        "Simulations are probabilistic estimates, not guaranteed season records."
        if language == "EN" else
        "A szimulációk valószínűségi becslések, nem garantált szezoneredmények."
    )
