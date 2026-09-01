"""Weekly Overview page."""

import pandas as pd
import streamlit as st

from src.dashboard.components import empty_state, probability_trend_badge, team_badge
from src.dashboard.i18n import DEFAULT_LANGUAGE, Language, tr
from src.dashboard.view_models import format_hungarian_kickoff, select_current_week
from src.dashboard.view_models import select_weekly_highlights


def _localized_number(value: float, language: Language) -> str:
    number = f"{float(value):.1f}"
    return number.replace(".", ",") if language == "HU" else number


def _matchup_card(row: pd.Series, language: Language) -> str:
    away = str(row["away_team"])
    home = str(row["home_team"])
    kickoff = format_hungarian_kickoff(
        row["gameday"], row["gametime"], language, include_timezone=False,
    )
    week_label = f"{int(row['week'])}. HÉT" if language == "HU" else f"WEEK {int(row['week'])}"
    score_label = "Várt eredmény" if language == "HU" else "Expected score"
    away_score = _localized_number(row["implied_away_score"], language)
    home_score = _localized_number(row["implied_home_score"], language)
    spread = _localized_number(row["predicted_home_margin"], language)
    total = _localized_number(row["predicted_total_points"], language)
    spread = f"+{spread}" if float(row["predicted_home_margin"]) >= 0 else spread
    away_trend = probability_trend_badge(
        row.get("away_probability_trend"), row.get("away_probability_change_pp"), language,
        compact=True,
        previous_probability=row.get("away_previous_win_probability"),
        current_probability=row.get("away_win_probability"),
        tooltip_align="left",
    )
    home_trend = probability_trend_badge(
        row.get("home_probability_trend"), row.get("home_probability_change_pp"), language,
        compact=True,
        previous_probability=row.get("home_previous_win_probability"),
        current_probability=row.get("home_win_probability"),
        tooltip_align="right",
    )
    return f"""
    <div class="nap-card nap-matchup-card">
      <div class="nap-matchup-meta">{week_label} · {kickoff}</div>
      <div class="nap-matchup-line">
        <div>{team_badge(away, 38)} <b>{away}</b><span>{row['away_win_probability']:.1%}</span></div>
        <div class="nap-at">@</div>
        <div>{team_badge(home, 38)} <b>{home}</b><span>{row['home_win_probability']:.1%}</span></div>
      </div>
      <div class="nap-matchup-trends"><span>{away_trend}</span><span>{home_trend}</span></div>
      <div class="nap-scoreline">{score_label} <b>{away_score} – {home_score}</b>
      <span>Spread {spread} · Total {total}</span></div>
    </div>
    """


def _open_game_center(game_id: str, language: Language) -> None:
    """Navigate internally without replacing the Streamlit browser session."""

    st.session_state["dashboard_page"] = "GAMES"
    st.session_state[f"dashboard_page_selector_{language}"] = "GAMES"
    st.session_state["dashboard_selected_game_id"] = str(game_id)
    st.query_params.from_dict({"language": language, "page": "GAMES"})


def render_weekly_overview(
    games: pd.DataFrame,
    language: Language = DEFAULT_LANGUAGE,
) -> None:
    if games.empty:
        empty_state(
            "Weekly predictions are not available" if language == "EN" else "A heti predictionök nem elérhetők",
            "Build the modeling pipeline to populate the forward matchup overview."
            if language == "EN" else
            "Futtasd a modeling pipeline-t a következő meccsek felépítéséhez.",
        )
        return

    default_week = select_current_week(games)
    weeks = sorted(int(value) for value in games["week"].dropna().unique())
    selected_week = st.selectbox(
        tr(language, "week"),
        weeks,
        index=weeks.index(default_week) if default_week in weeks else 0,
    )
    week_games = games.loc[games["week"] == selected_week].copy()
    featured, favorite, highest_total = select_weekly_highlights(week_games)

    summary = st.columns(4)
    summary[0].metric(tr(language, "games"), len(week_games))
    away_probability = float(featured["away_win_probability"])
    home_probability = float(featured["home_win_probability"])
    summary[1].metric(
        tr(language, "most_even"),
        f"{featured['away_team']} @ {featured['home_team']}",
        f"{away_probability:.1%} / {home_probability:.1%}",
        delta_color="off",
        help=(
            "The game whose model win probabilities are closest to 50/50 "
            "within the selected week."
            if language == "EN" else
            "Az a heti mérkőzés, amelynél a modell győzelmi valószínűségei a legközelebb vannak az 50–50%-hoz."
        ),
    )
    favorite_is_home = (
        float(favorite["home_win_probability"])
        >= float(favorite["away_win_probability"])
    )
    favorite_team = favorite["home_team"] if favorite_is_home else favorite["away_team"]
    favorite_probability = (
        favorite["home_win_probability"]
        if favorite_is_home else favorite["away_win_probability"]
    )
    summary[2].metric(
        "Biggest favorite" if language == "EN" else "Legnagyobb favorit",
        str(favorite_team),
        f"{float(favorite_probability):.1%} · {favorite['away_team']} @ {favorite['home_team']}",
        delta_color="off",
        help=(
            "The team with the highest model win probability in the selected week."
            if language == "EN" else
            "A kiválasztott hét legmagasabb modell szerinti győzelmi valószínűségével rendelkező csapata."
        ),
    )
    summary[3].metric(
        tr(language, "highest_total"),
        f"{highest_total['away_team']} @ {highest_total['home_team']}",
        f"{_localized_number(highest_total['predicted_total_points'], language)} "
        + ("points" if language == "EN" else "pont"),
        delta_color="off",
    )
    st.markdown(f"### {tr(language, 'matchups')}")
    for start in range(0, len(week_games), 2):
        columns = st.columns(2)
        for offset, column in enumerate(columns):
            index = start + offset
            if index < len(week_games):
                with column:
                    st.markdown(
                        _matchup_card(week_games.iloc[index], language),
                        unsafe_allow_html=True,
                    )
                    game_id = str(week_games.iloc[index]["game_id"])
                    st.button(
                        "Meccs részletei →" if language == "HU" else "Matchup details →",
                        key=f"overview_game_{game_id}",
                        on_click=_open_game_center,
                        args=(game_id, language),
                        use_container_width=True,
                    )
