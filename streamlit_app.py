"""Premium public shell for the NFL Analytics Platform."""

import streamlit as st

from src.dashboard.analytics import render_analytics
from src.dashboard.components import (
    empty_state,
    inject_styles,
    render_brand,
    render_page_header,
    status_pill,
)
from src.dashboard.i18n import Language, tr
from src.dashboard.repository import DashboardRepository
from src.dashboard.pages.betting import render_betting_board
from src.dashboard.pages.teams import render_teams
from src.dashboard.pages.about import render_about
from src.dashboard.pages.data_science_lab import render_data_science_lab
from src.dashboard.pages.game_center import render_game_center
from src.dashboard.pages.overview import render_weekly_overview
from src.dashboard.pages.season_simulator import render_season_simulator
from src.dashboard.styles import APP_CSS
from src.dashboard.view_models import format_refresh_timestamp


PAGES = {
    "OVERVIEW": ("⌂", "nav_overview", "subtitle_overview"),
    "GAMES": ("◫", "nav_games", "subtitle_games"),
    "BETTING": ("▦", "nav_betting", "subtitle_betting"),
    "TEAMS": ("♟", "nav_teams", "subtitle_teams"),
    "SIMULATOR": ("⌁", "nav_simulator", "subtitle_simulator"),
    "LAB": ("⚗", "nav_lab", "subtitle_lab"),
    "ABOUT": ("ⓘ", "nav_about", "subtitle_about"),
}


st.set_page_config(
    page_title="NFL Analytics Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles(APP_CSS)

repository = DashboardRepository()
health = repository.health()

query_page = st.query_params.get("page")
query_game_id = st.query_params.get("game_id")
query_language = st.query_params.get("language")
if query_page in PAGES and "dashboard_page" not in st.session_state:
    st.session_state["dashboard_page"] = query_page
if query_game_id:
    st.session_state["dashboard_page"] = "GAMES"
    st.session_state["dashboard_selected_game_id"] = str(query_game_id)
    del st.query_params["game_id"]
if query_language in ("EN", "HU") and "dashboard_language" not in st.session_state:
    st.session_state["dashboard_language"] = query_language


def _sync_navigation_query() -> None:
    """Persist canonical navigation state without resetting either widget."""

    st.query_params.from_dict({
        "language": st.session_state.get("dashboard_language", "EN"),
        "page": st.session_state.get("dashboard_page", "OVERVIEW"),
    })


def _select_dashboard_page(selector_key: str) -> None:
    st.session_state["dashboard_page"] = st.session_state[selector_key]
    _sync_navigation_query()

with st.sidebar:
    render_brand()
    language: Language = st.segmented_control(
        "Language / Nyelv",
        ("EN", "HU"),
        default="EN",
        key="dashboard_language",
        on_change=_sync_navigation_query,
        label_visibility="collapsed",
    )
    selector_key = f"dashboard_page_selector_{language}"
    if selector_key not in st.session_state:
        st.session_state[selector_key] = st.session_state.get(
            "dashboard_page", "OVERVIEW"
        )
    selected = st.radio(
        "Navigation",
        tuple(PAGES),
        format_func=lambda key: f"{PAGES[key][0]}  {tr(language, PAGES[key][1])}",
        key=selector_key,
        on_change=_select_dashboard_page,
        args=(selector_key,),
        label_visibility="collapsed",
    )
    selected = st.session_state.get("dashboard_page", selected)
    st.markdown("<div class='nap-divider'></div>", unsafe_allow_html=True)
    state = tr(language, "data_ready") if health.ready else tr(language, "refresh_required")
    st.markdown(status_pill(state, health.ready), unsafe_allow_html=True)
    st.caption(tr(language, "responsible"))
    st.markdown(
        '<div class="nap-attribution">'
        + ("Csapatmetaadatok: " if language == "HU" else "Team identity metadata: ")
        +
        '<a href="https://github.com/nflverse/nflverse-data/releases/tag/teams" '
        'target="_blank">nflverse teams</a>.</div>',
        unsafe_allow_html=True,
    )

page_key = selected
render_analytics(page_key, language)
page_title = tr(language, PAGES[selected][1])
refresh_label = None
if health.latest_refresh_at is not None:
    refresh_label = format_refresh_timestamp(health.latest_refresh_at, language)
render_page_header(
    tr(language, "eyebrow"),
    page_title,
    tr(language, PAGES[selected][2]),
    refresh_label,
)

st.write("")
if page_key == "OVERVIEW":
    render_weekly_overview(repository.load_weekly_games(), language)
elif page_key == "GAMES":
    preferred_game_id = st.session_state.pop(
        "dashboard_selected_game_id", None,
    )
    render_game_center(
        repository.load_game_center_games(),
        repository.load_current_betting_board(),
        language,
        preferred_game_id,
    )
elif page_key == "BETTING":
    render_betting_board(repository.load_current_betting_board(), language)
elif page_key == "TEAMS":
    render_teams(
        repository.load_current_team_rosters(),
        language=language,
        schedules=repository.load_current_team_schedule(),
    )
elif page_key == "SIMULATOR":
    simulation_summary, win_distribution, elo_benchmark = (
        repository.load_season_simulator()
    )
    render_season_simulator(
        simulation_summary,
        win_distribution,
        elo_benchmark,
        language,
    )
elif page_key == "LAB":
    render_data_science_lab(repository.load_data_science_lab(), language)
elif page_key == "ABOUT":
    render_about(language)
elif not health.ready:
    empty_state(
        "Dashboard data is not built locally",
        "Run the modeling pipeline or the audited in-season refresh. The public UI will populate automatically when its read-only analytics tables exist.",
    )
else:
    empty_state(
        f"{page_title} foundation ready" if language == "EN" else f"{page_title} alap kész",
        "This page is connected to the shared design and data-access system and will be populated in its delivery block."
        if language == "EN" else
        "Az oldal csatlakozik a közös design- és adatelérési rendszerhez.",
    )

st.markdown("<div class='nap-divider'></div>", unsafe_allow_html=True)
st.caption(
    (
        "Team identity metadata sourced from nflverse. Team names and marks belong "
        "to their respective owners. Independent analytics project; not affiliated "
        "with or endorsed by the NFL or its clubs."
    ) if language == "EN" else (
        "A csapatadatok forrása az nflverse. A csapatnevek, logók és védjegyek "
        "a megfelelő jogtulajdonosok tulajdonát képezik. Az NFL Analytics Platform "
        "független elemzési projekt, nem áll kapcsolatban az NFL-lel vagy annak csapataival."
    )
)
