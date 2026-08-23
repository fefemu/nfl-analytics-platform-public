"""Current NFL team and roster explorer."""

from html import escape

import pandas as pd
import streamlit as st

from src.config.nfl_team_mappings import normalize_franchise_code
from src.dashboard.components import empty_state, team_badge
from src.dashboard.i18n import DEFAULT_LANGUAGE, Language
from src.dashboard.team_branding import get_team_brand


DIVISIONS = {
    "BUF": "AFC East", "MIA": "AFC East", "NE": "AFC East", "NYJ": "AFC East",
    "BAL": "AFC North", "CIN": "AFC North", "CLE": "AFC North", "PIT": "AFC North",
    "HOU": "AFC South", "IND": "AFC South", "JAX": "AFC South", "TEN": "AFC South",
    "DEN": "AFC West", "KC": "AFC West", "LV": "AFC West", "LAC": "AFC West",
    "DAL": "NFC East", "NYG": "NFC East", "PHI": "NFC East", "WAS": "NFC East",
    "CHI": "NFC North", "DET": "NFC North", "GB": "NFC North", "MIN": "NFC North",
    "ATL": "NFC South", "CAR": "NFC South", "NO": "NFC South", "TB": "NFC South",
    "ARI": "NFC West", "LA": "NFC West", "SF": "NFC West", "SEA": "NFC West",
}

OFFENSE_GRID = {
    1: ("WR1", "wr1"), 2: ("WR2", "wr2"), 3: ("LT", "lt"),
    4: ("LG", "lg"), 5: ("C", "center"), 6: ("RG", "rg"),
    7: ("RT", "rt"), 8: ("SLOT", "slot"), 9: ("QB", "qb"),
    10: ("TE", "te"), 11: ("RB", "rb"),
}

DEFENSE_34_GRID = {
    1: ("DE", "dl1"), 2: ("NT", "dl2"), 3: ("DE", "dl3"),
    4: ("EDGE", "edge1"), 5: ("LB", "lb1"), 6: ("LB", "lb2"),
    7: ("EDGE", "edge2"), 8: ("CB", "cb1"), 9: ("S", "s1"),
    10: ("S", "s2"), 11: ("CB", "cb2"),
}

DEFENSE_43_GRID = {
    1: ("EDGE", "dl1"), 2: ("DT", "dl2"), 3: ("DT", "dl3"),
    4: ("EDGE", "dl4"), 5: ("LB", "lb1"), 6: ("LB", "lb2"),
    7: ("LB", "lb3"), 8: ("CB", "cb1"), 9: ("S", "s1"),
    10: ("S", "s2"), 11: ("CB", "cb2"),
}

POSITION_GROUPS = {
    "QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
    "LT": "OL", "LG": "OL", "C": "OL", "RG": "OL", "RT": "OL",
    "LDE": "EDGE", "RDE": "EDGE", "LDT": "DL", "RDT": "DL", "NT": "DL",
    "WLB": "LB", "SLB": "LB", "MLB": "LB", "LILB": "LB", "RILB": "LB",
    "LCB": "CB", "RCB": "CB", "NB": "CB", "FS": "S", "SS": "S",
}


def prepare_current_rosters(rosters: pd.DataFrame) -> pd.DataFrame:
    """Normalize identity and depth-chart fields used by the public page."""

    if rosters.empty:
        return rosters.copy()
    required = {"team", "player_name", "pos_grp", "pos_abb", "pos_slot", "pos_rank"}
    missing = sorted(required - set(rosters.columns))
    if missing:
        raise ValueError("Team roster data is missing columns: " + ", ".join(missing))
    result = rosters.copy()
    result["team"] = result["team"].astype(str).map(normalize_franchise_code)
    result["pos_rank"] = pd.to_numeric(result["pos_rank"], errors="coerce")
    result["pos_slot"] = pd.to_numeric(result["pos_slot"], errors="coerce")
    result = result.dropna(subset=["team", "player_name", "pos_rank", "pos_slot"])
    return result.drop_duplicates(
        ["team", "pos_grp", "pos_abb", "pos_slot", "pos_rank", "player_name"]
    ).reset_index(drop=True)


def select_starting_lineup(data: pd.DataFrame, unit: str) -> pd.DataFrame:
    """Select one real player for each of the eleven formation slots."""

    if unit not in {"offense", "defense"}:
        raise ValueError("unit must be offense or defense")
    if unit == "offense":
        unit_data = data.loc[data["pos_grp"].eq("3WR 1TE")]
    else:
        unit_data = data.loc[data["pos_grp"].isin(["Base 3-4 D", "Base 4-3 D"])]
    unit_data = unit_data.loc[unit_data["pos_slot"].between(1, 11)].copy()
    lineup = (
        unit_data.sort_values(["pos_slot", "pos_rank", "player_name"], kind="stable")
        .drop_duplicates("pos_slot", keep="first")
        .sort_values("pos_slot")
        .reset_index(drop=True)
    )
    return lineup


def _injury_badge(row: pd.Series) -> str:
    status = row.get("report_status")
    if pd.isna(status) or not str(status).strip():
        return ""
    injury = row.get("report_primary_injury")
    detail = "" if pd.isna(injury) else f" · {escape(str(injury))}"
    return f'<span class="nap-player-injury">{escape(str(status))}{detail}</span>'


def _formation_card(row: pd.Series, role: str, area: str) -> str:
    number = "" if pd.isna(row.get("jersey_number")) else f"#{escape(str(row['jersey_number']))}"
    headshot = row.get("headshot")
    image = (
        f'<img src="{escape(str(headshot))}" alt="" loading="lazy">'
        if pd.notna(headshot) and str(headshot).startswith("http") else ""
    )
    featured = " nap-player-featured" if role == "QB" else ""
    return (
        f'<div class="nap-formation-player{featured}" style="grid-area:{area}">'
        f'{image}<div class="nap-player-copy"><span class="nap-player-position">{role}</span>'
        f'<strong>{escape(str(row["player_name"]))}</strong>'
        f'<small>{number}</small>{_injury_badge(row)}</div></div>'
    )


def _render_formation(team_data: pd.DataFrame, unit: str, language: Language) -> None:
    lineup = select_starting_lineup(team_data, unit)
    if len(lineup) != 11:
        empty_state(
            "Incomplete starting lineup" if language == "EN" else "Hiányos kezdő felállás",
            f"Only {len(lineup)} of 11 source-backed formation slots are available."
            if language == "EN" else
            f"A forrásadatban a 11 formációhelyből csak {len(lineup)} tölthető fel valós játékossal.",
        )
        return
    if unit == "offense":
        grid = OFFENSE_GRID
        formation_class = "offense"
        label = "11 personnel visualization" if language == "EN" else "11 personnel vizualizáció"
    else:
        is_34 = lineup["pos_grp"].eq("Base 3-4 D").all()
        grid = DEFENSE_34_GRID if is_34 else DEFENSE_43_GRID
        formation_class = "defense defense-34" if is_34 else "defense defense-43"
        label = (
            ("Base 3–4 depth-chart visualization" if is_34 else "Base 4–3 depth-chart visualization")
            if language == "EN" else
            ("Base 3–4 felállás – depth chart vizualizáció" if is_34 else
             "Base 4–3 felállás – depth chart vizualizáció")
        )
    cards = []
    for _, row in lineup.iterrows():
        role, area = grid[int(row["pos_slot"])]
        cards.append(_formation_card(row, role, area))
    st.caption(label)
    st.markdown(
        f'<div class="nap-formation {formation_class}"><div class="nap-field-lines"></div>'
        + "".join(cards) + "</div>",
        unsafe_allow_html=True,
    )


def _depth_player(row: pd.Series) -> str:
    number = "" if pd.isna(row.get("jersey_number")) else f"#{escape(str(row['jersey_number']))} · "
    return (
        '<div class="nap-depth-player">'
        f'<span>{int(row["pos_rank"])}</span><strong>{escape(str(row["player_name"]))}</strong>'
        f'<small>{number}{escape(str(row["pos_abb"]))}</small>{_injury_badge(row)}</div>'
    )


def _render_full_depth(team_data: pd.DataFrame, unit: str) -> None:
    if unit == "offense":
        data = team_data.loc[team_data["pos_grp"].eq("3WR 1TE")].copy()
        group_order = ["QB", "RB", "WR", "TE", "OL"]
    else:
        data = team_data.loc[team_data["pos_grp"].isin(["Base 3-4 D", "Base 4-3 D"])].copy()
        group_order = ["DL", "EDGE", "LB", "CB", "S"]
    data["public_group"] = data["pos_abb"].map(POSITION_GROUPS)
    blocks = []
    for group in group_order:
        players = data.loc[data["public_group"].eq(group)].sort_values(
            ["pos_rank", "pos_slot", "player_name"], kind="stable"
        )
        if players.empty:
            continue
        blocks.append(
            '<section class="nap-depth-group"><h4>' + group + "</h4>"
            + "".join(_depth_player(row) for _, row in players.iterrows()) + "</section>"
        )
    st.markdown('<div class="nap-depth-groups">' + "".join(blocks) + "</div>", unsafe_allow_html=True)


def _render_team_header(team_data: pd.DataFrame, selected: str, language: Language) -> None:
    brand = get_team_brand(selected)
    snapshot = pd.to_datetime(team_data["snapshot_at"], utc=True, errors="coerce").max()
    snapshot_text = snapshot.strftime("%Y.%m.%d.") if pd.notna(snapshot) else "—"
    elo = pd.to_numeric(team_data.get("elo_rating"), errors="coerce").dropna()
    elo_rank = pd.to_numeric(team_data.get("elo_rank"), errors="coerce").dropna()
    elo_html = ""
    if not elo.empty:
        rank = f" · #{int(elo_rank.iloc[0])} NFL" if not elo_rank.empty else ""
        elo_help = (
            "A csapat aktuális erősségét becslő nfelo mutató, amelyet a platform "
            "prediction és szimulációs rétege is használ. A rating a mérkőzések "
            "után a teljesítmény és az ellenfél erőssége alapján változik. "
            "A magasabb érték erősebb csapatot jelez."
            if language == "HU" else
            "The current nfelo estimate of team strength used by the platform's prediction "
            "and simulation layers. It changes after games based on performance and opponent "
            "strength. A higher value indicates a stronger team."
        )
        elo_html = (
            '<div class="nap-roster-kpi"><span>Elo rating '
            f'<b class="nap-info" title="{escape(elo_help)}">ⓘ</b></span>'
            f'<strong>{elo.iloc[0]:.0f}</strong><small>{rank}</small></div>'
        )
    updated_label = "Depth chart frissítve" if language == "HU" else "Depth chart updated"
    st.markdown(
        '<div class="nap-card nap-roster-hero">'
        f'{team_badge(selected, 72)}<div class="nap-roster-identity"><h2>{escape(brand.display_name)}</h2>'
        f'<span>{escape(DIVISIONS.get(selected, "NFL"))}</span><small>{updated_label}: {snapshot_text}</small>'
        f'</div>{elo_html}</div>',
        unsafe_allow_html=True,
    )


def _first_rank(team_data: pd.DataFrame, column: str) -> int | None:
    values = pd.to_numeric(team_data.get(column), errors="coerce").dropna()
    return None if values.empty else int(values.iloc[0])


def _render_unit_strength(
    team_data: pd.DataFrame,
    unit: str,
    language: Language,
) -> None:
    if unit == "offense":
        metrics = (
            (
                "offense_rank", "Offense" if language == "EN" else "Támadóegység",
                "A csapat támadóerejének NFL-rangsora a nfelounits előrejelző unit ratingje alapján. A mutató a passz- és futójáték ellenfélhez igazított EPA-teljesítményét EWMA-simítással súlyozza. Az #1 jelenti a legerősebb támadóegységet."
                if language == "HU" else
                "NFL rank from the nfelounits predictive offense rating. It EWMA-smooths opponent-adjusted passing and rushing EPA performance. #1 is the strongest offense.",
            ),
            (
                "pass_offense_rank", "Pass offense" if language == "EN" else "Passzjáték",
                "A csapat passzjátékának NFL-rangsora az ellenfélhez igazított passz EPA-teljesítmény EWMA-simított nfelounits ratingje alapján. Az #1 jelenti a legerősebb passzjátékot."
                if language == "HU" else
                "NFL rank from the EWMA-smoothed nfelounits opponent-adjusted passing EPA rating. #1 is the strongest passing offense.",
            ),
            (
                "rush_offense_rank", "Rush offense" if language == "EN" else "Futójáték",
                "A csapat futójátékának NFL-rangsora az ellenfélhez igazított futás EPA-teljesítmény EWMA-simított nfelounits ratingje alapján. Az #1 jelenti a legerősebb futójátékot."
                if language == "HU" else
                "NFL rank from the EWMA-smoothed nfelounits opponent-adjusted rushing EPA rating. #1 is the strongest rushing offense.",
            ),
        )
    else:
        metrics = (
            (
                "defense_rank", "Defense" if language == "EN" else "Védőegység",
                "A csapat védelmi erejének NFL-rangsora a nfelounits előrejelző unit ratingje alapján. A mutató a passz és futás elleni, ellenfélhez igazított EPA-teljesítményt EWMA-simítással súlyozza. Az #1 jelenti a legerősebb védőegységet."
                if language == "HU" else
                "NFL rank from the nfelounits predictive defense rating. It EWMA-smooths opponent-adjusted passing and rushing defense EPA performance. #1 is the strongest defense.",
            ),
            (
                "pass_defense_rank", "Pass defense" if language == "EN" else "Passz elleni védelem",
                "A csapat passz elleni védelmének NFL-rangsora az ellenfélhez igazított pass defense EPA-teljesítmény EWMA-simított nfelounits ratingje alapján. Az #1 jelenti a legerősebb passz elleni védelmet."
                if language == "HU" else
                "NFL rank from the EWMA-smoothed nfelounits opponent-adjusted pass defense EPA rating. #1 is the strongest pass defense.",
            ),
            (
                "rush_defense_rank", "Rush defense" if language == "EN" else "Futás elleni védelem",
                "A csapat futás elleni védelmének NFL-rangsora az ellenfélhez igazított rush defense EPA-teljesítmény EWMA-simított nfelounits ratingje alapján. Az #1 jelenti a legerősebb futás elleni védelmet."
                if language == "HU" else
                "NFL rank from the EWMA-smoothed nfelounits opponent-adjusted rush defense EPA rating. #1 is the strongest rush defense.",
            ),
        )
    cards = []
    for column, label, help_text in metrics:
        rank = _first_rank(team_data, column)
        if rank is not None:
            cards.append(
                '<div class="nap-unit-rank"><span>' + escape(label)
                + f' <b class="nap-info" title="{escape(help_text)}">ⓘ</b></span>'
                + f'<strong>#{rank}</strong><small>/ 32 NFL</small></div>'
            )
    if cards:
        st.markdown('<div class="nap-unit-strip">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def render_teams(rosters: pd.DataFrame, language: Language = DEFAULT_LANGUAGE) -> None:
    """Render the team profile, formation and full depth-chart views."""

    data = prepare_current_rosters(rosters)
    if data.empty:
        empty_state(
            "Current depth charts are not available" if language == "EN" else "Az aktuális depth chartok nem érhetők el",
            "Run the roster/depth-chart pipeline to populate this page."
            if language == "EN" else "Futtasd a roster/depth chart pipeline-t az oldal feltöltéséhez.",
        )
        return

    teams = sorted(data["team"].unique())
    selected = st.selectbox(
        "Team" if language == "EN" else "Csapat",
        teams,
        format_func=lambda code: get_team_brand(code).display_name,
    )
    team_data = data.loc[data["team"].eq(selected)].copy()
    _render_team_header(team_data, selected, language)
    view = st.radio(
        "Roster view" if language == "EN" else "Nézet",
        ["STARTING", "FULL"], horizontal=True,
        format_func=lambda value: {
            ("STARTING", "EN"): "Starting lineup", ("FULL", "EN"): "Full depth chart",
            ("STARTING", "HU"): "Kezdő felállás", ("FULL", "HU"): "Teljes depth chart",
        }[(value, language)],
    )
    offense, defense = st.tabs(["Offense", "Defense"])
    with offense:
        _render_unit_strength(team_data, "offense", language)
        _render_formation(team_data, "offense", language) if view == "STARTING" else _render_full_depth(team_data, "offense")
    with defense:
        _render_unit_strength(team_data, "defense", language)
        _render_formation(team_data, "defense", language) if view == "STARTING" else _render_full_depth(team_data, "defense")
    st.caption(
        "Forrás: nflverse/ESPN depth chart, nflverse játékos-adatbázis és "
        "sérülésjelentések; nfelo Elo; nfelounits unit ratingek."
        if language == "HU" else
        "Source: nflverse/ESPN depth charts, nflverse player directory and injury reports; "
        "nfelo Elo; nfelounits unit ratings."
    )
