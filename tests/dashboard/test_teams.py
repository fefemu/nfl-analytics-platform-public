"""Tests for the Teams roster view."""

import pandas as pd
import pytest

from src.dashboard.pages.teams import (
    prepare_current_rosters,
    select_starting_lineup,
)
from src.dashboard.view_models import prepare_team_schedule


def test_prepare_current_rosters_normalizes_franchise_and_depth() -> None:
    data = pd.DataFrame({
        "team": ["LAR", "LAR"], "player_name": ["Starter", "Backup"],
        "pos_grp": ["Offense", "Offense"], "pos_abb": ["QB", "QB"],
        "pos_slot": [1, 1], "pos_rank": ["1", "2"],
    })
    result = prepare_current_rosters(data)
    assert result["team"].eq("LA").all()
    assert result["pos_rank"].tolist() == [1, 2]


def test_prepare_current_rosters_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        prepare_current_rosters(pd.DataFrame({"team": ["KC"]}))


def _formation_rows(group: str, positions: list[str]) -> pd.DataFrame:
    rows = []
    for slot, position in enumerate(positions, start=1):
        rows.append({
            "team": "DAL", "player_name": f"Starter {slot}",
            "pos_grp": group, "pos_abb": position,
            "pos_slot": slot, "pos_rank": slot if position == "WR" else 1,
        })
        rows.append({
            "team": "DAL", "player_name": f"Backup {slot}",
            "pos_grp": group, "pos_abb": position,
            "pos_slot": slot, "pos_rank": slot + 10,
        })
    return pd.DataFrame(rows)


def test_offense_lineup_uses_each_formation_slot_not_only_rank_one() -> None:
    data = _formation_rows(
        "3WR 1TE",
        ["WR", "WR", "LT", "LG", "C", "RG", "RT", "WR", "QB", "TE", "RB", "FB"],
    )
    lineup = select_starting_lineup(data, "offense")
    assert len(lineup) == 11
    assert lineup["pos_slot"].tolist() == list(range(1, 12))
    assert lineup.loc[lineup["pos_slot"].eq(2), "pos_rank"].item() == 2
    assert "FB" not in lineup["pos_abb"].tolist()


@pytest.mark.parametrize("group", ["Base 3-4 D", "Base 4-3 D"])
def test_defense_lineup_excludes_twelfth_nickel_slot(group: str) -> None:
    data = _formation_rows(
        group,
        ["LDE", "NT", "RDE", "WLB", "LILB", "RILB", "SLB", "LCB", "SS", "FS", "RCB", "NB"],
    )
    lineup = select_starting_lineup(data, "defense")
    assert len(lineup) == 11
    assert lineup["pos_slot"].tolist() == list(range(1, 12))
    assert "NB" not in lineup["pos_abb"].tolist()


def test_unknown_lineup_unit_is_rejected() -> None:
    with pytest.raises(ValueError, match="offense or defense"):
        select_starting_lineup(pd.DataFrame(), "special teams")


def _team_schedule_rows() -> pd.DataFrame:
    rows = []
    for week in range(1, 19):
        if week == 7:
            continue
        rows.append({
            "season": 2026,
            "week": week,
            "gameday": f"2026-09-{min(week + 6, 28):02d}",
            "gametime": "13:00",
            "team": "KC",
            "opponent": "BUF" if week == 1 else "DEN",
            "is_home": week % 2 == 0,
            "team_score": 27 if week == 1 else None,
            "opponent_score": 20 if week == 1 else None,
            "is_completed": week == 1,
            "opponent_elo": 1540.0,
            "current_week": 2,
        })
    return pd.DataFrame(rows)


def test_team_schedule_is_ordered_and_inserts_bye_week() -> None:
    result = prepare_team_schedule(_team_schedule_rows(), "KC")

    assert result["week"].tolist() == list(range(1, 19))
    bye = result.loc[result["week"].eq(7)].iloc[0]
    assert bool(bye["is_bye"])
    assert bye["schedule_state"] == "BYE"
    assert pd.isna(bye["opponent"])


def test_team_schedule_represents_home_away_and_game_states() -> None:
    result = prepare_team_schedule(_team_schedule_rows(), "KC")

    completed = result.loc[result["week"].eq(1)].iloc[0]
    current = result.loc[result["week"].eq(2)].iloc[0]
    upcoming = result.loc[result["week"].eq(3)].iloc[0]
    assert completed["venue"] == "@"
    assert completed["schedule_state"] == "COMPLETED"
    assert current["venue"] == "vs"
    assert current["schedule_state"] == "CURRENT"
    assert bool(current["is_current_week"])
    assert upcoming["schedule_state"] == "UPCOMING"
