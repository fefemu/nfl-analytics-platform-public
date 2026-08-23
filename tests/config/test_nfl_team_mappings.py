"""Tests for NFL team name mappings."""

import pytest

from src.config.nfl_team_mappings import (
    HISTORICAL_TEAM_TO_CURRENT,
    ODDS_TEAM_TO_NFLVERSE,
    map_odds_team_name,
    normalize_franchise_code,
)


def test_team_mapping_contains_all_current_nfl_teams() -> None:
    """Map all 32 current NFL teams."""

    assert len(ODDS_TEAM_TO_NFLVERSE) == 32
    assert len(set(ODDS_TEAM_TO_NFLVERSE.values())) == 32

    assert ODDS_TEAM_TO_NFLVERSE[
        "Los Angeles Rams"
    ] == "LA"
    assert ODDS_TEAM_TO_NFLVERSE[
        "Los Angeles Chargers"
    ] == "LAC"
    assert ODDS_TEAM_TO_NFLVERSE[
        "Washington Commanders"
    ] == "WAS"


def test_map_odds_team_name_strips_whitespace() -> None:
    """Ignore accidental whitespace around a team name."""

    result = map_odds_team_name(
        "  Buffalo Bills  "
    )

    assert result == "BUF"


def test_map_odds_team_name_rejects_unknown_team() -> None:
    """Fail when The Odds API returns an unknown team."""

    with pytest.raises(
        ValueError,
        match="Unknown Odds API NFL team",
    ):
        map_odds_team_name("Unknown NFL Team")


def test_historical_mapping_contains_franchise_aliases() -> None:
    """Map historical and external aliases to canonical team codes."""

    assert HISTORICAL_TEAM_TO_CURRENT == {
        "LAR": "LA",
        "OAK": "LV",
        "SD": "LAC",
        "STL": "LA",
    }


@pytest.mark.parametrize(
    ("historical_code", "current_code"),
    [
        ("LAR", "LA"),
        ("OAK", "LV"),
        ("SD", "LAC"),
        ("STL", "LA"),
        ("GB", "GB"),
    ],
)
def test_normalize_franchise_code(
    historical_code: str,
    current_code: str,
) -> None:
    """Normalize historical codes while preserving current codes."""

    result = normalize_franchise_code(
        historical_code
    )

    assert result == current_code


def test_normalize_franchise_code_cleans_input() -> None:
    """Normalize whitespace and letter casing in a team code."""

    result = normalize_franchise_code("  oak  ")

    assert result == "LV"
