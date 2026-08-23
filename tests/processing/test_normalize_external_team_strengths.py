import pandas as pd
import pytest

from src.processing.normalize_external_team_strengths import (
    UNIT_PRE_COLUMNS,
    WIN_TOTAL_COLUMNS,
    normalize_nfelounits_units,
    normalize_win_total_ratings,
)


def test_normalize_units_keeps_only_pregame_fields() -> None:
    row = {"season": 2026, "week": 1, "team": "LAR"}
    row.update({column: 1.0 for column in UNIT_PRE_COLUMNS})
    row["total_value_post"] = 99.0

    result = normalize_nfelounits_units(pd.DataFrame([row]))

    assert result.loc[0, "team"] == "LA"
    assert "total_value_post" not in result.columns


def test_normalize_units_rejects_duplicate_keys() -> None:
    row = {"season": 2026, "week": 1, "team": "OAK"}
    row.update({column: 1.0 for column in UNIT_PRE_COLUMNS})

    with pytest.raises(ValueError, match="duplicate"):
        normalize_nfelounits_units(pd.DataFrame([row, row]))


def test_normalize_units_drops_fully_uninitialized_rows() -> None:
    empty_row = {"season": 1999, "week": 1, "team": "KC"}
    empty_row.update({column: None for column in UNIT_PRE_COLUMNS})
    complete_row = {"season": 2018, "week": 1, "team": "KC"}
    complete_row.update({column: 1.0 for column in UNIT_PRE_COLUMNS})

    result = normalize_nfelounits_units(
        pd.DataFrame([empty_row, complete_row])
    )

    assert len(result) == 1
    assert result.loc[0, "season"] == 2018


def test_normalize_win_totals_maps_team_aliases() -> None:
    row = {"season": 2026, "team": "OAK"}
    row.update({column: 1.0 for column in WIN_TOTAL_COLUMNS})

    result = normalize_win_total_ratings(pd.DataFrame([row]))

    assert result.loc[0, "team"] == "LV"
    assert result.loc[0, "season"] == 2026


def test_normalize_win_totals_preserves_historical_oakland() -> None:
    row = {"season": 2019, "team": "OAK"}
    row.update({column: 1.0 for column in WIN_TOTAL_COLUMNS})

    result = normalize_win_total_ratings(pd.DataFrame([row]))

    assert result.loc[0, "team"] == "OAK"


def test_normalize_win_totals_rejects_missing_fields() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        normalize_win_total_ratings(
            pd.DataFrame([{"season": 2026, "team": "KC"}])
        )
