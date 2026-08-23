"""
Tests for nfelo game rating normalization.
"""

import numpy as np
import pandas as pd
import pytest

from src.processing.normalize_nfelo_game_ratings import (
    NORMALIZED_IDENTIFIER_COLUMNS,
    normalize_nfelo_game_ratings,
    normalize_team_code,
    parse_source_game_ids,
)


def create_source_data() -> pd.DataFrame:
    """Create representative nfelo source rows."""

    return pd.DataFrame(
        [
            {
                "game_id": "2019_01_OAK_DEN",
                "starting_nfelo_home": 1520.0,
                "starting_nfelo_away": 1480.0,
                "nfelo_dif_base": 40.0,
                "nfelo_home_probability_open": 0.58,
                "nfelo_home_probability_close": 0.60,
                "extra_source_column": "historical",
            },
            {
                "game_id": "2020_01_DEN_OAK",
                "starting_nfelo_home": 1450.0,
                "starting_nfelo_away": 1510.0,
                "nfelo_dif_base": -60.0,
                "nfelo_home_probability_open": 0.40,
                "nfelo_home_probability_close": 0.42,
                "extra_source_column": "relocated",
            },
            {
                "game_id": "2026_01_SF_LAR",
                "starting_nfelo_home": 1600.0,
                "starting_nfelo_away": 1580.0,
                "nfelo_dif_base": 20.0,
                "nfelo_home_probability_open": 0.55,
                "nfelo_home_probability_close": 0.56,
                "extra_source_column": "current",
            },
            {
                "game_id": "2026_01_WSH_PHI",
                "starting_nfelo_home": 1590.0,
                "starting_nfelo_away": 1510.0,
                "nfelo_dif_base": 80.0,
                "nfelo_home_probability_open": 0.62,
                "nfelo_home_probability_close": 0.63,
                "extra_source_column": "alias",
            },
        ]
    )


def test_normalize_team_code_is_season_aware(
) -> None:
    """Oakland changes to Las Vegas only from 2020."""

    assert normalize_team_code(
        "OAK",
        2019,
    ) == "OAK"

    assert normalize_team_code(
        "OAK",
        2020,
    ) == "LV"

    assert normalize_team_code(
        "LAR",
        2026,
    ) == "LA"

    assert normalize_team_code(
        "WSH",
        2026,
    ) == "WAS"

    assert normalize_team_code(
        "JAC",
        2026,
    ) == "JAX"


def test_parse_source_game_ids() -> None:
    """Parse season, week, away and home components."""

    source_ids = pd.Series(
        [
            "2026_01_DEN_KC",
            "2025_22_SEA_NE",
        ]
    )

    parsed = parse_source_game_ids(
        source_ids
    )

    assert parsed.iloc[0][
        "source_season"
    ] == 2026

    assert parsed.iloc[0][
        "source_week"
    ] == 1

    assert parsed.iloc[0][
        "away_team_source"
    ] == "DEN"

    assert parsed.iloc[0][
        "home_team_source"
    ] == "KC"


def test_normalize_nfelo_game_identifiers() -> None:
    """Create nflverse-compatible normalized IDs."""

    normalized = normalize_nfelo_game_ratings(
        create_source_data()
    )

    assert tuple(
        normalized.columns[
            :len(
                NORMALIZED_IDENTIFIER_COLUMNS
            )
        ]
    ) == NORMALIZED_IDENTIFIER_COLUMNS

    normalized_ids = set(
        normalized["normalized_game_id"]
    )

    assert "2019_01_OAK_DEN" in normalized_ids
    assert "2020_01_DEN_LV" in normalized_ids
    assert "2026_01_SF_LA" in normalized_ids
    assert "2026_01_WAS_PHI" in normalized_ids


def test_source_identifiers_and_columns_are_preserved(
) -> None:
    """Keep source IDs and additional source fields."""

    normalized = normalize_nfelo_game_ratings(
        create_source_data()
    )

    current_row = normalized.loc[
        normalized["normalized_game_id"]
        == "2026_01_SF_LA"
    ].iloc[0]

    assert current_row[
        "source_game_id"
    ] == "2026_01_SF_LAR"

    assert current_row[
        "source_name"
    ] == "nfelo_games"

    assert current_row[
        "extra_source_column"
    ] == "current"


def test_invalid_game_id_format_is_rejected() -> None:
    """Source IDs must contain four components."""

    source_data = create_source_data()

    source_data.loc[
        0,
        "game_id",
    ] = "invalid_game_id"

    with pytest.raises(
        ValueError,
        match="season_week_away_home",
    ):
        normalize_nfelo_game_ratings(
            source_data
        )


def test_duplicate_source_game_id_is_rejected(
) -> None:
    """Reject duplicate raw source identifiers."""

    source_data = create_source_data()

    source_data = pd.concat(
        [
            source_data,
            source_data.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate source game IDs",
    ):
        normalize_nfelo_game_ratings(
            source_data
        )


def test_duplicate_normalized_game_id_is_rejected(
) -> None:
    """Reject aliases that collapse to one game ID."""

    source_data = create_source_data()

    duplicate = source_data.loc[
        source_data["game_id"]
        == "2026_01_SF_LAR"
    ].copy()

    duplicate["game_id"] = "2026_01_SF_LA"

    source_data = pd.concat(
        [
            source_data,
            duplicate,
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="normalization creates duplicate",
    ):
        normalize_nfelo_game_ratings(
            source_data
        )


def test_non_finite_rating_is_rejected() -> None:
    """Starting ratings must be finite."""

    source_data = create_source_data()

    source_data.loc[
        0,
        "starting_nfelo_home",
    ] = np.nan

    with pytest.raises(
        ValueError,
        match="non-finite rating",
    ):
        normalize_nfelo_game_ratings(
            source_data
        )


def test_invalid_probability_is_rejected() -> None:
    """External probabilities must be inside zero and one."""

    source_data = create_source_data()

    source_data.loc[
        0,
        "nfelo_home_probability_open",
    ] = 1.0

    with pytest.raises(
        ValueError,
        match="invalid home probabilities",
    ):
        normalize_nfelo_game_ratings(
            source_data
        )


def test_missing_required_column_is_rejected(
) -> None:
    """Reject incomplete nfelo source schema."""

    source_data = create_source_data().drop(
        columns=[
            "starting_nfelo_away",
        ]
    )

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        normalize_nfelo_game_ratings(
            source_data
        )