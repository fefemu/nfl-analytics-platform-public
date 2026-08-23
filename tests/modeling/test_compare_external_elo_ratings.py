"""
Tests for internal and external Elo comparison.
"""

import pandas as pd
import pytest

from src.modeling.compare_external_elo_ratings import (
    COMPARISON_COLUMNS,
    SUMMARY_COLUMNS,
    compare_elo_ratings,
    prepare_latest_external_elo,
)


def create_internal_ratings() -> pd.DataFrame:
    """Create synthetic current internal ratings."""

    return pd.DataFrame(
        [
            {
                "team": "KC",
                "elo_rating": 1600.0,
                "last_completed_season": 2025,
                "last_game_id": "2025_KC",
                "as_of_gameday": pd.Timestamp(
                    "2026-01-04"
                ),
            },
            {
                "team": "LA",
                "elo_rating": 1500.0,
                "last_completed_season": 2025,
                "last_game_id": "2025_LA",
                "as_of_gameday": pd.Timestamp(
                    "2026-02-08"
                ),
            },
            {
                "team": "LV",
                "elo_rating": 1400.0,
                "last_completed_season": 2025,
                "last_game_id": "2025_LV",
                "as_of_gameday": pd.Timestamp(
                    "2026-01-04"
                ),
            },
        ]
    )


def create_external_history() -> pd.DataFrame:
    """Create historical and latest external ratings."""

    return pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 22,
                "team": "KC",
                "game_id": "2025_KC",
                "elo": 1500.0,
                "qb_adj": 5.0,
            },
            {
                "season": 2025,
                "week": 22,
                "team": "LAR",
                "game_id": "2025_LAR",
                "elo": 1510.0,
                "qb_adj": 2.0,
            },
            {
                "season": 2025,
                "week": 22,
                "team": "OAK",
                "game_id": "2025_OAK",
                "elo": 1390.0,
                "qb_adj": -3.0,
            },
            {
                "season": 2026,
                "week": 1,
                "team": "KC",
                "game_id": "2026_KC",
                "elo": 1580.0,
                "qb_adj": 0.0,
            },
            {
                "season": 2026,
                "week": 1,
                "team": "LAR",
                "game_id": "2026_LAR",
                "elo": 1550.0,
                "qb_adj": 0.0,
            },
            {
                "season": 2026,
                "week": 1,
                "team": "OAK",
                "game_id": "2026_OAK",
                "elo": 1380.0,
                "qb_adj": 0.0,
            },
        ]
    )


def test_prepare_latest_external_elo() -> None:
    """Select only the latest external season-week."""

    latest = prepare_latest_external_elo(
        create_external_history()
    )

    assert len(latest) == 3

    assert latest[
        "season"
    ].eq(2026).all()

    assert latest[
        "week"
    ].eq(1).all()

    assert set(latest["team"]) == {
        "KC",
        "LA",
        "LV",
    }


def test_legacy_team_codes_are_normalized() -> None:
    """Normalize external LAR and OAK team codes."""

    latest = prepare_latest_external_elo(
        create_external_history()
    )

    team_game_ids = latest.set_index(
        "team"
    )["game_id"].to_dict()

    assert team_game_ids["LA"] == "2026_LAR"
    assert team_game_ids["LV"] == "2026_OAK"


def test_compare_elo_ratings_schema_and_values(
) -> None:
    """Create matched rating and ranking differences."""

    comparison, summary = compare_elo_ratings(
        internal_ratings=(
            create_internal_ratings()
        ),
        external_history=(
            create_external_history()
        ),
    )

    assert tuple(comparison.columns) == (
        COMPARISON_COLUMNS
    )

    assert tuple(summary.columns) == (
        SUMMARY_COLUMNS
    )

    assert len(comparison) == 3
    assert len(summary) == 1

    kansas_city = comparison.loc[
        comparison["team"] == "KC"
    ].iloc[0]

    assert kansas_city[
        "internal_elo_rating"
    ] == pytest.approx(1600.0)

    assert kansas_city[
        "external_elo_rating"
    ] == pytest.approx(1580.0)

    assert kansas_city[
        "rating_delta_external_minus_internal"
    ] == pytest.approx(-20.0)

    assert kansas_city[
        "absolute_rating_delta"
    ] == pytest.approx(20.0)

    assert kansas_city[
        "internal_rank"
    ] == 1

    assert kansas_city[
        "external_rank"
    ] == 1


def test_comparison_summary_metrics() -> None:
    """Summarize matched team ratings."""

    _, summary = compare_elo_ratings(
        internal_ratings=(
            create_internal_ratings()
        ),
        external_history=(
            create_external_history()
        ),
    )

    row = summary.iloc[0]

    assert row["team_count"] == 3

    assert row[
        "external_season"
    ] == 2026

    assert row[
        "external_week"
    ] == 1

    assert row[
        "spearman_rank_correlation"
    ] == pytest.approx(1.0)

    assert row[
        "pearson_rating_correlation"
    ] > 0.9


def test_duplicate_normalized_external_team_is_rejected(
) -> None:
    """Reject collisions after team-code mapping."""

    external = create_external_history()

    duplicate = external.loc[
        (
            external["season"] == 2026
        )
        & (
            external["week"] == 1
        )
        & (
            external["team"] == "LAR"
        )
    ].copy()

    duplicate["team"] = "LA"
    duplicate["game_id"] = "duplicate_LA"

    external = pd.concat(
        [
            external,
            duplicate,
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate teams",
    ):
        prepare_latest_external_elo(
            external
        )


def test_unmatched_team_is_rejected() -> None:
    """Internal and external team sets must match."""

    external = create_external_history()

    external = external.loc[
        ~(
            (
                external["season"] == 2026
            )
            & (
                external["team"] == "KC"
            )
        )
    ].copy()

    with pytest.raises(
        ValueError,
        match="do not match",
    ):
        compare_elo_ratings(
            internal_ratings=(
                create_internal_ratings()
            ),
            external_history=external,
        )


def test_missing_external_column_is_rejected(
) -> None:
    """Reject incomplete external source schema."""

    external = (
        create_external_history().drop(
            columns=["qb_adj"]
        )
    )

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        prepare_latest_external_elo(
            external
        )


def test_duplicate_internal_team_is_rejected(
) -> None:
    """Reject duplicate internal team ratings."""

    internal = create_internal_ratings()

    internal = pd.concat(
        [
            internal,
            internal.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate teams",
    ):
        compare_elo_ratings(
            internal_ratings=internal,
            external_history=(
                create_external_history()
            ),
        )