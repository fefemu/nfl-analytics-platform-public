"""Tests for the model governance scorecard."""

import pandas as pd
import pytest

from src.modeling.run_model_governance_scorecard import (
    GOVERNANCE_FEATURE_COLUMNS,
    GOVERNANCE_VALIDATION_SEASONS,
    aggregate_governance_results,
    create_governance_fold,
    evaluate_governance_models,
)


def create_governance_data() -> pd.DataFrame:
    """Create chronological synthetic governance data."""

    rows: list[dict[str, object]] = []

    for season in range(
        2018,
        2026,
    ):
        for target in (
            0,
            1,
        ):
            direction = (
                1.0
                if target == 1
                else -1.0
            )

            row: dict[str, object] = {
                "game_id": (
                    f"{season}_{target}"
                ),
                "season": season,
                "game_date": pd.Timestamp(
                    f"{season}-09-{10 + target}"
                ),
                "split_name": "historical",
                "target_home_win": target,
                "elo_home_win_probability": (
                    0.65
                    if target == 1
                    else 0.35
                ),
                "has_complete_injury_data": True,
            }

            for feature_name in (
                GOVERNANCE_FEATURE_COLUMNS
            ):
                if "injury_burden" in feature_name:
                    row[feature_name] = (
                        -0.25 * direction
                    )
                elif "out_count" in feature_name:
                    row[feature_name] = (
                        -1.0 * direction
                    )
                else:
                    row[feature_name] = (
                        1.0 * direction
                    )

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


def test_create_governance_fold_is_chronological(
) -> None:
    """Train only before the validation season."""

    governance_data = (
        create_governance_data()
    )

    fold_data = create_governance_fold(
        governance_data=governance_data,
        validation_season=2023,
    )

    training_seasons = set(
        fold_data.loc[
            fold_data["split_name"] == "train",
            "season",
        ]
    )

    validation_seasons = set(
        fold_data.loc[
            fold_data["split_name"]
            == "validation",
            "season",
        ]
    )

    assert max(training_seasons) == 2022
    assert validation_seasons == {
        2023,
    }


def test_evaluate_governance_models_covers_every_season(
) -> None:
    """Evaluate Elo and all challengers every season."""

    governance_data = (
        create_governance_data()
    )

    results = evaluate_governance_models(
        governance_data
    )

    assert set(
        results["validation_season"]
    ) == set(
        GOVERNANCE_VALIDATION_SEASONS
    )

    assert results.shape[0] == 30

    assert set(
        results.groupby(
            "validation_season"
        ).size()
    ) == {
        5,
    }


def test_aggregate_governance_results_builds_scorecard(
) -> None:
    """Aggregate all seasons for every model."""

    governance_data = (
        create_governance_data()
    )

    season_results = (
        evaluate_governance_models(
            governance_data
        )
    )

    aggregate_results = (
        aggregate_governance_results(
            season_results
        )
    )

    assert aggregate_results.shape[0] == 5

    assert set(
        aggregate_results["season_count"]
    ) == {
        6,
    }

    assert set(
        aggregate_results["game_count"]
    ) == {
        12,
    }

    assert "brier_improvement_vs_elo" in (
        aggregate_results.columns
    )

    assert "worst_season_brier" in (
        aggregate_results.columns
    )

    assert "brier_season_std" in (
        aggregate_results.columns
    )


def test_aggregate_governance_results_rejects_empty(
) -> None:
    """Reject an empty governance result table."""

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        aggregate_governance_results(
            pd.DataFrame()
        )