"""Tests for the Elo-injury blend scorecard."""

import pandas as pd
import pytest

from src.modeling.run_elo_injury_blend_scorecard import (
    AUDIT_SELECTION_SEASONS,
    evaluate_blend_weights,
    evaluate_probability_models,
    select_best_blend_weight,
)


def create_oof_predictions() -> pd.DataFrame:
    """Create deterministic OOF blend predictions."""

    rows: list[dict[str, object]] = []

    for season in range(
        2020,
        2025,
    ):
        rows.extend(
            [
                {
                    "game_id": f"{season}_home",
                    "season": season,
                    "game_date": pd.Timestamp(
                        f"{season}-09-10"
                    ),
                    "target_home_win": 1,
                    "elo_probability": 0.55,
                    "injury_probability": 0.75,
                },
                {
                    "game_id": f"{season}_away",
                    "season": season,
                    "game_date": pd.Timestamp(
                        f"{season}-09-11"
                    ),
                    "target_home_win": 0,
                    "elo_probability": 0.45,
                    "injury_probability": 0.25,
                },
            ]
        )

    rows.extend(
        [
            {
                "game_id": "2025_home",
                "season": 2025,
                "game_date": pd.Timestamp(
                    "2025-09-10"
                ),
                "target_home_win": 1,
                "elo_probability": 0.75,
                "injury_probability": 0.45,
            },
            {
                "game_id": "2025_away",
                "season": 2025,
                "game_date": pd.Timestamp(
                    "2025-09-11"
                ),
                "target_home_win": 0,
                "elo_probability": 0.25,
                "injury_probability": 0.55,
            },
        ]
    )

    return pd.DataFrame(
        rows
    )


def test_evaluate_blend_weights_uses_selection_seasons(
) -> None:
    """Keep the 2025 audit outside weight selection."""

    predictions = create_oof_predictions()

    results = evaluate_blend_weights(
        predictions=predictions,
        selection_seasons=(
            AUDIT_SELECTION_SEASONS
        ),
        weight_grid=(
            0.0,
            0.5,
            1.0,
        ),
    )

    assert set(
        results["game_count"]
    ) == {
        10,
    }

    assert select_best_blend_weight(
        results
    ) == pytest.approx(
        1.0
    )


def test_evaluate_blend_weights_rejects_invalid_weight(
) -> None:
    """Reject a blend weight outside zero and one."""

    predictions = create_oof_predictions()

    with pytest.raises(
        ValueError,
        match="between zero and one",
    ):
        evaluate_blend_weights(
            predictions=predictions,
            selection_seasons=(
                2020,
            ),
            weight_grid=(
                -0.1,
                0.5,
            ),
        )


def test_select_best_blend_weight_rejects_empty(
) -> None:
    """Reject empty weight results."""

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        select_best_blend_weight(
            pd.DataFrame()
        )


def test_evaluate_probability_models_compares_three_models(
) -> None:
    """Compare Elo, injury and their fixed blend."""

    predictions = create_oof_predictions()

    results = evaluate_probability_models(
        predictions=predictions,
        seasons=(
            2025,
        ),
        injury_weight=0.40,
        evaluation_period=(
            "historical_audit_2025"
        ),
    )

    assert set(
        results["model_name"]
    ) == {
        "elo",
        "logistic_elo_qb_unit_burdens",
        "elo_injury_blend",
    }

    assert set(
        results["game_count"]
    ) == {
        2,
    }

    blend_row = results.loc[
        results["model_name"]
        == "elo_injury_blend"
    ].iloc[0]

    assert blend_row[
        "injury_weight"
    ] == pytest.approx(
        0.40
    )

    assert blend_row[
        "elo_weight"
    ] == pytest.approx(
        0.60
    )