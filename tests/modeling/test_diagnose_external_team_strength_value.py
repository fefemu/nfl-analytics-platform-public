import pandas as pd
import pytest

from src.modeling.diagnose_external_team_strength_value import (
    SUMMARY_COLUMNS,
    create_paired_oof_predictions,
    summarize_paired_unit_value,
)
from tests.modeling.test_backtest_external_team_strength_candidates import (
    create_source_data,
)


def test_paired_predictions_cover_each_validation_game_once() -> None:
    predictions = create_paired_oof_predictions(create_source_data())

    assert len(predictions) == 160
    assert not predictions["game_id"].duplicated().any()
    assert set(predictions["validation_season"]) == {2021, 2022, 2023, 2024}


def test_paired_summary_has_expected_schema() -> None:
    predictions = create_paired_oof_predictions(create_source_data())
    summary = summarize_paired_unit_value(
        predictions,
        bootstrap_iterations=100,
    )

    assert tuple(summary.columns) == SUMMARY_COLUMNS
    assert summary.loc[0, "validation_game_count"] == 160


def test_summary_delta_matches_paired_losses() -> None:
    predictions = pd.DataFrame(
        {
            "validation_season": [2021, 2021],
            "external_nfelo_qb_brier_loss": [0.25, 0.16],
            "external_nfelo_qb_units_brier_loss": [0.20, 0.18],
        }
    )
    summary = summarize_paired_unit_value(
        predictions,
        bootstrap_iterations=100,
    )

    assert summary.loc[0, "brier_score_delta"] == pytest.approx(-0.015)


def test_nonpositive_bootstrap_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        summarize_paired_unit_value(
            pd.DataFrame(), bootstrap_iterations=0
        )
