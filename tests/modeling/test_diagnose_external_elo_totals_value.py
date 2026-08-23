"""Tests for paired external Elo Totals diagnostics."""

import numpy as np
import pandas as pd
import pytest

from src.modeling.backtest_external_elo_totals_candidates import (
    CANDIDATES,
    EXTERNAL_ELO_SUM_FEATURE,
    EXTERNAL_QB_SUM_FEATURE,
    FALLBACK_BASE_FEATURES,
    PRIMARY_BASE_FEATURES,
    ROUTING_FALLBACK,
    ROUTING_PRIMARY,
)
from src.modeling.diagnose_external_elo_totals_value import (
    COMPARISONS,
    FOLD_RESULT_COLUMNS,
    PREDICTION_COLUMNS,
    SUMMARY_COLUMNS,
    bootstrap_paired_mean_delta,
    create_paired_comparison,
    create_routing_oof_predictions,
    diagnose_external_elo_totals_value,
)
from src.modeling.evaluate_totals_model_candidates import (
    TOTALS_TARGET_COLUMN,
)


TEST_VALIDATION_SEASONS = (
    2021,
    2022,
)


def create_base_data() -> pd.DataFrame:
    """Create chronological synthetic Totals data."""

    definitions = [
        (
            "2020_01_A_B",
            2020,
            "train",
            50.0,
            1.0,
        ),
        (
            "2020_02_C_D",
            2020,
            "train",
            38.0,
            -1.0,
        ),
        (
            "2020_03_E_F",
            2020,
            "train",
            47.0,
            0.7,
        ),
        (
            "2020_04_G_H",
            2020,
            "train",
            41.0,
            -0.6,
        ),
        (
            "2021_01_A_C",
            2021,
            "validation",
            49.0,
            0.8,
        ),
        (
            "2021_02_D_B",
            2021,
            "validation",
            40.0,
            -0.8,
        ),
        (
            "2022_01_E_G",
            2022,
            "validation",
            52.0,
            1.2,
        ),
        (
            "2022_02_H_F",
            2022,
            "validation",
            39.0,
            -1.2,
        ),
    ]

    rows: list[dict[str, object]] = []

    for row_index, (
        game_id,
        season,
        split_name,
        target_total,
        signal,
    ) in enumerate(definitions):
        row: dict[str, object] = {
            "game_id": game_id,
            "season": season,
            "split_name": split_name,
            TOTALS_TARGET_COLUMN: target_total,
        }

        for feature_index, feature_name in enumerate(
            PRIMARY_BASE_FEATURES
        ):
            row[feature_name] = (
                signal
                * (
                    feature_index + 1
                )
                + row_index * 0.01
            )

        row[
            "league_average_total_last_64"
        ] = 44.0 + row_index * 0.1

        row["is_indoor"] = float(
            row_index % 2
        )

        row["elo_rating_sum"] = (
            3000.0 + signal * 100.0
        )

        row[EXTERNAL_ELO_SUM_FEATURE] = (
            3050.0 + signal * 120.0
        )

        row[EXTERNAL_QB_SUM_FEATURE] = (
            signal * 15.0
        )

        rows.append(row)

    return pd.DataFrame(rows)


def create_primary_data() -> pd.DataFrame:
    """Create the complete primary sample."""

    return create_base_data()


def create_fallback_data() -> pd.DataFrame:
    """Create the complete fallback sample."""

    data = create_base_data()

    required_columns = {
        "game_id",
        "season",
        "split_name",
        TOTALS_TARGET_COLUMN,
        *FALLBACK_BASE_FEATURES,
        EXTERNAL_ELO_SUM_FEATURE,
        EXTERNAL_QB_SUM_FEATURE,
    }

    return data.loc[
        :,
        sorted(required_columns),
    ].copy()


def get_routing_candidates(
    routing_layer: str,
):
    """Return candidates from one routing layer."""

    return tuple(
        candidate
        for candidate in CANDIDATES
        if candidate.routing_layer
        == routing_layer
    )


def test_bootstrap_is_deterministic():
    """The same seed must return the same result."""

    deltas = np.array(
        [
            -1.0,
            -0.5,
            0.0,
            0.5,
            1.0,
        ]
    )

    first_result = bootstrap_paired_mean_delta(
        paired_deltas=deltas,
        iteration_count=500,
        random_seed=42,
    )

    second_result = bootstrap_paired_mean_delta(
        paired_deltas=deltas,
        iteration_count=500,
        random_seed=42,
    )

    assert first_result == second_result

    assert (
        first_result[
            "bootstrap_95_percent_lower"
        ]
        <= 0.0
    )

    assert (
        first_result[
            "bootstrap_95_percent_upper"
        ]
        >= 0.0
    )


def test_bootstrap_rejects_empty_input():
    """Bootstrap requires paired observations."""

    with pytest.raises(
        ValueError,
        match="non-empty",
    ):
        bootstrap_paired_mean_delta(
            paired_deltas=np.array([]),
            iteration_count=100,
        )


def test_oof_predictions_cover_primary_candidates():
    """Every primary candidate predicts every game."""

    primary_candidates = get_routing_candidates(
        ROUTING_PRIMARY
    )

    predictions = create_routing_oof_predictions(
        development_data=create_primary_data(),
        candidates=primary_candidates,
        validation_seasons=(
            TEST_VALIDATION_SEASONS
        ),
    )

    assert tuple(predictions.columns) == (
        PREDICTION_COLUMNS
    )

    assert len(predictions) == (
        4 * len(primary_candidates)
    )

    assert set(
        predictions["candidate_name"]
    ) == {
        candidate.candidate_name
        for candidate in primary_candidates
    }

    assert (
        predictions["routing_layer"]
        == ROUTING_PRIMARY
    ).all()

    assert predictions[
        [
            "candidate_name",
            "game_id",
        ]
    ].duplicated().sum() == 0


def test_oof_predictions_reject_holdout():
    """The diagnostic must not open holdout."""

    data = create_primary_data()

    data.loc[
        data.index[-1],
        "split_name",
    ] = "holdout"

    with pytest.raises(
        ValueError,
        match="must not contain holdout",
    ):
        create_routing_oof_predictions(
            development_data=data,
            candidates=get_routing_candidates(
                ROUTING_PRIMARY
            ),
            validation_seasons=(
                TEST_VALIDATION_SEASONS
            ),
        )


def test_paired_delta_matches_absolute_errors():
    """Delta must equal challenger minus base error."""

    predictions = create_routing_oof_predictions(
        development_data=create_primary_data(),
        candidates=get_routing_candidates(
            ROUTING_PRIMARY
        ),
        validation_seasons=(
            TEST_VALIDATION_SEASONS
        ),
    )

    paired = create_paired_comparison(
        predictions=predictions,
        routing_layer=ROUTING_PRIMARY,
        base_candidate="primary_current_locked",
        challenger_candidate=(
            "primary_external_qb_sum"
        ),
    )

    expected_delta = (
        paired["challenger_absolute_error"]
        - paired["base_absolute_error"]
    )

    np.testing.assert_allclose(
        paired["absolute_error_delta"],
        expected_delta,
    )

    assert len(paired) == 4


def test_missing_candidate_is_rejected():
    """Both comparison candidates must exist."""

    predictions = create_routing_oof_predictions(
        development_data=create_fallback_data(),
        candidates=get_routing_candidates(
            ROUTING_FALLBACK
        ),
        validation_seasons=(
            TEST_VALIDATION_SEASONS
        ),
    )

    with pytest.raises(
        ValueError,
        match="candidates are missing",
    ):
        create_paired_comparison(
            predictions=predictions,
            routing_layer=ROUTING_FALLBACK,
            base_candidate=(
                "fallback_current_locked"
            ),
            challenger_candidate=(
                "missing_candidate"
            ),
        )


def test_diagnostic_returns_every_comparison():
    """Every planned comparison must be returned."""

    (
        summary,
        fold_results,
        predictions,
    ) = diagnose_external_elo_totals_value(
        primary_data=create_primary_data(),
        fallback_data=create_fallback_data(),
        validation_seasons=(
            TEST_VALIDATION_SEASONS
        ),
        bootstrap_iterations=500,
        random_seed=42,
    )

    assert tuple(summary.columns) == (
        SUMMARY_COLUMNS
    )

    assert tuple(fold_results.columns) == (
        FOLD_RESULT_COLUMNS
    )

    assert tuple(predictions.columns) == (
        PREDICTION_COLUMNS
    )

    assert set(
        summary["comparison_name"]
    ) == set(COMPARISONS)

    assert len(summary) == 4
    assert len(fold_results) == 8

    assert (
        summary["fold_count"] == 2
    ).all()

    assert (
        summary["validation_game_count"] == 4
    ).all()


def test_summary_rates_form_complete_partition():
    """Win, loss and equal rates must total one."""

    summary, _, _ = (
        diagnose_external_elo_totals_value(
            primary_data=create_primary_data(),
            fallback_data=create_fallback_data(),
            validation_seasons=(
                TEST_VALIDATION_SEASONS
            ),
            bootstrap_iterations=500,
            random_seed=42,
        )
    )

    total_rate = (
        summary["challenger_win_rate"]
        + summary["challenger_loss_rate"]
        + summary["equal_error_rate"]
    )

    np.testing.assert_allclose(
        total_rate,
        1.0,
    )

    assert (
        summary["base_mae"] >= 0.0
    ).all()

    assert (
        summary["challenger_mae"] >= 0.0
    ).all()