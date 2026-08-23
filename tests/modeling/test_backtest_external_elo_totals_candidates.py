"""Tests for external Elo Totals candidate backtests."""

import pandas as pd
import pytest

from src.modeling.backtest_external_elo_totals_candidates import (
    CANDIDATES,
    EXTERNAL_ELO_SUM_FEATURE,
    EXTERNAL_QB_SUM_FEATURE,
    FALLBACK_BASE_FEATURES,
    FOLD_RESULT_COLUMNS,
    PRIMARY_BASE_FEATURES,
    ROUTING_FALLBACK,
    ROUTING_PRIMARY,
    SUMMARY_COLUMNS,
    add_external_totals_features,
    evaluate_external_elo_totals_candidates,
    evaluate_routing_candidates,
    prepare_routing_sample,
)
from src.modeling.evaluate_totals_model_candidates import (
    TOTALS_TARGET_COLUMN,
)
from src.modeling.production_totals_model import (
    PRODUCTION_TOTALS_MODEL,
)


TEST_VALIDATION_SEASONS = (
    2021,
    2022,
)


def create_base_rows() -> list[
    dict[str, object]
]:
    """Create chronological synthetic game rows."""

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

    return rows


def create_primary_data() -> pd.DataFrame:
    """Create a complete primary candidate sample."""

    return pd.DataFrame(
        create_base_rows()
    )


def create_fallback_data() -> pd.DataFrame:
    """Create a complete fallback candidate sample."""

    data = pd.DataFrame(
        create_base_rows()
    )

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


def test_external_features_join_one_to_one():
    """External features should join by game ID."""

    development_data = create_primary_data().drop(
        columns=[
            EXTERNAL_ELO_SUM_FEATURE,
            EXTERNAL_QB_SUM_FEATURE,
        ]
    )

    external_features = create_primary_data().loc[
        :,
        [
            "game_id",
            EXTERNAL_ELO_SUM_FEATURE,
            EXTERNAL_QB_SUM_FEATURE,
        ],
    ]

    merged = add_external_totals_features(
        development_data=development_data,
        external_features=external_features,
    )

    assert len(merged) == len(
        development_data
    )

    assert merged[
        EXTERNAL_ELO_SUM_FEATURE
    ].notna().all()

    assert merged[
        EXTERNAL_QB_SUM_FEATURE
    ].notna().all()


def test_missing_external_coverage_is_rejected():
    """Every development game needs external ratings."""

    development_data = create_primary_data().drop(
        columns=[
            EXTERNAL_ELO_SUM_FEATURE,
            EXTERNAL_QB_SUM_FEATURE,
        ]
    )

    external_features = (
        create_primary_data()
        .iloc[:-1]
        .loc[
            :,
            [
                "game_id",
                EXTERNAL_ELO_SUM_FEATURE,
                EXTERNAL_QB_SUM_FEATURE,
            ],
        ]
    )

    with pytest.raises(
        RuntimeError,
        match="coverage is missing",
    ):
        add_external_totals_features(
            development_data=development_data,
            external_features=external_features,
        )


def test_routing_sample_rejects_holdout():
    """The protected 2025 holdout must remain closed."""

    data = create_primary_data()

    data.loc[
        data.index[-1],
        "split_name",
    ] = "holdout"

    primary_candidates = tuple(
        candidate
        for candidate in CANDIDATES
        if candidate.routing_layer
        == ROUTING_PRIMARY
    )

    with pytest.raises(
        ValueError,
        match="must not contain holdout",
    ):
        prepare_routing_sample(
            development_data=data,
            candidates=primary_candidates,
        )


def test_routing_sample_rejects_2025():
    """No 2025 data may enter development evaluation."""

    data = create_fallback_data()

    data.loc[
        data.index[-1],
        "season",
    ] = 2025

    fallback_candidates = tuple(
        candidate
        for candidate in CANDIDATES
        if candidate.routing_layer
        == ROUTING_FALLBACK
    )

    with pytest.raises(
        ValueError,
        match="must end before the 2025 holdout",
    ):
        prepare_routing_sample(
            development_data=data,
            candidates=fallback_candidates,
        )


def test_candidate_grid_preserves_locked_alphas():
    """Candidate extensions must retain locked alphas."""

    primary_candidates = [
        candidate
        for candidate in CANDIDATES
        if candidate.routing_layer
        == ROUTING_PRIMARY
    ]

    fallback_candidates = [
        candidate
        for candidate in CANDIDATES
        if candidate.routing_layer
        == ROUTING_FALLBACK
    ]

    assert len(primary_candidates) == 4
    assert len(fallback_candidates) == 5

    assert all(
        candidate.ridge_alpha
        == PRODUCTION_TOTALS_MODEL.ridge_alpha
        for candidate in primary_candidates
    )

    assert all(
        candidate.ridge_alpha
        == (
            PRODUCTION_TOTALS_MODEL
            .fallback_ridge_alpha
        )
        for candidate in fallback_candidates
    )


def test_primary_routing_returns_every_candidate():
    """Primary candidates should cover every fold."""

    primary_candidates = tuple(
        candidate
        for candidate in CANDIDATES
        if candidate.routing_layer
        == ROUTING_PRIMARY
    )

    summary, fold_results = (
        evaluate_routing_candidates(
            development_data=create_primary_data(),
            candidates=primary_candidates,
            validation_seasons=(
                TEST_VALIDATION_SEASONS
            ),
        )
    )

    assert tuple(summary.columns) == (
        SUMMARY_COLUMNS
    )

    assert tuple(fold_results.columns) == (
        FOLD_RESULT_COLUMNS
    )

    assert len(summary) == 4
    assert len(fold_results) == 8

    assert set(
        summary["candidate_name"]
    ) == {
        candidate.candidate_name
        for candidate in primary_candidates
    }

    assert (
        summary["validation_game_count"] == 4
    ).all()

    assert (
        summary["fold_count"] == 2
    ).all()


def test_fallback_routing_returns_every_candidate():
    """Fallback candidates should cover every fold."""

    fallback_candidates = tuple(
        candidate
        for candidate in CANDIDATES
        if candidate.routing_layer
        == ROUTING_FALLBACK
    )

    summary, fold_results = (
        evaluate_routing_candidates(
            development_data=create_fallback_data(),
            candidates=fallback_candidates,
            validation_seasons=(
                TEST_VALIDATION_SEASONS
            ),
        )
    )

    assert len(summary) == 5
    assert len(fold_results) == 10

    assert set(
        summary["candidate_name"]
    ) == {
        candidate.candidate_name
        for candidate in fallback_candidates
    }

    assert (
        summary["routing_layer"]
        == ROUTING_FALLBACK
    ).all()


def test_combined_backtest_keeps_routing_layers_separate():
    """Primary and fallback panels must remain separate."""

    summary, fold_results = (
        evaluate_external_elo_totals_candidates(
            primary_data=create_primary_data(),
            fallback_data=create_fallback_data(),
            validation_seasons=(
                TEST_VALIDATION_SEASONS
            ),
        )
    )

    assert len(summary) == 9
    assert len(fold_results) == 18

    assert set(
        summary["routing_layer"]
    ) == {
        ROUTING_PRIMARY,
        ROUTING_FALLBACK,
    }

    candidate_counts = (
        summary.groupby(
            "routing_layer"
        )["candidate_name"].nunique()
        .to_dict()
    )

    assert candidate_counts == {
        ROUTING_PRIMARY: 4,
        ROUTING_FALLBACK: 5,
    }

    assert (
        summary["validation_mae"] >= 0.0
    ).all()

    assert (
        summary["validation_rmse"] >= 0.0
    ).all()