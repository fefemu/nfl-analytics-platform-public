"""Tests for the external probability fallback holdout."""

import numpy as np
import pandas as pd
import pytest

from src.modeling.backtest_external_elo_probability_champion import (
    EXTERNAL_ELO_FEATURE,
    EXTERNAL_QB_FEATURE,
)
from src.modeling.backtest_external_probability_fallback import (
    CURRENT_FALLBACK_CANDIDATE,
    EXTERNAL_LOGISTIC_CANDIDATE,
    PRIMARY_ELIGIBILITY_COLUMN,
)
from src.modeling.evaluate_external_probability_fallback_holdout import (
    PREDICTION_COLUMNS,
    SUMMARY_COLUMNS,
    evaluate_locked_probability_fallback_holdout,
    load_probability_fallback_holdout_data,
    prepare_probability_fallback_holdout,
)
from src.modeling.external_probability_holdout_component import (
    HOLDOUT_SEASON,
    HOLDOUT_SPLIT,
)
from src.modeling.run_logistic_injury_time_cv import (
    INJURY_AVAILABILITY_COLUMN,
    UNIT_BURDEN_FEATURES,
)
from src.modeling.train_logistic_baseline import (
    TARGET_COLUMN,
)


def create_test_data() -> pd.DataFrame:
    """Create synthetic fallback holdout data."""

    definitions = [
        (
            "2020_01_A_B",
            2020,
            "train",
            True,
            True,
            1,
            80.0,
            100.0,
            20.0,
            15.0,
            0.68,
        ),
        (
            "2020_02_C_D",
            2020,
            "train",
            True,
            True,
            0,
            -60.0,
            -90.0,
            -15.0,
            -12.0,
            0.32,
        ),
        (
            "2021_01_E_F",
            2021,
            "validation",
            False,
            False,
            1,
            30.0,
            50.0,
            8.0,
            6.0,
            0.58,
        ),
        (
            "2021_02_G_H",
            2021,
            "validation",
            False,
            False,
            0,
            -20.0,
            -40.0,
            -6.0,
            -5.0,
            0.42,
        ),
        (
            "2025_01_A_C",
            HOLDOUT_SEASON,
            HOLDOUT_SPLIT,
            True,
            True,
            1,
            45.0,
            70.0,
            12.0,
            9.0,
            0.62,
        ),
        (
            "2025_02_D_B",
            HOLDOUT_SEASON,
            HOLDOUT_SPLIT,
            False,
            False,
            0,
            -35.0,
            -55.0,
            -10.0,
            -8.0,
            0.38,
        ),
        (
            "2025_03_E_G",
            HOLDOUT_SEASON,
            HOLDOUT_SPLIT,
            True,
            False,
            1,
            55.0,
            85.0,
            14.0,
            11.0,
            0.65,
        ),
    ]

    rows: list[dict[str, object]] = []

    for row_index, (
        game_id,
        season,
        split_name,
        core_eligible,
        injury_complete,
        target,
        internal_elo,
        external_elo,
        listed_qb,
        external_qb,
        internal_probability,
    ) in enumerate(definitions):
        row: dict[str, object] = {
            "game_id": game_id,
            "season": season,
            "split_name": split_name,
            "is_core_model_eligible": (
                core_eligible
            ),
            TARGET_COLUMN: target,
            INJURY_AVAILABILITY_COLUMN: (
                injury_complete
            ),
            "elo_rating_difference": (
                internal_elo
            ),
            "listed_qb_rating_difference": (
                listed_qb
            ),
            EXTERNAL_ELO_FEATURE: (
                external_elo
            ),
            EXTERNAL_QB_FEATURE: (
                external_qb
            ),
            "elo_home_win_probability": (
                internal_probability
            ),
        }

        for (
            feature_index,
            feature_name,
        ) in enumerate(
            UNIT_BURDEN_FEATURES
        ):
            direction = (
                1.0
                if target == 1
                else -1.0
            )

            row[feature_name] = (
                direction
                * (
                    feature_index + 1
                )
                + row_index * 0.01
            )

        rows.append(row)

    return pd.DataFrame(rows)


def test_prepare_selects_only_fallback_holdout():
    """Primary-eligible holdout games are excluded."""

    (
        training_data,
        fallback_holdout,
    ) = prepare_probability_fallback_holdout(
        create_test_data()
    )

    assert len(training_data) == 4
    assert len(fallback_holdout) == 2

    assert set(
        fallback_holdout["game_id"]
    ) == {
        "2025_02_D_B",
        "2025_03_E_G",
    }

    assert (
        ~fallback_holdout[
            PRIMARY_ELIGIBILITY_COLUMN
        ]
    ).all()


def test_core_and_injury_both_control_routing():
    """Either missing condition triggers fallback."""

    _, fallback_holdout = (
        prepare_probability_fallback_holdout(
            create_test_data()
        )
    )

    non_core_game = fallback_holdout.loc[
        fallback_holdout["game_id"]
        == "2025_02_D_B"
    ].iloc[0]

    incomplete_injury_game = (
        fallback_holdout.loc[
            fallback_holdout["game_id"]
            == "2025_03_E_G"
        ].iloc[0]
    )

    assert not bool(
        non_core_game[
            "is_core_model_eligible"
        ]
    )

    assert not bool(
        incomplete_injury_game[
            INJURY_AVAILABILITY_COLUMN
        ]
    )


def test_prepare_rejects_duplicate_games():
    """Every game identifier must be unique."""

    data = create_test_data()

    data.loc[
        data.index[-1],
        "game_id",
    ] = data.loc[
        data.index[0],
        "game_id",
    ]

    with pytest.raises(
        ValueError,
        match="duplicate game identifiers",
    ):
        prepare_probability_fallback_holdout(
            data
        )


def test_prepare_rejects_post_holdout_games():
    """No post-2025 data may enter evaluation."""

    data = create_test_data()

    data.loc[
        data.index[-1],
        "season",
    ] = 2026

    with pytest.raises(
        ValueError,
        match="post-2025",
    ):
        prepare_probability_fallback_holdout(
            data
        )


def test_prepare_requires_fallback_holdout_games():
    """At least one routed fallback game is required."""

    data = create_test_data()

    holdout_mask = (
        data["season"] == HOLDOUT_SEASON
    )

    data.loc[
        holdout_mask,
        "is_core_model_eligible",
    ] = True

    data.loc[
        holdout_mask,
        INJURY_AVAILABILITY_COLUMN,
    ] = True

    with pytest.raises(
        RuntimeError,
        match="No routed 2025",
    ):
        prepare_probability_fallback_holdout(
            data
        )


def test_evaluation_returns_locked_candidates():
    """Current and external fallbacks are evaluated."""

    (
        summary,
        paired_summary,
        predictions,
    ) = evaluate_locked_probability_fallback_holdout(
        create_test_data()
    )

    assert tuple(summary.columns) == (
        SUMMARY_COLUMNS
    )

    assert tuple(predictions.columns) == (
        PREDICTION_COLUMNS
    )

    assert set(
        summary["candidate_name"]
    ) == {
        CURRENT_FALLBACK_CANDIDATE,
        EXTERNAL_LOGISTIC_CANDIDATE,
    }

    assert len(summary) == 2
    assert len(predictions) == 2
    assert len(paired_summary) == 1

    assert (
        summary[
            "fallback_holdout_game_count"
        ] == 2
    ).all()

    assert summary["brier_score"].between(
        0.0,
        1.0,
    ).all()


def test_prediction_delta_matches_brier_losses():
    """Paired loss delta must be external minus current."""

    (
        _,
        paired_summary,
        predictions,
    ) = evaluate_locked_probability_fallback_holdout(
        create_test_data()
    )

    expected_delta = (
        predictions["external_brier_loss"]
        - predictions["current_brier_loss"]
    )

    np.testing.assert_allclose(
        predictions[
            "external_brier_loss_delta"
        ],
        expected_delta,
    )

    assert (
        paired_summary.iloc[0][
            "external_mean_loss_delta"
        ]
        == pytest.approx(
            expected_delta.mean()
        )
    )


def test_loader_does_not_filter_out_fallback_rows():
    """SQL must load core and non-core games."""

    class RecordingConnection:
        """Record one DuckDB query."""

        def __init__(self):
            self.query = ""

        def execute(self, query):
            self.query = query
            return self

        def fetchdf(self):
            return pd.DataFrame(
                {
                    "game_id": [
                        "synthetic_game",
                    ],
                }
            )

    connection = RecordingConnection()

    result = (
        load_probability_fallback_holdout_data(
            connection
        )
    )

    normalized_query = " ".join(
        connection.query.split()
    ).lower()

    assert not result.empty

    assert (
        "splits.is_core_model_eligible,"
        in normalized_query
    )

    assert (
        "and splits.is_core_model_eligible"
        not in normalized_query
    )

    assert (
        "dataset.season <= 2025"
        in normalized_query
    )