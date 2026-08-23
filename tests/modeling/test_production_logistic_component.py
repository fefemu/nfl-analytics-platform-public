"""Tests for the production logistic component."""

import numpy as np
import pandas as pd
import pytest

from src.modeling.production_logistic_component import (
    LOGISTIC_FEATURE_CONTRIBUTION_COLUMNS,
    LOGISTIC_FEATURE_COVERAGE_COLUMN,
    LOGISTIC_INTERCEPT_COLUMN,
    LOGISTIC_PROBABILITY_COLUMN,
    LOGISTIC_RECONSTRUCTED_PROBABILITY_COLUMN,
    LOGISTIC_TOTAL_LOG_ODDS_COLUMN,
    LOG_ODDS_CONTRIBUTION_SUFFIX,
    STANDARDIZED_VALUE_SUFFIX,
    calculate_logistic_feature_contributions,
    create_long_logistic_feature_contributions,
    prepare_production_training_data,
    score_current_logistic_component,
    train_production_logistic_component,
)
from src.modeling.production_probability_model import (
    PRODUCTION_PROBABILITY_MODEL,
)
from src.modeling.train_logistic_baseline import (
    TARGET_COLUMN,
    TRAIN_SPLIT,
)


FEATURE_COLUMNS = (
    PRODUCTION_PROBABILITY_MODEL
    .logistic_feature_columns
)


def create_historical_data() -> pd.DataFrame:
    """Create a small production training dataset."""

    rows: list[dict[str, object]] = []

    for index in range(12):
        home_win = index % 2

        rows.append(
            {
                "game_id": f"game_{index}",
                "season": 2020 + index // 3,
                TARGET_COLUMN: home_win,
                "has_complete_injury_data": True,
                "external_nfelo_rating_difference": (
                    80.0
                    if home_win
                    else -80.0
                ),
                "listed_qb_rating_difference": (
                    4.0
                    if home_win
                    else -4.0
                ),
                "external_nfelo_qb_adjustment_difference": (
                    6.0
                    if home_win
                    else -6.0
                ),
                "offense_injury_burden_difference": (
                    -0.20
                    if home_win
                    else 0.20
                ),
                "defense_injury_burden_difference": (
                    -0.15
                    if home_win
                    else 0.15
                ),
                "special_teams_injury_burden_difference": (
                    -0.05
                    if home_win
                    else 0.05
                ),
            }
        )

    rows.append(
        {
            "game_id": "future_2026",
            "season": 2026,
            TARGET_COLUMN: 1,
            "has_complete_injury_data": True,
            "external_nfelo_rating_difference": 100.0,
            "listed_qb_rating_difference": 5.0,
            "external_nfelo_qb_adjustment_difference": 8.0,
            "offense_injury_burden_difference": -0.2,
            "defense_injury_burden_difference": -0.1,
            "special_teams_injury_burden_difference": -0.1,
        }
    )

    rows.append(
        {
            "game_id": "missing_injury",
            "season": 2025,
            TARGET_COLUMN: 0,
            "has_complete_injury_data": False,
            "external_nfelo_rating_difference": -50.0,
            "listed_qb_rating_difference": -2.0,
            "external_nfelo_qb_adjustment_difference": -4.0,
            "offense_injury_burden_difference": 0.1,
            "defense_injury_burden_difference": 0.1,
            "special_teams_injury_burden_difference": 0.1,
        }
    )

    rows.append(
        {
            "game_id": "missing_feature",
            "season": 2025,
            TARGET_COLUMN: 1,
            "has_complete_injury_data": True,
            "external_nfelo_rating_difference": 50.0,
            "listed_qb_rating_difference": np.nan,
            "external_nfelo_qb_adjustment_difference": 3.0,
            "offense_injury_burden_difference": -0.1,
            "defense_injury_burden_difference": -0.1,
            "special_teams_injury_burden_difference": -0.1,
        }
    )

    return pd.DataFrame(rows)


def create_current_features() -> pd.DataFrame:
    """Create complete and incomplete current games."""

    return pd.DataFrame(
        [
            {
                "game_id": "complete_game",
                "has_complete_injury_data": True,
                "external_nfelo_rating_difference": 60.0,
                "listed_qb_rating_difference": 3.0,
                "external_nfelo_qb_adjustment_difference": 5.0,
                "offense_injury_burden_difference": -0.2,
                "defense_injury_burden_difference": -0.1,
                "special_teams_injury_burden_difference": -0.05,
            },
            {
                "game_id": "missing_injury_game",
                "has_complete_injury_data": False,
                "external_nfelo_rating_difference": 20.0,
                "listed_qb_rating_difference": 1.0,
                "external_nfelo_qb_adjustment_difference": 2.0,
                "offense_injury_burden_difference": 0.0,
                "defense_injury_burden_difference": 0.0,
                "special_teams_injury_burden_difference": 0.0,
            },
            {
                "game_id": "missing_qb_game",
                "has_complete_injury_data": True,
                "external_nfelo_rating_difference": 10.0,
                "listed_qb_rating_difference": np.nan,
                "external_nfelo_qb_adjustment_difference": 1.0,
                "offense_injury_burden_difference": 0.0,
                "defense_injury_burden_difference": 0.0,
                "special_teams_injury_burden_difference": 0.0,
            },
        ]
    )


def test_prepare_training_data_uses_eligible_history(
) -> None:
    """Keep only complete games before 2026."""

    training_data = (
        prepare_production_training_data(
            create_historical_data()
        )
    )

    assert len(training_data) == 12

    assert (
        training_data["season"].max()
        < 2026
    )

    assert (
        training_data[
            "has_complete_injury_data"
        ].all()
    )

    assert (
        training_data[
            list(FEATURE_COLUMNS)
        ].notna().all().all()
    )

    assert set(
        training_data["split_name"]
    ) == {
        TRAIN_SPLIT,
    }


def test_prepare_training_data_rejects_missing_columns(
) -> None:
    """Reject an incomplete historical schema."""

    historical_data = (
        create_historical_data().drop(
            columns=[
                "external_nfelo_rating_difference",
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match="is missing columns",
    ):
        prepare_production_training_data(
            historical_data
        )


def test_prepare_training_data_rejects_duplicates(
) -> None:
    """Reject duplicate historical game identifiers."""

    historical_data = (
        create_historical_data()
    )

    duplicated_data = pd.concat(
        [
            historical_data,
            historical_data.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate game identifiers",
    ):
        prepare_production_training_data(
            duplicated_data
        )


def test_prepare_training_data_rejects_empty_sample(
) -> None:
    """Reject history without eligible games."""

    historical_data = (
        create_historical_data()
    )

    historical_data[
        "has_complete_injury_data"
    ] = False

    with pytest.raises(
        RuntimeError,
        match="No eligible historical games",
    ):
        prepare_production_training_data(
            historical_data
        )


def test_train_production_logistic_component(
) -> None:
    """Train the selected logistic component."""

    model = (
        train_production_logistic_component(
            create_historical_data()
        )
    )

    assert hasattr(
        model,
        "predict_proba",
    )


def test_score_current_features_preserves_all_games(
) -> None:
    """Score complete rows and preserve fallback rows."""

    model = (
        train_production_logistic_component(
            create_historical_data()
        )
    )

    scored = (
        score_current_logistic_component(
            current_features=(
                create_current_features()
            ),
            trained_model=model,
        )
    )

    assert list(
        scored["game_id"]
    ) == [
        "complete_game",
        "missing_injury_game",
        "missing_qb_game",
    ]

    complete_row = scored.loc[
        scored["game_id"]
        == "complete_game"
    ].iloc[0]

    assert bool(
        complete_row[
            LOGISTIC_FEATURE_COVERAGE_COLUMN
        ]
    )

    assert (
        0.0
        <= complete_row[
            LOGISTIC_PROBABILITY_COLUMN
        ]
        <= 1.0
    )

    fallback_rows = scored.loc[
        scored["game_id"].isin(
            [
                "missing_injury_game",
                "missing_qb_game",
            ]
        )
    ]

    assert not fallback_rows[
        LOGISTIC_FEATURE_COVERAGE_COLUMN
    ].any()

    assert fallback_rows[
        LOGISTIC_PROBABILITY_COLUMN
    ].isna().all()


def test_score_current_features_rejects_duplicates(
) -> None:
    """Reject duplicate current game identifiers."""

    model = (
        train_production_logistic_component(
            create_historical_data()
        )
    )

    current_features = (
        create_current_features()
    )

    duplicated_features = pd.concat(
        [
            current_features,
            current_features.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate game identifiers",
    ):
        score_current_logistic_component(
            current_features=(
                duplicated_features
            ),
            trained_model=model,
        )


def test_score_current_features_rejects_missing_columns(
) -> None:
    """Reject an incomplete current feature schema."""

    model = (
        train_production_logistic_component(
            create_historical_data()
        )
    )

    current_features = (
        create_current_features().drop(
            columns=[
                "listed_qb_rating_difference",
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match="is missing columns",
    ):
        score_current_logistic_component(
            current_features=(
                current_features
            ),
            trained_model=model,
        )


def test_calculate_logistic_feature_contributions(
) -> None:
    """Reconstruct the fitted logistic probability."""

    model = (
        train_production_logistic_component(
            create_historical_data()
        )
    )

    contributions = (
        calculate_logistic_feature_contributions(
            current_features=(
                create_current_features()
            ),
            trained_model=model,
        )
    )

    assert list(
        contributions["game_id"]
    ) == [
        "complete_game",
    ]

    contribution_row = contributions.iloc[0]

    contribution_sum = sum(
        contribution_row[
            (
                feature_name
                + LOG_ODDS_CONTRIBUTION_SUFFIX
            )
        ]
        for feature_name in FEATURE_COLUMNS
    )

    expected_log_odds = (
        contribution_row[
            LOGISTIC_INTERCEPT_COLUMN
        ]
        + contribution_sum
    )

    assert contribution_row[
        LOGISTIC_TOTAL_LOG_ODDS_COLUMN
    ] == pytest.approx(
        expected_log_odds
    )

    expected_probability = (
        model.predict_proba(
            create_current_features().loc[
                [0],
                list(FEATURE_COLUMNS),
            ]
        )[0, 1]
    )

    assert contribution_row[
        LOGISTIC_RECONSTRUCTED_PROBABILITY_COLUMN
    ] == pytest.approx(
        expected_probability
    )


def test_contributions_equal_value_times_coefficient(
) -> None:
    """Calculate each exact standardized contribution."""

    model = (
        train_production_logistic_component(
            create_historical_data()
        )
    )

    contributions = (
        calculate_logistic_feature_contributions(
            current_features=(
                create_current_features()
            ),
            trained_model=model,
        )
    )

    contribution_row = contributions.iloc[0]

    coefficients = (
        model.named_steps[
            "model"
        ].coef_[0]
    )

    for feature_index, feature_name in enumerate(
        FEATURE_COLUMNS
    ):
        standardized_value = contribution_row[
            (
                feature_name
                + STANDARDIZED_VALUE_SUFFIX
            )
        ]

        actual_contribution = contribution_row[
            (
                feature_name
                + LOG_ODDS_CONTRIBUTION_SUFFIX
            )
        ]

        assert actual_contribution == pytest.approx(
            standardized_value
            * coefficients[feature_index]
        )


def test_contributions_support_no_complete_games(
) -> None:
    """Return a stable empty contribution frame."""

    model = (
        train_production_logistic_component(
            create_historical_data()
        )
    )

    current_features = (
        create_current_features()
    )

    current_features[
        "has_complete_injury_data"
    ] = False

    contributions = (
        calculate_logistic_feature_contributions(
            current_features=(
                current_features
            ),
            trained_model=model,
        )
    )

    assert contributions.empty

    assert (
        LOGISTIC_TOTAL_LOG_ODDS_COLUMN
        in contributions.columns
    )

    assert (
        LOGISTIC_RECONSTRUCTED_PROBABILITY_COLUMN
        in contributions.columns
    )


def test_create_long_feature_contributions(
) -> None:
    """Create one exact row per complete game-feature."""

    model = (
        train_production_logistic_component(
            create_historical_data()
        )
    )

    contributions = (
        create_long_logistic_feature_contributions(
            current_features=(
                create_current_features()
            ),
            trained_model=model,
        )
    )

    assert tuple(
        contributions.columns
    ) == LOGISTIC_FEATURE_CONTRIBUTION_COLUMNS

    assert set(
        contributions["game_id"]
    ) == {
        "complete_game",
    }

    assert len(contributions) == len(
        FEATURE_COLUMNS
    )

    assert set(
        contributions["feature_name"]
    ) == set(FEATURE_COLUMNS)


def test_long_contributions_preserve_exact_math(
) -> None:
    """Preserve value times coefficient mathematics."""

    model = (
        train_production_logistic_component(
            create_historical_data()
        )
    )

    contributions = (
        create_long_logistic_feature_contributions(
            current_features=(
                create_current_features()
            ),
            trained_model=model,
        )
    )

    expected_contributions = (
        contributions[
            "standardized_feature_value"
        ]
        * contributions["coefficient"]
    )

    assert np.allclose(
        contributions[
            "log_odds_contribution"
        ],
        expected_contributions,
    )

    assert np.allclose(
        contributions[
            "absolute_log_odds_contribution"
        ],
        contributions[
            "log_odds_contribution"
        ].abs(),
    )


def test_long_contributions_rank_absolute_impact(
) -> None:
    """Rank the strongest absolute contribution first."""

    model = (
        train_production_logistic_component(
            create_historical_data()
        )
    )

    contributions = (
        create_long_logistic_feature_contributions(
            current_features=(
                create_current_features()
            ),
            trained_model=model,
        )
    )

    assert list(
        contributions["contribution_rank"]
    ) == list(
        range(
            1,
            len(FEATURE_COLUMNS) + 1,
        )
    )

    absolute_contributions = list(
        contributions[
            "absolute_log_odds_contribution"
        ]
    )

    assert absolute_contributions == sorted(
        absolute_contributions,
        reverse=True,
    )


def test_long_contributions_support_fallback_only(
) -> None:
    """Return stable schema without eligible games."""

    model = (
        train_production_logistic_component(
            create_historical_data()
        )
    )

    current_features = (
        create_current_features()
    )

    current_features[
        "has_complete_injury_data"
    ] = False

    contributions = (
        create_long_logistic_feature_contributions(
            current_features=(
                current_features
            ),
            trained_model=model,
        )
    )

    assert contributions.empty

    assert tuple(
        contributions.columns
    ) == LOGISTIC_FEATURE_CONTRIBUTION_COLUMNS


def test_long_contributions_reconstruct_probability(
) -> None:
    """Reconstruct probability from intercept and impacts."""

    model = (
        train_production_logistic_component(
            create_historical_data()
        )
    )

    contributions = (
        create_long_logistic_feature_contributions(
            current_features=(
                create_current_features()
            ),
            trained_model=model,
        )
    )

    intercept = contributions[
        LOGISTIC_INTERCEPT_COLUMN
    ].iloc[0]

    total_log_odds = contributions[
        LOGISTIC_TOTAL_LOG_ODDS_COLUMN
    ].iloc[0]

    reconstructed_probability = contributions[
        LOGISTIC_RECONSTRUCTED_PROBABILITY_COLUMN
    ].iloc[0]

    expected_log_odds = (
        intercept
        + contributions[
            "log_odds_contribution"
        ].sum()
    )

    expected_probability = (
        1.0
        / (
            1.0
            + np.exp(-expected_log_odds)
        )
    )

    assert contributions[
        LOGISTIC_INTERCEPT_COLUMN
    ].nunique() == 1

    assert contributions[
        LOGISTIC_TOTAL_LOG_ODDS_COLUMN
    ].nunique() == 1

    assert total_log_odds == pytest.approx(
        expected_log_odds
    )

    assert reconstructed_probability == pytest.approx(
        expected_probability
    )