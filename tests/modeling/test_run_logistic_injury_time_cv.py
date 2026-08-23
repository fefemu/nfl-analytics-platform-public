"""Tests for logistic injury feature time-CV."""

import duckdb
import pytest

from src.modeling.run_logistic_injury_time_cv import (
    INJURY_DEVELOPMENT_COLUMNS,
    INJURY_FEATURE_GROUPS,
    load_injury_development_data,
    validate_injury_dataset_columns,
    validate_injury_feature_groups,
)


def create_injury_experiment_sources(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create minimal injury experiment sources."""

    feature_definitions = ",\n".join(
        f"{column_name} DOUBLE"
        for column_name in INJURY_DEVELOPMENT_COLUMNS
    )

    connection.execute(
        f"""
        CREATE SCHEMA analytics;

        CREATE TABLE analytics.game_modeling_dataset (
            game_id VARCHAR,
            season INTEGER,
            game_date DATE,
            target_home_win INTEGER,
            elo_home_win_probability DOUBLE,
            has_complete_injury_data BOOLEAN,
            {feature_definitions}
        );

        CREATE TABLE analytics.modeling_game_splits (
            game_id VARCHAR,
            split_name VARCHAR,
            is_core_model_eligible BOOLEAN
        );
        """
    )

    feature_count = len(
        INJURY_DEVELOPMENT_COLUMNS
    )

    dataset_placeholders = ", ".join(
        "?"
        for _ in range(
            6 + feature_count
        )
    )

    base_features = tuple(
        float(index + 1)
        for index in range(
            feature_count
        )
    )

    dataset_rows = [
        (
            "train_complete",
            2020,
            "2020-09-10",
            1,
            0.60,
            True,
            *base_features,
        ),
        (
            "train_incomplete",
            2021,
            "2021-09-10",
            0,
            0.45,
            False,
            *base_features,
        ),
        (
            "validation_complete",
            2023,
            "2023-09-10",
            1,
            0.65,
            True,
            *base_features,
        ),
        (
            "holdout_complete",
            2025,
            "2025-09-10",
            0,
            0.40,
            True,
            *base_features,
        ),
        (
            "non_core_complete",
            2022,
            "2022-09-10",
            1,
            0.55,
            True,
            *base_features,
        ),
    ]

    connection.executemany(
        f"""
        INSERT INTO analytics.game_modeling_dataset
        VALUES ({dataset_placeholders})
        """,
        dataset_rows,
    )

    connection.execute(
        """
        INSERT INTO analytics.modeling_game_splits
        VALUES
            (
                'train_complete',
                'train',
                TRUE
            ),
            (
                'train_incomplete',
                'train',
                TRUE
            ),
            (
                'validation_complete',
                'validation',
                TRUE
            ),
            (
                'holdout_complete',
                'holdout',
                TRUE
            ),
            (
                'non_core_complete',
                'train',
                FALSE
            );
        """
    )


def test_validate_injury_feature_groups_accepts_defaults(
) -> None:
    """Accept the configured injury feature groups."""

    validate_injury_feature_groups()


def test_validate_injury_feature_groups_rejects_empty(
) -> None:
    """Reject an empty injury experiment."""

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        validate_injury_feature_groups({})


def test_validate_injury_feature_groups_rejects_unknown(
) -> None:
    """Reject an unknown feature column."""

    with pytest.raises(
        ValueError,
        match="Unknown injury features",
    ):
        validate_injury_feature_groups(
            {
                "invalid": (
                    "unknown_feature",
                ),
            }
        )


def test_feature_groups_keep_qb_burden_separate(
) -> None:
    """Do not double count the QB injury layer."""

    selected_features = {
        feature_name
        for feature_columns
        in INJURY_FEATURE_GROUPS.values()
        for feature_name in feature_columns
    }

    assert "qb_injury_burden_difference" not in (
        selected_features
    )
    assert "qb_out_count_difference" not in (
        selected_features
    )


def test_validate_injury_dataset_columns_accepts_source(
) -> None:
    """Accept a modeling dataset with injury columns."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_injury_experiment_sources(
            connection
        )

        validate_injury_dataset_columns(
            connection
        )


def test_load_injury_development_data_filters_safely(
) -> None:
    """Exclude incomplete, non-core and holdout games."""

    with duckdb.connect(
        ":memory:"
    ) as connection:
        create_injury_experiment_sources(
            connection
        )

        data = load_injury_development_data(
            connection
        )

    assert data["game_id"].tolist() == [
        "train_complete",
        "validation_complete",
    ]

    assert set(
        data["split_name"]
    ) == {
        "train",
        "validation",
    }

    assert data[
        "has_complete_injury_data"
    ].all()

    assert "holdout" not in set(
        data["split_name"]
    )