"""
Tests for the logistic regression baseline trainer.
"""

from collections.abc import Iterator

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.modeling.train_logistic_baseline import (
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    create_logistic_pipeline,
    evaluate_probabilities,
    load_development_data,
    train_logistic_model,
    DEVELOPMENT_FEATURE_COLUMNS,
)


@pytest.fixture
def connection() -> Iterator[duckdb.DuckDBPyConnection]:
    """Create in-memory model development sources."""

    with duckdb.connect(":memory:") as database:
        create_source_tables(database)
        yield database


def create_source_tables(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create minimal modeling dataset and split tables."""

    connection.execute(
        """
        CREATE SCHEMA analytics;
        """
    )

    feature_columns_sql = ",\n".join(
        f"{column_name} DOUBLE"
        for column_name in DEVELOPMENT_FEATURE_COLUMNS
    )

    connection.execute(
        f"""
        CREATE TABLE analytics.game_modeling_dataset (
            game_id VARCHAR,
            season INTEGER,
            game_date DATE,
            target_home_win BOOLEAN,
            elo_home_win_probability DOUBLE,
            {feature_columns_sql}
        );

        CREATE TABLE analytics.modeling_game_splits (
            game_id VARCHAR,
            split_name VARCHAR,
            is_core_model_eligible BOOLEAN
        );
        """
    )

    dataset_placeholders = ", ".join(
        "?"
        for _ in range(
            5 + len(DEVELOPMENT_FEATURE_COLUMNS)
        )
    )

    rows = []

    game_definitions = [
        (
            "2018_01_A_B",
            2018,
            "2018-09-06",
            True,
            0.60,
            0.50,
        ),
        (
            "2019_01_C_D",
            2019,
            "2019-09-05",
            False,
            0.45,
            -0.40,
        ),
        (
            "2020_01_E_F",
            2020,
            "2020-09-10",
            True,
            0.65,
            0.60,
        ),
        (
            "2023_01_G_H",
            2023,
            "2023-09-07",
            False,
            0.40,
            -0.30,
        ),
        (
            "2024_01_I_J",
            2024,
            "2024-09-05",
            True,
            0.62,
            0.45,
        ),
        (
            "2025_01_K_L",
            2025,
            "2025-09-04",
            True,
            0.70,
            0.80,
        ),
    ]

    for (
        game_id,
        season,
        game_date,
        target,
        elo_probability,
        feature_value,
    ) in game_definitions:
        rows.append(
            (
                game_id,
                season,
                game_date,
                target,
                elo_probability,
                *(
                    feature_value
                    for _ in DEVELOPMENT_FEATURE_COLUMNS
                ),
            )
        )

    connection.executemany(
        f"""
        INSERT INTO analytics.game_modeling_dataset
        VALUES ({dataset_placeholders})
        """,
        rows,
    )

    connection.execute(
        """
        INSERT INTO analytics.modeling_game_splits
        VALUES
            ('2018_01_A_B', 'train', TRUE),
            ('2019_01_C_D', 'train', TRUE),
            ('2020_01_E_F', 'train', TRUE),
            ('2023_01_G_H', 'validation', TRUE),
            ('2024_01_I_J', 'validation', TRUE),
            ('2025_01_K_L', 'holdout', TRUE);
        """
    )


def create_development_frame() -> pd.DataFrame:
    """Create a small two-class train/validation dataset."""

    rows = []

    for index in range(8):
        split_name = (
            "train"
            if index < 6
            else "validation"
        )

        target = index % 2

        row = {
            "game_id": f"game_{index}",
            "season": 2020 + index,
            "game_date": pd.Timestamp(
                "2020-01-01"
            )
            + pd.Timedelta(days=index),
            "split_name": split_name,
            TARGET_COLUMN: target,
            "elo_home_win_probability": (
                0.65 if target else 0.35
            ),
        }

        for feature_index, column_name in enumerate(
            MODEL_FEATURE_COLUMNS
        ):
            row[column_name] = (
                float(target)
                + 0.01 * feature_index
            )

        rows.append(row)

    return pd.DataFrame(rows)


def test_load_development_data_excludes_holdout(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Never load holdout games during development."""

    data = load_development_data(connection)

    assert set(data["split_name"]) == {
        "train",
        "validation",
    }

    assert "2025_01_K_L" not in set(data["game_id"])
    assert len(data) == 5


def test_create_logistic_pipeline_predicts_probabilities() -> None:
    """Fit the complete preprocessing and model pipeline."""

    data = create_development_frame()

    pipeline = create_logistic_pipeline()

    pipeline.fit(
        data.loc[:, MODEL_FEATURE_COLUMNS],
        data[TARGET_COLUMN],
    )

    probabilities = pipeline.predict_proba(
        data.loc[:, MODEL_FEATURE_COLUMNS]
    )[:, 1]

    assert probabilities.shape == (8,)
    assert np.all(probabilities > 0.0)
    assert np.all(probabilities < 1.0)


def test_train_logistic_model_uses_train_split() -> None:
    """Train a valid two-class logistic model."""

    data = create_development_frame()

    model = train_logistic_model(data)

    probabilities = model.predict_proba(
        data.loc[
            data["split_name"] == "validation",
            MODEL_FEATURE_COLUMNS,
        ]
    )[:, 1]

    assert len(probabilities) == 2


def test_train_logistic_model_requires_both_classes() -> None:
    """Reject training data containing only one outcome."""

    data = create_development_frame()

    data.loc[
        data["split_name"] == "train",
        TARGET_COLUMN,
    ] = 1

    with pytest.raises(
        RuntimeError,
        match="both target classes",
    ):
        train_logistic_model(data)


def test_evaluate_probabilities_returns_expected_metrics() -> None:
    """Calculate accuracy, Brier score and log loss."""

    evaluation = evaluate_probabilities(
        actual_values=pd.Series([0, 1, 1, 0]),
        probabilities=np.array(
            [0.10, 0.90, 0.80, 0.20]
        ),
    )

    assert evaluation.game_count == 4
    assert evaluation.accuracy == pytest.approx(1.0)
    assert evaluation.brier_score == pytest.approx(
        0.025
    )
    assert evaluation.log_loss > 0.0


def test_evaluate_probabilities_rejects_length_mismatch() -> None:
    """Reject probability arrays with a different length."""

    with pytest.raises(
        ValueError,
        match="equal length",
    ):
        evaluate_probabilities(
            actual_values=pd.Series([0, 1]),
            probabilities=np.array([0.5]),
        )


def test_create_logistic_pipeline_supports_feature_subset() -> None:
    """Fit a logistic model using only selected features."""

    data = create_development_frame()

    feature_columns = (
        "elo_rating_difference",
        "listed_qb_rating_difference",
    )

    pipeline = create_logistic_pipeline(
        feature_columns=feature_columns,
        regularization_c=0.1,
    )

    pipeline.fit(
        data.loc[:, feature_columns],
        data[TARGET_COLUMN],
    )

    probabilities = pipeline.predict_proba(
        data.loc[:, feature_columns]
    )[:, 1]

    assert probabilities.shape == (8,)
    assert (
        pipeline.named_steps["model"].C
        == pytest.approx(0.1)
    )


@pytest.mark.parametrize(
    "regularization_c",
    [
        0.0,
        -1.0,
    ],
)
def test_create_logistic_pipeline_rejects_invalid_c(
    regularization_c: float,
) -> None:
    """Reject non-positive regularization values."""

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        create_logistic_pipeline(
            regularization_c=regularization_c,
        )


def test_train_logistic_model_supports_feature_subset() -> None:
    """Train using a selected feature group."""

    data = create_development_frame()

    feature_columns = (
        "elo_rating_difference",
    )

    model = train_logistic_model(
        development_data=data,
        feature_columns=feature_columns,
        regularization_c=0.01,
    )

    probabilities = model.predict_proba(
        data.loc[
            data["split_name"] == "validation",
            feature_columns,
        ]
    )[:, 1]

    assert len(probabilities) == 2