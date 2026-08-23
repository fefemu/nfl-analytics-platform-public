"""Tests for persisted current spread predictions."""

from datetime import datetime

import duckdb
import pandas as pd
import pytest

from src.modeling.build_current_spread_predictions import (
    TARGET_FULL_NAME,
    create_current_spread_predictions_table,
    validate_current_spread_predictions_table,
)


@pytest.fixture
def connection() -> duckdb.DuckDBPyConnection:
    """Create an in-memory DuckDB connection."""

    database = duckdb.connect(":memory:")

    yield database

    database.close()


def create_predictions() -> pd.DataFrame:
    """Create valid primary and fallback predictions."""

    generated_at = datetime(
        2026,
        8,
        6,
        15,
        15,
        0,
    )

    return pd.DataFrame(
        [
            {
                "game_id": "game_1",
                "season": 2026,
                "game_type": "REG",
                "week": 1,
                "gameday": "2026-09-10",
                "gametime": "20:20",
                "home_team": "BUF",
                "away_team": "NYJ",
                "is_neutral": False,
                "model_name": "external_nfelo_external_qb_spread",
                "model_version": "0.2.0",
                "prediction_mode": "EXTERNAL_NFELO_QB_RIDGE",
                "prediction_mode_reason": (
                    "complete_external_nfelo_qb_features"
                ),
                "ridge_alpha": 10.0,
                "primary_training_game_count": 1906,
                "fallback_training_game_count": 1927,
                "external_nfelo_rating_difference": 80.0,
                "listed_qb_rating_difference": 4.0,
                "external_nfelo_qb_adjustment_difference": 3.0,
                "both_listed_qb_ratings_available": True,
                "predicted_home_margin": 6.5,
                "predicted_away_margin": -6.5,
                "predicted_winner": "BUF",
                "prediction_generated_at": generated_at,
            },
            {
                "game_id": "game_2",
                "season": 2026,
                "game_type": "REG",
                "week": 1,
                "gameday": "2026-09-13",
                "gametime": "13:00",
                "home_team": "LV",
                "away_team": "DEN",
                "is_neutral": False,
                "model_name": "external_nfelo_external_qb_spread",
                "model_version": "0.2.0",
                "prediction_mode": (
                    "EXTERNAL_NFELO_QB_RIDGE"
                ),
                "prediction_mode_reason": (
                    "complete_external_nfelo_qb_features"
                ),
                "ridge_alpha": 10.0,
                "primary_training_game_count": 1906,
                "fallback_training_game_count": 1927,
                "external_nfelo_rating_difference": -60.0,
                "listed_qb_rating_difference": None,
                "external_nfelo_qb_adjustment_difference": -2.0,
                "both_listed_qb_ratings_available": False,
                "predicted_home_margin": -1.5,
                "predicted_away_margin": 1.5,
                "predicted_winner": "DEN",
                "prediction_generated_at": generated_at,
            },
        ]
    )


def test_create_and_validate_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Persist valid primary and fallback rows."""

    create_current_spread_predictions_table(
        connection=connection,
        predictions=create_predictions(),
    )

    validate_current_spread_predictions_table(
        connection=connection,
        expected_row_count=2,
    )

    stored = connection.execute(
        f"""
        SELECT
            game_id,
            prediction_mode,
            predicted_home_margin
        FROM {TARGET_FULL_NAME}
        ORDER BY game_id
        """
    ).fetchdf()

    assert len(stored) == 2

    assert set(
        stored["prediction_mode"]
    ) == {
        "EXTERNAL_NFELO_QB_RIDGE",
    }


def test_create_empty_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create a stable empty target table."""

    create_current_spread_predictions_table(
        connection=connection,
        predictions=(
            create_predictions().iloc[0:0]
        ),
    )

    validate_current_spread_predictions_table(
        connection=connection,
        expected_row_count=0,
    )


def test_writer_rejects_missing_column(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Reject an incomplete prediction schema."""

    predictions = create_predictions().drop(
        columns=[
            "predicted_home_margin",
        ]
    )

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        create_current_spread_predictions_table(
            connection=connection,
            predictions=predictions,
        )


def test_validator_rejects_asymmetric_margins(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Require opposite home and away margins."""

    create_current_spread_predictions_table(
        connection=connection,
        predictions=create_predictions(),
    )

    connection.execute(
        f"""
        UPDATE {TARGET_FULL_NAME}
        SET predicted_away_margin = 2.0
        WHERE game_id = 'game_1'
        """
    )

    with pytest.raises(
        RuntimeError,
        match="margins",
    ):
        validate_current_spread_predictions_table(
            connection=connection,
            expected_row_count=2,
        )


def test_validator_rejects_wrong_winner(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Require winner and margin agreement."""

    create_current_spread_predictions_table(
        connection=connection,
        predictions=create_predictions(),
    )

    connection.execute(
        f"""
        UPDATE {TARGET_FULL_NAME}
        SET predicted_winner = 'NYJ'
        WHERE game_id = 'game_1'
        """
    )

    with pytest.raises(
        RuntimeError,
        match="winners",
    ):
        validate_current_spread_predictions_table(
            connection=connection,
            expected_row_count=2,
        )


def test_validator_rejects_invalid_routing(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Require mode, feature coverage and alpha agreement."""

    create_current_spread_predictions_table(
        connection=connection,
        predictions=create_predictions(),
    )

    connection.execute(
        f"""
        UPDATE {TARGET_FULL_NAME}
        SET ridge_alpha = 100.0
        WHERE prediction_mode = 'EXTERNAL_NFELO_QB_RIDGE'
        """
    )

    with pytest.raises(
        RuntimeError,
        match="routing",
    ):
        validate_current_spread_predictions_table(
            connection=connection,
            expected_row_count=2,
        )
