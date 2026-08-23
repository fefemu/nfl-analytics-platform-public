"""Tests for persisted bilingual game narratives."""

from datetime import datetime

import duckdb
import pandas as pd
import pytest

from src.modeling.build_current_game_prediction_narratives import (
    TARGET_FULL_NAME,
    create_current_game_prediction_narratives_table,
    validate_current_game_prediction_narratives_table,
)
from src.modeling.current_game_prediction_narratives import (
    NARRATIVE_COLUMNS,
)


@pytest.fixture
def connection() -> duckdb.DuckDBPyConnection:
    """Create an in-memory DuckDB connection."""

    database = duckdb.connect(":memory:")

    yield database

    database.close()


def create_narratives() -> pd.DataFrame:
    """Create valid blend and fallback narratives."""

    generated_at = datetime(
        2026,
        8,
        6,
        14,
        0,
        0,
    )

    return pd.DataFrame(
        [
            {
                "game_id": "blend_game",
                "narrative_version": "1.0.0",
                "model_name": (
                    "elo_injury_logistic_blend"
                ),
                "model_version": "0.2.0",
                "prediction_generated_at": (
                    generated_at
                ),
                "headline_en": (
                    "PHI win probability: 67.0%."
                ),
                "headline_hu": (
                    "PHI győzelmi valószínűsége: 67,0%."
                ),
                "summary_en": (
                    "PHI is favored over DAL."
                ),
                "summary_hu": (
                    "PHI a favorit DAL ellen."
                ),
                "model_context_en": (
                    "70% logistic and 30% Elo blend."
                ),
                "model_context_hu": (
                    "70% logistic és 30% Elo blend."
                ),
                "top_factor_feature": (
                    "elo_rating_difference"
                ),
                "top_factor_direction": (
                    "supports_favorite"
                ),
                "top_factor_en": (
                    "Elo rating difference supports PHI."
                ),
                "top_factor_hu": (
                    "Az Elo rating difference PHI felé hat."
                ),
            },
            {
                "game_id": "fallback_game",
                "narrative_version": "1.0.0",
                "model_name": (
                    "elo_injury_logistic_blend"
                ),
                "model_version": "0.2.0",
                "prediction_generated_at": (
                    generated_at
                ),
                "headline_en": (
                    "NE win probability: 55.0%."
                ),
                "headline_hu": (
                    "NE győzelmi valószínűsége: 55,0%."
                ),
                "summary_en": (
                    "NE is favored using Elo."
                ),
                "summary_hu": (
                    "NE a favorit az Elo alapján."
                ),
                "model_context_en": (
                    "Elo fallback is active."
                ),
                "model_context_hu": (
                    "Elo fallback aktív."
                ),
                "top_factor_feature": None,
                "top_factor_direction": None,
                "top_factor_en": None,
                "top_factor_hu": None,
            },
        ],
        columns=NARRATIVE_COLUMNS,
    )


def test_create_and_validate_narrative_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Persist valid bilingual narratives."""

    narratives = create_narratives()

    create_current_game_prediction_narratives_table(
        connection=connection,
        narratives=narratives,
    )

    validate_current_game_prediction_narratives_table(
        connection=connection,
        expected_row_count=2,
    )

    stored = connection.execute(
        f"""
        SELECT *
        FROM {TARGET_FULL_NAME}
        ORDER BY game_id
        """
    ).fetchdf()

    assert len(stored) == 2

    assert set(
        stored["game_id"]
    ) == {
        "blend_game",
        "fallback_game",
    }


def test_empty_narrative_table_is_supported(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create a stable empty narrative table."""

    narratives = pd.DataFrame(
        columns=NARRATIVE_COLUMNS
    )

    create_current_game_prediction_narratives_table(
        connection=connection,
        narratives=narratives,
    )

    validate_current_game_prediction_narratives_table(
        connection=connection,
        expected_row_count=0,
    )


def test_builder_rejects_missing_columns(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Reject incomplete narrative schema."""

    narratives = create_narratives().drop(
        columns=[
            "headline_hu",
        ]
    )

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        create_current_game_prediction_narratives_table(
            connection=connection,
            narratives=narratives,
        )


def test_validator_rejects_missing_text(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Reject an empty translated narrative."""

    create_current_game_prediction_narratives_table(
        connection=connection,
        narratives=create_narratives(),
    )

    connection.execute(
        f"""
        UPDATE {TARGET_FULL_NAME}
        SET summary_hu = ''
        WHERE game_id = 'blend_game'
        """
    )

    with pytest.raises(
        RuntimeError,
        match="bilingual narrative",
    ):
        validate_current_game_prediction_narratives_table(
            connection=connection,
            expected_row_count=2,
        )


def test_validator_rejects_partial_top_factor(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Reject an incomplete top-factor explanation."""

    create_current_game_prediction_narratives_table(
        connection=connection,
        narratives=create_narratives(),
    )

    connection.execute(
        f"""
        UPDATE {TARGET_FULL_NAME}
        SET top_factor_hu = NULL
        WHERE game_id = 'blend_game'
        """
    )

    with pytest.raises(
        RuntimeError,
        match="top-factor",
    ):
        validate_current_game_prediction_narratives_table(
            connection=connection,
            expected_row_count=2,
        )