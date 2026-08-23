"""
NFL Analytics Platform
Current Game Prediction Narrative Persistence

Purpose:
    Persist and validate accessible English and
    Hungarian production prediction narratives.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import duckdb
import pandas as pd

from src.modeling.current_game_prediction_narratives import (
    NARRATIVE_COLUMNS,
)


TARGET_SCHEMA = "analytics"
TARGET_TABLE = "current_game_prediction_narratives"
TARGET_FULL_NAME = (
    f"{TARGET_SCHEMA}.{TARGET_TABLE}"
)


def create_current_game_prediction_narratives_table(
    connection: duckdb.DuckDBPyConnection,
    narratives: pd.DataFrame,
) -> None:
    """Create the bilingual narrative table."""

    missing_columns = sorted(
        set(NARRATIVE_COLUMNS)
        - set(narratives.columns)
    )

    if missing_columns:
        raise ValueError(
            "Current game prediction narratives are "
            "missing columns: "
            + ", ".join(missing_columns)
        )

    connection.execute(
        f"CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA}"
    )

    connection.execute(
        f"""
        CREATE OR REPLACE TABLE {TARGET_FULL_NAME} (
            game_id VARCHAR,
            narrative_version VARCHAR,
            model_name VARCHAR,
            model_version VARCHAR,
            prediction_generated_at TIMESTAMP,
            headline_en VARCHAR,
            headline_hu VARCHAR,
            summary_en VARCHAR,
            summary_hu VARCHAR,
            model_context_en VARCHAR,
            model_context_hu VARCHAR,
            top_factor_feature VARCHAR,
            top_factor_direction VARCHAR,
            top_factor_en VARCHAR,
            top_factor_hu VARCHAR
        )
        """
    )

    if narratives.empty:
        return

    rows = [
        tuple(
            (
                None
                if pd.isna(
                    row[column_name]
                )
                else row[column_name]
            )
            for column_name in NARRATIVE_COLUMNS
        )
        for row in narratives.to_dict(
            orient="records"
        )
    ]

    placeholders = ", ".join(
        "?"
        for _ in NARRATIVE_COLUMNS
    )

    connection.executemany(
        f"""
        INSERT INTO {TARGET_FULL_NAME}
        VALUES ({placeholders})
        """,
        rows,
    )


def validate_current_game_prediction_narratives_table(
    connection: duckdb.DuckDBPyConnection,
    expected_row_count: int,
) -> None:
    """Validate persisted bilingual narratives."""

    actual_row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    if actual_row_count != expected_row_count:
        raise RuntimeError(
            "Current game narrative row count does not "
            f"match: expected {expected_row_count}, "
            f"found {actual_row_count}."
        )

    duplicate_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT game_id
            FROM {TARGET_FULL_NAME}
            GROUP BY game_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_count > 0:
        raise RuntimeError(
            "Duplicate current game narratives found."
        )

    invalid_text_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE game_id IS NULL
           OR narrative_version IS NULL
           OR model_name IS NULL
           OR model_version IS NULL
           OR prediction_generated_at IS NULL
           OR headline_en IS NULL
           OR headline_hu IS NULL
           OR summary_en IS NULL
           OR summary_hu IS NULL
           OR model_context_en IS NULL
           OR model_context_hu IS NULL
           OR LENGTH(TRIM(headline_en)) = 0
           OR LENGTH(TRIM(headline_hu)) = 0
           OR LENGTH(TRIM(summary_en)) = 0
           OR LENGTH(TRIM(summary_hu)) = 0
           OR LENGTH(TRIM(model_context_en)) = 0
           OR LENGTH(TRIM(model_context_hu)) = 0
        """
    ).fetchone()[0]

    if invalid_text_count > 0:
        raise RuntimeError(
            "Invalid or incomplete bilingual narrative "
            "text found."
        )

    invalid_factor_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE (
                top_factor_feature IS NULL
                AND (
                    top_factor_direction IS NOT NULL
                    OR top_factor_en IS NOT NULL
                    OR top_factor_hu IS NOT NULL
                )
              )
           OR (
                top_factor_feature IS NOT NULL
                AND (
                    top_factor_direction IS NULL
                    OR top_factor_en IS NULL
                    OR top_factor_hu IS NULL
                )
              )
           OR (
                top_factor_direction IS NOT NULL
                AND top_factor_direction NOT IN (
                    'supports_favorite',
                    'opposes_favorite'
                )
              )
        """
    ).fetchone()[0]

    if invalid_factor_count > 0:
        raise RuntimeError(
            "Invalid current game top-factor narrative "
            "found."
        )