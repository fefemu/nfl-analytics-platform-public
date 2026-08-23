"""
Tests for the modeling game split builder.
"""

from collections.abc import Iterator

import duckdb
import pytest

from src.modeling.build_modeling_splits import (
    TARGET_FULL_NAME,
    assign_split_name,
    create_modeling_splits_table,
    validate_modeling_splits_table,
    validate_source_table,
)


@pytest.fixture
def connection() -> Iterator[duckdb.DuckDBPyConnection]:
    """Create an in-memory modeling dataset."""

    with duckdb.connect(":memory:") as database:
        create_source_table(database)
        yield database


def create_source_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create a minimal game modeling dataset source."""

    connection.execute(
        """
        CREATE SCHEMA analytics;

        CREATE TABLE analytics.game_modeling_dataset (
            game_id VARCHAR,
            season INTEGER,
            game_type VARCHAR,
            week INTEGER,
            game_date DATE,
            target_home_win BOOLEAN,
            target_home_result DOUBLE,
            both_short_windows_complete BOOLEAN,
            both_long_windows_complete BOOLEAN,
            both_listed_qb_ratings_available BOOLEAN
        );

        INSERT INTO analytics.game_modeling_dataset
        VALUES
            (
                '2018_01_A_B',
                2018,
                'REG',
                1,
                DATE '2018-09-06',
                TRUE,
                1.0,
                TRUE,
                FALSE,
                TRUE
            ),
            (
                '2022_18_C_D',
                2022,
                'REG',
                18,
                DATE '2023-01-08',
                FALSE,
                0.0,
                TRUE,
                TRUE,
                TRUE
            ),
            (
                '2023_01_E_F',
                2023,
                'REG',
                1,
                DATE '2023-09-07',
                TRUE,
                1.0,
                TRUE,
                TRUE,
                TRUE
            ),
            (
                '2024_18_G_H',
                2024,
                'REG',
                18,
                DATE '2025-01-05',
                FALSE,
                0.0,
                FALSE,
                FALSE,
                TRUE
            ),
            (
                '2025_01_I_J',
                2025,
                'REG',
                1,
                DATE '2025-09-04',
                NULL,
                0.5,
                TRUE,
                TRUE,
                TRUE
            ),
            (
                '2025_02_K_L',
                2025,
                'REG',
                2,
                DATE '2025-09-11',
                TRUE,
                1.0,
                TRUE,
                TRUE,
                FALSE
            );
        """
    )


@pytest.mark.parametrize(
    ("season", "expected_split"),
    [
        (2018, "train"),
        (2022, "train"),
        (2023, "validation"),
        (2024, "validation"),
        (2025, "holdout"),
    ],
)
def test_assign_split_name(
    season: int,
    expected_split: str,
) -> None:
    """Assign configured seasons to the correct split."""

    assert assign_split_name(season) == expected_split


def test_assign_split_name_rejects_unknown_season() -> None:
    """Reject seasons outside the configured split range."""

    with pytest.raises(
        ValueError,
        match="outside the configured split range",
    ):
        assign_split_name(2017)


def test_create_modeling_splits_assigns_time_order(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create train, validation and holdout assignments."""

    create_modeling_splits_table(connection)

    rows = connection.execute(
        f"""
        SELECT DISTINCT
            split_order,
            split_name
        FROM {TARGET_FULL_NAME}
        ORDER BY split_order
        """
    ).fetchall()

    assert rows == [
        (1, "train"),
        (2, "validation"),
        (3, "holdout"),
    ]


def test_create_modeling_splits_builds_eligibility_flags(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Build model eligibility without hiding missing history."""

    create_modeling_splits_table(connection)

    rows = connection.execute(
        f"""
        SELECT
            game_id,
            is_binary_target_eligible,
            is_core_model_eligible,
            is_extended_model_eligible
        FROM {TARGET_FULL_NAME}
        WHERE season = 2025
        ORDER BY game_id
        """
    ).fetchall()

    assert rows == [
        (
            "2025_01_I_J",
            False,
            False,
            False,
        ),
        (
            "2025_02_K_L",
            True,
            False,
            False,
        ),
    ]


def test_created_modeling_splits_pass_validation(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Accept consistent time-based modeling splits."""

    validate_source_table(connection)
    create_modeling_splits_table(connection)
    validate_modeling_splits_table(connection)