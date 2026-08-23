"""
Tests for the current Spread value builder.
"""

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src.betting.build_current_spread_value import (
    TARGET_FULL_NAME,
    build_current_spread_value,
    create_current_spread_value_table,
    load_current_spread_market,
    load_current_spread_predictions,
    load_external_spread_development_data,
    validate_current_spread_value_table,
    validate_source_tables,
)
from src.betting.calibrate_spread_cover_probabilities import (
    create_spread_calibration_residuals,
)
from src.betting.current_spread_value import (
    create_current_spread_value,
)
from src.modeling.evaluate_spread_model_candidates import (
    SPREAD_CORE_FEATURES,
)


def create_source_database(
    database_file: Path,
) -> None:
    """Create minimal Spread value source tables."""

    with duckdb.connect(
        str(database_file)
    ) as connection:
        connection.execute(
            """
            CREATE SCHEMA analytics;
            CREATE SCHEMA processed;

            CREATE TABLE analytics.current_market_board (
                snapshot_id VARCHAR,
                fetched_at TIMESTAMPTZ,
                game_id VARCHAR,
                season INTEGER,
                game_type VARCHAR,
                week INTEGER,
                gameday DATE,
                commence_time TIMESTAMPTZ,
                home_team VARCHAR,
                away_team VARCHAR,
                market_key VARCHAR,
                market_name VARCHAR,
                outcome_name VARCHAR,
                outcome_type VARCHAR,
                point DOUBLE,
                market_line DOUBLE,
                best_bookmaker_key VARCHAR,
                best_bookmaker_title VARCHAR,
                best_american_price INTEGER,
                best_decimal_odds DOUBLE,
                best_implied_probability DOUBLE,
                bookmaker_count INTEGER,
                consensus_no_vig_probability DOUBLE
            );

            CREATE TABLE
                analytics.current_game_spread_predictions (
                    game_id VARCHAR,
                    model_name VARCHAR,
                    model_version VARCHAR,
                    prediction_mode VARCHAR,
                    predicted_home_margin DOUBLE,
                    predicted_away_margin DOUBLE,
                    prediction_generated_at TIMESTAMP
                );

            CREATE TABLE analytics.modeling_game_splits (
                game_id VARCHAR,
                split_name VARCHAR
            );

            CREATE TABLE processed.external_nfelo_game_ratings (
                normalized_game_id VARCHAR,
                starting_nfelo_home DOUBLE,
                starting_nfelo_away DOUBLE,
                home_538_qb_adj DOUBLE,
                away_538_qb_adj DOUBLE
            );
            """
        )

        connection.executemany(
            """
            INSERT INTO analytics.current_market_board
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                (
                    "snapshot_001",
                    "2026-08-09T09:00:00Z",
                    "current_game",
                    2026,
                    "REG",
                    1,
                    "2026-09-10",
                    "2026-09-11T00:20:00Z",
                    "PHI",
                    "DAL",
                    "spreads",
                    "Spread",
                    "PHI",
                    "home",
                    -0.5,
                    0.5,
                    "book_a",
                    "Book A",
                    100,
                    2.00,
                    0.50,
                    5,
                    0.50,
                ),
                (
                    "snapshot_001",
                    "2026-08-09T09:00:00Z",
                    "current_game",
                    2026,
                    "REG",
                    1,
                    "2026-09-10",
                    "2026-09-11T00:20:00Z",
                    "PHI",
                    "DAL",
                    "spreads",
                    "Spread",
                    "DAL",
                    "away",
                    0.5,
                    0.5,
                    "book_b",
                    "Book B",
                    100,
                    2.00,
                    0.50,
                    4,
                    0.50,
                ),
            ],
        )

        connection.execute(
            """
            INSERT INTO
                analytics.current_game_spread_predictions
            VALUES (
                'current_game',
                'external_nfelo_external_qb_spread',
                '0.2.0',
                'EXTERNAL_NFELO_QB_RIDGE',
                0.0,
                0.0,
                TIMESTAMP '2026-08-09 09:05:00'
            )
            """
        )

        development_rows = []

        targets = {
            2020: 7.0,
            2021: -7.0,
            2022: 7.0,
            2023: -7.0,
            2024: 7.0,
        }

        for season, target in targets.items():
            row: dict[str, object] = {
                feature_name: 0.0
                for feature_name
                in SPREAD_CORE_FEATURES
            }

            row.update(
                {
                    "game_id": (
                        f"{season}_game"
                    ),
                    "season": season,
                    "game_date": pd.Timestamp(
                        f"{season}-09-10"
                    ),
                    "has_complete_injury_data": True,
                    "target_point_differential": (
                        target
                    ),
                }
            )

            development_rows.append(row)

        development_data = pd.DataFrame(
            development_rows
        )

        connection.register(
            "_development_data",
            development_data,
        )

        connection.execute(
            """
            CREATE TABLE
                analytics.game_modeling_dataset
            AS
            SELECT *
            FROM _development_data
            """
        )

        connection.unregister(
            "_development_data"
        )

        connection.executemany(
            """
            INSERT INTO analytics.modeling_game_splits
            VALUES (?, ?)
            """,
            [
                (
                    "2020_game",
                    "train",
                ),
                (
                    "2021_game",
                    "train",
                ),
                (
                    "2022_game",
                    "train",
                ),
                (
                    "2023_game",
                    "validation",
                ),
                (
                    "2024_game",
                    "validation",
                ),
            ],
        )

        connection.executemany(
            """
            INSERT INTO processed.external_nfelo_game_ratings
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (f"{season}_game", 1525.0, 1475.0, 5.0, 0.0)
                for season in targets
            ],
        )


def create_value_frame(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Calculate a valid Spread value frame."""

    market_board = load_current_spread_market(
        connection
    )

    predictions = (
        load_current_spread_predictions(
            connection
        )
    )

    development_data = (
            load_external_spread_development_data(
            connection
        )
    )

    residuals = (
        create_spread_calibration_residuals(
            development_data
        )
    )

    return create_current_spread_value(
        market_board=market_board,
        predictions=predictions,
        residuals=residuals,
    )


def test_validate_source_tables_rejects_missing_tables(
) -> None:
    """Fail when required Spread sources are absent."""

    with duckdb.connect(":memory:") as connection:
        with pytest.raises(
            RuntimeError,
            match="Missing Spread value source tables",
        ):
            validate_source_tables(connection)


def test_load_current_spread_sources(
    tmp_path: Path,
) -> None:
    """Load current Spread market and predictions."""

    database_file = tmp_path / "sources.duckdb"

    create_source_database(database_file)

    with duckdb.connect(
        str(database_file)
    ) as connection:
        validate_source_tables(connection)

        market_board = (
            load_current_spread_market(
                connection
            )
        )

        predictions = (
            load_current_spread_predictions(
                connection
            )
        )

    assert len(market_board) == 2

    assert set(
        market_board["outcome_type"]
    ) == {
        "home",
        "away",
    }

    assert len(predictions) == 1

    assert predictions.iloc[0][
        "prediction_mode"
    ] == "EXTERNAL_NFELO_QB_RIDGE"


def test_create_and_validate_spread_value_table(
    tmp_path: Path,
) -> None:
    """Persist and validate current Spread value."""

    database_file = tmp_path / "value.duckdb"

    create_source_database(database_file)

    with duckdb.connect(
        str(database_file)
    ) as connection:
        value = create_value_frame(
            connection
        )

        create_current_spread_value_table(
            connection=connection,
            value=value,
        )

        validate_current_spread_value_table(
            connection=connection,
            expected_row_count=2,
            expected_game_count=1,
            expected_line_count=1,
        )

        row_count = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM {TARGET_FULL_NAME}
            """
        ).fetchone()[0]

        probability_sums = connection.execute(
            f"""
            SELECT
                cover_probability
                + push_probability
                + loss_probability
            FROM {TARGET_FULL_NAME}
            """
        ).fetchall()

    assert row_count == 2

    assert all(
        probability_sum[0]
        == pytest.approx(1.0)
        for probability_sum
        in probability_sums
    )


def test_invalid_persisted_value_is_rejected(
    tmp_path: Path,
) -> None:
    """Reject an inconsistent persisted Spread EV."""

    database_file = tmp_path / "invalid.duckdb"

    create_source_database(database_file)

    with duckdb.connect(
        str(database_file)
    ) as connection:
        value = create_value_frame(
            connection
        )

        create_current_spread_value_table(
            connection=connection,
            value=value,
        )

        connection.execute(
            f"""
            UPDATE {TARGET_FULL_NAME}
            SET expected_value_per_unit = 10.0
            WHERE outcome_type = 'home'
            """
        )

        with pytest.raises(
            RuntimeError,
            match="value calculations",
        ):
            validate_current_spread_value_table(
                connection=connection,
                expected_row_count=2,
                expected_game_count=1,
                expected_line_count=1,
            )


def test_missing_output_column_is_rejected(
    tmp_path: Path,
) -> None:
    """Reject an incomplete Spread value frame."""

    database_file = tmp_path / "missing.duckdb"

    create_source_database(database_file)

    with duckdb.connect(
        str(database_file)
    ) as connection:
        value = create_value_frame(
            connection
        ).drop(
            columns=[
                "cover_probability",
            ]
        )

        with pytest.raises(
            ValueError,
            match="missing columns",
        ):
            create_current_spread_value_table(
                connection=connection,
                value=value,
            )


def test_build_current_spread_value_end_to_end(
    tmp_path: Path,
) -> None:
    """Build current Spread value from all sources."""

    database_file = tmp_path / "build.duckdb"

    create_source_database(database_file)

    result = build_current_spread_value(
        database_file
    )

    with duckdb.connect(
        str(database_file)
    ) as connection:
        persisted = connection.execute(
            f"""
            SELECT
                outcome_type,
                calibration_sample_count,
                cover_probability,
                push_probability,
                loss_probability,
                expected_value_per_unit
            FROM {TARGET_FULL_NAME}
            ORDER BY outcome_type
            """
        ).fetchdf()

    assert len(result) == 2
    assert len(persisted) == 2

    assert persisted[
        "calibration_sample_count"
    ].eq(4).all()

    assert persisted[
        "cover_probability"
    ].between(
        0.0,
        1.0,
    ).all()

    assert persisted[
        "expected_value_per_unit"
    ].notna().all()
