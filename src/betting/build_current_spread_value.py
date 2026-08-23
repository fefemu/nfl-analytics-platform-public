"""
NFL Analytics Platform
Current Spread Value Builder

Purpose:
    Load current Spread market lines, production margin
    predictions and leakage-safe calibration residuals,
    then persist validated Spread expected values.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging
from pathlib import Path

import duckdb
import pandas as pd

from src.betting.calibrate_spread_cover_probabilities import (
    create_spread_calibration_residuals,
    load_external_spread_development_data,
)
from src.betting.current_spread_value import (
    SPREAD_VALUE_COLUMNS,
    create_current_spread_value,
)
from src.modeling.evaluate_spread_model_candidates import (
    DATASET_FULL_NAME,
    SPLIT_FULL_NAME,
)
from src.modeling.train_logistic_baseline import (
    DATABASE_FILE,
    validate_database_file,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

MARKET_BOARD_FULL_NAME = (
    "analytics.current_market_board"
)

SPREAD_PREDICTION_FULL_NAME = (
    "analytics.current_game_spread_predictions"
)

TARGET_SCHEMA = "analytics"
TARGET_TABLE = "current_spread_value"
TARGET_FULL_NAME = (
    f"{TARGET_SCHEMA}.{TARGET_TABLE}"
)


def validate_source_tables(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate all required Spread value sources."""

    required_tables = {
        (
            "analytics",
            "current_market_board",
        ),
        (
            "analytics",
            "current_game_spread_predictions",
        ),
        tuple(
            DATASET_FULL_NAME.split(
                ".",
                maxsplit=1,
            )
        ),
        tuple(
            SPLIT_FULL_NAME.split(
                ".",
                maxsplit=1,
            )
        ),
    }

    existing_tables = {
        (row[0], row[1])
        for row in connection.execute(
            """
            SELECT
                table_schema,
                table_name
            FROM information_schema.tables
            """
        ).fetchall()
    }

    missing_tables = (
        required_tables - existing_tables
    )

    if missing_tables:
        missing_names = ", ".join(
            f"{schema}.{table}"
            for schema, table
            in sorted(missing_tables)
        )

        raise RuntimeError(
            "Missing Spread value source tables: "
            + missing_names
        )

    logger.info(
        "Spread value sources validated."
    )


def load_current_spread_market(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Load current paired Spread offers."""

    market_board = connection.execute(
        f"""
        SELECT
            snapshot_id,
            fetched_at,
            game_id,
            season,
            game_type,
            week,
            gameday,
            commence_time,
            home_team,
            away_team,
            market_key,
            market_name,
            outcome_name,
            outcome_type,
            point,
            market_line,
            best_bookmaker_key,
            best_bookmaker_title,
            best_american_price,
            best_decimal_odds,
            best_implied_probability,
            bookmaker_count,
            consensus_no_vig_probability

        FROM {MARKET_BOARD_FULL_NAME}

        WHERE market_key = 'spreads'

        ORDER BY
            commence_time,
            game_id,
            market_line,
            outcome_type
        """
    ).fetchdf()

    if market_board.empty:
        raise RuntimeError(
            "Current market board contains no "
            "Spread offers."
        )

    logger.info(
        "Current Spread market loaded: "
        "%s offers across %s games and %s lines.",
        len(market_board),
        market_board["game_id"].nunique(),
        market_board[
            [
                "game_id",
                "market_line",
            ]
        ].drop_duplicates().shape[0],
    )

    return market_board


def load_current_spread_predictions(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Load current production spread predictions."""

    predictions = connection.execute(
        f"""
        SELECT
            game_id,
            model_name,
            model_version,
            prediction_mode,
            predicted_home_margin,
            predicted_away_margin,
            prediction_generated_at

        FROM {SPREAD_PREDICTION_FULL_NAME}

        ORDER BY game_id
        """
    ).fetchdf()

    if predictions.empty:
        raise RuntimeError(
            "Current Spread prediction source is empty."
        )

    logger.info(
        "Current Spread predictions loaded: %s games.",
        len(predictions),
    )

    return predictions


def create_current_spread_value_table(
    connection: duckdb.DuckDBPyConnection,
    value: pd.DataFrame,
) -> None:
    """Persist current Spread value rows."""

    missing_columns = sorted(
        set(SPREAD_VALUE_COLUMNS)
        - set(value.columns)
    )

    if missing_columns:
        raise ValueError(
            "Current Spread value output is missing "
            "columns: "
            + ", ".join(missing_columns)
        )

    connection.execute(
        f"CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA}"
    )

    connection.register(
        "_current_spread_value",
        value.loc[
            :,
            SPREAD_VALUE_COLUMNS,
        ],
    )

    try:
        connection.execute(
            f"""
            CREATE OR REPLACE TABLE
                {TARGET_FULL_NAME}
            AS
            SELECT *
            FROM _current_spread_value
            """
        )
    finally:
        connection.unregister(
            "_current_spread_value"
        )


def validate_current_spread_value_table(
    connection: duckdb.DuckDBPyConnection,
    expected_row_count: int,
    expected_game_count: int,
    expected_line_count: int,
) -> None:
    """Validate persisted Spread probabilities and EV."""

    actual_row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    if actual_row_count != expected_row_count:
        raise RuntimeError(
            "Current Spread value row count does "
            f"not match: expected {expected_row_count}, "
            f"found {actual_row_count}."
        )

    actual_game_count = connection.execute(
        f"""
        SELECT COUNT(DISTINCT game_id)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    if actual_game_count != expected_game_count:
        raise RuntimeError(
            "Current Spread value game count does "
            f"not match: expected {expected_game_count}, "
            f"found {actual_game_count}."
        )

    actual_line_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT DISTINCT
                game_id,
                home_spread_line
            FROM {TARGET_FULL_NAME}
        )
        """
    ).fetchone()[0]

    if actual_line_count != expected_line_count:
        raise RuntimeError(
            "Current Spread value line count does "
            f"not match: expected {expected_line_count}, "
            f"found {actual_line_count}."
        )

    duplicate_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT
                game_id,
                outcome_type,
                point
            FROM {TARGET_FULL_NAME}
            GROUP BY
                game_id,
                outcome_type,
                point
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_count > 0:
        raise RuntimeError(
            "Duplicate current Spread offers found."
        )

    invalid_pair_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT
                game_id,
                home_spread_line,
                COUNT(*) AS outcome_count,
                COUNT(
                    DISTINCT outcome_type
                ) AS unique_outcome_count,
                SUM(point) AS point_sum
            FROM {TARGET_FULL_NAME}
            GROUP BY
                game_id,
                home_spread_line
            HAVING outcome_count <> 2
                OR unique_outcome_count <> 2
                OR ABS(point_sum) > 0.000001
        )
        """
    ).fetchone()[0]

    if invalid_pair_count > 0:
        raise RuntimeError(
            "Invalid current Spread line pairs found."
        )

    invalid_probability_count = (
        connection.execute(
            f"""
            SELECT COUNT(*)
            FROM {TARGET_FULL_NAME}
            WHERE cover_probability < 0.0
               OR cover_probability > 1.0
               OR push_probability < 0.0
               OR push_probability > 1.0
               OR loss_probability < 0.0
               OR loss_probability > 1.0
               OR no_push_cover_probability <= 0.0
               OR no_push_cover_probability >= 1.0
               OR ABS(
                    cover_probability
                    + push_probability
                    + loss_probability
                    - 1.0
               ) > 0.000001
               OR calibration_sample_count <= 0
            """
        ).fetchone()[0]
    )

    if invalid_probability_count > 0:
        raise RuntimeError(
            "Invalid current Spread probabilities found."
        )

    invalid_calculation_count = (
        connection.execute(
            f"""
            SELECT COUNT(*)
            FROM {TARGET_FULL_NAME}
            WHERE NOT isfinite(
                    expected_value_per_unit
                  )
               OR NOT isfinite(
                    expected_value_percent
                  )
               OR NOT isfinite(
                    fair_decimal_odds
                  )
               OR NOT isfinite(
                    full_kelly_fraction
                  )
               OR ABS(
                    expected_value_per_unit
                    - (
                        cover_probability
                        * (
                            best_decimal_odds - 1.0
                        )
                        - loss_probability
                    )
               ) > 0.000001
               OR ABS(
                    no_push_cover_probability
                    - (
                        cover_probability
                        / NULLIF(
                            cover_probability
                            + loss_probability,
                            0.0
                        )
                    )
               ) > 0.000001
               OR ABS(
                    probability_edge
                    - (
                        no_push_cover_probability
                        - consensus_no_vig_probability
                    )
               ) > 0.000001
               OR full_kelly_fraction < 0.0
            """
        ).fetchone()[0]
    )

    if invalid_calculation_count > 0:
        raise RuntimeError(
            "Invalid current Spread value "
            "calculations found."
        )

    invalid_flag_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE positive_expected_value
            <> (
                expected_value_per_unit > 0.0
            )
        """
    ).fetchone()[0]

    if invalid_flag_count > 0:
        raise RuntimeError(
            "Invalid current Spread positive-EV "
            "flags found."
        )

    invalid_metadata_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE snapshot_id IS NULL
           OR fetched_at IS NULL
           OR game_id IS NULL
           OR season IS NULL
           OR week IS NULL
           OR commence_time IS NULL
           OR home_team IS NULL
           OR away_team IS NULL
           OR home_team = away_team
           OR market_key <> 'spreads'
           OR market_name <> 'Spread'
           OR outcome_type NOT IN (
                'home',
                'away'
           )
           OR point IS NULL
           OR market_line IS NULL
           OR best_decimal_odds <= 1.0
           OR best_bookmaker_key IS NULL
           OR bookmaker_count <= 0
           OR model_name IS NULL
           OR model_version IS NULL
           OR prediction_mode IS NULL
           OR prediction_generated_at IS NULL
        """
    ).fetchone()[0]

    if invalid_metadata_count > 0:
        raise RuntimeError(
            "Invalid current Spread metadata found."
        )

    logger.info(
        "Current Spread value table validated: "
        "%s offers across %s games and %s lines.",
        actual_row_count,
        actual_game_count,
        actual_line_count,
    )


def build_current_spread_value(
    database_file: Path = DATABASE_FILE,
) -> pd.DataFrame:
    """Build and persist current Spread value."""

    validate_database_file(database_file)

    logger.info(
        "Starting current Spread value build..."
    )

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

        logger.info(
            "Spread calibration prepared: "
            "%s residuals across %s modes.",
            len(residuals),
            residuals[
                "prediction_mode"
            ].nunique(),
        )

        value = create_current_spread_value(
            market_board=market_board,
            predictions=predictions,
            residuals=residuals,
        )

        expected_line_count = value[
            [
                "game_id",
                "home_spread_line",
            ]
        ].drop_duplicates().shape[0]

        connection.execute("BEGIN TRANSACTION")

        try:
            create_current_spread_value_table(
                connection=connection,
                value=value,
            )

            validate_current_spread_value_table(
                connection=connection,
                expected_row_count=len(value),
                expected_game_count=(
                    value["game_id"].nunique()
                ),
                expected_line_count=(
                    expected_line_count
                ),
            )

            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise

    positive_ev_count = int(
        value["positive_expected_value"].sum()
    )

    logger.info(
        "Current Spread value build completed: "
        "%s offers across %s games and %s lines, "
        "%s positive-EV offers.",
        len(value),
        value["game_id"].nunique(),
        expected_line_count,
        positive_ev_count,
    )

    return value


def main() -> None:
    """Run the current Spread value builder."""

    try:
        build_current_spread_value()
    except Exception:
        logger.exception(
            "Current Spread value build failed."
        )
        raise


if __name__ == "__main__":
    main()
