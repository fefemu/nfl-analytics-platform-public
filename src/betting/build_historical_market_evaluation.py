"""Build and persist historical ROI and closing-line benchmark tables."""

import logging
from pathlib import Path

import duckdb
import pandas as pd

from src.betting.historical_market_evaluation import (
    LEDGER_COLUMNS,
    SUMMARY_COLUMNS,
    create_historical_betting_ledger,
    create_historical_betting_summary,
)
from src.modeling.backtest_external_elo_probability_champion import (
    create_champion_oof_predictions,
    load_champion_development_data,
)
from src.modeling.backtest_external_probability_fallback import (
    create_fallback_oof_predictions,
    load_probability_fallback_data,
)
from src.modeling.backtest_external_elo_totals_candidates import (
    CANDIDATES,
    ROUTING_FALLBACK,
    ROUTING_PRIMARY,
)
from src.modeling.diagnose_elo_rating_source_value import (
    create_oof_candidate_predictions,
)
from src.modeling.diagnose_external_elo_totals_value import (
    create_routing_oof_predictions,
)
from src.modeling.backtest_elo_rating_sources import (
    load_rating_source_backtest_data,
)
from src.modeling.evaluate_totals_fallback_candidates import (
    load_totals_fallback_development_data,
)
from src.modeling.evaluate_totals_model_candidates import (
    load_totals_development_data,
)
from src.modeling.train_logistic_baseline import DATABASE_FILE, validate_database_file

logger = logging.getLogger(__name__)
LEDGER_TABLE = "analytics.historical_betting_ledger"
SUMMARY_TABLE = "analytics.historical_betting_performance"

PRIMARY_PROBABILITY_MODEL = "external_nfelo_injury_blend"
FALLBACK_PROBABILITY_MODEL = "external_elo_qb_logistic_fallback"
SPREAD_MODEL = "external_nfelo_external_qb"
PRIMARY_TOTALS_MODEL = "primary_current_locked"
FALLBACK_TOTALS_MODEL = "fallback_current_locked"


def load_historical_market_data(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load completed development games with normalized nfelo closing markets."""
    return connection.execute(
        """
        SELECT
            model.game_id,
            model.season,
            CAST(model.target_point_differential AS DOUBLE) AS actual_home_margin,
            CAST(model.target_total_points AS DOUBLE) AS actual_total,
            CAST(ext.home_implied_win_probability_open AS DOUBLE)
                AS home_market_probability_open,
            CAST(ext.home_implied_win_probability_close AS DOUBLE)
                AS home_market_probability_close,
            CAST(ext.home_line_open AS DOUBLE) AS home_line_open,
            CAST(ext.home_line_close AS DOUBLE) AS home_line_close,
            CAST(ext.home_line_close_price AS DOUBLE) AS home_line_close_price,
            CAST(ext.away_line_close_price AS DOUBLE) AS away_line_close_price,
            CAST(ext.total_line_open AS DOUBLE) AS total_line_open,
            CAST(ext.total_line_close AS DOUBLE) AS total_line_close,
            CAST(ext.over_price_close AS DOUBLE) AS over_price_close,
            CAST(ext.under_price_close AS DOUBLE) AS under_price_close
        FROM analytics.game_modeling_dataset AS model
        INNER JOIN processed.external_nfelo_game_ratings AS ext
            ON model.game_id = ext.normalized_game_id
        WHERE model.season BETWEEN 2021 AND 2024
          AND model.target_point_differential IS NOT NULL
          AND model.target_total_points IS NOT NULL
          AND ext.home_implied_win_probability_open BETWEEN 0.0 AND 1.0
          AND ext.home_implied_win_probability_close BETWEEN 0.0 AND 1.0
          AND ext.home_line_close IS NOT NULL
          AND ext.home_line_close_price IS NOT NULL
          AND ext.away_line_close_price IS NOT NULL
          AND ext.total_line_close IS NOT NULL
          AND ext.over_price_close IS NOT NULL
          AND ext.under_price_close IS NOT NULL
        ORDER BY model.season, model.game_id
        """
    ).fetchdf()


def create_selected_oof_predictions(
    connection: duckdb.DuckDBPyConnection,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reuse locked expanding-window components to create selected OOF routes."""
    primary_source = load_champion_development_data(connection)
    primary_predictions, _ = create_champion_oof_predictions(primary_source)
    primary = primary_predictions.loc[
        primary_predictions["candidate_name"].eq(PRIMARY_PROBABILITY_MODEL),
        ["game_id", "validation_season", "home_win_probability"],
    ]

    fallback_source = load_probability_fallback_data(connection)
    fallback_predictions, _ = create_fallback_oof_predictions(fallback_source)
    fallback = fallback_predictions.loc[
        fallback_predictions["candidate_name"].eq(FALLBACK_PROBABILITY_MODEL),
        ["game_id", "validation_season", "home_win_probability"],
    ]
    fallback = fallback.loc[~fallback["game_id"].isin(primary["game_id"])]
    probability = pd.concat([primary, fallback], ignore_index=True).rename(
        columns={"validation_season": "season"}
    )
    probability["model_name"] = probability["game_id"].map(
        dict(zip(primary["game_id"], [PRIMARY_PROBABILITY_MODEL] * len(primary), strict=True))
    ).fillna(FALLBACK_PROBABILITY_MODEL)

    rating_data = load_rating_source_backtest_data(connection)
    spread_all = create_oof_candidate_predictions(rating_data)
    spread = spread_all.loc[
        spread_all["candidate_name"].eq(SPREAD_MODEL),
        ["game_id", "validation_season", "predicted_home_margin"],
    ].rename(columns={"validation_season": "season"})
    spread["model_name"] = SPREAD_MODEL

    primary_candidate = tuple(
        candidate for candidate in CANDIDATES
        if candidate.routing_layer == ROUTING_PRIMARY
        and candidate.candidate_name == PRIMARY_TOTALS_MODEL
    )
    fallback_candidate = tuple(
        candidate for candidate in CANDIDATES
        if candidate.routing_layer == ROUTING_FALLBACK
        and candidate.candidate_name == FALLBACK_TOTALS_MODEL
    )
    primary_totals = create_routing_oof_predictions(
        load_totals_development_data(connection), primary_candidate
    )
    fallback_totals = create_routing_oof_predictions(
        load_totals_fallback_development_data(connection), fallback_candidate
    )
    primary_totals = primary_totals.loc[:, ["game_id", "validation_season", "predicted_total"]]
    fallback_totals = fallback_totals.loc[
        ~fallback_totals["game_id"].isin(primary_totals["game_id"]),
        ["game_id", "validation_season", "predicted_total"],
    ]
    totals = pd.concat([primary_totals, fallback_totals], ignore_index=True).rename(
        columns={"validation_season": "season"}
    )
    totals["model_name"] = totals["game_id"].map(
        dict(zip(primary_totals["game_id"], [PRIMARY_TOTALS_MODEL] * len(primary_totals), strict=True))
    ).fillna(FALLBACK_TOTALS_MODEL)
    return probability, spread, totals


def persist_historical_market_evaluation(
    connection: duckdb.DuckDBPyConnection,
    ledger: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Persist validated historical reporting tables."""
    connection.execute("CREATE SCHEMA IF NOT EXISTS analytics")
    for name, frame, columns in (
        (LEDGER_TABLE, ledger, LEDGER_COLUMNS),
        (SUMMARY_TABLE, summary, SUMMARY_COLUMNS),
    ):
        connection.register("_historical_output", frame.loc[:, columns])
        try:
            connection.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _historical_output")
        finally:
            connection.unregister("_historical_output")


def validate_historical_market_evaluation(connection: duckdb.DuckDBPyConnection) -> None:
    """Validate row uniqueness, settlement arithmetic and aggregate reconciliation."""
    invalid = connection.execute(
        f"""
        SELECT COUNT(*) FROM {LEDGER_TABLE}
        WHERE market_key NOT IN ('h2h', 'spreads', 'totals')
           OR pricing_basis NOT IN ('SYNTHETIC_CLOSE_FAIR', 'CLOSING_PRICE')
           OR result NOT IN ('WIN', 'LOSS', 'PUSH')
           OR decimal_odds <= 1.0
           OR (result = 'WIN' AND profit_per_unit <= 0.0)
           OR (result = 'LOSS' AND profit_per_unit <> -1.0)
           OR (result = 'PUSH' AND profit_per_unit <> 0.0)
        """
    ).fetchone()[0]
    duplicates = connection.execute(
        f"SELECT COUNT(*) FROM (SELECT game_id, market_key FROM {LEDGER_TABLE} GROUP BY 1,2 HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    if invalid or duplicates:
        raise RuntimeError("Historical market evaluation validation failed.")


def build_historical_market_evaluation(database_file: Path = DATABASE_FILE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build historical closing-line benchmark and flat-stake performance."""
    validate_database_file(database_file)
    with duckdb.connect(str(database_file)) as connection:
        market = load_historical_market_data(connection)
        probability, spread, totals = create_selected_oof_predictions(connection)
        ledger = create_historical_betting_ledger(probability, spread, totals, market)
        summary = create_historical_betting_summary(ledger)
        connection.execute("BEGIN TRANSACTION")
        try:
            persist_historical_market_evaluation(connection, ledger, summary)
            validate_historical_market_evaluation(connection)
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
    logger.info("Historical market evaluation completed: %s settled selections.", len(ledger))
    return ledger, summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    _, performance = build_historical_market_evaluation()
    print("\nHISTORICAL MARKET PERFORMANCE\n")
    print(performance.to_string(index=False))
