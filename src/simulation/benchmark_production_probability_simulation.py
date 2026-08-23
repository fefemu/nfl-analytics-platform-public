"""
NFL Analytics Platform
Production Probability Monte Carlo Benchmark

Purpose:
    Compare the current internal Elo-only Monte Carlo with
    production game-probability Frozen and Dynamic candidates.

    The benchmark is read-only and does not replace persisted
    production simulation tables.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.modeling.diagnose_prediction_dispersion import (
    load_current_market_expected_wins,
)
from src.modeling.train_logistic_baseline import (
    DATABASE_FILE,
    validate_database_file,
)
from src.simulation.run_current_season_simulation import (
    load_current_team_records,
    load_latest_regular_season_schedule,
    validate_prediction_source,
)
from src.simulation.run_elo_monte_carlo import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_SIMULATION_COUNT,
    DYNAMIC_ELO_MODE,
    FROZEN_ELO_MODE,
    INTERNAL_ELO_PROBABILITY_SOURCE,
    PRODUCTION_PROBABILITY_SOURCE,
    MonteCarloSimulationResult,
    run_elo_monte_carlo,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


SUMMARY_COLUMNS = (
    "candidate_name",
    "probability_source",
    "simulation_mode",
    "simulation_count",
    "team_count",
    "mean_expected_wins",
    "expected_wins_standard_deviation",
    "minimum_expected_wins",
    "maximum_expected_wins",
    "p10_expected_wins",
    "p90_expected_wins",
    "market_expected_wins_mae",
    "market_expected_wins_correlation",
)

TEAM_COMPARISON_COLUMNS = (
    "team",
    "current_internal_elo_expected_wins",
    "production_frozen_expected_wins",
    "production_dynamic_expected_wins",
    "market_expected_wins",
    "production_dynamic_vs_current_delta",
    "production_dynamic_vs_market_delta",
)

PROBABILITY_BENCHMARK_COLUMNS = (
    "game_count",
    "model_probability_mean",
    "market_probability_mean",
    "model_probability_standard_deviation",
    "market_probability_standard_deviation",
    "model_to_market_dispersion_ratio",
    "market_on_model_intercept",
    "market_on_model_slope",
    "mean_absolute_probability_difference",
    "model_minimum_probability",
    "model_maximum_probability",
    "market_minimum_probability",
    "market_maximum_probability",
)


def load_current_probability_market_comparison(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Load one home probability pair per current game."""

    return connection.execute(
        """
        SELECT
            game_id,
            model_probability,
            consensus_no_vig_probability
                AS market_probability
        FROM analytics.current_moneyline_value
        WHERE outcome_type = 'home'
        ORDER BY game_id
        """
    ).fetchdf()


def summarize_probability_market_comparison(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Measure game-level production probability dispersion."""

    required_columns = {
        "model_probability",
        "market_probability",
    }
    missing_columns = sorted(
        required_columns - set(comparison.columns)
    )

    if missing_columns:
        raise ValueError(
            "Probability comparison is missing columns: "
            + ", ".join(missing_columns)
        )

    model = pd.to_numeric(
        comparison["model_probability"],
        errors="coerce",
    ).to_numpy(dtype=float)
    market = pd.to_numeric(
        comparison["market_probability"],
        errors="coerce",
    ).to_numpy(dtype=float)
    valid_mask = np.isfinite(model) & np.isfinite(market)
    model = model[valid_mask]
    market = market[valid_mask]

    if len(model) < 2:
        raise ValueError(
            "At least two probability pairs are required."
        )

    model_standard_deviation = float(np.std(model, ddof=0))
    market_standard_deviation = float(np.std(market, ddof=0))
    slope = float(
        np.cov(model, market, ddof=0)[0, 1]
        / np.var(model)
    )
    intercept = float(
        np.mean(market) - slope * np.mean(model)
    )

    return pd.DataFrame(
        [
            {
                "game_count": len(model),
                "model_probability_mean": float(np.mean(model)),
                "market_probability_mean": float(np.mean(market)),
                "model_probability_standard_deviation": (
                    model_standard_deviation
                ),
                "market_probability_standard_deviation": (
                    market_standard_deviation
                ),
                "model_to_market_dispersion_ratio": (
                    model_standard_deviation
                    / market_standard_deviation
                ),
                "market_on_model_intercept": intercept,
                "market_on_model_slope": slope,
                "mean_absolute_probability_difference": float(
                    np.mean(np.abs(model - market))
                ),
                "model_minimum_probability": float(np.min(model)),
                "model_maximum_probability": float(np.max(model)),
                "market_minimum_probability": float(np.min(market)),
                "market_maximum_probability": float(np.max(market)),
            }
        ],
        columns=PROBABILITY_BENCHMARK_COLUMNS,
    )


def summarize_simulation_candidate(
    *,
    candidate_name: str,
    result: MonteCarloSimulationResult,
    market_expected_wins: pd.DataFrame,
) -> pd.DataFrame:
    """Create one benchmark summary row."""

    comparison = result.team_summary.loc[
        :,
        ["team", "expected_wins"],
    ].merge(
        market_expected_wins.loc[:, ["team", "wins"]],
        on="team",
        how="inner",
        validate="one_to_one",
    )

    if len(comparison) != len(result.team_summary):
        raise ValueError(
            "Market expected wins do not cover every "
            "simulated team."
        )

    expected_wins = result.team_summary[
        "expected_wins"
    ].to_numpy(dtype=float)

    correlation = float(
        comparison["expected_wins"].corr(
            comparison["wins"]
        )
    )

    return pd.DataFrame(
        [
            {
                "candidate_name": candidate_name,
                "probability_source": result.probability_source,
                "simulation_mode": result.simulation_mode,
                "simulation_count": result.simulation_count,
                "team_count": len(expected_wins),
                "mean_expected_wins": float(
                    np.mean(expected_wins)
                ),
                "expected_wins_standard_deviation": float(
                    np.std(expected_wins, ddof=0)
                ),
                "minimum_expected_wins": float(
                    np.min(expected_wins)
                ),
                "maximum_expected_wins": float(
                    np.max(expected_wins)
                ),
                "p10_expected_wins": float(
                    np.quantile(expected_wins, 0.10)
                ),
                "p90_expected_wins": float(
                    np.quantile(expected_wins, 0.90)
                ),
                "market_expected_wins_mae": float(
                    (
                        comparison["expected_wins"]
                        - comparison["wins"]
                    ).abs().mean()
                ),
                "market_expected_wins_correlation": correlation,
            }
        ],
        columns=SUMMARY_COLUMNS,
    )


def create_team_comparison(
    *,
    current_result: MonteCarloSimulationResult,
    production_frozen_result: MonteCarloSimulationResult,
    production_dynamic_result: MonteCarloSimulationResult,
    market_expected_wins: pd.DataFrame,
) -> pd.DataFrame:
    """Align team expected wins across all candidates."""

    def select_expected(
        result: MonteCarloSimulationResult,
        column_name: str,
    ) -> pd.DataFrame:
        return result.team_summary.loc[
            :,
            ["team", "expected_wins"],
        ].rename(columns={"expected_wins": column_name})

    comparison = select_expected(
        current_result,
        "current_internal_elo_expected_wins",
    ).merge(
        select_expected(
            production_frozen_result,
            "production_frozen_expected_wins",
        ),
        on="team",
        validate="one_to_one",
    ).merge(
        select_expected(
            production_dynamic_result,
            "production_dynamic_expected_wins",
        ),
        on="team",
        validate="one_to_one",
    ).merge(
        market_expected_wins.loc[
            :,
            ["team", "wins"],
        ].rename(columns={"wins": "market_expected_wins"}),
        on="team",
        validate="one_to_one",
    )

    comparison["production_dynamic_vs_current_delta"] = (
        comparison["production_dynamic_expected_wins"]
        - comparison["current_internal_elo_expected_wins"]
    )
    comparison["production_dynamic_vs_market_delta"] = (
        comparison["production_dynamic_expected_wins"]
        - comparison["market_expected_wins"]
    )

    return comparison.loc[
        :,
        TEAM_COMPARISON_COLUMNS,
    ].sort_values(
        by=[
            "production_dynamic_expected_wins",
            "team",
        ],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)


def run_production_probability_simulation_benchmark(
    database_file: Path = DATABASE_FILE,
    simulation_count: int = DEFAULT_SIMULATION_COUNT,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run all three candidates with common random numbers."""

    validate_database_file(database_file)

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        validate_prediction_source(connection)
        schedule = load_latest_regular_season_schedule(connection)
        season = int(schedule["season"].iloc[0])
        current_records = load_current_team_records(
            connection=connection,
            season=season,
        )
        market_expected_wins = load_current_market_expected_wins(
            connection
        )
        probability_market_comparison = (
            load_current_probability_market_comparison(connection)
        )

    candidate_definitions = (
        (
            "current_internal_elo_dynamic",
            DYNAMIC_ELO_MODE,
            INTERNAL_ELO_PROBABILITY_SOURCE,
        ),
        (
            "production_probability_frozen",
            FROZEN_ELO_MODE,
            PRODUCTION_PROBABILITY_SOURCE,
        ),
        (
            "production_probability_dynamic",
            DYNAMIC_ELO_MODE,
            PRODUCTION_PROBABILITY_SOURCE,
        ),
    )

    results: dict[str, MonteCarloSimulationResult] = {}
    summary_frames = []

    for candidate_name, mode, source in candidate_definitions:
        result = run_elo_monte_carlo(
            schedule=schedule,
            simulation_count=simulation_count,
            random_seed=random_seed,
            current_records=current_records,
            simulation_mode=mode,
            probability_source=source,
        )
        results[candidate_name] = result
        summary_frames.append(
            summarize_simulation_candidate(
                candidate_name=candidate_name,
                result=result,
                market_expected_wins=market_expected_wins,
            )
        )

    summary = pd.concat(
        summary_frames,
        ignore_index=True,
    ).sort_values(
        by=["market_expected_wins_mae", "candidate_name"],
        kind="stable",
    ).reset_index(drop=True)

    team_comparison = create_team_comparison(
        current_result=results["current_internal_elo_dynamic"],
        production_frozen_result=(
            results["production_probability_frozen"]
        ),
        production_dynamic_result=(
            results["production_probability_dynamic"]
        ),
        market_expected_wins=market_expected_wins,
    )

    logger.info(
        "Production probability simulation benchmark "
        "completed: %s simulations per candidate.",
        simulation_count,
    )

    probability_benchmark = (
        summarize_probability_market_comparison(
            probability_market_comparison
        )
    )

    return summary, team_comparison, probability_benchmark


def main() -> None:
    """Run and print the production probability benchmark."""

    summary, team_comparison, probability_benchmark = (
        run_production_probability_simulation_benchmark()
    )

    print("\nPRODUCTION PROBABILITY SIMULATION BENCHMARK\n")
    print(summary.to_string(index=False))

    print("\nTEAM EXPECTED WINS COMPARISON\n")
    print(team_comparison.to_string(index=False))

    print("\nGAME PROBABILITY MARKET BENCHMARK\n")
    print(probability_benchmark.to_string(index=False))


if __name__ == "__main__":
    main()
