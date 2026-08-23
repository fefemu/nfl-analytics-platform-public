"""
NFL Analytics Platform
Frozen Versus Dynamic Elo Simulation Benchmark

Purpose:
    Compare frozen and dynamically updated Elo season
    simulations using common random numbers.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

from dataclasses import dataclass

import pandas as pd

from src.simulation.run_elo_monte_carlo import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_SIMULATION_COUNT,
    DYNAMIC_ELO_MODE,
    FROZEN_ELO_MODE,
    MonteCarloSimulationResult,
    run_elo_monte_carlo,
)


TEAM_COMPARISON_COLUMNS = (
    "team",
    "games",
    "dynamic_expected_wins",
    "frozen_expected_wins",
    "expected_wins_delta",
    "absolute_expected_wins_delta",
    "dynamic_p10_wins",
    "frozen_p10_wins",
    "dynamic_p90_wins",
    "frozen_p90_wins",
    "dynamic_win_range",
    "frozen_win_range",
    "win_range_delta",
    "dynamic_expected_final_elo",
    "frozen_expected_final_elo",
    "expected_final_elo_delta",
    "dynamic_probability_14_plus_wins",
    "frozen_probability_14_plus_wins",
    "probability_14_plus_wins_delta",
    "dynamic_probability_3_or_fewer_wins",
    "frozen_probability_3_or_fewer_wins",
    "probability_3_or_fewer_wins_delta",
    "dynamic_probability_2_or_fewer_wins",
    "frozen_probability_2_or_fewer_wins",
    "probability_2_or_fewer_wins_delta",
    "win_distribution_total_variation",
)

BENCHMARK_SUMMARY_COLUMNS = (
    "season",
    "simulation_count",
    "random_seed",
    "comparison_method",
    "dynamic_mode",
    "frozen_mode",
    "team_count",
    "mean_absolute_expected_wins_delta",
    "maximum_absolute_expected_wins_delta",
    "team_with_maximum_expected_wins_delta",
    "mean_win_distribution_total_variation",
    "maximum_win_distribution_total_variation",
)


@dataclass(frozen=True)
class EloSimulationBenchmarkResult:
    """Store paired simulation and comparison outputs."""

    team_comparison: pd.DataFrame
    benchmark_summary: pd.DataFrame
    dynamic_result: MonteCarloSimulationResult
    frozen_result: MonteCarloSimulationResult


def calculate_threshold_probability(
    distribution: pd.DataFrame,
    team: str,
    minimum_wins: int | None = None,
    maximum_wins: int | None = None,
) -> float:
    """Calculate one team win-threshold probability."""

    if (
        minimum_wins is None
        and maximum_wins is None
    ):
        raise ValueError(
            "At least one win threshold is required."
        )

    team_distribution = distribution.loc[
        distribution["team"] == team
    ]

    threshold_mask = pd.Series(
        True,
        index=team_distribution.index,
    )

    if minimum_wins is not None:
        threshold_mask &= (
            team_distribution["wins"]
            >= minimum_wins
        )

    if maximum_wins is not None:
        threshold_mask &= (
            team_distribution["wins"]
            <= maximum_wins
        )

    return float(
        team_distribution.loc[
            threshold_mask,
            "probability",
        ].sum()
    )


def calculate_total_variation_distance(
    dynamic_distribution: pd.DataFrame,
    frozen_distribution: pd.DataFrame,
    team: str,
) -> float:
    """Compare two discrete team win distributions."""

    dynamic_team = dynamic_distribution.loc[
        dynamic_distribution["team"] == team,
        [
            "wins",
            "probability",
        ],
    ].rename(
        columns={
            "probability": "dynamic_probability",
        }
    )

    frozen_team = frozen_distribution.loc[
        frozen_distribution["team"] == team,
        [
            "wins",
            "probability",
        ],
    ].rename(
        columns={
            "probability": "frozen_probability",
        }
    )

    comparison = dynamic_team.merge(
        frozen_team,
        on="wins",
        how="outer",
        validate="one_to_one",
    ).fillna(0.0)

    return float(
        0.5
        * (
            comparison["dynamic_probability"]
            - comparison["frozen_probability"]
        ).abs().sum()
    )


def run_elo_simulation_benchmark(
    schedule: pd.DataFrame,
    simulation_count: int = (
        DEFAULT_SIMULATION_COUNT
    ),
    random_seed: int = DEFAULT_RANDOM_SEED,
    current_records: pd.DataFrame | None = None,
) -> EloSimulationBenchmarkResult:
    """Run paired dynamic and frozen Elo simulations."""

    dynamic_result = run_elo_monte_carlo(
        schedule=schedule,
        simulation_count=simulation_count,
        random_seed=random_seed,
        current_records=current_records,
        simulation_mode=DYNAMIC_ELO_MODE,
    )

    frozen_result = run_elo_monte_carlo(
        schedule=schedule,
        simulation_count=simulation_count,
        random_seed=random_seed,
        current_records=current_records,
        simulation_mode=FROZEN_ELO_MODE,
    )

    if (
        dynamic_result.season
        != frozen_result.season
        or dynamic_result.simulation_count
        != frozen_result.simulation_count
        or dynamic_result.random_seed
        != frozen_result.random_seed
    ):
        raise RuntimeError(
            "Paired Elo simulations do not share "
            "identical benchmark metadata."
        )

    dynamic_summary = (
        dynamic_result.team_summary.rename(
            columns={
                "expected_wins": (
                    "dynamic_expected_wins"
                ),
                "p10_wins": "dynamic_p10_wins",
                "p90_wins": "dynamic_p90_wins",
                "expected_final_elo": (
                    "dynamic_expected_final_elo"
                ),
            }
        )
    )

    frozen_summary = (
        frozen_result.team_summary.rename(
            columns={
                "expected_wins": (
                    "frozen_expected_wins"
                ),
                "p10_wins": "frozen_p10_wins",
                "p90_wins": "frozen_p90_wins",
                "expected_final_elo": (
                    "frozen_expected_final_elo"
                ),
            }
        )
    )

    comparison = dynamic_summary[
        [
            "team",
            "games",
            "dynamic_expected_wins",
            "dynamic_p10_wins",
            "dynamic_p90_wins",
            "dynamic_expected_final_elo",
        ]
    ].merge(
        frozen_summary[
            [
                "team",
                "games",
                "frozen_expected_wins",
                "frozen_p10_wins",
                "frozen_p90_wins",
                "frozen_expected_final_elo",
            ]
        ],
        on="team",
        how="inner",
        validate="one_to_one",
        suffixes=(
            "_dynamic",
            "_frozen",
        ),
    )

    if not (
        comparison["games_dynamic"]
        == comparison["games_frozen"]
    ).all():
        raise RuntimeError(
            "Paired simulations disagree on team games."
        )

    comparison["games"] = comparison[
        "games_dynamic"
    ].astype(int)

    comparison["expected_wins_delta"] = (
        comparison["dynamic_expected_wins"]
        - comparison["frozen_expected_wins"]
    )

    comparison[
        "absolute_expected_wins_delta"
    ] = comparison[
        "expected_wins_delta"
    ].abs()

    comparison["dynamic_win_range"] = (
        comparison["dynamic_p90_wins"]
        - comparison["dynamic_p10_wins"]
    )

    comparison["frozen_win_range"] = (
        comparison["frozen_p90_wins"]
        - comparison["frozen_p10_wins"]
    )

    comparison["win_range_delta"] = (
        comparison["dynamic_win_range"]
        - comparison["frozen_win_range"]
    )

    comparison["expected_final_elo_delta"] = (
        comparison["dynamic_expected_final_elo"]
        - comparison["frozen_expected_final_elo"]
    )

    threshold_definitions = (
        (
            "probability_14_plus_wins",
            14,
            None,
        ),
        (
            "probability_3_or_fewer_wins",
            None,
            3,
        ),
        (
            "probability_2_or_fewer_wins",
            None,
            2,
        ),
    )

    for (
        metric_name,
        minimum_wins,
        maximum_wins,
    ) in threshold_definitions:
        comparison[
            f"dynamic_{metric_name}"
        ] = [
            calculate_threshold_probability(
                distribution=(
                    dynamic_result
                    .win_distribution
                ),
                team=team,
                minimum_wins=minimum_wins,
                maximum_wins=maximum_wins,
            )
            for team in comparison["team"]
        ]

        comparison[
            f"frozen_{metric_name}"
        ] = [
            calculate_threshold_probability(
                distribution=(
                    frozen_result
                    .win_distribution
                ),
                team=team,
                minimum_wins=minimum_wins,
                maximum_wins=maximum_wins,
            )
            for team in comparison["team"]
        ]

        comparison[
            f"{metric_name}_delta"
        ] = (
            comparison[
                f"dynamic_{metric_name}"
            ]
            - comparison[
                f"frozen_{metric_name}"
            ]
        )

    comparison[
        "win_distribution_total_variation"
    ] = [
        calculate_total_variation_distance(
            dynamic_distribution=(
                dynamic_result.win_distribution
            ),
            frozen_distribution=(
                frozen_result.win_distribution
            ),
            team=team,
        )
        for team in comparison["team"]
    ]

    comparison = comparison.sort_values(
        by=[
            "absolute_expected_wins_delta",
            "win_distribution_total_variation",
            "team",
        ],
        ascending=[
            False,
            False,
            True,
        ],
        kind="stable",
    ).reset_index(drop=True)

    team_comparison = comparison.loc[
        :,
        TEAM_COMPARISON_COLUMNS,
    ]

    maximum_row = team_comparison.iloc[0]

    benchmark_summary = pd.DataFrame(
        [
            {
                "season": dynamic_result.season,
                "simulation_count": (
                    simulation_count
                ),
                "random_seed": random_seed,
                "comparison_method": (
                    "common_random_numbers"
                ),
                "dynamic_mode": (
                    DYNAMIC_ELO_MODE
                ),
                "frozen_mode": (
                    FROZEN_ELO_MODE
                ),
                "team_count": len(
                    team_comparison
                ),
                "mean_absolute_expected_wins_delta": float(
                    team_comparison[
                        "absolute_expected_wins_delta"
                    ].mean()
                ),
                "maximum_absolute_expected_wins_delta": float(
                    maximum_row[
                        "absolute_expected_wins_delta"
                    ]
                ),
                "team_with_maximum_expected_wins_delta": str(
                    maximum_row["team"]
                ),
                "mean_win_distribution_total_variation": float(
                    team_comparison[
                        "win_distribution_total_variation"
                    ].mean()
                ),
                "maximum_win_distribution_total_variation": float(
                    team_comparison[
                        "win_distribution_total_variation"
                    ].max()
                ),
            }
        ],
        columns=BENCHMARK_SUMMARY_COLUMNS,
    )

    return EloSimulationBenchmarkResult(
        team_comparison=team_comparison,
        benchmark_summary=benchmark_summary,
        dynamic_result=dynamic_result,
        frozen_result=frozen_result,
    )