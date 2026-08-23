"""Tests for frozen versus dynamic Elo benchmarking."""

import pandas as pd
import pytest

from src.simulation.benchmark_elo_simulation_modes import (
    BENCHMARK_SUMMARY_COLUMNS,
    TEAM_COMPARISON_COLUMNS,
    calculate_threshold_probability,
    calculate_total_variation_distance,
    run_elo_simulation_benchmark,
)
from src.simulation.run_elo_monte_carlo import (
    DYNAMIC_ELO_MODE,
    FROZEN_ELO_MODE,
)


def create_schedule() -> pd.DataFrame:
    """Create repeated matchups for Elo feedback."""

    rows: list[dict[str, object]] = []

    for week in range(1, 7):
        home_team = (
            "NE"
            if week % 2 == 1
            else "NYJ"
        )
        away_team = (
            "NYJ"
            if home_team == "NE"
            else "NE"
        )

        rows.append(
            {
                "game_id": f"game_{week}",
                "season": 2026,
                "week": week,
                "gameday": pd.Timestamp(
                    "2026-09-01"
                )
                + pd.Timedelta(
                    days=7 * week
                ),
                "gametime": "13:00",
                "home_team": home_team,
                "away_team": away_team,
                "is_neutral": False,
                "home_rating_pregame": (
                    1550.0
                    if home_team == "NE"
                    else 1450.0
                ),
                "away_rating_pregame": (
                    1450.0
                    if away_team == "NYJ"
                    else 1550.0
                ),
            }
        )

    return pd.DataFrame(rows)


def test_benchmark_creates_stable_outputs(
) -> None:
    """Create comparison and aggregate summary schemas."""

    result = run_elo_simulation_benchmark(
        schedule=create_schedule(),
        simulation_count=1_000,
        random_seed=42,
    )

    assert tuple(
        result.team_comparison.columns
    ) == TEAM_COMPARISON_COLUMNS

    assert tuple(
        result.benchmark_summary.columns
    ) == BENCHMARK_SUMMARY_COLUMNS

    assert set(
        result.team_comparison["team"]
    ) == {
        "NE",
        "NYJ",
    }


def test_benchmark_uses_common_random_numbers(
) -> None:
    """Use identical simulation metadata for both modes."""

    result = run_elo_simulation_benchmark(
        schedule=create_schedule(),
        simulation_count=500,
        random_seed=17,
    )

    assert (
        result.dynamic_result.random_seed
        == result.frozen_result.random_seed
        == 17
    )

    assert (
        result.dynamic_result.simulation_count
        == result.frozen_result.simulation_count
        == 500
    )

    assert (
        result.dynamic_result.simulation_mode
        == DYNAMIC_ELO_MODE
    )

    assert (
        result.frozen_result.simulation_mode
        == FROZEN_ELO_MODE
    )

    assert result.benchmark_summary.iloc[0][
        "comparison_method"
    ] == "common_random_numbers"


def test_benchmark_preserves_total_expected_wins(
) -> None:
    """Allocate the same total wins in both modes."""

    result = run_elo_simulation_benchmark(
        schedule=create_schedule(),
        simulation_count=1_000,
        random_seed=42,
    )

    assert result.team_comparison[
        "expected_wins_delta"
    ].sum() == pytest.approx(0.0)


def test_threshold_probability(
) -> None:
    """Calculate upper and lower tail probability."""

    distribution = pd.DataFrame(
        {
            "team": [
                "NE",
                "NE",
                "NE",
            ],
            "wins": [
                2,
                3,
                14,
            ],
            "probability": [
                0.20,
                0.30,
                0.50,
            ],
        }
    )

    assert calculate_threshold_probability(
        distribution=distribution,
        team="NE",
        minimum_wins=14,
    ) == pytest.approx(0.50)

    assert calculate_threshold_probability(
        distribution=distribution,
        team="NE",
        maximum_wins=3,
    ) == pytest.approx(0.50)

    assert calculate_threshold_probability(
        distribution=distribution,
        team="NE",
        maximum_wins=2,
    ) == pytest.approx(0.20)


def test_total_variation_distance(
) -> None:
    """Measure discrete distribution divergence."""

    dynamic = pd.DataFrame(
        {
            "team": [
                "NE",
                "NE",
            ],
            "wins": [
                1,
                2,
            ],
            "probability": [
                0.25,
                0.75,
            ],
        }
    )

    frozen = pd.DataFrame(
        {
            "team": [
                "NE",
                "NE",
            ],
            "wins": [
                1,
                2,
            ],
            "probability": [
                0.50,
                0.50,
            ],
        }
    )

    assert calculate_total_variation_distance(
        dynamic_distribution=dynamic,
        frozen_distribution=frozen,
        team="NE",
    ) == pytest.approx(0.25)