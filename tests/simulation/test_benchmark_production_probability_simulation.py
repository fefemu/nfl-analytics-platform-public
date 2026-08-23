import pandas as pd
import pytest

from src.simulation.benchmark_production_probability_simulation import (
    SUMMARY_COLUMNS,
    TEAM_COMPARISON_COLUMNS,
    PROBABILITY_BENCHMARK_COLUMNS,
    create_team_comparison,
    summarize_probability_market_comparison,
    summarize_simulation_candidate,
)
from src.simulation.run_elo_monte_carlo import (
    DYNAMIC_ELO_MODE,
    FROZEN_ELO_MODE,
    INTERNAL_ELO_PROBABILITY_SOURCE,
    PRODUCTION_PROBABILITY_SOURCE,
    run_elo_monte_carlo,
)


def create_schedule() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "g1",
                "season": 2026,
                "week": 1,
                "gameday": pd.Timestamp("2026-09-10"),
                "gametime": "20:20",
                "home_team": "A",
                "away_team": "B",
                "is_neutral": False,
                "home_rating_pregame": 1500.0,
                "away_rating_pregame": 1500.0,
                "home_win_probability": 0.80,
            },
            {
                "game_id": "g2",
                "season": 2026,
                "week": 2,
                "gameday": pd.Timestamp("2026-09-17"),
                "gametime": "20:20",
                "home_team": "B",
                "away_team": "A",
                "is_neutral": False,
                "home_rating_pregame": 1500.0,
                "away_rating_pregame": 1500.0,
                "home_win_probability": 0.20,
            },
        ]
    )


def create_result(mode: str, source: str):
    return run_elo_monte_carlo(
        schedule=create_schedule(),
        simulation_count=2_000,
        random_seed=42,
        simulation_mode=mode,
        probability_source=source,
    )


def market_wins() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2026, 2026],
            "team": ["A", "B"],
            "wins": [1.6, 0.4],
        }
    )


def test_production_probability_changes_expected_wins() -> None:
    result = create_result(
        FROZEN_ELO_MODE,
        PRODUCTION_PROBABILITY_SOURCE,
    )

    expected = result.team_summary.set_index("team")[
        "expected_wins"
    ]

    assert expected["A"] == pytest.approx(1.6, abs=0.06)
    assert expected["B"] == pytest.approx(0.4, abs=0.06)


def test_candidate_summary_schema() -> None:
    result = create_result(
        DYNAMIC_ELO_MODE,
        PRODUCTION_PROBABILITY_SOURCE,
    )
    summary = summarize_simulation_candidate(
        candidate_name="production_dynamic",
        result=result,
        market_expected_wins=market_wins(),
    )

    assert tuple(summary.columns) == SUMMARY_COLUMNS
    assert summary.loc[0, "team_count"] == 2
    assert summary.loc[0, "mean_expected_wins"] == pytest.approx(1.0)


def test_team_comparison_schema_and_deltas() -> None:
    current = create_result(
        DYNAMIC_ELO_MODE,
        INTERNAL_ELO_PROBABILITY_SOURCE,
    )
    frozen = create_result(
        FROZEN_ELO_MODE,
        PRODUCTION_PROBABILITY_SOURCE,
    )
    dynamic = create_result(
        DYNAMIC_ELO_MODE,
        PRODUCTION_PROBABILITY_SOURCE,
    )

    comparison = create_team_comparison(
        current_result=current,
        production_frozen_result=frozen,
        production_dynamic_result=dynamic,
        market_expected_wins=market_wins(),
    )

    assert tuple(comparison.columns) == TEAM_COMPARISON_COLUMNS
    assert len(comparison) == 2
    assert comparison[
        "production_dynamic_vs_current_delta"
    ].notna().all()


def test_production_source_requires_probabilities() -> None:
    with pytest.raises(
        ValueError,
        match="requires home_win_probability",
    ):
        run_elo_monte_carlo(
            schedule=create_schedule().drop(
                columns="home_win_probability"
            ),
            simulation_count=10,
            probability_source=PRODUCTION_PROBABILITY_SOURCE,
        )


def test_probability_market_summary_detects_compression() -> None:
    comparison = pd.DataFrame(
        {
            "model_probability": [0.40, 0.45, 0.55, 0.60],
            "market_probability": [0.30, 0.40, 0.60, 0.70],
        }
    )

    result = summarize_probability_market_comparison(comparison)

    assert tuple(result.columns) == PROBABILITY_BENCHMARK_COLUMNS
    assert result.loc[
        0,
        "model_to_market_dispersion_ratio",
    ] < 1.0
    assert result.loc[0, "market_on_model_slope"] == pytest.approx(2.0)
