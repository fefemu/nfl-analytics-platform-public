"""Tests for dynamic Elo Monte Carlo simulation."""

import pandas as pd
import pytest

from src.simulation.run_elo_monte_carlo import (
    DYNAMIC_ELO_MODE,
    FROZEN_ELO_MODE,
    calculate_most_likely_wins,
    run_elo_monte_carlo,
)


def create_monte_carlo_schedule() -> pd.DataFrame:
    """Create a small two-team simulation schedule."""

    return pd.DataFrame(
        [
            {
                "game_id": "game_1",
                "season": 2026,
                "week": 1,
                "gameday": pd.Timestamp(
                    "2026-09-10"
                ),
                "gametime": "20:20",
                "home_team": "NE",
                "away_team": "NYJ",
                "is_neutral": False,
                "home_rating_pregame": 1550.0,
                "away_rating_pregame": 1450.0,
            },
            {
                "game_id": "game_2",
                "season": 2026,
                "week": 2,
                "gameday": pd.Timestamp(
                    "2026-09-17"
                ),
                "gametime": "20:20",
                "home_team": "NYJ",
                "away_team": "NE",
                "is_neutral": False,
                "home_rating_pregame": 1450.0,
                "away_rating_pregame": 1550.0,
            },
        ]
    )


def test_calculate_most_likely_wins() -> None:
    """Return the modal simulated win total."""

    result = calculate_most_likely_wins(
        win_values=pd.Series(
            [
                0,
                1,
                1,
                1,
                2,
            ]
        ).to_numpy()
    )

    assert result == 1


def test_run_monte_carlo_creates_team_summary() -> None:
    """Create one summary row for every team."""

    result = run_elo_monte_carlo(
        schedule=create_monte_carlo_schedule(),
        simulation_count=100,
        random_seed=42,
    )

    assert len(result.team_summary) == 2
    assert set(result.team_summary["team"]) == {
        "NE",
        "NYJ",
    }
    assert set(result.team_summary["games"]) == {
        2
    }
    assert result.season == 2026
    assert result.simulation_count == 100
    assert result.random_seed == 42


def test_run_monte_carlo_preserves_total_wins() -> None:
    """Allocate one win for every scheduled game."""

    result = run_elo_monte_carlo(
        schedule=create_monte_carlo_schedule(),
        simulation_count=100,
        random_seed=42,
    )

    assert result.team_summary[
        "expected_wins"
    ].sum() == pytest.approx(2.0)


def test_win_distribution_sums_to_one() -> None:
    """Create a complete probability distribution."""

    result = run_elo_monte_carlo(
        schedule=create_monte_carlo_schedule(),
        simulation_count=100,
        random_seed=42,
    )

    probability_sums = (
        result.win_distribution.groupby(
            "team"
        )["probability"].sum()
    )

    assert all(
        probability_sum
        == pytest.approx(1.0)
        for probability_sum
        in probability_sums
    )


def test_monte_carlo_is_reproducible() -> None:
    """Produce identical aggregates with one seed."""

    schedule = create_monte_carlo_schedule()

    first_result = run_elo_monte_carlo(
        schedule=schedule,
        simulation_count=50,
        random_seed=7,
    )

    second_result = run_elo_monte_carlo(
        schedule=schedule,
        simulation_count=50,
        random_seed=7,
    )

    pd.testing.assert_frame_equal(
        first_result.team_summary,
        second_result.team_summary,
    )

    pd.testing.assert_frame_equal(
        first_result.win_distribution,
        second_result.win_distribution,
    )


def test_monte_carlo_rejects_invalid_count() -> None:
    """Reject a non-positive simulation count."""

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        run_elo_monte_carlo(
            schedule=create_monte_carlo_schedule(),
            simulation_count=0,
        )


def test_monte_carlo_includes_current_records() -> None:
    """Combine completed and simulated game records."""

    current_records = pd.DataFrame(
        [
            {
                "team": "NE",
                "wins": 1,
                "losses": 0,
            },
            {
                "team": "NYJ",
                "wins": 0,
                "losses": 1,
            },
        ]
    )

    result = run_elo_monte_carlo(
        schedule=create_monte_carlo_schedule(),
        simulation_count=100,
        random_seed=42,
        current_records=current_records,
    )

    assert set(result.team_summary["games"]) == {
        3
    }

    assert result.team_summary[
        "expected_wins"
    ].sum() == pytest.approx(3.0)

    patriots = result.team_summary.loc[
        result.team_summary["team"] == "NE"
    ].iloc[0]

    assert patriots["minimum_wins"] >= 1


def test_monte_carlo_rejects_unknown_record_team() -> None:
    """Reject standings teams outside the schedule."""

    current_records = pd.DataFrame(
        [
            {
                "team": "UNKNOWN",
                "wins": 1,
                "losses": 0,
            }
        ]
    )

    with pytest.raises(
        ValueError,
        match="unknown teams",
    ):
        run_elo_monte_carlo(
            schedule=create_monte_carlo_schedule(),
            simulation_count=10,
            current_records=current_records,
        )


def test_monte_carlo_preserves_current_ties() -> None:
    """Preserve completed ties in final records."""

    current_records = pd.DataFrame(
        [
            {
                "team": "NE",
                "wins": 0,
                "losses": 0,
                "ties": 1,
            },
            {
                "team": "NYJ",
                "wins": 0,
                "losses": 0,
                "ties": 1,
            },
        ]
    )

    result = run_elo_monte_carlo(
        schedule=create_monte_carlo_schedule(),
        simulation_count=100,
        random_seed=42,
        current_records=current_records,
    )

    assert set(result.team_summary["games"]) == {
        3
    }

    assert set(
        result.team_summary["expected_ties"]
    ) == {
        1
    }

    for row in result.team_summary.itertuples(
        index=False
    ):
        assert (
            row.expected_wins
            + row.expected_losses
            + row.expected_ties
        ) == pytest.approx(row.games)


def test_monte_carlo_rejects_multiple_seasons() -> None:
    """Reject a schedule spanning multiple seasons."""

    schedule = create_monte_carlo_schedule()

    schedule.loc[
        schedule["game_id"] == "game_2",
        "season",
    ] = 2027

    with pytest.raises(
        ValueError,
        match="exactly one season",
    ):
        run_elo_monte_carlo(
            schedule=schedule,
            simulation_count=10,
        )


def test_dynamic_mode_remains_default(
) -> None:
    """Preserve the existing production behavior."""

    result = run_elo_monte_carlo(
        schedule=create_monte_carlo_schedule(),
        simulation_count=100,
        random_seed=42,
    )

    assert (
        result.simulation_mode
        == DYNAMIC_ELO_MODE
    )


def test_frozen_mode_preserves_initial_ratings(
) -> None:
    """Keep Elo unchanged across every simulation."""

    result = run_elo_monte_carlo(
        schedule=create_monte_carlo_schedule(),
        simulation_count=100,
        random_seed=42,
        simulation_mode=FROZEN_ELO_MODE,
    )

    expected_ratings = {
        "NE": 1550.0,
        "NYJ": 1450.0,
    }

    for row in result.team_summary.itertuples(
        index=False
    ):
        assert row.expected_final_elo == pytest.approx(
            expected_ratings[row.team]
        )

    assert (
        result.simulation_mode
        == FROZEN_ELO_MODE
    )


def test_dynamic_and_frozen_modes_can_diverge(
) -> None:
    """Allow simulated results to affect later games."""

    schedule = pd.concat(
        [
            create_monte_carlo_schedule(),
            create_monte_carlo_schedule().assign(
                game_id=[
                    "game_3",
                    "game_4",
                ],
                week=[
                    3,
                    4,
                ],
            ),
        ],
        ignore_index=True,
    )

    dynamic = run_elo_monte_carlo(
        schedule=schedule,
        simulation_count=1_000,
        random_seed=42,
        simulation_mode=DYNAMIC_ELO_MODE,
    )

    frozen = run_elo_monte_carlo(
        schedule=schedule,
        simulation_count=1_000,
        random_seed=42,
        simulation_mode=FROZEN_ELO_MODE,
    )

    dynamic_wins = (
        dynamic.team_summary.set_index(
            "team"
        )["expected_wins"]
    )

    frozen_wins = (
        frozen.team_summary.set_index(
            "team"
        )["expected_wins"]
    )

    assert not dynamic_wins.equals(
        frozen_wins
    )


def test_rejects_unknown_simulation_mode(
) -> None:
    """Reject unsupported benchmark modes."""

    with pytest.raises(
        ValueError,
        match="Unsupported Elo simulation mode",
    ):
        run_elo_monte_carlo(
            schedule=create_monte_carlo_schedule(),
            simulation_count=10,
            simulation_mode="UNKNOWN",
        )
