"""
NFL Analytics Platform
Dynamic Elo Monte Carlo Simulation

Purpose:
    Repeat the dynamic Elo regular-season simulation and
    summarize team record distributions.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.modeling.production_model import (
    PRODUCTION_MODEL,
)
from src.models.elo import (
    calculate_expected_probability,
    update_ratings,
)
from src.simulation.simulate_elo_season import (
    extract_initial_ratings,
)


DEFAULT_SIMULATION_COUNT = 10_000
DEFAULT_RANDOM_SEED = 42

DYNAMIC_ELO_MODE = "DYNAMIC_ELO"
FROZEN_ELO_MODE = "FROZEN_ELO"

INTERNAL_ELO_PROBABILITY_SOURCE = (
    "INTERNAL_ELO_PROBABILITY"
)
PRODUCTION_PROBABILITY_SOURCE = (
    "PRODUCTION_GAME_PROBABILITY"
)

SUPPORTED_PROBABILITY_SOURCES = {
    INTERNAL_ELO_PROBABILITY_SOURCE,
    PRODUCTION_PROBABILITY_SOURCE,
}

SUPPORTED_SIMULATION_MODES = {
    DYNAMIC_ELO_MODE,
    FROZEN_ELO_MODE,
}


@dataclass
class MonteCarloSimulationResult:
    """Store aggregated Monte Carlo outputs."""

    team_summary: pd.DataFrame
    win_distribution: pd.DataFrame
    season: int
    simulation_count: int
    random_seed: int
    simulation_mode: str
    probability_source: str = (
        INTERNAL_ELO_PROBABILITY_SOURCE
    )


def adjust_baseline_probability_for_elo_change(
    *,
    baseline_home_win_probability: float,
    home_rating_change: float,
    away_rating_change: float,
) -> float:
    """Apply simulated Elo movement to a production baseline."""

    if not 0.0 < baseline_home_win_probability < 1.0:
        raise ValueError(
            "Baseline home win probability must be "
            "strictly between zero and one."
        )

    baseline_log_odds = np.log(
        baseline_home_win_probability
        / (1.0 - baseline_home_win_probability)
    )
    elo_log_odds_change = (
        np.log(10.0)
        / 400.0
        * (home_rating_change - away_rating_change)
    )
    adjusted_log_odds = (
        baseline_log_odds + elo_log_odds_change
    )

    return float(
        1.0 / (1.0 + np.exp(-adjusted_log_odds))
    )


def calculate_most_likely_wins(
    win_values: np.ndarray,
) -> int:
    """Return the most frequent simulated win total."""

    if win_values.size == 0:
        raise ValueError(
            "Win values must not be empty."
        )

    integer_wins = win_values.astype(int)

    frequencies = np.bincount(
        integer_wins
    )

    return int(np.argmax(frequencies))


def run_elo_monte_carlo(
    schedule: pd.DataFrame,
    simulation_count: int = (
        DEFAULT_SIMULATION_COUNT
    ),
    random_seed: int = DEFAULT_RANDOM_SEED,
    current_records: pd.DataFrame | None = None,
    simulation_mode: str = DYNAMIC_ELO_MODE,
    probability_source: str = (
        INTERNAL_ELO_PROBABILITY_SOURCE
    ),
) -> MonteCarloSimulationResult:
    """
    Run optimized Elo season simulations.

    Dynamic mode updates Elo after every simulated game.
    Frozen mode keeps every team at its initial rating.
    Identical seeds consume identical random-number
    sequences, enabling common-random-number comparison.
    """

    if simulation_mode not in (
        SUPPORTED_SIMULATION_MODES
    ):
        raise ValueError(
            "Unsupported Elo simulation mode: "
            f"{simulation_mode}"
        )

    if probability_source not in SUPPORTED_PROBABILITY_SOURCES:
        raise ValueError(
            "Unsupported probability source: "
            f"{probability_source}"
        )

    if simulation_count <= 0:
        raise ValueError(
            "Simulation count must be greater than zero."
        )

    if schedule.empty:
        raise ValueError(
            "Monte Carlo schedule must not be empty."
        )

    available_seasons = set(
        schedule["season"].astype(int)
    )

    if len(available_seasons) != 1:
        raise ValueError(
            "Monte Carlo schedule must contain exactly "
            "one season."
        )

    simulation_season = next(
        iter(available_seasons)
    )

    initial_rating_map = extract_initial_ratings(
        schedule
    )

    teams = sorted(initial_rating_map)

    team_index = {
        team: index
        for index, team in enumerate(teams)
    }

    current_wins = np.zeros(
        len(teams),
        dtype=np.int16,
    )

    current_losses = np.zeros(
        len(teams),
        dtype=np.int16,
    )

    current_ties = np.zeros(
        len(teams),
        dtype=np.int16,
    )

    if current_records is not None:
        required_record_columns = {
            "team",
            "wins",
            "losses",
        }

        missing_record_columns = sorted(
            required_record_columns
            - set(current_records.columns)
        )

        if missing_record_columns:
            raise ValueError(
                "Current records are missing columns: "
                + ", ".join(missing_record_columns)
            )

        if current_records["team"].duplicated().any():
            raise ValueError(
                "Current records contain duplicate teams."
            )

        unknown_teams = sorted(
            set(current_records["team"])
            - set(teams)
        )

        if unknown_teams:
            raise ValueError(
                "Current records contain unknown teams: "
                + ", ".join(unknown_teams)
            )

        if "ties" not in current_records.columns:
            current_records = (
                current_records.copy()
            )
            current_records["ties"] = 0

        if (
            (
                current_records["wins"] < 0
            ).any()
            or (
                current_records["losses"] < 0
            ).any()
            or (
                current_records["ties"] < 0
            ).any()
        ):
            raise ValueError(
                "Current wins, losses, and ties must "
                "not be negative."
            )

        for record in current_records.itertuples(
            index=False
        ):
            column_index = team_index[
                str(record.team)
            ]

            current_wins[column_index] = int(
                record.wins
            )
            current_losses[column_index] = int(
                record.losses
            )
            current_ties[column_index] = int(
                record.ties
            )

    initial_ratings = np.array(
        [
            initial_rating_map[team]
            for team in teams
        ],
        dtype=float,
    )

    ordered_schedule = schedule.sort_values(
        by=[
            "week",
            "gameday",
            "gametime",
            "game_id",
        ]
    ).reset_index(drop=True)

    if probability_source == PRODUCTION_PROBABILITY_SOURCE:
        if "home_win_probability" not in ordered_schedule.columns:
            raise ValueError(
                "Production-probability simulation requires "
                "home_win_probability."
            )

        baseline_home_probabilities = pd.to_numeric(
            ordered_schedule["home_win_probability"],
            errors="coerce",
        ).to_numpy(dtype=float)

        if (
            ~np.isfinite(baseline_home_probabilities)
        ).any() or (
            (baseline_home_probabilities <= 0.0)
            | (baseline_home_probabilities >= 1.0)
        ).any():
            raise ValueError(
                "Production home win probabilities must be "
                "finite and strictly between zero and one."
            )
    else:
        baseline_home_probabilities = np.full(
            len(ordered_schedule),
            np.nan,
            dtype=float,
        )

    home_team_indices = np.array(
        [
            team_index[str(team)]
            for team in ordered_schedule[
                "home_team"
            ]
        ],
        dtype=np.int16,
    )

    away_team_indices = np.array(
        [
            team_index[str(team)]
            for team in ordered_schedule[
                "away_team"
            ]
        ],
        dtype=np.int16,
    )

    neutral_flags = ordered_schedule[
        "is_neutral"
    ].astype(bool).to_numpy()

    win_matrix = np.zeros(
        shape=(
            simulation_count,
            len(teams),
        ),
        dtype=np.int16,
    )

    final_rating_matrix = np.zeros(
        shape=(
            simulation_count,
            len(teams),
        ),
        dtype=float,
    )

    random_generator = np.random.default_rng(
        random_seed
    )

    for simulation_index in range(
        simulation_count
    ):
        current_ratings = initial_ratings.copy()

        simulated_wins = current_wins.copy()

        for game_index in range(
            len(ordered_schedule)
        ):
            home_index = int(
                home_team_indices[game_index]
            )
            away_index = int(
                away_team_indices[game_index]
            )

            home_rating = float(
                current_ratings[home_index]
            )
            away_rating = float(
                current_ratings[away_index]
            )

            if (
                probability_source
                == PRODUCTION_PROBABILITY_SOURCE
            ):
                baseline_probability = float(
                    baseline_home_probabilities[game_index]
                )

                if simulation_mode == DYNAMIC_ELO_MODE:
                    home_win_probability = (
                        adjust_baseline_probability_for_elo_change(
                            baseline_home_win_probability=(
                                baseline_probability
                            ),
                            home_rating_change=(
                                home_rating
                                - initial_ratings[home_index]
                            ),
                            away_rating_change=(
                                away_rating
                                - initial_ratings[away_index]
                            ),
                        )
                    )
                else:
                    home_win_probability = baseline_probability
            else:
                applied_home_advantage = (
                    0.0
                    if neutral_flags[game_index]
                    else PRODUCTION_MODEL.home_advantage
                )

                home_win_probability = (
                    calculate_expected_probability(
                        team_rating=home_rating,
                        opponent_rating=away_rating,
                        rating_advantage=(
                            applied_home_advantage
                        ),
                    )
                )

            simulated_home_win = bool(
                random_generator.random()
                < home_win_probability
            )

            if simulation_mode == DYNAMIC_ELO_MODE:
                actual_home_score = (
                    1.0
                    if simulated_home_win
                    else 0.0
                )

                (
                    home_rating_post,
                    away_rating_post,
                ) = update_ratings(
                    team_rating=home_rating,
                    opponent_rating=away_rating,
                    actual_score=actual_home_score,
                    expected_probability=(
                        home_win_probability
                    ),
                    k_factor=(
                        PRODUCTION_MODEL.k_factor
                    ),
                )

                current_ratings[home_index] = (
                    home_rating_post
                )
                current_ratings[away_index] = (
                    away_rating_post
                )

            winner_index = (
                home_index
                if simulated_home_win
                else away_index
            )

            simulated_wins[winner_index] += 1

        win_matrix[
            simulation_index,
            :,
        ] = simulated_wins

        final_rating_matrix[
            simulation_index,
            :,
        ] = current_ratings

    summary_rows: list[dict[str, object]] = []
    distribution_rows: list[
        dict[str, object]
    ] = []

    for team in teams:
        column_index = team_index[team]

        team_wins = win_matrix[
            :,
            column_index,
        ]

        team_final_ratings = final_rating_matrix[
            :,
            column_index,
        ]

        completed_games = int(
            current_wins[column_index]
            + current_losses[column_index]
            + current_ties[column_index]
        )

        remaining_games = int(
            (
                schedule["home_team"] == team
            ).sum()
            + (
                schedule["away_team"] == team
            ).sum()
        )

        games_played = (
            completed_games
            + remaining_games
        )

        expected_wins = float(
            team_wins.mean()
        )

        summary_rows.append(
            {
                "team": team,
                "games": games_played,
                "expected_wins": expected_wins,
                "expected_losses": float(
                    games_played
                    - expected_wins
                    - current_ties[column_index]
                ),
                "expected_ties": int(
                    current_ties[column_index]
                ),
                "median_wins": float(
                    np.median(team_wins)
                ),
                "p10_wins": float(
                    np.percentile(
                        team_wins,
                        10,
                    )
                ),
                "p90_wins": float(
                    np.percentile(
                        team_wins,
                        90,
                    )
                ),
                "most_likely_wins": (
                    calculate_most_likely_wins(
                        team_wins
                    )
                ),
                "minimum_wins": int(
                    team_wins.min()
                ),
                "maximum_wins": int(
                    team_wins.max()
                ),
                "expected_final_elo": float(
                    team_final_ratings.mean()
                ),
            }
        )

        unique_wins, counts = np.unique(
            team_wins,
            return_counts=True,
        )

        for win_total, count in zip(
            unique_wins,
            counts,
            strict=True,
        ):
            distribution_rows.append(
                {
                    "team": team,
                    "wins": int(win_total),
                    "simulation_count": int(
                        count
                    ),
                    "probability": float(
                        count / simulation_count
                    ),
                }
            )

    team_summary = pd.DataFrame(
        summary_rows
    ).sort_values(
        by=[
            "expected_wins",
            "expected_final_elo",
            "team",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)

    win_distribution = pd.DataFrame(
        distribution_rows
    ).sort_values(
        by=[
            "team",
            "wins",
        ]
    ).reset_index(drop=True)

    return MonteCarloSimulationResult(
        team_summary=team_summary,
        win_distribution=win_distribution,
        season=simulation_season,
        simulation_count=simulation_count,
        random_seed=random_seed,
        simulation_mode=simulation_mode,
        probability_source=probability_source,
    )
