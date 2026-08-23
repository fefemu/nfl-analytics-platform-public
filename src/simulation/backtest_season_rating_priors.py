"""Backtest preseason rating priors against actual season wins."""

import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.modeling.train_logistic_baseline import DATABASE_FILE, validate_database_file
from src.processing.build_external_team_strengths import download_csv_with_retries
from src.simulation.backtest_simulation_rating_inputs import normalize_nfelounits_elo
from src.simulation.run_elo_monte_carlo import (
    DYNAMIC_ELO_MODE,
    FROZEN_ELO_MODE,
    run_elo_monte_carlo,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

BACKTEST_SEASONS = (2021, 2022, 2023, 2024)
DEFAULT_BACKTEST_SIMULATIONS = 2_500
DEFAULT_RANDOM_SEED = 42
NFELOUNITS_ELO_URL = (
    "https://raw.githubusercontent.com/greerreNFL/"
    "nfelounits/refs/heads/main/Output/elo.csv"
)

SUMMARY_COLUMNS = (
    "candidate_name",
    "simulation_mode",
    "season_count",
    "team_season_count",
    "expected_wins_mae",
    "expected_wins_rmse",
    "expected_wins_correlation",
    "mean_expected_wins_standard_deviation",
    "mean_actual_wins_standard_deviation",
    "dispersion_ratio",
)


def load_season_backtest_sources(
    connection: duckdb.DuckDBPyConnection,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load historical schedules, outcomes and local rating sources."""

    schedules = connection.execute(
        """
        SELECT
            game_id, season, game_type, week, gameday, gametime,
            home_team, away_team,
            CASE WHEN LOWER(COALESCE(location, '')) = 'neutral'
                THEN TRUE ELSE FALSE END AS is_neutral
        FROM processed.schedule
        WHERE season BETWEEN 2021 AND 2024
          AND game_type = 'REG'
          AND is_completed = TRUE
        ORDER BY season, week, gameday, gametime, game_id
        """
    ).fetchdf()
    actual = connection.execute(
        """
        WITH team_games AS (
            SELECT season, home_team AS team,
                CASE WHEN home_score > away_score THEN 1.0
                     WHEN home_score = away_score THEN 0.5 ELSE 0.0 END AS wins
            FROM processed.schedule
            WHERE season BETWEEN 2021 AND 2024 AND game_type = 'REG'
              AND is_completed = TRUE
            UNION ALL
            SELECT season, away_team AS team,
                CASE WHEN away_score > home_score THEN 1.0
                     WHEN away_score = home_score THEN 0.5 ELSE 0.0 END AS wins
            FROM processed.schedule
            WHERE season BETWEEN 2021 AND 2024 AND game_type = 'REG'
              AND is_completed = TRUE
        )
        SELECT season, team, SUM(wins) AS actual_wins
        FROM team_games GROUP BY season, team ORDER BY season, team
        """
    ).fetchdf()
    current = connection.execute(
        """
        WITH ratings AS (
            SELECT source_season AS season, source_week AS week,
                home_team AS team, starting_nfelo_home AS rating
            FROM processed.external_nfelo_game_ratings
            UNION ALL
            SELECT source_season, source_week, away_team, starting_nfelo_away
            FROM processed.external_nfelo_game_ratings
        )
        SELECT season, team, rating
        FROM ratings
        WHERE season BETWEEN 2021 AND 2024 AND week = 1
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY season, team ORDER BY rating
        ) = 1
        ORDER BY season, team
        """
    ).fetchdf()
    win_totals = connection.execute(
        """
        SELECT season, team, wt_rating_elo AS rating
        FROM processed.external_win_total_ratings
        WHERE season BETWEEN 2021 AND 2024
        ORDER BY season, team
        """
    ).fetchdf()
    return schedules, actual, current, win_totals


def create_season_schedule(
    schedule: pd.DataFrame,
    ratings: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    """Attach one preseason rating per team to every season game."""

    season_schedule = schedule.loc[schedule["season"] == season].copy()
    season_ratings = ratings.loc[ratings["season"] == season]
    if season_ratings["team"].duplicated().any():
        raise ValueError(
            f"Season {season} ratings contain duplicate teams."
        )
    rating_map = season_ratings.set_index("team")["rating"]
    teams = set(season_schedule["home_team"]) | set(season_schedule["away_team"])
    missing = sorted(teams - set(rating_map.index))
    if missing:
        raise RuntimeError(
            f"Season {season} ratings are missing teams: " + ", ".join(missing)
        )
    season_schedule["home_rating_pregame"] = season_schedule["home_team"].map(
        rating_map
    )
    season_schedule["away_rating_pregame"] = season_schedule["away_team"].map(
        rating_map
    )
    return season_schedule


def blend_rating_sources(
    base: pd.DataFrame,
    challenger: pd.DataFrame,
    challenger_weight: float,
) -> pd.DataFrame:
    """Blend two complete, same-scale season-team rating sources."""

    if not 0.0 <= challenger_weight <= 1.0:
        raise ValueError("Challenger weight must be between zero and one.")
    merged = base.merge(
        challenger,
        on=["season", "team"],
        how="inner",
        suffixes=("_base", "_challenger"),
        validate="one_to_one",
    )
    if len(merged) != len(base) or len(merged) != len(challenger):
        raise ValueError("Rating sources do not have identical coverage.")
    merged["rating"] = (
        (1.0 - challenger_weight) * merged["rating_base"]
        + challenger_weight * merged["rating_challenger"]
    )
    return merged.loc[:, ["season", "team", "rating"]]


def summarize_season_prior_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate team-season prediction quality and dispersion."""

    rows = []
    for (candidate, mode), group in results.groupby(
        ["candidate_name", "simulation_mode"], sort=False
    ):
        residual = group["expected_wins"] - group["actual_wins"]
        season_dispersion = group.groupby("season").agg(
            expected_sd=("expected_wins", lambda values: values.std(ddof=0)),
            actual_sd=("actual_wins", lambda values: values.std(ddof=0)),
        )
        expected_sd = float(season_dispersion["expected_sd"].mean())
        actual_sd = float(season_dispersion["actual_sd"].mean())
        rows.append(
            {
                "candidate_name": candidate,
                "simulation_mode": mode,
                "season_count": group["season"].nunique(),
                "team_season_count": len(group),
                "expected_wins_mae": float(residual.abs().mean()),
                "expected_wins_rmse": float(np.sqrt(np.mean(residual**2))),
                "expected_wins_correlation": float(
                    group["expected_wins"].corr(group["actual_wins"])
                ),
                "mean_expected_wins_standard_deviation": expected_sd,
                "mean_actual_wins_standard_deviation": actual_sd,
                "dispersion_ratio": expected_sd / actual_sd,
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS).sort_values(
        ["expected_wins_mae", "candidate_name", "simulation_mode"],
        kind="stable",
    ).reset_index(drop=True)


def run_season_prior_backtest(
    database_file: Path = DATABASE_FILE,
    simulation_count: int = DEFAULT_BACKTEST_SIMULATIONS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest all priors without accessing the 2025 holdout."""

    validate_database_file(database_file)
    with duckdb.connect(str(database_file), read_only=True) as connection:
        schedules, actual, current, win_totals = load_season_backtest_sources(
            connection
        )
    nfelounits = normalize_nfelounits_elo(
        download_csv_with_retries(
            source_url=NFELOUNITS_ELO_URL, source_name="nfelounits_elo"
        )
    )
    nfelounits = nfelounits.loc[
        nfelounits["week"] == 1,
        ["season", "team", "elo"],
    ].rename(columns={"elo": "rating"})

    rating_sources = {
        "current_nfelo_prior": current,
        "nfelounits_elo_prior": nfelounits,
        "win_total_elo_prior": win_totals,
        "current_75_win_total_25": blend_rating_sources(
            current, win_totals, 0.25
        ),
        "current_50_win_total_50": blend_rating_sources(
            current, win_totals, 0.50
        ),
        "current_25_win_total_75": blend_rating_sources(
            current, win_totals, 0.75
        ),
    }
    rows = []
    for season in BACKTEST_SEASONS:
        season_actual = actual.loc[
            actual["season"] == season,
            ["team", "actual_wins"],
        ]
        for candidate_name, ratings in rating_sources.items():
            candidate_schedule = create_season_schedule(schedules, ratings, season)
            for mode in (FROZEN_ELO_MODE, DYNAMIC_ELO_MODE):
                result = run_elo_monte_carlo(
                    schedule=candidate_schedule,
                    simulation_count=simulation_count,
                    random_seed=random_seed,
                    simulation_mode=mode,
                )
                comparison = result.team_summary.loc[
                    :, ["team", "expected_wins"]
                ].merge(season_actual, on="team", how="inner", validate="one_to_one")
                if len(comparison) != 32:
                    raise RuntimeError(
                        f"Season {season} comparison does not contain 32 teams."
                    )
                comparison.insert(0, "season", season)
                comparison.insert(0, "simulation_mode", mode)
                comparison.insert(0, "candidate_name", candidate_name)
                rows.append(comparison)

    results = pd.concat(rows, ignore_index=True)
    summary = summarize_season_prior_results(results)
    logger.info(
        "Season-prior backtest completed: %s seasons, %s simulations per "
        "candidate-mode without opening holdout.",
        len(BACKTEST_SEASONS), simulation_count
    )
    return summary, results


def main() -> None:
    """Run and print the season-prior backtest."""

    summary, _ = run_season_prior_backtest()
    print("\nSEASON RATING-PRIOR BACKTEST\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
