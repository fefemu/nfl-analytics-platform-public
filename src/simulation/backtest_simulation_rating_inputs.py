"""Historical validation of candidate Monte Carlo rating inputs."""

import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.modeling.backtest_elo_rating_sources import BACKTEST_VALIDATION_SEASONS
from src.modeling.train_logistic_baseline import (
    DATABASE_FILE,
    evaluate_probabilities,
    validate_database_file,
)
from src.models.elo import calculate_expected_probability
from src.processing.build_external_team_strengths import download_csv_with_retries
from src.processing.normalize_external_team_strengths import normalize_team

NFELOUNITS_ELO_URL = (
    "https://raw.githubusercontent.com/greerreNFL/"
    "nfelounits/refs/heads/main/Output/elo.csv"
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

HOME_ADVANTAGE = 50.0
SUMMARY_COLUMNS = (
    "candidate_name",
    "validation_game_count",
    "accuracy",
    "brier_score",
    "log_loss",
    "probability_standard_deviation",
)


def normalize_nfelounits_elo(source_data: pd.DataFrame) -> pd.DataFrame:
    """Normalize leakage-safe pregame Elo rows."""

    required = {"season", "week", "team", "elo"}
    missing = sorted(required - set(source_data.columns))
    if missing:
        raise ValueError("nfelounits Elo is missing columns: " + ", ".join(missing))
    data = source_data.loc[:, ["season", "week", "team", "elo"]].copy()
    data["season"] = pd.to_numeric(data["season"], errors="raise").astype(int)
    data["week"] = pd.to_numeric(data["week"], errors="raise").astype(int)
    data["team"] = [
        normalize_team(team, int(season))
        for team, season in zip(data["team"], data["season"], strict=True)
    ]
    data["elo"] = pd.to_numeric(data["elo"], errors="raise").astype(float)
    if data[["season", "week", "team"]].duplicated().any():
        raise ValueError("nfelounits Elo contains duplicate keys.")
    return data


def load_historical_simulation_inputs(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Load development regular seasons without holdout."""

    return connection.execute(
        """
        SELECT
            dataset.game_id,
            dataset.season,
            dataset.week,
            dataset.home_team,
            dataset.away_team,
            CAST(dataset.target_home_win AS INTEGER) AS target_home_win,
            external.starting_nfelo_home AS current_home_rating,
            external.starting_nfelo_away AS current_away_rating
        FROM analytics.game_modeling_dataset AS dataset
        INNER JOIN processed.schedule AS schedule USING (game_id)
        INNER JOIN processed.external_nfelo_game_ratings AS external
            ON dataset.game_id = external.normalized_game_id
        WHERE schedule.game_type = 'REG'
          AND dataset.season < 2025
          AND dataset.target_home_win IS NOT NULL
        ORDER BY dataset.season, dataset.week, dataset.game_id
        """
    ).fetchdf()


def attach_nfelounits_ratings(
    games: pd.DataFrame,
    ratings: pd.DataFrame,
) -> pd.DataFrame:
    """Attach exact pregame home and away nfelounits Elo values."""

    home = ratings.rename(columns={"team": "home_team", "elo": "unit_home_rating"})
    away = ratings.rename(columns={"team": "away_team", "elo": "unit_away_rating"})
    data = games.merge(
        home, on=["season", "week", "home_team"], how="inner", validate="many_to_one"
    ).merge(
        away, on=["season", "week", "away_team"], how="inner", validate="many_to_one"
    )
    if len(data) != len(games):
        raise RuntimeError(
            f"nfelounits historical coverage mismatch: {len(data)}/{len(games)}."
        )
    return data


def evaluate_rating_input_candidates(data: pd.DataFrame) -> pd.DataFrame:
    """Evaluate current and nfelounits inputs on identical seasons."""

    validation = data.loc[data["season"].isin(BACKTEST_VALIDATION_SEASONS)].copy()
    candidates = {
        "current_nfelo_rating_input": ("current_home_rating", "current_away_rating"),
        "nfelounits_elo_rating_input": ("unit_home_rating", "unit_away_rating"),
    }
    rows = []
    for name, (home_column, away_column) in candidates.items():
        probabilities = np.array(
            [
                calculate_expected_probability(
                    float(home), float(away), HOME_ADVANTAGE
                )
                for home, away in zip(
                    validation[home_column], validation[away_column], strict=True
                )
            ]
        )
        metrics = evaluate_probabilities(validation["target_home_win"], probabilities)
        rows.append(
            {
                "candidate_name": name,
                "validation_game_count": len(validation),
                "accuracy": metrics.accuracy,
                "brier_score": metrics.brier_score,
                "log_loss": metrics.log_loss,
                "probability_standard_deviation": float(probabilities.std(ddof=0)),
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS).sort_values(
        ["brier_score", "log_loss"], kind="stable"
    ).reset_index(drop=True)


def run_historical_rating_input_backtest(
    database_file: Path = DATABASE_FILE,
) -> pd.DataFrame:
    """Download candidate ratings and evaluate without holdout."""

    validate_database_file(database_file)
    with duckdb.connect(str(database_file), read_only=True) as connection:
        games = load_historical_simulation_inputs(connection)
    ratings = normalize_nfelounits_elo(
        download_csv_with_retries(
            source_url=NFELOUNITS_ELO_URL, source_name="nfelounits_elo"
        )
    )
    data = attach_nfelounits_ratings(games, ratings)
    summary = evaluate_rating_input_candidates(data)
    logger.info(
        "Historical simulation rating-input backtest completed on %s games "
        "without opening holdout.", len(data)
    )
    return summary


def main() -> None:
    """Run and print the historical benchmark."""

    print("\nHISTORICAL SIMULATION RATING-INPUT BACKTEST\n")
    print(run_historical_rating_input_backtest().to_string(index=False))


if __name__ == "__main__":
    main()
