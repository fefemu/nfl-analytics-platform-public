"""Benchmark current and nfelounits Elo simulation inputs."""

import logging
from pathlib import Path

import duckdb
import pandas as pd

from src.modeling.diagnose_prediction_dispersion import (
    load_current_market_expected_wins,
)
from src.modeling.train_logistic_baseline import (
    DATABASE_FILE,
    validate_database_file,
)
from src.processing.build_external_team_strengths import (
    download_csv_with_retries,
)
from src.processing.normalize_external_team_strengths import normalize_team
from src.simulation.benchmark_production_probability_simulation import (
    summarize_simulation_candidate,
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
    run_elo_monte_carlo,
)
from src.simulation.backtest_season_rating_priors import blend_rating_sources


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

NFELOUNITS_ELO_URL = (
    "https://raw.githubusercontent.com/greerreNFL/"
    "nfelounits/refs/heads/main/Output/elo.csv"
)


def prepare_latest_nfelounits_ratings(
    source_data: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    """Prepare one latest pregame composite rating per team."""

    required = {"season", "week", "team", "elo"}
    missing = sorted(required - set(source_data.columns))
    if missing:
        raise ValueError(
            "nfelounits Elo data is missing columns: " + ", ".join(missing)
        )

    ratings = source_data.loc[
        pd.to_numeric(source_data["season"], errors="coerce") == season,
        ["season", "week", "team", "elo"],
    ].copy()
    ratings["season"] = ratings["season"].astype(int)
    ratings["week"] = pd.to_numeric(ratings["week"], errors="raise").astype(int)
    ratings["team"] = [
        normalize_team(team, season) for team in ratings["team"]
    ]
    ratings["rating"] = pd.to_numeric(ratings["elo"], errors="raise")
    ratings = ratings.sort_values(["week", "team"], kind="stable")
    ratings = ratings.drop_duplicates("team", keep="last")

    if len(ratings) != 32 or ratings["team"].duplicated().any():
        raise RuntimeError(
            "nfelounits Elo must provide exactly 32 current teams."
        )
    return ratings.loc[:, ["team", "rating"]].reset_index(drop=True)


def replace_schedule_ratings(
    schedule: pd.DataFrame,
    ratings: pd.DataFrame,
) -> pd.DataFrame:
    """Replace both team-rating columns while preserving the schedule."""

    rating_map = ratings.set_index("team")["rating"]
    teams = set(schedule["home_team"]) | set(schedule["away_team"])
    missing = sorted(teams - set(rating_map.index))
    if missing:
        raise ValueError("Ratings are missing teams: " + ", ".join(missing))

    updated = schedule.copy()
    updated["home_rating_pregame"] = updated["home_team"].map(rating_map)
    updated["away_rating_pregame"] = updated["away_team"].map(rating_map)
    return updated


def run_rating_input_benchmark(
    database_file: Path = DATABASE_FILE,
    simulation_count: int = DEFAULT_SIMULATION_COUNT,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run paired rating-input and dynamic/frozen simulations."""

    validate_database_file(database_file)
    with duckdb.connect(str(database_file), read_only=True) as connection:
        validate_prediction_source(connection)
        current_schedule = load_latest_regular_season_schedule(connection)
        season = int(current_schedule["season"].iloc[0])
        current_records = load_current_team_records(connection, season)
        market_expected_wins = load_current_market_expected_wins(connection)
        win_total_ratings = connection.execute(
            """
            SELECT team, wt_rating_elo AS rating
            FROM processed.external_win_total_ratings
            WHERE season = ?
            ORDER BY team
            """,
            [season],
        ).fetchdf()

    source = download_csv_with_retries(
        source_url=NFELOUNITS_ELO_URL,
        source_name="nfelounits_elo",
    )
    nfelounits_ratings = prepare_latest_nfelounits_ratings(source, season)
    nfelounits_schedule = replace_schedule_ratings(
        current_schedule, nfelounits_ratings
    )
    win_total_schedule = replace_schedule_ratings(
        current_schedule, win_total_ratings
    )
    current_ratings = pd.concat(
        [
            current_schedule.loc[:, ["home_team", "home_rating_pregame"]]
            .rename(columns={"home_team": "team", "home_rating_pregame": "rating"}),
            current_schedule.loc[:, ["away_team", "away_rating_pregame"]]
            .rename(columns={"away_team": "team", "away_rating_pregame": "rating"}),
        ],
        ignore_index=True,
    ).drop_duplicates("team")
    current_ratings.insert(0, "season", season)
    blend_schedules = {}
    win_total_with_season = win_total_ratings.copy()
    win_total_with_season.insert(0, "season", season)
    for weight in (0.25, 0.50, 0.75):
        blended = blend_rating_sources(
            current_ratings, win_total_with_season, weight
        ).loc[:, ["team", "rating"]]
        blend_schedules[weight] = replace_schedule_ratings(
            current_schedule, blended
        )

    candidates = {
        "current_nfelo_frozen": (current_schedule, FROZEN_ELO_MODE),
        "current_nfelo_dynamic": (current_schedule, DYNAMIC_ELO_MODE),
        "nfelounits_elo_frozen": (nfelounits_schedule, FROZEN_ELO_MODE),
        "nfelounits_elo_dynamic": (nfelounits_schedule, DYNAMIC_ELO_MODE),
        "win_total_elo_frozen": (win_total_schedule, FROZEN_ELO_MODE),
        "win_total_elo_dynamic": (win_total_schedule, DYNAMIC_ELO_MODE),
    }
    for weight, schedule in blend_schedules.items():
        label = int(weight * 100)
        candidates[f"current_win_total_{label}_frozen"] = (
            schedule, FROZEN_ELO_MODE
        )
        candidates[f"current_win_total_{label}_dynamic"] = (
            schedule, DYNAMIC_ELO_MODE
        )
    summaries = []
    team_results = []
    for name, (schedule, mode) in candidates.items():
        result = run_elo_monte_carlo(
            schedule=schedule,
            simulation_count=simulation_count,
            random_seed=random_seed,
            current_records=current_records,
            simulation_mode=mode,
        )
        summaries.append(
            summarize_simulation_candidate(
                candidate_name=name,
                result=result,
                market_expected_wins=market_expected_wins,
            )
        )
        team = result.team_summary.loc[:, ["team", "expected_wins"]].copy()
        team.insert(0, "candidate_name", name)
        team_results.append(team)

    summary = pd.concat(summaries, ignore_index=True).sort_values(
        ["market_expected_wins_mae", "candidate_name"], kind="stable"
    ).reset_index(drop=True)
    teams = pd.concat(team_results, ignore_index=True)
    logger.info(
        "Simulation rating-input benchmark completed: %s simulations per "
        "candidate.", simulation_count
    )
    return summary, teams


def main() -> None:
    """Run and print the rating-input benchmark."""

    summary, teams = run_rating_input_benchmark()
    print("\nSIMULATION RATING-INPUT BENCHMARK\n")
    print(summary.to_string(index=False))
    print("\nNFELOUNITS DYNAMIC EXPECTED WINS\n")
    print(
        teams.loc[
            teams["candidate_name"] == "nfelounits_elo_dynamic"
        ].sort_values("expected_wins", ascending=False).to_string(index=False)
    )


if __name__ == "__main__":
    main()
