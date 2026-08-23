"""
NFL Analytics Platform
External Elo Rating Comparison

Purpose:
    Compare the current internal Elo state with the
    latest nfelounits composite Elo ratings before any
    external rating is admitted to production modeling.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging
from pathlib import Path

import duckdb
import pandas as pd

from src.modeling.train_logistic_baseline import (
    DATABASE_FILE,
    validate_database_file,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

EXTERNAL_ELO_URL = (
    "https://raw.githubusercontent.com/"
    "greerreNFL/nfelounits/refs/heads/"
    "main/Output/elo.csv"
)

EXTERNAL_SOURCE_NAME = "nfelounits_elo"

INTERNAL_ELO_FULL_NAME = (
    "analytics.current_elo_ratings"
)

CURRENT_TEAM_CODE_MAP = {
    "LAR": "LA",
    "OAK": "LV",
    "JAC": "JAX",
    "WSH": "WAS",
}

REQUIRED_EXTERNAL_COLUMNS = {
    "season",
    "week",
    "team",
    "game_id",
    "elo",
    "qb_adj",
}

REQUIRED_INTERNAL_COLUMNS = {
    "team",
    "elo_rating",
    "last_completed_season",
    "last_game_id",
    "as_of_gameday",
}

COMPARISON_COLUMNS = (
    "team",
    "internal_elo_rating",
    "external_elo_rating",
    "external_qb_adjustment",
    "internal_rank",
    "external_rank",
    "rank_change_external_vs_internal",
    "rating_delta_external_minus_internal",
    "absolute_rating_delta",
    "internal_last_completed_season",
    "internal_last_game_id",
    "internal_as_of_gameday",
    "external_season",
    "external_week",
    "external_game_id",
    "external_source_name",
)

SUMMARY_COLUMNS = (
    "team_count",
    "internal_mean_rating",
    "external_mean_rating",
    "mean_absolute_rating_delta",
    "maximum_absolute_rating_delta",
    "team_with_maximum_rating_delta",
    "pearson_rating_correlation",
    "spearman_rank_correlation",
    "external_season",
    "external_week",
)


def validate_required_columns(
    data: pd.DataFrame,
    required_columns: set[str],
    data_name: str,
) -> None:
    """Validate required DataFrame columns."""

    missing_columns = sorted(
        required_columns - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{data_name} is missing columns: "
            + ", ".join(missing_columns)
        )


def prepare_latest_external_elo(
    external_history: pd.DataFrame,
) -> pd.DataFrame:
    """Select and normalize the latest external ratings."""

    validate_required_columns(
        data=external_history,
        required_columns=REQUIRED_EXTERNAL_COLUMNS,
        data_name="External Elo history",
    )

    if external_history.empty:
        raise RuntimeError(
            "External Elo history is empty."
        )

    external = external_history.copy()

    external["team"] = (
        external["team"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace(CURRENT_TEAM_CODE_MAP)
    )

    latest_season = int(
        external["season"].max()
    )

    latest_week = int(
        external.loc[
            external["season"] == latest_season,
            "week",
        ].max()
    )

    latest = external.loc[
        (
            external["season"] == latest_season
        )
        & (
            external["week"] == latest_week
        )
    ].copy()

    if latest.empty:
        raise RuntimeError(
            "No latest external Elo ratings exist."
        )

    if latest["team"].duplicated().any():
        duplicate_teams = ", ".join(
            sorted(
                latest.loc[
                    latest["team"].duplicated(
                        keep=False
                    ),
                    "team",
                ].unique()
            )
        )

        raise ValueError(
            "Latest external Elo ratings contain "
            f"duplicate teams: {duplicate_teams}"
        )

    if latest["elo"].isna().any():
        raise ValueError(
            "Latest external Elo ratings contain "
            "missing rating values."
        )

    latest["external_source_name"] = (
        EXTERNAL_SOURCE_NAME
    )

    return latest.loc[
        :,
        [
            "season",
            "week",
            "team",
            "game_id",
            "elo",
            "qb_adj",
            "external_source_name",
        ],
    ].sort_values(
        by="team",
        kind="stable",
    ).reset_index(drop=True)


def validate_internal_elo(
    internal_ratings: pd.DataFrame,
) -> None:
    """Validate the internal current Elo state."""

    validate_required_columns(
        data=internal_ratings,
        required_columns=REQUIRED_INTERNAL_COLUMNS,
        data_name="Internal Elo ratings",
    )

    if internal_ratings.empty:
        raise RuntimeError(
            "Internal Elo ratings are empty."
        )

    if internal_ratings["team"].duplicated().any():
        raise ValueError(
            "Internal Elo ratings contain "
            "duplicate teams."
        )

    if internal_ratings["elo_rating"].isna().any():
        raise ValueError(
            "Internal Elo ratings contain "
            "missing rating values."
        )


def compare_elo_ratings(
    internal_ratings: pd.DataFrame,
    external_history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare current internal and external Elo states."""

    validate_internal_elo(
        internal_ratings
    )

    external = prepare_latest_external_elo(
        external_history
    )

    internal = internal_ratings.rename(
        columns={
            "elo_rating": (
                "internal_elo_rating"
            ),
            "last_completed_season": (
                "internal_last_completed_season"
            ),
            "last_game_id": (
                "internal_last_game_id"
            ),
            "as_of_gameday": (
                "internal_as_of_gameday"
            ),
        }
    )

    external = external.rename(
        columns={
            "season": "external_season",
            "week": "external_week",
            "game_id": "external_game_id",
            "elo": "external_elo_rating",
            "qb_adj": (
                "external_qb_adjustment"
            ),
        }
    )

    comparison = internal.merge(
        external,
        on="team",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = comparison.loc[
        comparison["_merge"] != "both"
    ]

    if not unmatched.empty:
        unmatched_rows = ", ".join(
            unmatched.apply(
                lambda row: (
                    f"{row['team']}:{row['_merge']}"
                ),
                axis=1,
            )
        )

        raise ValueError(
            "Internal and external Elo teams do "
            f"not match: {unmatched_rows}"
        )

    comparison = comparison.drop(
        columns=["_merge"]
    )

    comparison["internal_rank"] = (
        comparison["internal_elo_rating"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    comparison["external_rank"] = (
        comparison["external_elo_rating"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    comparison[
        "rank_change_external_vs_internal"
    ] = (
        comparison["internal_rank"]
        - comparison["external_rank"]
    )

    comparison[
        "rating_delta_external_minus_internal"
    ] = (
        comparison["external_elo_rating"]
        - comparison["internal_elo_rating"]
    )

    comparison["absolute_rating_delta"] = (
        comparison[
            "rating_delta_external_minus_internal"
        ].abs()
    )

    comparison = comparison.loc[
        :,
        COMPARISON_COLUMNS,
    ].sort_values(
        by=[
            "absolute_rating_delta",
            "team",
        ],
        ascending=[
            False,
            True,
        ],
        kind="stable",
    ).reset_index(drop=True)

    maximum_row = comparison.iloc[0]

    summary = pd.DataFrame(
        [
            {
                "team_count": len(
                    comparison
                ),
                "internal_mean_rating": float(
                    comparison[
                        "internal_elo_rating"
                    ].mean()
                ),
                "external_mean_rating": float(
                    comparison[
                        "external_elo_rating"
                    ].mean()
                ),
                "mean_absolute_rating_delta": (
                    float(
                        comparison[
                            "absolute_rating_delta"
                        ].mean()
                    )
                ),
                "maximum_absolute_rating_delta": (
                    float(
                        maximum_row[
                            "absolute_rating_delta"
                        ]
                    )
                ),
                "team_with_maximum_rating_delta": (
                    maximum_row["team"]
                ),
                "pearson_rating_correlation": (
                    float(
                        comparison[
                            "internal_elo_rating"
                        ].corr(
                            comparison[
                                "external_elo_rating"
                            ],
                            method="pearson",
                        )
                    )
                ),
                "spearman_rank_correlation": (
                    float(
                        comparison[
                            "internal_elo_rating"
                        ].corr(
                            comparison[
                                "external_elo_rating"
                            ],
                            method="spearman",
                        )
                    )
                ),
                "external_season": int(
                    comparison[
                        "external_season"
                    ].iloc[0]
                ),
                "external_week": int(
                    comparison[
                        "external_week"
                    ].iloc[0]
                ),
            }
        ],
        columns=SUMMARY_COLUMNS,
    )

    return comparison, summary


def load_internal_elo(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Load the current internal Elo state."""

    return connection.execute(
        f"""
        SELECT
            team,
            elo_rating,
            last_completed_season,
            last_game_id,
            as_of_gameday
        FROM {INTERNAL_ELO_FULL_NAME}
        ORDER BY team
        """
    ).fetchdf()


def run_external_elo_comparison(
    database_file: Path = DATABASE_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download external ratings and compare them."""

    validate_database_file(database_file)

    external_history = pd.read_csv(
        EXTERNAL_ELO_URL
    )

    with duckdb.connect(
        str(database_file),
        read_only=True,
    ) as connection:
        internal_ratings = load_internal_elo(
            connection
        )

    comparison, summary = compare_elo_ratings(
        internal_ratings=internal_ratings,
        external_history=external_history,
    )

    logger.info(
        "External Elo comparison completed: "
        "%s matched teams from external season %s "
        "week %s.",
        len(comparison),
        int(summary.iloc[0]["external_season"]),
        int(summary.iloc[0]["external_week"]),
    )

    return comparison, summary


def main() -> None:
    """Run and print the external Elo comparison."""

    comparison, summary = (
        run_external_elo_comparison()
    )

    print("\nEXTERNAL ELO COMPARISON SUMMARY\n")

    print(
        summary.to_string(
            index=False
        )
    )

    print("\nLARGEST INTERNAL-EXTERNAL DIFFERENCES\n")

    print(
        comparison.head(15).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()