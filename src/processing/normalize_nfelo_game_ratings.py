"""
NFL Analytics Platform
nfelo Game Rating Normalization

Purpose:
    Validate nfelo game-level model outputs and normalize
    source game identifiers to nflverse-compatible team
    codes without losing the original source identifier.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import numpy as np
import pandas as pd


SOURCE_NAME = "nfelo_games"

REQUIRED_SOURCE_COLUMNS = {
    "game_id",
    "starting_nfelo_home",
    "starting_nfelo_away",
    "nfelo_dif_base",
    "nfelo_home_probability_open",
    "nfelo_home_probability_close",
}

NORMALIZED_IDENTIFIER_COLUMNS = (
    "source_name",
    "source_game_id",
    "normalized_game_id",
    "source_season",
    "source_week",
    "away_team",
    "home_team",
)

ALWAYS_NORMALIZED_TEAM_CODES = {
    "LAR": "LA",
    "WSH": "WAS",
    "JAC": "JAX",
}


def normalize_team_code(
    team: str,
    season: int,
) -> str:
    """Normalize one nfelo team code by season."""

    normalized_team = (
        str(team).strip().upper()
    )

    normalized_team = (
        ALWAYS_NORMALIZED_TEAM_CODES.get(
            normalized_team,
            normalized_team,
        )
    )

    if (
        normalized_team == "OAK"
        and season >= 2020
    ):
        return "LV"

    return normalized_team


def parse_source_game_ids(
    source_game_ids: pd.Series,
) -> pd.DataFrame:
    """Parse and validate nfelo game identifiers."""

    if source_game_ids.isna().any():
        raise ValueError(
            "nfelo source contains missing game IDs."
        )

    game_ids = (
        source_game_ids.astype(str).str.strip()
    )

    valid_format_mask = game_ids.str.fullmatch(
        r"\d{4}_\d{2}_[A-Za-z0-9]+_[A-Za-z0-9]+"
    )

    if not valid_format_mask.all():
        raise ValueError(
            "nfelo game IDs must use "
            "season_week_away_home format."
        )

    parts = game_ids.str.split(
        "_",
        expand=True,
    )

    if parts.shape[1] != 4:
        raise ValueError(
            "nfelo game IDs must use "
            "season_week_away_home format."
        )

    parts.columns = [
        "season_text",
        "week_text",
        "away_team_source",
        "home_team_source",
    ]

    season_values = pd.to_numeric(
        parts["season_text"],
        errors="coerce",
    )

    week_values = pd.to_numeric(
        parts["week_text"],
        errors="coerce",
    )

    invalid_numeric_mask = (
        season_values.isna()
        | week_values.isna()
    )

    if invalid_numeric_mask.any():
        raise ValueError(
            "nfelo game IDs contain invalid "
            "season or week values."
        )

    parsed = pd.DataFrame(
        {
            "source_season": (
                season_values.astype(int)
            ),
            "source_week": (
                week_values.astype(int)
            ),
            "away_team_source": (
                parts[
                    "away_team_source"
                ].astype(str)
            ),
            "home_team_source": (
                parts[
                    "home_team_source"
                ].astype(str)
            ),
        },
        index=source_game_ids.index,
    )

    invalid_range_mask = (
        parsed["source_season"].lt(1900)
        | parsed["source_week"].lt(1)
        | parsed["source_week"].gt(30)
    )

    if invalid_range_mask.any():
        raise ValueError(
            "nfelo game IDs contain out-of-range "
            "season or week values."
        )

    empty_team_mask = (
        parsed["away_team_source"]
        .str.strip()
        .eq("")
        | parsed["home_team_source"]
        .str.strip()
        .eq("")
    )

    if empty_team_mask.any():
        raise ValueError(
            "nfelo game IDs contain empty team codes."
        )

    return parsed


def normalize_nfelo_game_ratings(
    source_data: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize nfelo game IDs and preserve source data."""

    missing_columns = sorted(
        REQUIRED_SOURCE_COLUMNS
        - set(source_data.columns)
    )

    if missing_columns:
        raise ValueError(
            "nfelo game data is missing columns: "
            + ", ".join(missing_columns)
        )

    if source_data.empty:
        raise RuntimeError(
            "nfelo game data is empty."
        )

    if source_data["game_id"].duplicated().any():
        raise ValueError(
            "nfelo game data contains duplicate "
            "source game IDs."
        )

    parsed_ids = parse_source_game_ids(
        source_data["game_id"]
    )

    normalized = source_data.copy()

    normalized = normalized.rename(
        columns={
            "game_id": "source_game_id",
        }
    )

    normalized["source_name"] = SOURCE_NAME

    normalized["source_season"] = (
        parsed_ids["source_season"]
    )

    normalized["source_week"] = (
        parsed_ids["source_week"]
    )

    normalized["away_team"] = [
        normalize_team_code(
            team=team,
            season=int(season),
        )
        for team, season in zip(
            parsed_ids["away_team_source"],
            parsed_ids["source_season"],
            strict=True,
        )
    ]

    normalized["home_team"] = [
        normalize_team_code(
            team=team,
            season=int(season),
        )
        for team, season in zip(
            parsed_ids["home_team_source"],
            parsed_ids["source_season"],
            strict=True,
        )
    ]

    normalized["normalized_game_id"] = (
        normalized[
            "source_season"
        ].astype(str)
        + "_"
        + normalized[
            "source_week"
        ].astype(str).str.zfill(2)
        + "_"
        + normalized["away_team"]
        + "_"
        + normalized["home_team"]
    )

    if normalized[
        "normalized_game_id"
    ].duplicated().any():
        duplicate_game_ids = ", ".join(
            normalized.loc[
                normalized[
                    "normalized_game_id"
                ].duplicated(
                    keep=False
                ),
                "normalized_game_id",
            ].unique()
        )

        raise ValueError(
            "nfelo normalization creates duplicate "
            f"game IDs: {duplicate_game_ids}"
        )

    rating_columns = [
        "starting_nfelo_home",
        "starting_nfelo_away",
        "nfelo_dif_base",
    ]

    rating_values = normalized[
        rating_columns
    ].to_numpy(dtype=float)

    if not np.isfinite(
        rating_values
    ).all():
        raise ValueError(
            "nfelo game data contains non-finite "
            "rating values."
        )

    probability_columns = [
        "nfelo_home_probability_open",
        "nfelo_home_probability_close",
    ]

    probability_values = normalized[
        probability_columns
    ].to_numpy(dtype=float)

    if (
        not np.isfinite(
            probability_values
        ).all()
        or (
            probability_values <= 0.0
        ).any()
        or (
            probability_values >= 1.0
        ).any()
    ):
        raise ValueError(
            "nfelo game data contains invalid "
            "home probabilities."
        )

    remaining_columns = [
        column_name
        for column_name in normalized.columns
        if column_name
        not in NORMALIZED_IDENTIFIER_COLUMNS
    ]

    return normalized.loc[
        :,
        [
            *NORMALIZED_IDENTIFIER_COLUMNS,
            *remaining_columns,
        ],
    ].sort_values(
        by=[
            "source_season",
            "source_week",
            "normalized_game_id",
        ],
        kind="stable",
    ).reset_index(drop=True)