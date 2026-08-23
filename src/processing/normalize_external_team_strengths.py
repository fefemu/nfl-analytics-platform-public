"""Normalize external unit and win-total team ratings."""

import pandas as pd


TEAM_ALIASES = {
    "JAC": "JAX",
    "LAR": "LA",
    "WSH": "WAS",
}

UNIT_PRE_COLUMNS = (
    "pass_off_value_pre",
    "rush_off_value_pre",
    "st_off_value_pre",
    "pass_def_value_pre",
    "rush_def_value_pre",
    "st_def_value_pre",
    "off_value_pre",
    "def_value_pre",
    "total_value_pre",
)

WIN_TOTAL_COLUMNS = (
    "wt_rating",
    "wt_rating_elo",
    "sos",
    "line",
    "over_probability",
    "under_probability",
    "line_adj",
)


def normalize_team(team: object, season: int | None = None) -> str:
    """Normalize one external team abbreviation."""

    value = str(team).strip().upper()

    if not value:
        raise ValueError("Team abbreviation must not be empty.")

    if value == "OAK" and season is not None and season >= 2020:
        return "LV"

    return TEAM_ALIASES.get(value, value)


def normalize_nfelounits_units(
    source_data: pd.DataFrame,
) -> pd.DataFrame:
    """Keep leakage-safe pregame unit rating columns."""

    required_columns = {
        "season",
        "week",
        "team",
        *UNIT_PRE_COLUMNS,
    }
    missing_columns = sorted(
        required_columns - set(source_data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Unit rating data is missing columns: "
            + ", ".join(missing_columns)
        )

    normalized = source_data.loc[
        :,
        ["season", "week", "team", *UNIT_PRE_COLUMNS],
    ].copy()
    normalized["season"] = pd.to_numeric(
        normalized["season"], errors="raise"
    ).astype(int)
    normalized["week"] = pd.to_numeric(
        normalized["week"], errors="raise"
    ).astype(int)
    normalized["team"] = [
        normalize_team(team, int(season))
        for team, season in zip(
            normalized["team"],
            normalized["season"],
            strict=True,
        )
    ]

    for column in UNIT_PRE_COLUMNS:
        normalized[column] = pd.to_numeric(
            normalized[column], errors="raise"
        ).astype(float)

    all_unit_values_missing = normalized[
        list(UNIT_PRE_COLUMNS)
    ].isna().all(axis=1)
    normalized = normalized.loc[
        ~all_unit_values_missing
    ].copy()

    if normalized[["season", "week", "team"]].duplicated().any():
        raise ValueError(
            "Normalized unit ratings contain duplicate "
            "season-week-team rows."
        )

    if normalized[list(UNIT_PRE_COLUMNS)].isna().any().any():
        raise ValueError(
            "Normalized unit ratings contain partially null rows."
        )

    return normalized.sort_values(
        ["season", "week", "team"], kind="stable"
    ).reset_index(drop=True)


def normalize_win_total_ratings(
    source_data: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize one preseason win-total rating per team-season."""

    required_columns = {"season", "team", *WIN_TOTAL_COLUMNS}
    missing_columns = sorted(
        required_columns - set(source_data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Win-total rating data is missing columns: "
            + ", ".join(missing_columns)
        )

    normalized = source_data.loc[
        :,
        ["season", "team", *WIN_TOTAL_COLUMNS],
    ].copy()
    normalized["season"] = pd.to_numeric(
        normalized["season"], errors="raise"
    ).astype(int)
    normalized["team"] = [
        normalize_team(team, int(season))
        for team, season in zip(
            normalized["team"],
            normalized["season"],
            strict=True,
        )
    ]

    for column in WIN_TOTAL_COLUMNS:
        normalized[column] = pd.to_numeric(
            normalized[column], errors="raise"
        ).astype(float)

    if normalized[["season", "team"]].duplicated().any():
        raise ValueError(
            "Normalized win-total ratings contain duplicate "
            "season-team rows."
        )

    if normalized[list(WIN_TOTAL_COLUMNS)].isna().any().any():
        raise ValueError(
            "Normalized win-total ratings contain null values."
        )

    return normalized.sort_values(
        ["season", "team"], kind="stable"
    ).reset_index(drop=True)
