"""
NFL Analytics Platform
Game Modeling Dataset Builder

Purpose:
    Build a leakage-safe game-level modeling dataset
    from schedule, Elo, rolling team and quarterback features.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

import logging
from pathlib import Path

import duckdb


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_FILE = PROJECT_ROOT / "data" / "nfl_analytics.duckdb"


# ---------------------------------------------------------
# Source tables
# ---------------------------------------------------------

SCHEDULE_SCHEMA = "processed"
SCHEDULE_TABLE = "schedule"
SCHEDULE_FULL_NAME = (
    f"{SCHEDULE_SCHEMA}.{SCHEDULE_TABLE}"
)

ELO_SCHEMA = "analytics"
ELO_TABLE = "elo_game_predictions"
ELO_FULL_NAME = f"{ELO_SCHEMA}.{ELO_TABLE}"

ROLLING_SCHEMA = "analytics"
ROLLING_TABLE = "rolling_team_features"
ROLLING_FULL_NAME = (
    f"{ROLLING_SCHEMA}.{ROLLING_TABLE}"
)

QB_SCHEMA = "analytics"
QB_TABLE = "game_qb_features"
QB_FULL_NAME = f"{QB_SCHEMA}.{QB_TABLE}"
SCHEDULE_FEATURE_SCHEMA = "analytics"
SCHEDULE_FEATURE_TABLE = "game_schedule_features"
SCHEDULE_FEATURE_FULL_NAME = (
    f"{SCHEDULE_FEATURE_SCHEMA}."
    f"{SCHEDULE_FEATURE_TABLE}"
)

INJURY_SCHEMA = "analytics"
INJURY_TABLE = "game_injury_features"
INJURY_FULL_NAME = (
    f"{INJURY_SCHEMA}.{INJURY_TABLE}"
)

WEATHER_SCHEMA = "analytics"
WEATHER_TABLE = "game_weather_features"
WEATHER_FULL_NAME = (
    f"{WEATHER_SCHEMA}.{WEATHER_TABLE}"
)

SCORING_ENVIRONMENT_SCHEMA = "analytics"
SCORING_ENVIRONMENT_TABLE = (
    "game_scoring_environment_features"
)
SCORING_ENVIRONMENT_FULL_NAME = (
    f"{SCORING_ENVIRONMENT_SCHEMA}."
    f"{SCORING_ENVIRONMENT_TABLE}"
)

# ---------------------------------------------------------
# Target table
# ---------------------------------------------------------

TARGET_SCHEMA = "analytics"
TARGET_TABLE = "game_modeling_dataset"
TARGET_FULL_NAME = f"{TARGET_SCHEMA}.{TARGET_TABLE}"


# ---------------------------------------------------------
# Rolling feature configuration
# ---------------------------------------------------------

SHORT_WINDOW = 4
LONG_WINDOW = 8

ROLLING_METRICS = (
    "offensive_plays",
    "points_scored",
    "points_allowed",
    "offensive_epa_per_play",
    "competitive_epa_per_play",
    "dropback_epa_per_play",
    "designed_rush_epa_per_play",
    "early_down_epa_per_play",
    "success_rate",
    "explosive_play_rate",
    "sack_rate",
    "turnover_rate",
    "defensive_epa_allowed_per_play",
    "competitive_defensive_epa_allowed_per_play",
    "defensive_success_rate_allowed",
    "explosive_play_rate_allowed",
    "sack_rate_generated",
    "turnover_rate_generated",
)


# ---------------------------------------------------------
# Required source columns
# ---------------------------------------------------------

REQUIRED_SCHEDULE_COLUMNS = {
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "is_completed",
    "home_win",
    "point_differential",
    "total_points",
}

REQUIRED_ELO_COLUMNS = {
    "game_id",
    "gameday",
    "home_team",
    "away_team",
    "home_advantage",
    "home_rating_pre",
    "away_rating_pre",
    "home_win_probability",
}

REQUIRED_ROLLING_COLUMNS = {
    "game_id",
    "season",
    "season_type",
    "week",
    "game_date",
    "team",
    "opponent",
    "is_home",
    "season_games_played_before",
    "short_window_games",
    "long_window_games",
} | {
    f"pregame_{metric}_last_{window}"
    for metric in ROLLING_METRICS
    for window in (SHORT_WINDOW, LONG_WINDOW)
}

REQUIRED_QB_COLUMNS = {
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "home_listed_qb_id",
    "home_listed_qb_name",
    "home_listed_qb_rating",
    "home_listed_qb_effective_dropbacks",
    "home_listed_qb_prior_weight",
    "home_listed_qb_rating_standard_error",
    "home_listed_qb_rating_available",
    "away_listed_qb_id",
    "away_listed_qb_name",
    "away_listed_qb_rating",
    "away_listed_qb_effective_dropbacks",
    "away_listed_qb_prior_weight",
    "away_listed_qb_rating_standard_error",
    "away_listed_qb_rating_available",
    "both_listed_qb_ratings_available",
    "listed_qb_rating_difference",
    "listed_qb_rating_difference_standard_error",
}

REQUIRED_SCHEDULE_FEATURE_COLUMNS = {
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "home_rest_days",
    "away_rest_days",
    "rest_days_difference",
    "home_short_week",
    "away_short_week",
    "short_week_difference",
    "home_extended_rest",
    "away_extended_rest",
    "extended_rest_difference",
    "home_post_bye",
    "away_post_bye",
    "post_bye_difference",
}

REQUIRED_INJURY_COLUMNS = {
    "game_id",
    "home_team",
    "away_team",
    "home_has_injury_report_data",
    "away_has_injury_report_data",
    "has_complete_injury_data",
    "home_out_player_count",
    "away_out_player_count",
    "out_player_count_difference",
    "home_doubtful_player_count",
    "away_doubtful_player_count",
    "doubtful_player_count_difference",
    "home_questionable_player_count",
    "away_questionable_player_count",
    "questionable_player_count_difference",
    "home_starter_out_count",
    "away_starter_out_count",
    "starter_out_count_difference",
    "home_qb_out_count",
    "away_qb_out_count",
    "qb_out_count_difference",
    "home_total_injury_burden",
    "away_total_injury_burden",
    "total_injury_burden_difference",
    "home_qb_injury_burden",
    "away_qb_injury_burden",
    "qb_injury_burden_difference",
    "home_non_qb_injury_burden",
    "away_non_qb_injury_burden",
    "non_qb_injury_burden_difference",
    "home_offense_injury_burden",
    "away_offense_injury_burden",
    "offense_injury_burden_difference",
    "home_defense_injury_burden",
    "away_defense_injury_burden",
    "defense_injury_burden_difference",
    "home_special_teams_injury_burden",
    "away_special_teams_injury_burden",
    "special_teams_injury_burden_difference",
}


REQUIRED_WEATHER_COLUMNS = {
    "game_id",
    "season",
    "game_date",
    "home_team",
    "away_team",
    "roof_type",
    "surface_type",
    "stadium_id",
    "stadium",
    "is_indoor",
    "is_weather_exposed",
    "has_game_weather",
    "raw_temperature_f",
    "raw_wind_mph",
    "modeled_temperature_f",
    "modeled_wind_mph",
    "is_freezing",
    "is_high_wind",
    "is_extreme_heat",
    "cold_degrees_below_50",
    "heat_degrees_above_80",
    "wind_mph_above_10",
}


REQUIRED_SCORING_ENVIRONMENT_COLUMNS = {
    "game_id",
    "season",
    "game_date",
    "home_team",
    "away_team",
    "league_game_count_last_32",
    "league_average_total_last_32",
    "league_total_standard_deviation_last_32",
    "league_game_count_last_64",
    "league_average_total_last_64",
    "league_total_standard_deviation_last_64",
    "league_game_count_last_128",
    "league_average_total_last_128",
    "league_total_standard_deviation_last_128",
}


def validate_database_file(
    database_file: Path = DATABASE_FILE,
) -> None:
    """Validate that the DuckDB database file exists."""

    if not database_file.exists():
        raise FileNotFoundError(
            f"Database file does not exist: {database_file}"
        )

    if not database_file.is_file():
        raise RuntimeError(
            f"Database path is not a file: {database_file}"
        )

    logger.info(
        "Database file validated: %s",
        database_file,
    )


def validate_source_tables(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate all modeling dataset source tables."""

    required_tables = {
        (
            SCHEDULE_SCHEMA,
            SCHEDULE_TABLE,
        ): REQUIRED_SCHEDULE_COLUMNS,
        (
            ELO_SCHEMA,
            ELO_TABLE,
        ): REQUIRED_ELO_COLUMNS,
        (
            ROLLING_SCHEMA,
            ROLLING_TABLE,
        ): REQUIRED_ROLLING_COLUMNS,
        (
            QB_SCHEMA,
            QB_TABLE,
        ): REQUIRED_QB_COLUMNS,
        (
            SCHEDULE_FEATURE_SCHEMA,
            SCHEDULE_FEATURE_TABLE,
        ): REQUIRED_SCHEDULE_FEATURE_COLUMNS,
        (
            INJURY_SCHEMA,
            INJURY_TABLE,
        ): REQUIRED_INJURY_COLUMNS,
        (
            WEATHER_SCHEMA,
            WEATHER_TABLE,
        ): REQUIRED_WEATHER_COLUMNS,
        (
            SCORING_ENVIRONMENT_SCHEMA,
            SCORING_ENVIRONMENT_TABLE,
        ): REQUIRED_SCORING_ENVIRONMENT_COLUMNS,
    }

    for (
        schema_name,
        table_name,
    ), required_columns in required_tables.items():
        table_exists = connection.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = ?
              AND table_name = ?
            """,
            [schema_name, table_name],
        ).fetchone()[0]

        full_name = f"{schema_name}.{table_name}"

        if table_exists == 0:
            raise RuntimeError(
                f"Source table does not exist: {full_name}"
            )

        available_columns = {
            row[0]
            for row in connection.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = ?
                  AND table_name = ?
                """,
                [schema_name, table_name],
            ).fetchall()
        }

        missing_columns = sorted(
            required_columns - available_columns
        )

        if missing_columns:
            missing_names = ", ".join(missing_columns)

            raise RuntimeError(
                f"Missing columns in {full_name}: "
                f"{missing_names}"
            )

    logger.info(
        "Modeling dataset sources validated: "
        "%s, %s, %s, %s, %s, %s, %s and %s.",
        SCHEDULE_FULL_NAME,
        ELO_FULL_NAME,
        ROLLING_FULL_NAME,
        QB_FULL_NAME,
        SCHEDULE_FEATURE_FULL_NAME,
        INJURY_FULL_NAME,
        WEATHER_FULL_NAME,
        SCORING_ENVIRONMENT_FULL_NAME,
    )


def build_rolling_feature_expressions(
    table_alias: str,
    output_prefix: str,
) -> str:
    """Build SQL expressions for one team's rolling features."""

    expressions: list[str] = []

    for metric in ROLLING_METRICS:
        for window in (SHORT_WINDOW, LONG_WINDOW):
            source_column = (
                f"pregame_{metric}_last_{window}"
            )

            output_column = (
                f"{output_prefix}_{metric}_last_{window}"
            )

            expressions.append(
                f"{table_alias}.{source_column} "
                f"AS {output_column}"
            )

    return ",\n            ".join(expressions)


def build_rolling_difference_expressions() -> str:
    """Build home-minus-away rolling feature differences."""

    expressions: list[str] = []

    for metric in ROLLING_METRICS:
        for window in (SHORT_WINDOW, LONG_WINDOW):
            home_column = (
                f"home_features."
                f"pregame_{metric}_last_{window}"
            )

            away_column = (
                f"away_features."
                f"pregame_{metric}_last_{window}"
            )

            output_column = (
                f"{metric}_difference_last_{window}"
            )

            expressions.append(
                f"{home_column} - {away_column} "
                f"AS {output_column}"
            )

    return ",\n            ".join(expressions)


def create_game_modeling_dataset(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create the leakage-safe game-level modeling dataset."""

    home_rolling_expressions = (
        build_rolling_feature_expressions(
            table_alias="home_features",
            output_prefix="home",
        )
    )

    away_rolling_expressions = (
        build_rolling_feature_expressions(
            table_alias="away_features",
            output_prefix="away",
        )
    )

    rolling_difference_expressions = (
        build_rolling_difference_expressions()
    )

    connection.execute(
        f"CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA}"
    )

    connection.execute(
        f"""
        CREATE OR REPLACE TABLE {TARGET_FULL_NAME} AS

        SELECT
            schedule.game_id,
            schedule.season,
            schedule.game_type,
            schedule.week,
            CAST(schedule.gameday AS DATE) AS game_date,

            schedule.home_team,
            schedule.away_team,

            -- ---------------------------------------------
            -- Leakage-safe schedule context
            -- ---------------------------------------------

            schedule_features.home_rest_days,
            schedule_features.away_rest_days,
            schedule_features.rest_days_difference,

            schedule_features.home_short_week,
            schedule_features.away_short_week,
            schedule_features.short_week_difference,

            schedule_features.home_extended_rest,
            schedule_features.away_extended_rest,
            schedule_features.extended_rest_difference,

            schedule_features.home_post_bye,
            schedule_features.away_post_bye,
            schedule_features.post_bye_difference,

            -- ---------------------------------------------
            -- Pregame Elo features
            -- ---------------------------------------------

            elo.home_advantage
                AS elo_home_advantage,

            elo.home_rating_pre
                AS home_elo_rating,

            elo.away_rating_pre
                AS away_elo_rating,

            elo.home_rating_pre
                - elo.away_rating_pre
                AS elo_rating_difference,

            elo.home_win_probability
                AS elo_home_win_probability,

            -- ---------------------------------------------
            -- Rolling history availability
            -- ---------------------------------------------

            home_features.season_games_played_before
                AS home_season_games_played_before,

            away_features.season_games_played_before
                AS away_season_games_played_before,

            home_features.short_window_games
                AS home_short_window_games,

            away_features.short_window_games
                AS away_short_window_games,

            home_features.long_window_games
                AS home_long_window_games,

            away_features.long_window_games
                AS away_long_window_games,

            home_features.short_window_games
                = {SHORT_WINDOW}
                AS home_short_window_complete,

            away_features.short_window_games
                = {SHORT_WINDOW}
                AS away_short_window_complete,

            home_features.long_window_games
                = {LONG_WINDOW}
                AS home_long_window_complete,

            away_features.long_window_games
                = {LONG_WINDOW}
                AS away_long_window_complete,

            (
                home_features.short_window_games
                    = {SHORT_WINDOW}
                AND away_features.short_window_games
                    = {SHORT_WINDOW}
            ) AS both_short_windows_complete,

            (
                home_features.long_window_games
                    = {LONG_WINDOW}
                AND away_features.long_window_games
                    = {LONG_WINDOW}
            ) AS both_long_windows_complete,

            -- ---------------------------------------------
            -- Home rolling features
            -- ---------------------------------------------

            {home_rolling_expressions},

            -- ---------------------------------------------
            -- Away rolling features
            -- ---------------------------------------------

            {away_rolling_expressions},

            -- ---------------------------------------------
            -- Home-minus-away rolling differences
            -- ---------------------------------------------

            {rolling_difference_expressions},

            -- ---------------------------------------------
            -- Leakage-safe listed-QB features
            -- ---------------------------------------------

            qb.home_listed_qb_id,
            qb.home_listed_qb_name,
            qb.home_listed_qb_rating,
            qb.home_listed_qb_effective_dropbacks,
            qb.home_listed_qb_prior_weight,
            qb.home_listed_qb_rating_standard_error,
            qb.home_listed_qb_rating_available,

            qb.away_listed_qb_id,
            qb.away_listed_qb_name,
            qb.away_listed_qb_rating,
            qb.away_listed_qb_effective_dropbacks,
            qb.away_listed_qb_prior_weight,
            qb.away_listed_qb_rating_standard_error,
            qb.away_listed_qb_rating_available,

            qb.both_listed_qb_ratings_available,
            qb.listed_qb_rating_difference,
            qb.listed_qb_rating_difference_standard_error,
            -- ---------------------------------------------
            -- Pregame injury availability and burden
            -- ---------------------------------------------

            injury.home_has_injury_report_data,
            injury.away_has_injury_report_data,
            injury.has_complete_injury_data,

            injury.home_out_player_count,
            injury.away_out_player_count,
            injury.out_player_count_difference,

            injury.home_doubtful_player_count,
            injury.away_doubtful_player_count,
            injury.doubtful_player_count_difference,

            injury.home_questionable_player_count,
            injury.away_questionable_player_count,
            injury.questionable_player_count_difference,

            injury.home_starter_out_count,
            injury.away_starter_out_count,
            injury.starter_out_count_difference,

            injury.home_qb_out_count,
            injury.away_qb_out_count,
            injury.qb_out_count_difference,

            injury.home_total_injury_burden,
            injury.away_total_injury_burden,
            injury.total_injury_burden_difference,

            injury.home_qb_injury_burden,
            injury.away_qb_injury_burden,
            injury.qb_injury_burden_difference,

            injury.home_non_qb_injury_burden,
            injury.away_non_qb_injury_burden,
            injury.non_qb_injury_burden_difference,

            injury.home_offense_injury_burden,
            injury.away_offense_injury_burden,
            injury.offense_injury_burden_difference,

            injury.home_defense_injury_burden,
            injury.away_defense_injury_burden,
            injury.defense_injury_burden_difference,

            injury.home_special_teams_injury_burden,
            injury.away_special_teams_injury_burden,
            injury.special_teams_injury_burden_difference,

            -- ---------------------------------------------
            -- Venue and game-time weather context
            -- ---------------------------------------------

            weather.roof_type,
            weather.surface_type,
            weather.stadium_id,
            weather.stadium,
            weather.is_indoor,
            weather.is_weather_exposed,
            weather.has_game_weather,
            weather.raw_temperature_f,
            weather.raw_wind_mph,
            weather.modeled_temperature_f,
            weather.modeled_wind_mph,
            weather.is_freezing,
            weather.is_high_wind,
            weather.is_extreme_heat,
            weather.cold_degrees_below_50,
            weather.heat_degrees_above_80,
            weather.wind_mph_above_10,

            -- ---------------------------------------------
            -- Pregame league scoring environment
            -- ---------------------------------------------

            scoring_environment.league_game_count_last_32,
            scoring_environment.league_average_total_last_32,
            scoring_environment
                .league_total_standard_deviation_last_32,

            scoring_environment.league_game_count_last_64,
            scoring_environment.league_average_total_last_64,
            scoring_environment
                .league_total_standard_deviation_last_64,

            scoring_environment.league_game_count_last_128,
            scoring_environment.league_average_total_last_128,
            scoring_environment
                .league_total_standard_deviation_last_128,

            -- ---------------------------------------------
            -- Targets known only after the game
            -- ---------------------------------------------

            schedule.home_score
                AS target_home_score,

            schedule.away_score
                AS target_away_score,

            CASE
                WHEN schedule.home_score
                    = schedule.away_score
                    THEN NULL
                ELSE schedule.home_win
            END AS target_home_win,

            CASE
                WHEN schedule.home_score
                    > schedule.away_score
                    THEN 1.0
                WHEN schedule.home_score
                    < schedule.away_score
                    THEN 0.0
                ELSE 0.5
            END AS target_home_result,

            schedule.point_differential
                AS target_point_differential,

            schedule.total_points
                AS target_total_points

        FROM {SCHEDULE_FULL_NAME} AS schedule

        INNER JOIN {SCHEDULE_FEATURE_FULL_NAME}
            AS schedule_features
            ON schedule.game_id
                = schedule_features.game_id

        INNER JOIN {ELO_FULL_NAME} AS elo
            ON schedule.game_id = elo.game_id

        INNER JOIN {ROLLING_FULL_NAME}
            AS home_features
            ON schedule.game_id = home_features.game_id
           AND home_features.is_home = TRUE

        INNER JOIN {ROLLING_FULL_NAME}
            AS away_features
            ON schedule.game_id = away_features.game_id
           AND away_features.is_home = FALSE

        INNER JOIN {QB_FULL_NAME} AS qb
            ON schedule.game_id = qb.game_id

        INNER JOIN {INJURY_FULL_NAME} AS injury
            ON schedule.game_id = injury.game_id
           AND schedule.home_team = injury.home_team
           AND schedule.away_team = injury.away_team

        INNER JOIN {WEATHER_FULL_NAME}
            AS weather
            ON schedule.game_id = weather.game_id
           AND schedule.home_team = weather.home_team
           AND schedule.away_team = weather.away_team

        INNER JOIN {SCORING_ENVIRONMENT_FULL_NAME}
            AS scoring_environment
            ON schedule.game_id
                = scoring_environment.game_id
           AND schedule.home_team
                = scoring_environment.home_team
           AND schedule.away_team
                = scoring_environment.away_team

        WHERE schedule.is_completed = TRUE

        ORDER BY
            schedule.gameday,
            schedule.game_id
        """
    )

    row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    logger.info(
        "Game modeling dataset created: %s rows in %s.",
        row_count,
        TARGET_FULL_NAME,
    )


def build_rolling_difference_validation_predicate() -> str:
    """Build validation rules for rolling feature differences."""

    predicates: list[str] = []

    for metric in ROLLING_METRICS:
        for window in (SHORT_WINDOW, LONG_WINDOW):
            home_column = (
                f"home_{metric}_last_{window}"
            )

            away_column = (
                f"away_{metric}_last_{window}"
            )

            difference_column = (
                f"{metric}_difference_last_{window}"
            )

            predicates.append(
                f"""
                (
                    {home_column} IS NOT NULL
                    AND {away_column} IS NOT NULL
                    AND (
                        {difference_column} IS NULL
                        OR ABS(
                            {difference_column}
                            - (
                                {home_column}
                                - {away_column}
                            )
                        ) > 0.000000001
                    )
                )
                OR
                (
                    (
                        {home_column} IS NULL
                        OR {away_column} IS NULL
                    )
                    AND {difference_column} IS NOT NULL
                )
                """.strip()
            )

    return "\n            OR ".join(predicates)


def validate_game_modeling_dataset(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Validate the game-level modeling dataset."""

    row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        """
    ).fetchone()[0]

    if row_count == 0:
        raise RuntimeError(
            "The game modeling dataset is empty."
        )

    expected_row_count = connection.execute(
        f"""
        SELECT COUNT(*)

        FROM {SCHEDULE_FULL_NAME} AS schedule

        INNER JOIN {SCHEDULE_FEATURE_FULL_NAME}
            AS schedule_features
            ON schedule.game_id
                = schedule_features.game_id

        INNER JOIN {ELO_FULL_NAME} AS elo
            ON schedule.game_id = elo.game_id

        INNER JOIN {ROLLING_FULL_NAME}
            AS home_features
            ON schedule.game_id = home_features.game_id
           AND home_features.is_home = TRUE

        INNER JOIN {ROLLING_FULL_NAME}
            AS away_features
            ON schedule.game_id = away_features.game_id
           AND away_features.is_home = FALSE

        INNER JOIN {QB_FULL_NAME} AS qb
            ON schedule.game_id = qb.game_id

        INNER JOIN {INJURY_FULL_NAME} AS injury
            ON schedule.game_id = injury.game_id
           AND schedule.home_team = injury.home_team
           AND schedule.away_team = injury.away_team

        INNER JOIN {WEATHER_FULL_NAME}
            AS weather
            ON schedule.game_id = weather.game_id
           AND schedule.home_team = weather.home_team
           AND schedule.away_team = weather.away_team

        INNER JOIN {SCORING_ENVIRONMENT_FULL_NAME}
            AS scoring_environment
            ON schedule.game_id
                = scoring_environment.game_id
           AND schedule.home_team
                = scoring_environment.home_team
           AND schedule.away_team
                = scoring_environment.away_team

        WHERE schedule.is_completed = TRUE
        """
    ).fetchone()[0]

    if row_count != expected_row_count:
        raise RuntimeError(
            "Modeling dataset row count does not match: "
            f"expected {expected_row_count}, "
            f"found {row_count}."
        )

    duplicate_game_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT game_id
            FROM {TARGET_FULL_NAME}
            GROUP BY game_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_game_count > 0:
        raise RuntimeError(
            "Duplicate games found in the modeling dataset: "
            f"{duplicate_game_count}"
        )

    invalid_team_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE home_team IS NULL
           OR away_team IS NULL
           OR home_team = away_team
        """
    ).fetchone()[0]

    if invalid_team_count > 0:
        raise RuntimeError(
            "Invalid teams found in the modeling dataset: "
            f"{invalid_team_count}"
        )

    invalid_elo_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE home_elo_rating IS NULL
           OR away_elo_rating IS NULL
           OR elo_rating_difference IS NULL
           OR elo_home_win_probability
                NOT BETWEEN 0.0 AND 1.0
           OR ABS(
                elo_rating_difference
                - (
                    home_elo_rating
                    - away_elo_rating
                )
           ) > 0.000000001
        """
    ).fetchone()[0]

    if invalid_elo_count > 0:
        raise RuntimeError(
            "Invalid Elo features found: "
            f"{invalid_elo_count}"
        )

    invalid_window_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE home_short_window_games
                NOT BETWEEN 0 AND {SHORT_WINDOW}
           OR away_short_window_games
                NOT BETWEEN 0 AND {SHORT_WINDOW}
           OR home_long_window_games
                NOT BETWEEN 0 AND {LONG_WINDOW}
           OR away_long_window_games
                NOT BETWEEN 0 AND {LONG_WINDOW}

           OR home_short_window_complete
                IS DISTINCT FROM (
                    home_short_window_games
                    = {SHORT_WINDOW}
                )

           OR away_short_window_complete
                IS DISTINCT FROM (
                    away_short_window_games
                    = {SHORT_WINDOW}
                )

           OR home_long_window_complete
                IS DISTINCT FROM (
                    home_long_window_games
                    = {LONG_WINDOW}
                )

           OR away_long_window_complete
                IS DISTINCT FROM (
                    away_long_window_games
                    = {LONG_WINDOW}
                )

           OR both_short_windows_complete
                IS DISTINCT FROM (
                    home_short_window_games
                        = {SHORT_WINDOW}
                    AND away_short_window_games
                        = {SHORT_WINDOW}
                )

           OR both_long_windows_complete
                IS DISTINCT FROM (
                    home_long_window_games
                        = {LONG_WINDOW}
                    AND away_long_window_games
                        = {LONG_WINDOW}
                )
        """
    ).fetchone()[0]

    if invalid_window_count > 0:
        raise RuntimeError(
            "Invalid rolling window metadata found: "
            f"{invalid_window_count}"
        )

    difference_predicate = (
        build_rolling_difference_validation_predicate()
    )

    invalid_difference_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE {difference_predicate}
        """
    ).fetchone()[0]

    if invalid_difference_count > 0:
        raise RuntimeError(
            "Invalid rolling feature differences found: "
            f"{invalid_difference_count}"
        )

    invalid_injury_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE home_has_injury_report_data IS NULL
           OR away_has_injury_report_data IS NULL
           OR has_complete_injury_data IS NULL

           OR has_complete_injury_data
                IS DISTINCT FROM (
                    home_has_injury_report_data
                    AND away_has_injury_report_data
                )

           OR home_total_injury_burden < 0
           OR away_total_injury_burden < 0
           OR home_qb_injury_burden < 0
           OR away_qb_injury_burden < 0
           OR home_non_qb_injury_burden < 0
           OR away_non_qb_injury_burden < 0
           OR home_offense_injury_burden < 0
           OR away_offense_injury_burden < 0
           OR home_defense_injury_burden < 0
           OR away_defense_injury_burden < 0
           OR home_special_teams_injury_burden < 0
           OR away_special_teams_injury_burden < 0

           OR ABS(
                total_injury_burden_difference
                - (
                    home_total_injury_burden
                    - away_total_injury_burden
                )
           ) > 0.000000001

           OR ABS(
                qb_injury_burden_difference
                - (
                    home_qb_injury_burden
                    - away_qb_injury_burden
                )
           ) > 0.000000001

           OR ABS(
                non_qb_injury_burden_difference
                - (
                    home_non_qb_injury_burden
                    - away_non_qb_injury_burden
                )
           ) > 0.000000001

           OR ABS(
                offense_injury_burden_difference
                - (
                    home_offense_injury_burden
                    - away_offense_injury_burden
                )
           ) > 0.000000001

           OR ABS(
                defense_injury_burden_difference
                - (
                    home_defense_injury_burden
                    - away_defense_injury_burden
                )
           ) > 0.000000001

           OR ABS(
                special_teams_injury_burden_difference
                - (
                    home_special_teams_injury_burden
                    - away_special_teams_injury_burden
                )
           ) > 0.000000001

           OR out_player_count_difference
                != (
                    home_out_player_count
                    - away_out_player_count
                )

           OR doubtful_player_count_difference
                != (
                    home_doubtful_player_count
                    - away_doubtful_player_count
                )

           OR questionable_player_count_difference
                != (
                    home_questionable_player_count
                    - away_questionable_player_count
                )

           OR starter_out_count_difference
                != (
                    home_starter_out_count
                    - away_starter_out_count
                )

           OR qb_out_count_difference
                != (
                    home_qb_out_count
                    - away_qb_out_count
                )
        """
    ).fetchone()[0]

    if invalid_injury_count > 0:
        raise RuntimeError(
            "Invalid injury features found: "
            f"{invalid_injury_count}"
        )

    invalid_weather_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE roof_type IS NULL
           OR is_indoor IS NULL
           OR is_weather_exposed IS NULL
           OR has_game_weather IS NULL
           OR modeled_temperature_f IS NULL
           OR modeled_wind_mph IS NULL
           OR is_freezing IS NULL
           OR is_high_wind IS NULL
           OR is_extreme_heat IS NULL
           OR cold_degrees_below_50 IS NULL
           OR heat_degrees_above_80 IS NULL
           OR wind_mph_above_10 IS NULL
           OR (
                is_indoor
                AND is_weather_exposed
              )
           OR (
                has_game_weather
                AND NOT is_weather_exposed
              )
        """
    ).fetchone()[0]

    if invalid_weather_count > 0:
        raise RuntimeError(
            "Invalid modeling weather features found: "
            f"{invalid_weather_count}"
        )

    invalid_target_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {TARGET_FULL_NAME}
        WHERE target_home_score IS NULL
           OR target_away_score IS NULL

           OR target_point_differential
                IS DISTINCT FROM (
                    target_home_score
                    - target_away_score
                )

           OR target_total_points
                IS DISTINCT FROM (
                    target_home_score
                    + target_away_score
                )

           OR target_home_result
                IS DISTINCT FROM (
                    CASE
                        WHEN target_home_score
                            > target_away_score
                            THEN 1.0
                        WHEN target_home_score
                            < target_away_score
                            THEN 0.0
                        ELSE 0.5
                    END
                )

           OR target_home_win
                IS DISTINCT FROM (
                    CASE
                        WHEN target_home_score
                            = target_away_score
                            THEN NULL
                        ELSE target_home_score
                            > target_away_score
                    END
                )
        """
    ).fetchone()[0]

    if invalid_target_count > 0:
        raise RuntimeError(
            "Invalid modeling targets found: "
            f"{invalid_target_count}"
        )

    forbidden_column_count = connection.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_schema = ?
          AND table_name = ?
          AND (
                column_name IN (
                    'home_score',
                    'away_score',
                    'home_win',
                    'point_differential',
                    'total_points'
                )
                OR column_name LIKE '%actual_primary%'
                OR column_name LIKE '%postgame%'
          )
        """,
        [TARGET_SCHEMA, TARGET_TABLE],
    ).fetchone()[0]

    if forbidden_column_count > 0:
        raise RuntimeError(
            "Unprefixed target or postgame columns were found "
            "in the modeling dataset."
        )

    logger.info(
        "Game modeling dataset validated successfully: %s rows.",
        row_count,
    )


def build_game_modeling_dataset(
    database_file: Path = DATABASE_FILE,
) -> None:
    """Build the leakage-safe game-level modeling dataset."""

    validate_database_file(database_file)

    logger.info(
        "Starting game modeling dataset build..."
    )

    with duckdb.connect(str(database_file)) as connection:
        validate_source_tables(connection)

        connection.execute("BEGIN TRANSACTION")

        try:
            create_game_modeling_dataset(connection)

            validate_game_modeling_dataset(connection)

            connection.execute("COMMIT")

            logger.info(
                "Game modeling dataset transaction committed."
            )

        except Exception:
            connection.execute("ROLLBACK")

            logger.exception(
                "Game modeling dataset build failed; "
                "transaction rolled back."
            )
            raise

    logger.info(
        "Game modeling dataset build completed: %s.",
        TARGET_FULL_NAME,
    )


def main() -> None:
    """Run the game modeling dataset builder."""

    try:
        build_game_modeling_dataset()

    except Exception:
        logger.exception(
            "Game modeling dataset builder failed."
        )
        raise


if __name__ == "__main__":
    main()