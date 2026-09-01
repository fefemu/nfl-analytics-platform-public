"""Read-only DuckDB access and health state for public dashboard pages."""

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from src.deployment.dashboard_database import resolve_dashboard_database_file


CORE_TABLES = (
    "current_game_predictions",
    "current_game_spread_predictions",
    "current_game_total_predictions",
    "current_game_score_predictions",
    "current_season_simulation_summary",
)


@dataclass(frozen=True)
class DashboardHealth:
    database_available: bool
    available_tables: tuple[str, ...]
    missing_core_tables: tuple[str, ...]
    latest_refresh_status: str | None
    latest_refresh_at: pd.Timestamp | None

    @property
    def ready(self) -> bool:
        return self.database_available and not self.missing_core_tables


class DashboardRepository:
    """Central query boundary; dashboard pages never open writable connections."""

    def __init__(self, database_file: Path | None = None) -> None:
        self.database_file = Path(
            database_file
            if database_file is not None
            else resolve_dashboard_database_file()
        )

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.database_file), read_only=True)

    def health(self) -> DashboardHealth:
        if not self.database_file.is_file():
            return DashboardHealth(False, (), CORE_TABLES, None, None)

        connection = None
        try:
            connection = self._connect()
            tables = tuple(
                row[0]
                for row in connection.execute(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'analytics'
                    ORDER BY table_name
                    """
                ).fetchall()
            )
            status = None
            refreshed_at = None
            if "refresh_run_history" in tables:
                latest = connection.execute(
                    """
                    SELECT status, CAST(completed_at AS VARCHAR)
                    FROM analytics.refresh_run_history
                    ORDER BY started_at DESC
                    LIMIT 1
                    """
                ).fetchone()
                if latest:
                    status, refreshed_at = latest
            if refreshed_at is None:
                timestamp_sources = (
                    (
                        "current_game_predictions",
                        "prediction_generated_at",
                    ),
                    (
                        "current_season_simulation_summary",
                        "simulation_generated_at",
                    ),
                )
                output_timestamps: list[object] = []
                for table_name, column_name in timestamp_sources:
                    if table_name in tables:
                        value = connection.execute(
                            f"SELECT CAST(MAX({column_name}) AS VARCHAR) "
                            f"FROM analytics.{table_name}"
                        ).fetchone()[0]
                        if value is not None:
                            output_timestamps.append(value)
                if output_timestamps:
                    normalized_timestamps = []
                    for value in output_timestamps:
                        timestamp_value = pd.Timestamp(value)
                        if timestamp_value.tzinfo is None:
                            timestamp_value = timestamp_value.tz_localize("UTC")
                        else:
                            timestamp_value = timestamp_value.tz_convert("UTC")
                        normalized_timestamps.append(timestamp_value)
                    refreshed_at = max(normalized_timestamps)
                    status = status or "MODEL_BUILD"
        except (duckdb.Error, OSError):
            return DashboardHealth(False, (), CORE_TABLES, None, None)
        finally:
            if connection is not None:
                connection.close()

        missing = tuple(table for table in CORE_TABLES if table not in tables)
        timestamp = pd.Timestamp(refreshed_at) if refreshed_at is not None else None
        return DashboardHealth(True, tables, missing, status, timestamp)

    def read_table(
        self,
        table_name: str,
        columns: tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        """Read a validated analytics table without accepting arbitrary SQL."""

        if not table_name.replace("_", "").isalnum():
            raise ValueError("Invalid analytics table name.")
        selected = "*" if columns is None else ", ".join(columns)
        if columns and any(not column.replace("_", "").isalnum() for column in columns):
            raise ValueError("Invalid analytics column name.")
        connection = self._connect()
        try:
            return connection.execute(
                f"SELECT {selected} FROM analytics.{table_name}"
            ).fetchdf()
        finally:
            connection.close()

    def load_weekly_games(self) -> pd.DataFrame:
        """Load the unified prediction product used by Weekly Overview."""

        required = {
            "current_game_predictions",
            "current_game_spread_predictions",
            "current_game_total_predictions",
            "current_game_score_predictions",
        }
        if not required.issubset(self.health().available_tables):
            return pd.DataFrame()
        has_trends = (
            "current_game_probability_trends"
            in self.health().available_tables
        )
        trend_fields = """
                    , trends.home_previous_win_probability
                    , trends.away_previous_win_probability
                    , trends.home_probability_change_pp
                    , trends.away_probability_change_pp
                    , trends.home_probability_trend
                    , trends.away_probability_trend
                    , trends.previous_prediction_generated_at
                    , trends.neutral_threshold_pp
        """ if has_trends else ""
        trend_join = """
                LEFT JOIN analytics.current_game_probability_trends AS trends
                    USING (game_id)
        """ if has_trends else ""
        connection = self._connect()
        try:
            return connection.execute(
                f"""
                SELECT
                    probability.game_id,
                    probability.season,
                    probability.week,
                    probability.gameday,
                    probability.gametime,
                    probability.away_team,
                    probability.home_team,
                    probability.away_win_probability,
                    probability.home_win_probability,
                    probability.predicted_winner,
                    probability.model_name AS probability_model_name,
                    probability.prediction_mode AS probability_prediction_mode,
                    spread.predicted_home_margin,
                    spread.prediction_mode AS spread_prediction_mode,
                    totals.predicted_total_points,
                    totals.prediction_mode AS totals_prediction_mode,
                    score.implied_away_score,
                    score.implied_home_score
                    {trend_fields}
                FROM analytics.current_game_predictions AS probability
                INNER JOIN analytics.current_game_spread_predictions AS spread
                    USING (game_id)
                INNER JOIN analytics.current_game_total_predictions AS totals
                    USING (game_id)
                INNER JOIN analytics.current_game_score_predictions AS score
                    USING (game_id)
                {trend_join}
                ORDER BY probability.week, probability.gameday,
                         probability.gametime, probability.game_id
                """
            ).fetchdf()
        finally:
            connection.close()

    def load_current_betting_board(self) -> pd.DataFrame:
        """Load the current standardized betting board when available."""

        if "current_betting_board" not in self.health().available_tables:
            return pd.DataFrame()
        return self.read_table("current_betting_board")

    def load_current_team_rosters(self) -> pd.DataFrame:
        """Load the latest timestamped depth chart with available player context."""

        if not self.database_file.is_file():
            return pd.DataFrame()
        connection = self._connect()
        try:
            source_tables = {
                (row[0], row[1])
                for row in connection.execute(
                    "SELECT table_schema, table_name FROM information_schema.tables"
                ).fetchall()
            }
            if ("raw", "depth_charts_espn") not in source_tables:
                return pd.DataFrame()
            has_directory = ("raw", "player_directory") in source_tables
            has_injuries = ("raw", "injury_reports") in source_tables
            has_external_elo = (
                "processed", "external_nfelo_game_ratings"
            ) in source_tables
            has_units = (
                "processed", "external_nfelounits_units"
            ) in source_tables
            directory_join = """
                LEFT JOIN directory AS player
                  ON depth.gsis_id = player.gsis_id
                  OR (depth.gsis_id IS NULL AND CAST(depth.espn_id AS VARCHAR) = CAST(player.espn_id AS VARCHAR))
            """ if has_directory else ""
            injury_join = """
                LEFT JOIN injuries AS injury
                  ON depth.gsis_id = injury.gsis_id AND depth.team = injury.team
                 AND depth.source_season = injury.season
            """ if has_injuries else ""
            directory_cte = """
                , directory AS (
                    SELECT * FROM raw.player_directory
                    QUALIFY ROW_NUMBER() OVER (
                        PARTITION BY COALESCE(
                            gsis_id,
                            'ESPN:' || CAST(espn_id AS VARCHAR)
                        )
                        ORDER BY last_season DESC NULLS LAST
                    ) = 1
                )
            """ if has_directory else ""
            injury_cte = """
                , injuries AS (
                    SELECT season, team, gsis_id, report_status,
                           report_primary_injury, practice_status
                    FROM raw.injury_reports
                    QUALIFY ROW_NUMBER() OVER (
                        PARTITION BY season, team, gsis_id
                        ORDER BY week DESC, date_modified DESC NULLS LAST
                    ) = 1
                )
            """ if has_injuries else ""
            elo_cte = """
                , external_elo_rows AS (
                    SELECT source_season AS season, source_week AS week,
                           CASE home_team WHEN 'OAK' THEN 'LV'
                               WHEN 'LAR' THEN 'LA' ELSE home_team END AS team,
                           starting_nfelo_home AS rating,
                           source_fetched_at
                    FROM processed.external_nfelo_game_ratings
                    UNION ALL
                    SELECT source_season, source_week,
                           CASE away_team WHEN 'OAK' THEN 'LV'
                               WHEN 'LAR' THEN 'LA' ELSE away_team END,
                           starting_nfelo_away, source_fetched_at
                    FROM processed.external_nfelo_game_ratings
                )
                , external_elo_values AS (
                    SELECT team, AVG(rating) AS rating,
                           MAX(source_fetched_at) AS source_fetched_at
                    FROM external_elo_rows
                    WHERE season = (SELECT MAX(season) FROM external_elo_rows)
                      AND week = (
                          SELECT MIN(week) FROM external_elo_rows
                          WHERE season = (SELECT MAX(season) FROM external_elo_rows)
                      )
                    GROUP BY team
                )
                , external_elo AS (
                    SELECT team, rating,
                           RANK() OVER (ORDER BY rating DESC) AS rating_rank,
                           source_fetched_at
                    FROM external_elo_values
                )
            """ if has_external_elo else ""
            elo_join = """
                LEFT JOIN external_elo AS elo
                  ON depth.team = elo.team
            """ if has_external_elo else ""
            elo_fields = (
                "elo.rating AS elo_rating, elo.rating_rank AS elo_rank, "
                "elo.source_fetched_at AS elo_source_fetched_at"
                if has_external_elo else
                "CAST(NULL AS DOUBLE) AS elo_rating, "
                "CAST(NULL AS INTEGER) AS elo_rank, "
                "CAST(NULL AS TIMESTAMPTZ) AS elo_source_fetched_at"
            )
            units_cte = """
                , current_unit_values AS (
                    SELECT CASE team WHEN 'OAK' THEN 'LV'
                               WHEN 'LAR' THEN 'LA' ELSE team END AS team,
                           off_value_pre, def_value_pre,
                           pass_off_value_pre, rush_off_value_pre,
                           pass_def_value_pre, rush_def_value_pre,
                           source_fetched_at
                    FROM processed.external_nfelounits_units
                    WHERE season = (
                        SELECT MAX(season)
                        FROM processed.external_nfelounits_units
                    )
                      AND week = (
                        SELECT MIN(week)
                        FROM processed.external_nfelounits_units
                        WHERE season = (
                            SELECT MAX(season)
                            FROM processed.external_nfelounits_units
                        )
                    )
                )
                , current_unit_ranks AS (
                    SELECT *,
                           RANK() OVER (ORDER BY off_value_pre DESC) AS offense_rank,
                           RANK() OVER (ORDER BY def_value_pre DESC) AS defense_rank,
                           RANK() OVER (ORDER BY pass_off_value_pre DESC) AS pass_offense_rank,
                           RANK() OVER (ORDER BY rush_off_value_pre DESC) AS rush_offense_rank,
                           RANK() OVER (ORDER BY pass_def_value_pre DESC) AS pass_defense_rank,
                           RANK() OVER (ORDER BY rush_def_value_pre DESC) AS rush_defense_rank
                    FROM current_unit_values
                )
            """ if has_units else ""
            units_join = """
                LEFT JOIN current_unit_ranks AS units
                  ON depth.team = units.team
            """ if has_units else ""
            unit_fields = (
                "units.offense_rank, units.defense_rank, "
                "units.pass_offense_rank, units.rush_offense_rank, "
                "units.pass_defense_rank, units.rush_defense_rank, "
                "units.source_fetched_at AS units_source_fetched_at"
                if has_units else
                "CAST(NULL AS INTEGER) AS offense_rank, "
                "CAST(NULL AS INTEGER) AS defense_rank, "
                "CAST(NULL AS INTEGER) AS pass_offense_rank, "
                "CAST(NULL AS INTEGER) AS rush_offense_rank, "
                "CAST(NULL AS INTEGER) AS pass_defense_rank, "
                "CAST(NULL AS INTEGER) AS rush_defense_rank, "
                "CAST(NULL AS TIMESTAMPTZ) AS units_source_fetched_at"
            )
            player_fields = (
                "player.jersey_number, player.headshot, player.status AS roster_status"
                if has_directory else
                "CAST(NULL AS VARCHAR) AS jersey_number, CAST(NULL AS VARCHAR) AS headshot, CAST(NULL AS VARCHAR) AS roster_status"
            )
            injury_fields = (
                "injury.report_status, injury.report_primary_injury, injury.practice_status"
                if has_injuries else
                "CAST(NULL AS VARCHAR) AS report_status, CAST(NULL AS VARCHAR) AS report_primary_injury, CAST(NULL AS VARCHAR) AS practice_status"
            )
            return connection.execute(
                f"""
                WITH latest_depth AS (
                    SELECT source_season, CAST(dt AS TIMESTAMPTZ) AS snapshot_at,
                           team, player_name, espn_id, gsis_id, pos_grp,
                           pos_name, pos_abb, pos_slot, pos_rank
                    FROM raw.depth_charts_espn
                    WHERE source_season = (SELECT MAX(source_season) FROM raw.depth_charts_espn)
                    QUALIFY snapshot_at = MAX(snapshot_at) OVER (PARTITION BY team)
                )
                {directory_cte}
                {injury_cte}
                {elo_cte}
                {units_cte}
                SELECT depth.source_season, depth.snapshot_at, depth.team,
                       depth.player_name, depth.gsis_id, depth.espn_id,
                       depth.pos_grp, depth.pos_name, depth.pos_abb,
                       depth.pos_slot, depth.pos_rank,
                       {player_fields}, {injury_fields}, {elo_fields},
                       {unit_fields}
                FROM latest_depth AS depth
                {directory_join}
                {injury_join}
                {elo_join}
                {units_join}
                ORDER BY depth.team, depth.pos_grp, depth.pos_slot, depth.pos_rank
                """
            ).fetchdf()
        finally:
            connection.close()

    def load_current_team_schedule(self) -> pd.DataFrame:
        """Load the current regular-season schedule in a team-centric shape."""

        if not self.database_file.is_file():
            return pd.DataFrame()
        connection = self._connect()
        try:
            source_tables = {
                (row[0], row[1])
                for row in connection.execute(
                    "SELECT table_schema, table_name FROM information_schema.tables"
                ).fetchall()
            }
            if ("processed", "schedule") not in source_tables:
                return pd.DataFrame()
            has_elo = ("analytics", "current_elo_ratings") in source_tables
            elo_cte = """
                , elo AS (
                    SELECT CASE team WHEN 'OAK' THEN 'LV'
                               WHEN 'LAR' THEN 'LA' ELSE team END AS team,
                           elo_rating
                    FROM analytics.current_elo_ratings
                )
            """ if has_elo else ""
            elo_join = "LEFT JOIN elo ON games.opponent = elo.team" if has_elo else ""
            elo_field = (
                "elo.elo_rating AS opponent_elo" if has_elo
                else "CAST(NULL AS DOUBLE) AS opponent_elo"
            )
            return connection.execute(
                f"""
                WITH current_schedule AS (
                    SELECT *
                    FROM processed.schedule
                    WHERE season = (
                        SELECT MAX(season) FROM processed.schedule
                        WHERE game_type = 'REG'
                    )
                      AND game_type = 'REG'
                ), team_games AS (
                    SELECT season, week, gameday, gametime,
                           CASE home_team WHEN 'OAK' THEN 'LV'
                               WHEN 'LAR' THEN 'LA' ELSE home_team END AS team,
                           CASE away_team WHEN 'OAK' THEN 'LV'
                               WHEN 'LAR' THEN 'LA' ELSE away_team END AS opponent,
                           TRUE AS is_home,
                           home_score AS team_score,
                           away_score AS opponent_score,
                           is_completed
                    FROM current_schedule
                    UNION ALL
                    SELECT season, week, gameday, gametime,
                           CASE away_team WHEN 'OAK' THEN 'LV'
                               WHEN 'LAR' THEN 'LA' ELSE away_team END AS team,
                           CASE home_team WHEN 'OAK' THEN 'LV'
                               WHEN 'LAR' THEN 'LA' ELSE home_team END AS opponent,
                           FALSE AS is_home,
                           away_score AS team_score,
                           home_score AS opponent_score,
                           is_completed
                    FROM current_schedule
                ), current_week AS (
                    SELECT MIN(week) AS week
                    FROM current_schedule
                    WHERE is_completed = FALSE
                )
                {elo_cte}
                SELECT games.*, {elo_field}, current_week.week AS current_week
                FROM team_games AS games
                CROSS JOIN current_week
                {elo_join}
                ORDER BY games.team, games.week, games.gameday, games.gametime
                """
            ).fetchdf()
        finally:
            connection.close()

    def load_game_center_games(self) -> pd.DataFrame:
        """Load matchup predictions plus optional public narrative fields."""

        games = self.load_weekly_games()
        if games.empty:
            return games
        if "current_game_prediction_narratives" not in self.health().available_tables:
            return games
        narratives = self.read_table(
            "current_game_prediction_narratives",
            (
                "game_id", "headline_en", "headline_hu", "summary_en",
                "summary_hu", "model_context_en", "model_context_hu",
                "top_factor_en", "top_factor_hu",
            ),
        )
        return games.merge(
            narratives,
            on="game_id",
            how="left",
            validate="one_to_one",
        )

    def load_season_simulator(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load simulation summary, distributions and optional Elo benchmark."""

        available = set(self.health().available_tables)
        required = {
            "current_season_simulation_summary",
            "current_season_win_distribution",
        }
        if not required.issubset(available):
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        summary = self.read_table("current_season_simulation_summary")
        distribution = self.read_table("current_season_win_distribution")
        benchmark = pd.DataFrame()
        if "current_season_elo_benchmark_team_comparison" in available:
            benchmark = self.read_table(
                "current_season_elo_benchmark_team_comparison"
            )
        return summary, distribution, benchmark

    def load_data_science_lab(self) -> dict[str, pd.DataFrame]:
        """Load only curated governance products exposed by the Lab."""

        available = set(self.health().available_tables)
        requested = (
            "production_model_registry",
            "model_governance_scorecard",
            "model_governance_season_results",
            "model_blend_scorecard",
        )
        products = {
            table: self.read_table(table)
            for table in requested
            if table in available
        }
        if "game_modeling_dataset" in available:
            connection = self._connect()
            try:
                products["historical_impact_summary"] = connection.execute(
                    """
                    WITH completed AS (
                        SELECT
                            *,
                            home_rest_days - away_rest_days AS rest_difference,
                            home_listed_qb_rating - away_listed_qb_rating
                                AS qb_rating_difference,
                            home_total_injury_burden - away_total_injury_burden
                                AS injury_burden_difference
                        FROM analytics.game_modeling_dataset
                        WHERE target_home_score IS NOT NULL
                          AND target_away_score IS NOT NULL
                    ), groups AS (
                        SELECT 'home_field' AS topic, 'all_games' AS segment,
                               COUNT(*) AS game_count,
                               AVG(CAST(target_home_win AS INTEGER)) AS win_rate,
                               AVG(target_point_differential) AS average_margin,
                               NULL::DOUBLE AS average_total
                        FROM completed
                        UNION ALL
                        SELECT 'rest', 'home_3_plus', COUNT(*),
                               AVG(CAST(target_home_win AS INTEGER)),
                               AVG(target_point_differential), NULL::DOUBLE
                        FROM completed WHERE rest_difference >= 3
                        UNION ALL
                        SELECT 'rest', 'away_3_plus', COUNT(*),
                               AVG(CAST(target_home_win AS INTEGER)),
                               AVG(target_point_differential), NULL::DOUBLE
                        FROM completed WHERE rest_difference <= -3
                        UNION ALL
                        SELECT 'rest', 'similar', COUNT(*),
                               AVG(CAST(target_home_win AS INTEGER)),
                               AVG(target_point_differential), NULL::DOUBLE
                        FROM completed WHERE ABS(rest_difference) < 3
                        UNION ALL
                        SELECT 'rest', 'home_post_bye', COUNT(*),
                               AVG(CAST(target_home_win AS INTEGER)),
                               AVG(target_point_differential), NULL::DOUBLE
                        FROM completed
                        WHERE home_post_bye AND NOT away_post_bye
                        UNION ALL
                        SELECT 'qb', 'home_advantage', COUNT(*),
                               AVG(CAST(target_home_win AS INTEGER)),
                               AVG(target_point_differential), NULL::DOUBLE
                        FROM completed WHERE qb_rating_difference >= 0.75
                        UNION ALL
                        SELECT 'qb', 'away_advantage', COUNT(*),
                               AVG(CAST(target_home_win AS INTEGER)),
                               AVG(target_point_differential), NULL::DOUBLE
                        FROM completed WHERE qb_rating_difference <= -0.75
                        UNION ALL
                        SELECT 'qb', 'similar', COUNT(*),
                               AVG(CAST(target_home_win AS INTEGER)),
                               AVG(target_point_differential), NULL::DOUBLE
                        FROM completed WHERE ABS(qb_rating_difference) < 0.75
                        UNION ALL
                        SELECT 'injury', 'home_more_injured', COUNT(*),
                               AVG(CAST(target_home_win AS INTEGER)),
                               AVG(target_point_differential), NULL::DOUBLE
                        FROM completed WHERE injury_burden_difference >= 2
                        UNION ALL
                        SELECT 'injury', 'away_more_injured', COUNT(*),
                               AVG(CAST(target_home_win AS INTEGER)),
                               AVG(target_point_differential), NULL::DOUBLE
                        FROM completed WHERE injury_burden_difference <= -2
                        UNION ALL
                        SELECT 'injury', 'similar', COUNT(*),
                               AVG(CAST(target_home_win AS INTEGER)),
                               AVG(target_point_differential), NULL::DOUBLE
                        FROM completed WHERE ABS(injury_burden_difference) < 2
                        UNION ALL
                        SELECT 'weather', 'indoor', COUNT(*), NULL::DOUBLE,
                               NULL::DOUBLE, AVG(target_total_points)
                        FROM completed WHERE is_indoor
                        UNION ALL
                        SELECT 'weather', 'freezing', COUNT(*), NULL::DOUBLE,
                               NULL::DOUBLE, AVG(target_total_points)
                        FROM completed WHERE is_freezing
                        UNION ALL
                        SELECT 'weather', 'high_wind', COUNT(*), NULL::DOUBLE,
                               NULL::DOUBLE, AVG(target_total_points)
                        FROM completed WHERE is_high_wind
                        UNION ALL
                        SELECT 'weather', 'other_exposed', COUNT(*), NULL::DOUBLE,
                               NULL::DOUBLE, AVG(target_total_points)
                        FROM completed
                        WHERE is_weather_exposed
                          AND NOT is_freezing AND NOT is_high_wind
                    )
                    SELECT * FROM groups ORDER BY topic, segment
                    """
                ).fetchdf()
            finally:
                connection.close()
        return products
