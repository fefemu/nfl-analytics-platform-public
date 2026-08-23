"""
Tests for the game modeling dataset builder.
"""

from collections.abc import Iterator

import duckdb
import pytest

from src.modeling.build_game_modeling_dataset import (
    LONG_WINDOW,
    ROLLING_METRICS,
    SHORT_WINDOW,
    TARGET_FULL_NAME,
    build_rolling_difference_expressions,
    build_rolling_feature_expressions,
    create_game_modeling_dataset,
    validate_game_modeling_dataset,
    validate_source_tables,
)


@pytest.fixture
def connection() -> Iterator[duckdb.DuckDBPyConnection]:
    """Create an in-memory database with modeling sources."""

    with duckdb.connect(":memory:") as database:
        create_source_tables(database)
        yield database


def create_source_tables(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create minimal source tables required by the builder."""

    connection.execute(
        """
        CREATE SCHEMA processed;
        CREATE SCHEMA analytics;

        CREATE TABLE processed.schedule (
            game_id VARCHAR,
            season INTEGER,
            game_type VARCHAR,
            week INTEGER,
            gameday DATE,
            home_team VARCHAR,
            away_team VARCHAR,
            home_score INTEGER,
            away_score INTEGER,
            is_completed BOOLEAN,
            home_win BOOLEAN,
            point_differential INTEGER,
            total_points INTEGER
        );

        CREATE TABLE analytics.elo_game_predictions (
            game_id VARCHAR,
            gameday DATE,
            home_team VARCHAR,
            away_team VARCHAR,
            home_advantage DOUBLE,
            home_rating_pre DOUBLE,
            away_rating_pre DOUBLE,
            home_win_probability DOUBLE
        );
        """
    )
    connection.execute(
        """
        CREATE TABLE analytics.game_schedule_features (
            game_id VARCHAR,
            game_date DATE,
            home_team VARCHAR,
            away_team VARCHAR,
            home_rest_days INTEGER,
            away_rest_days INTEGER,
            rest_days_difference INTEGER,
            home_short_week BOOLEAN,
            away_short_week BOOLEAN,
            short_week_difference INTEGER,
            home_extended_rest BOOLEAN,
            away_extended_rest BOOLEAN,
            extended_rest_difference INTEGER,
            home_post_bye BOOLEAN,
            away_post_bye BOOLEAN,
            post_bye_difference INTEGER
        );

        INSERT INTO analytics.game_schedule_features
        VALUES
            (
                '2025_05_A_B',
                DATE '2025-10-05',
                'A',
                'B',
                7,
                7,
                0,
                FALSE,
                FALSE,
                0,
                FALSE,
                FALSE,
                0,
                FALSE,
                FALSE,
                0
            ),
            (
                '2025_06_C_D',
                DATE '2025-10-12',
                'C',
                'D',
                10,
                6,
                4,
                FALSE,
                TRUE,
                -1,
                TRUE,
                FALSE,
                1,
                FALSE,
                FALSE,
                0
            );
        """
    )

    connection.execute(
        """
        CREATE TABLE analytics.game_weather_features (
            game_id VARCHAR,
            season INTEGER,
            game_date DATE,
            home_team VARCHAR,
            away_team VARCHAR,
            roof_type VARCHAR,
            surface_type VARCHAR,
            stadium_id VARCHAR,
            stadium VARCHAR,
            is_indoor BOOLEAN,
            is_weather_exposed BOOLEAN,
            has_game_weather BOOLEAN,
            raw_temperature_f DOUBLE,
            raw_wind_mph DOUBLE,
            modeled_temperature_f DOUBLE,
            modeled_wind_mph DOUBLE,
            is_freezing BOOLEAN,
            is_high_wind BOOLEAN,
            is_extreme_heat BOOLEAN,
            cold_degrees_below_50 DOUBLE,
            heat_degrees_above_80 DOUBLE,
            wind_mph_above_10 DOUBLE
        );

        INSERT INTO analytics.game_weather_features
        VALUES
            (
                '2025_05_A_B',
                2025,
                DATE '2025-10-05',
                'A',
                'B',
                'outdoors',
                'grass',
                'stadium_1',
                'Outdoor Stadium',
                FALSE,
                TRUE,
                TRUE,
                30.0,
                18.0,
                30.0,
                18.0,
                TRUE,
                TRUE,
                FALSE,
                20.0,
                0.0,
                8.0
            ),
            (
                '2025_06_C_D',
                2025,
                DATE '2025-10-12',
                'C',
                'D',
                'dome',
                'fieldturf',
                'stadium_2',
                'Indoor Stadium',
                TRUE,
                FALSE,
                FALSE,
                NULL,
                NULL,
                65.0,
                0.0,
                FALSE,
                FALSE,
                FALSE,
                0.0,
                0.0,
                0.0
            );
        """
    )

    connection.execute(
        """
        CREATE TABLE
            analytics.game_scoring_environment_features (
                game_id VARCHAR,
                season INTEGER,
                game_date DATE,
                home_team VARCHAR,
                away_team VARCHAR,
                league_game_count_last_32 INTEGER,
                league_average_total_last_32 DOUBLE,
                league_total_standard_deviation_last_32 DOUBLE,
                league_game_count_last_64 INTEGER,
                league_average_total_last_64 DOUBLE,
                league_total_standard_deviation_last_64 DOUBLE,
                league_game_count_last_128 INTEGER,
                league_average_total_last_128 DOUBLE,
                league_total_standard_deviation_last_128 DOUBLE
            );

        INSERT INTO
            analytics.game_scoring_environment_features
        VALUES
            (
                '2025_05_A_B',
                2025,
                DATE '2025-10-05',
                'A',
                'B',
                32,
                45.5,
                13.0,
                64,
                45.8,
                13.2,
                128,
                46.1,
                13.5
            ),
            (
                '2025_06_C_D',
                2025,
                DATE '2025-10-12',
                'C',
                'D',
                32,
                44.5,
                12.5,
                64,
                45.0,
                12.8,
                128,
                45.7,
                13.1
            );
        """
    )

    rolling_metric_columns = ",\n".join(
        (
            f"pregame_{metric}_last_{window} DOUBLE"
        )
        for metric in ROLLING_METRICS
        for window in (SHORT_WINDOW, LONG_WINDOW)
    )

    connection.execute(
        f"""
        CREATE TABLE analytics.rolling_team_features (
            game_id VARCHAR,
            season INTEGER,
            season_type VARCHAR,
            week INTEGER,
            game_date DATE,
            team VARCHAR,
            opponent VARCHAR,
            is_home BOOLEAN,
            season_games_played_before INTEGER,
            short_window_games INTEGER,
            long_window_games INTEGER,
            {rolling_metric_columns}
        );
        """
    )

    connection.execute(
        """
        CREATE TABLE analytics.game_qb_features (
            game_id VARCHAR,
            game_date DATE,
            home_team VARCHAR,
            away_team VARCHAR,

            home_listed_qb_id VARCHAR,
            home_listed_qb_name VARCHAR,
            home_listed_qb_rating DOUBLE,
            home_listed_qb_effective_dropbacks DOUBLE,
            home_listed_qb_prior_weight DOUBLE,
            home_listed_qb_rating_standard_error DOUBLE,
            home_listed_qb_rating_available BOOLEAN,

            away_listed_qb_id VARCHAR,
            away_listed_qb_name VARCHAR,
            away_listed_qb_rating DOUBLE,
            away_listed_qb_effective_dropbacks DOUBLE,
            away_listed_qb_prior_weight DOUBLE,
            away_listed_qb_rating_standard_error DOUBLE,
            away_listed_qb_rating_available BOOLEAN,

            both_listed_qb_ratings_available BOOLEAN,
            listed_qb_rating_difference DOUBLE,
            listed_qb_rating_difference_standard_error DOUBLE
        );
        """
    )

    connection.execute(
        """
        CREATE TABLE analytics.game_injury_features (
            game_id VARCHAR,
            home_team VARCHAR,
            away_team VARCHAR,

            home_has_injury_report_data BOOLEAN,
            away_has_injury_report_data BOOLEAN,
            has_complete_injury_data BOOLEAN,

            home_out_player_count INTEGER,
            away_out_player_count INTEGER,
            out_player_count_difference INTEGER,

            home_doubtful_player_count INTEGER,
            away_doubtful_player_count INTEGER,
            doubtful_player_count_difference INTEGER,

            home_questionable_player_count INTEGER,
            away_questionable_player_count INTEGER,
            questionable_player_count_difference INTEGER,

            home_starter_out_count INTEGER,
            away_starter_out_count INTEGER,
            starter_out_count_difference INTEGER,

            home_qb_out_count INTEGER,
            away_qb_out_count INTEGER,
            qb_out_count_difference INTEGER,

            home_total_injury_burden DOUBLE,
            away_total_injury_burden DOUBLE,
            total_injury_burden_difference DOUBLE,

            home_qb_injury_burden DOUBLE,
            away_qb_injury_burden DOUBLE,
            qb_injury_burden_difference DOUBLE,

            home_non_qb_injury_burden DOUBLE,
            away_non_qb_injury_burden DOUBLE,
            non_qb_injury_burden_difference DOUBLE,

            home_offense_injury_burden DOUBLE,
            away_offense_injury_burden DOUBLE,
            offense_injury_burden_difference DOUBLE,

            home_defense_injury_burden DOUBLE,
            away_defense_injury_burden DOUBLE,
            defense_injury_burden_difference DOUBLE,

            home_special_teams_injury_burden DOUBLE,
            away_special_teams_injury_burden DOUBLE,
            special_teams_injury_burden_difference DOUBLE
        );

        INSERT INTO analytics.game_injury_features
        VALUES
            (
                '2025_05_A_B',
                'A',
                'B',
                TRUE,
                TRUE,
                TRUE,
                2,
                1,
                1,
                1,
                0,
                1,
                3,
                2,
                1,
                1,
                0,
                1,
                0,
                0,
                0,
                1.80,
                0.90,
                0.90,
                0.00,
                0.00,
                0.00,
                1.80,
                0.90,
                0.90,
                1.10,
                0.40,
                0.70,
                0.60,
                0.40,
                0.20,
                0.10,
                0.10,
                0.00
            ),
            (
                '2025_06_C_D',
                'C',
                'D',
                TRUE,
                FALSE,
                FALSE,
                1,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0.70,
                0.00,
                0.70,
                0.00,
                0.00,
                0.00,
                0.70,
                0.00,
                0.70,
                0.30,
                0.00,
                0.30,
                0.30,
                0.00,
                0.30,
                0.10,
                0.00,
                0.10
            );
        """
    )

    connection.execute(
        """
        INSERT INTO processed.schedule
        VALUES
            (
                '2025_05_A_B',
                2025,
                'REG',
                5,
                DATE '2025-10-05',
                'A',
                'B',
                24,
                17,
                TRUE,
                TRUE,
                7,
                41
            ),
            (
                '2025_06_C_D',
                2025,
                'REG',
                6,
                DATE '2025-10-12',
                'C',
                'D',
                20,
                20,
                TRUE,
                FALSE,
                0,
                40
            );
        """
    )

    connection.execute(
        """
        INSERT INTO analytics.elo_game_predictions
        VALUES
            (
                '2025_05_A_B',
                DATE '2025-10-05',
                'A',
                'B',
                50.0,
                1550.0,
                1500.0,
                0.64
            ),
            (
                '2025_06_C_D',
                DATE '2025-10-12',
                'C',
                'D',
                50.0,
                1490.0,
                1510.0,
                0.54
            );
        """
    )

    rolling_column_count = (
        len(ROLLING_METRICS) * 2
    )

    rolling_placeholders = ", ".join(
        "?"
        for _ in range(11 + rolling_column_count)
    )

    complete_home_values = [
        0.20
        for _ in range(rolling_column_count)
    ]

    complete_away_values = [
        0.10
        for _ in range(rolling_column_count)
    ]

    missing_values = [
        None
        for _ in range(rolling_column_count)
    ]

    rolling_rows = [
        (
            "2025_05_A_B",
            2025,
            "REG",
            5,
            "2025-10-05",
            "A",
            "B",
            True,
            4,
            4,
            4,
            *complete_home_values,
        ),
        (
            "2025_05_A_B",
            2025,
            "REG",
            5,
            "2025-10-05",
            "B",
            "A",
            False,
            4,
            4,
            4,
            *complete_away_values,
        ),
        (
            "2025_06_C_D",
            2025,
            "REG",
            6,
            "2025-10-12",
            "C",
            "D",
            True,
            0,
            0,
            0,
            *missing_values,
        ),
        (
            "2025_06_C_D",
            2025,
            "REG",
            6,
            "2025-10-12",
            "D",
            "C",
            False,
            0,
            0,
            0,
            *missing_values,
        ),
    ]

    connection.executemany(
        f"""
        INSERT INTO analytics.rolling_team_features
        VALUES ({rolling_placeholders})
        """,
        rolling_rows,
    )

    connection.execute(
        """
        INSERT INTO analytics.game_qb_features
        VALUES
            (
                '2025_05_A_B',
                DATE '2025-10-05',
                'A',
                'B',
                'QB_A',
                'Quarterback A',
                105.0,
                500.0,
                0.29,
                0.65,
                TRUE,
                'QB_B',
                'Quarterback B',
                100.0,
                450.0,
                0.31,
                0.70,
                TRUE,
                TRUE,
                5.0,
                0.96
            ),
            (
                '2025_06_C_D',
                DATE '2025-10-12',
                'C',
                'D',
                'QB_C',
                'Quarterback C',
                NULL,
                NULL,
                NULL,
                NULL,
                FALSE,
                'QB_D',
                'Quarterback D',
                101.0,
                300.0,
                0.40,
                0.80,
                TRUE,
                FALSE,
                NULL,
                NULL
            );
        """
    )


def test_build_rolling_feature_expressions_uses_prefix() -> None:
    """Generate prefixed rolling feature aliases."""

    expressions = build_rolling_feature_expressions(
        table_alias="home_features",
        output_prefix="home",
    )

    assert (
        "home_features."
        "pregame_offensive_epa_per_play_last_4 "
        "AS home_offensive_epa_per_play_last_4"
        in expressions
    )


def test_build_rolling_difference_expressions() -> None:
    """Generate home-minus-away difference expressions."""

    expressions = (
        build_rolling_difference_expressions()
    )

    assert (
        "home_features."
        "pregame_offensive_epa_per_play_last_4"
        in expressions
    )

    assert (
        "AS offensive_epa_per_play_difference_last_4"
        in expressions
    )


def test_create_dataset_builds_features_and_targets(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Create game features and clearly prefixed targets."""

    create_game_modeling_dataset(connection)

    row = connection.execute(
        f"""
        SELECT
            elo_rating_difference,
            offensive_epa_per_play_difference_last_4,
            listed_qb_rating_difference,
            target_home_win,
            target_home_result,
            target_point_differential,
            target_total_points
        FROM {TARGET_FULL_NAME}
        WHERE game_id = '2025_05_A_B'
        """
    ).fetchone()

    assert row[0] == pytest.approx(50.0)
    assert row[1] == pytest.approx(0.10)
    assert row[2] == pytest.approx(5.0)
    assert row[3:] == (
        True,
        1.0,
        7,
        41,
    )


def test_tied_game_has_null_binary_target(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Represent a tie without treating it as a home loss."""

    create_game_modeling_dataset(connection)

    row = connection.execute(
        f"""
        SELECT
            target_home_win,
            target_home_result,
            target_point_differential
        FROM {TARGET_FULL_NAME}
        WHERE game_id = '2025_06_C_D'
        """
    ).fetchone()

    assert row == (
        None,
        0.5,
        0,
    )


def test_created_dataset_passes_validation(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Accept a consistent leakage-safe modeling dataset."""

    validate_source_tables(connection)
    create_game_modeling_dataset(connection)
    validate_game_modeling_dataset(connection)


def test_create_dataset_includes_schedule_context(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Include leakage-safe pregame rest features."""

    create_game_modeling_dataset(connection)

    row = connection.execute(
        f"""
        SELECT
            home_rest_days,
            away_rest_days,
            rest_days_difference,
            home_short_week,
            away_short_week,
            short_week_difference,
            home_extended_rest,
            away_extended_rest,
            extended_rest_difference,
            home_post_bye,
            away_post_bye,
            post_bye_difference
        FROM {TARGET_FULL_NAME}
        WHERE game_id = '2025_06_C_D'
        """
    ).fetchone()

    assert row == (
        10,
        6,
        4,
        False,
        True,
        -1,
        True,
        False,
        1,
        False,
        False,
        0,
    )


def test_create_dataset_includes_weather_context(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Include normalized venue and weather features."""

    create_game_modeling_dataset(connection)

    outdoor = connection.execute(
        f"""
        SELECT
            roof_type,
            is_indoor,
            is_weather_exposed,
            has_game_weather,
            modeled_temperature_f,
            modeled_wind_mph,
            is_freezing,
            is_high_wind,
            cold_degrees_below_50,
            wind_mph_above_10
        FROM {TARGET_FULL_NAME}
        WHERE game_id = '2025_05_A_B'
        """
    ).fetchone()

    indoor = connection.execute(
        f"""
        SELECT
            roof_type,
            is_indoor,
            is_weather_exposed,
            has_game_weather,
            modeled_temperature_f,
            modeled_wind_mph
        FROM {TARGET_FULL_NAME}
        WHERE game_id = '2025_06_C_D'
        """
    ).fetchone()

    assert outdoor == (
        "outdoors",
        False,
        True,
        True,
        30.0,
        18.0,
        True,
        True,
        20.0,
        8.0,
    )

    assert indoor == (
        "dome",
        True,
        False,
        False,
        65.0,
        0.0,
    )


def test_create_dataset_includes_scoring_environment(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Include strictly pregame league scoring context."""

    create_game_modeling_dataset(connection)

    row = connection.execute(
        f"""
        SELECT
            league_game_count_last_32,
            league_average_total_last_32,
            league_total_standard_deviation_last_32,
            league_game_count_last_64,
            league_average_total_last_64,
            league_total_standard_deviation_last_64,
            league_game_count_last_128,
            league_average_total_last_128,
            league_total_standard_deviation_last_128
        FROM {TARGET_FULL_NAME}
        WHERE game_id = '2025_05_A_B'
        """
    ).fetchone()

    assert row[0] == 32
    assert row[1] == pytest.approx(45.5)
    assert row[2] == pytest.approx(13.0)
    assert row[3] == 64
    assert row[4] == pytest.approx(45.8)
    assert row[5] == pytest.approx(13.2)
    assert row[6] == 128
    assert row[7] == pytest.approx(46.1)
    assert row[8] == pytest.approx(13.5)


def test_create_dataset_includes_injury_features(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Include pregame injury burden and availability."""

    create_game_modeling_dataset(
        connection
    )

    complete_row = connection.execute(
        f"""
        SELECT
            home_has_injury_report_data,
            away_has_injury_report_data,
            has_complete_injury_data,
            home_non_qb_injury_burden,
            away_non_qb_injury_burden,
            non_qb_injury_burden_difference,
            offense_injury_burden_difference,
            defense_injury_burden_difference,
            starter_out_count_difference
        FROM {TARGET_FULL_NAME}
        WHERE game_id = '2025_05_A_B'
        """
    ).fetchone()

    incomplete_row = connection.execute(
        f"""
        SELECT
            home_has_injury_report_data,
            away_has_injury_report_data,
            has_complete_injury_data,
            non_qb_injury_burden_difference
        FROM {TARGET_FULL_NAME}
        WHERE game_id = '2025_06_C_D'
        """
    ).fetchone()

    assert complete_row[:3] == (
        True,
        True,
        True,
    )
    assert complete_row[3] == pytest.approx(
        1.80
    )
    assert complete_row[4] == pytest.approx(
        0.90
    )
    assert complete_row[5] == pytest.approx(
        0.90
    )
    assert complete_row[6] == pytest.approx(
        0.70
    )
    assert complete_row[7] == pytest.approx(
        0.20
    )
    assert complete_row[8] == 1

    assert incomplete_row[:3] == (
        True,
        False,
        False,
    )
    assert incomplete_row[3] == pytest.approx(
        0.70
    )