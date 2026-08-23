"""Tests for totals model candidate evaluation."""

import numpy as np
import pandas as pd
import pytest

from src.modeling.evaluate_totals_model_candidates import (
    RESULT_COLUMNS,
    TOTALS_CANDIDATE_SAMPLE_FEATURES,
    TOTALS_CORE_FEATURES,
    create_totals_aggregate_features,
    evaluate_totals_model_candidates,
    prepare_common_totals_sample,
)


def create_development_data() -> pd.DataFrame:
    """Create synthetic totals development games."""

    rows: list[dict[str, object]] = []

    for index in range(80):
        split_name = (
            "train"
            if index < 60
            else "validation"
        )

        home_plays = float(
            58 + index % 8
        )

        away_plays = float(
            57 + index % 7
        )

        home_points_scored = float(
            18 + index % 12
        )

        away_points_scored = float(
            17 + index % 10
        )

        home_points_allowed = float(
            16 + index % 11
        )

        away_points_allowed = float(
            19 + index % 9
        )

        home_offense = (
            -0.10 + index * 0.004
        )

        away_offense = (
            0.08 - index * 0.002
        )

        home_defense = (
            -0.05 + (index % 7) * 0.02
        )

        away_defense = (
            0.06 - (index % 5) * 0.015
        )

        home_qb = (
            -2.0 + index * 0.08
        )

        away_qb = (
            3.0 - index * 0.03
        )

        home_success = (
            0.40 + (index % 6) * 0.01
        )

        away_success = (
            0.42 + (index % 4) * 0.01
        )

        home_defensive_success = (
            0.43 + (index % 3) * 0.01
        )

        away_defensive_success = (
            0.44 + (index % 5) * 0.008
        )

        home_explosive = (
            0.10 + (index % 4) * 0.01
        )

        away_explosive = (
            0.11 + (index % 3) * 0.01
        )

        home_explosive_allowed = (
            0.12 + (index % 5) * 0.008
        )

        away_explosive_allowed = (
            0.10 + (index % 6) * 0.007
        )

        is_indoor = (
            index % 4 == 0
        )

        has_game_weather = (
            not is_indoor
        )

        cold_degrees = float(
            15
            if index % 7 == 0
            and has_game_weather
            else 0
        )

        heat_degrees = float(
            10
            if index % 11 == 0
            and has_game_weather
            else 0
        )

        wind_above_10 = float(
            8
            if index % 6 == 0
            and has_game_weather
            else 0
        )

        is_freezing = (
            cold_degrees >= 18
        )

        is_high_wind = (
            wind_above_10 >= 5
        )

        is_extreme_heat = (
            heat_degrees >= 5
        )

        league_average_total_32 = (
            45.0 + 0.04 * index
        )

        league_average_total_64 = (
            45.5 + 0.03 * index
        )

        league_average_total_128 = (
            46.0 + 0.02 * index
        )

        target_total = (
            44.0
            + 12.0 * (
                home_offense
                + away_offense
            )
            + 8.0 * (
                home_defense
                + away_defense
            )
            + 0.6 * (
                home_qb
                + away_qb
            )
            + 15.0 * (
                home_success
                + away_success
            )
            + 0.10 * (
                home_plays
                + away_plays
            )
            + 0.35 * (
                home_points_scored
                + away_points_scored
                + home_points_allowed
                + away_points_allowed
            )
            + (
                2.0
                if is_indoor
                else 0.0
            )
            - 0.12 * cold_degrees
            - 0.18 * wind_above_10
            - 0.05 * heat_degrees
        )

        rows.append(
            {
                "game_id": f"game_{index}",
                "season": (
                    2022
                    if split_name == "train"
                    else 2023
                ),
                "split_name": split_name,
                "both_short_windows_complete": True,
                "target_total_points": target_total,
                "home_offensive_plays_last_4": (
                    home_plays
                ),
                "away_offensive_plays_last_4": (
                    away_plays
                ),
                "home_points_scored_last_4": (
                    home_points_scored
                ),
                "away_points_scored_last_4": (
                    away_points_scored
                ),
                "home_points_allowed_last_4": (
                    home_points_allowed
                ),
                "away_points_allowed_last_4": (
                    away_points_allowed
                ),
                "home_offensive_epa_per_play_last_4": (
                    home_offense
                ),
                "away_offensive_epa_per_play_last_4": (
                    away_offense
                ),
                "home_defensive_epa_allowed_per_play_last_4": (
                    home_defense
                ),
                "away_defensive_epa_allowed_per_play_last_4": (
                    away_defense
                ),
                "home_listed_qb_rating": home_qb,
                "away_listed_qb_rating": away_qb,
                "home_success_rate_last_4": (
                    home_success
                ),
                "away_success_rate_last_4": (
                    away_success
                ),
                "home_defensive_success_rate_allowed_last_4": (
                    home_defensive_success
                ),
                "away_defensive_success_rate_allowed_last_4": (
                    away_defensive_success
                ),
                "home_explosive_play_rate_last_4": (
                    home_explosive
                ),
                "away_explosive_play_rate_last_4": (
                    away_explosive
                ),
                "home_explosive_play_rate_allowed_last_4": (
                    home_explosive_allowed
                ),
                "away_explosive_play_rate_allowed_last_4": (
                    away_explosive_allowed
                ),
                "is_indoor": is_indoor,
                "has_game_weather": (
                    has_game_weather
                ),
                "cold_degrees_below_50": (
                    cold_degrees
                ),
                "heat_degrees_above_80": (
                    heat_degrees
                ),
                "wind_mph_above_10": (
                    wind_above_10
                ),
                "is_freezing": is_freezing,
                "is_high_wind": is_high_wind,
                "is_extreme_heat": (
                    is_extreme_heat
                ),
                "league_average_total_last_32": (
                    league_average_total_32
                ),
                "league_average_total_last_64": (
                    league_average_total_64
                ),
                "league_average_total_last_128": (
                    league_average_total_128
                ),
            }
        )

    incomplete = {
        **rows[-1],
        "game_id": "incomplete_game",
        "home_listed_qb_rating": np.nan,
    }

    holdout = {
        **rows[-1],
        "game_id": "holdout_game",
        "split_name": "holdout",
    }

    rows.extend(
        [
            incomplete,
            holdout,
        ]
    )

    return pd.DataFrame(rows)


def test_create_aggregate_features() -> None:
    """Create symmetric sums from home and away values."""

    source = create_development_data().iloc[
        [0]
    ]

    features = create_totals_aggregate_features(
        source
    )

    assert (
        features.iloc[0][
            "offensive_epa_sum_last_4"
        ]
        == pytest.approx(-0.02)
    )

    assert (
        features.iloc[0][
            "listed_qb_rating_sum"
        ]
        == pytest.approx(1.0)
    )
    assert (
        features.iloc[0][
            "offensive_plays_sum_last_4"
        ]
        == pytest.approx(115.0)
    )

    assert (
        features.iloc[0][
            "points_scored_sum_last_4"
        ]
        == pytest.approx(35.0)
    )

    assert (
        features.iloc[0][
            "points_allowed_sum_last_4"
        ]
        == pytest.approx(35.0)
    )


def test_prepare_common_sample() -> None:
    """Use complete train and validation games only."""

    sample = prepare_common_totals_sample(
        create_development_data()
    )

    assert len(sample) == 80

    assert set(
        sample["split_name"]
    ) == {
        "train",
        "validation",
    }

    assert sample[
        list(TOTALS_CANDIDATE_SAMPLE_FEATURES)
    ].notna().all().all()


def test_evaluate_expected_candidates() -> None:
    """Evaluate every candidate on identical games."""

    results = evaluate_totals_model_candidates(
        create_development_data()
    )

    assert tuple(
        results.columns
    ) == RESULT_COLUMNS

    assert set(
        results["candidate_name"]
    ) == {
        "constant_train_mean",
        "ridge_epa",
        "ridge_epa_indoor",
        "ridge_epa_weather_continuous",
        "ridge_epa_weather_extremes",
        "ridge_epa_weather_continuous_qb",
        (
            "ridge_epa_weather_continuous_qb_"
            "success_explosive"
        ),
        "ridge_epa_weather_qb_league_32",
        "ridge_epa_weather_qb_league_64",
        "ridge_epa_weather_qb_league_128",
    }

    assert set(
        results["train_game_count"]
    ) == {
        60,
    }

    assert set(
        results["validation_game_count"]
    ) == {
        20,
    }


def test_signal_model_beats_constant() -> None:
    """Recover the synthetic totals signal."""

    results = evaluate_totals_model_candidates(
        create_development_data()
    ).set_index(
        "candidate_name"
    )

    assert (
        results.loc[
            "ridge_epa_weather_continuous",
            "validation_mae",
        ]
        < results.loc[
            "constant_train_mean",
            "validation_mae",
        ]
    )


def test_team_order_does_not_change_aggregates(
) -> None:
    """Preserve totals features when teams are swapped."""

    original = create_development_data().iloc[
        [0]
    ].copy()

    swapped = original.copy()

    swap_pairs = (
        (
            "home_offensive_plays_last_4",
            "away_offensive_plays_last_4",
        ),
        (
            "home_points_scored_last_4",
            "away_points_scored_last_4",
        ),
        (
            "home_points_allowed_last_4",
            "away_points_allowed_last_4",
        ),
        (
            "home_offensive_epa_per_play_last_4",
            "away_offensive_epa_per_play_last_4",
        ),
        (
            "home_defensive_epa_allowed_per_play_last_4",
            "away_defensive_epa_allowed_per_play_last_4",
        ),
        (
            "home_listed_qb_rating",
            "away_listed_qb_rating",
        ),
        (
            "home_success_rate_last_4",
            "away_success_rate_last_4",
        ),
        (
            "home_defensive_success_rate_allowed_last_4",
            "away_defensive_success_rate_allowed_last_4",
        ),
        (
            "home_explosive_play_rate_last_4",
            "away_explosive_play_rate_last_4",
        ),
        (
            "home_explosive_play_rate_allowed_last_4",
            "away_explosive_play_rate_allowed_last_4",
        ),
    )

    for home_column, away_column in swap_pairs:
        home_value = original.iloc[0][
            home_column
        ]

        away_value = original.iloc[0][
            away_column
        ]

        swapped.loc[
            swapped.index[0],
            home_column,
        ] = away_value

        swapped.loc[
            swapped.index[0],
            away_column,
        ] = home_value

    original_features = (
        create_totals_aggregate_features(
            original
        )
    )

    swapped_features = (
        create_totals_aggregate_features(
            swapped
        )
    )

    assert np.allclose(
        original_features[
            list(TOTALS_CORE_FEATURES)
        ].to_numpy(
            dtype=float
        ),
        swapped_features[
            list(TOTALS_CORE_FEATURES)
        ].to_numpy(
            dtype=float
        ),
    )


def test_missing_source_column_is_rejected() -> None:
    """Reject an incomplete totals source schema."""

    data = create_development_data().drop(
        columns=[
            "home_success_rate_last_4",
        ]
    )

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        create_totals_aggregate_features(
            data
        )