"""Tests for production probability routing."""

import pytest

from src.modeling.production_probability_predictions import (
    BLEND_PREDICTION_MODE,
    COMPLETE_MODEL_FEATURES_REASON,
    ELO_FALLBACK_PREDICTION_MODE,
    EXTERNAL_ELO_QB_FALLBACK_PREDICTION_MODE,
    EXTERNAL_NFELO_BLEND_PREDICTION_MODE,
    INCOMPLETE_MODEL_FEATURES_REASON,
    create_production_probability_prediction,
    validate_probability,
)


def test_complete_features_apply_external_blend(
) -> None:
    """Blend primary logistic and published nfelo."""

    prediction = (
        create_production_probability_prediction(
            published_nfelo_home_probability=0.60,
            primary_logistic_home_win_probability=0.70,
            fallback_logistic_home_win_probability=None,
            has_complete_primary_features=True,
        )
    )

    assert (
        prediction.model_name
        == "external_nfelo_probability_routing"
    )

    assert prediction.model_version == "0.3.0"

    assert (
        prediction.prediction_mode
        == EXTERNAL_NFELO_BLEND_PREDICTION_MODE
    )

    assert (
        prediction.prediction_mode
        == BLEND_PREDICTION_MODE
    )

    assert (
        prediction.prediction_mode_reason
        == COMPLETE_MODEL_FEATURES_REASON
    )

    assert (
        prediction.home_win_probability
        == pytest.approx(0.67)
    )

    assert (
        prediction.away_win_probability
        == pytest.approx(0.33)
    )

    assert (
        prediction.published_nfelo_home_probability
        == pytest.approx(0.60)
    )

    assert (
        prediction
        .primary_logistic_home_win_probability
        == pytest.approx(0.70)
    )

    assert (
        prediction
        .fallback_logistic_home_win_probability
        is None
    )

    assert (
        prediction
        .applied_primary_logistic_weight
        == pytest.approx(0.70)
    )

    assert (
        prediction
        .applied_published_nfelo_weight
        == pytest.approx(0.30)
    )


def test_incomplete_features_use_external_fallback(
) -> None:
    """Use the external Elo-QB logistic fallback."""

    prediction = (
        create_production_probability_prediction(
            published_nfelo_home_probability=0.58,
            primary_logistic_home_win_probability=None,
            fallback_logistic_home_win_probability=0.63,
            has_complete_primary_features=False,
        )
    )

    assert (
        prediction.prediction_mode
        == (
            EXTERNAL_ELO_QB_FALLBACK_PREDICTION_MODE
        )
    )

    assert (
        prediction.prediction_mode
        == ELO_FALLBACK_PREDICTION_MODE
    )

    assert (
        prediction.prediction_mode_reason
        == INCOMPLETE_MODEL_FEATURES_REASON
    )

    assert (
        prediction.home_win_probability
        == pytest.approx(0.63)
    )

    assert (
        prediction.away_win_probability
        == pytest.approx(0.37)
    )

    assert (
        prediction
        .primary_logistic_home_win_probability
        is None
    )

    assert (
        prediction
        .fallback_logistic_home_win_probability
        == pytest.approx(0.63)
    )

    assert (
        prediction
        .applied_primary_logistic_weight
        == pytest.approx(0.0)
    )

    assert (
        prediction
        .applied_published_nfelo_weight
        == pytest.approx(0.0)
    )


def test_fallback_allows_missing_published_probability(
) -> None:
    """Route future games without exact nfelo output."""

    prediction = (
        create_production_probability_prediction(
            published_nfelo_home_probability=None,
            primary_logistic_home_win_probability=None,
            fallback_logistic_home_win_probability=0.57,
            has_complete_primary_features=False,
        )
    )

    assert (
        prediction.home_win_probability
        == pytest.approx(0.57)
    )
    assert (
        prediction.published_nfelo_home_probability
        is None
    )


def test_fallback_ignores_primary_probability(
) -> None:
    """Ignore a partial primary output in fallback mode."""

    prediction = (
        create_production_probability_prediction(
            published_nfelo_home_probability=0.55,
            primary_logistic_home_win_probability=0.90,
            fallback_logistic_home_win_probability=0.61,
            has_complete_primary_features=False,
        )
    )

    assert (
        prediction.home_win_probability
        == pytest.approx(0.61)
    )

    assert (
        prediction
        .primary_logistic_home_win_probability
        is None
    )

    assert (
        prediction
        .fallback_logistic_home_win_probability
        == pytest.approx(0.61)
    )


def test_primary_requires_primary_probability(
) -> None:
    """Reject a missing primary logistic output."""

    with pytest.raises(
        ValueError,
        match=(
            "Complete primary features require"
        ),
    ):
        create_production_probability_prediction(
            published_nfelo_home_probability=0.60,
            primary_logistic_home_win_probability=None,
            fallback_logistic_home_win_probability=0.62,
            has_complete_primary_features=True,
        )


def test_fallback_requires_fallback_probability(
) -> None:
    """Reject a missing fallback logistic output."""

    with pytest.raises(
        ValueError,
        match=(
            "Incomplete primary features require"
        ),
    ):
        create_production_probability_prediction(
            published_nfelo_home_probability=0.60,
            primary_logistic_home_win_probability=None,
            fallback_logistic_home_win_probability=None,
            has_complete_primary_features=False,
        )


def test_transitional_audit_aliases(
) -> None:
    """Keep legacy audit properties during migration."""

    prediction = (
        create_production_probability_prediction(
            published_nfelo_home_probability=0.60,
            primary_logistic_home_win_probability=0.70,
            fallback_logistic_home_win_probability=None,
            has_complete_primary_features=True,
        )
    )

    assert (
        prediction.elo_home_win_probability
        == pytest.approx(0.60)
    )

    assert (
        prediction.logistic_home_win_probability
        == pytest.approx(0.70)
    )

    assert (
        prediction.applied_logistic_weight
        == pytest.approx(0.70)
    )

    assert (
        prediction.applied_elo_weight
        == pytest.approx(0.30)
    )


@pytest.mark.parametrize(
    "invalid_probability",
    (
        -0.01,
        1.01,
        float("inf"),
        float("-inf"),
        float("nan"),
    ),
)
def test_validate_probability_rejects_invalid_values(
    invalid_probability: float,
) -> None:
    """Reject out-of-range or non-finite values."""

    with pytest.raises(
        ValueError,
    ):
        validate_probability(
            probability=invalid_probability,
            probability_name="Test probability",
        )


@pytest.mark.parametrize(
    "valid_probability",
    (
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ),
)
def test_validate_probability_accepts_valid_values(
    valid_probability: float,
) -> None:
    """Accept valid probability boundaries."""

    assert (
        validate_probability(
            probability=valid_probability,
            probability_name="Test probability",
        )
        == pytest.approx(
            valid_probability
        )
    )
