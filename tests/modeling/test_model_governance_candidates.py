"""Tests for frozen model governance candidates."""

from src.modeling.model_governance_candidates import (
    GOVERNANCE_CANDIDATES,
)


def test_governance_candidates_are_unique() -> None:
    """Require unique versioned candidate identities."""

    identities = [
        (
            candidate.model_name,
            candidate.model_version,
        )
        for candidate in GOVERNANCE_CANDIDATES
    ]

    assert len(identities) == len(
        set(identities)
    )


def test_governance_candidates_have_valid_features(
) -> None:
    """Require non-empty unique feature definitions."""

    for candidate in GOVERNANCE_CANDIDATES:
        assert candidate.feature_columns

        assert len(
            candidate.feature_columns
        ) == len(
            set(candidate.feature_columns)
        )

        assert candidate.regularization_c > 0.0


def test_injury_candidate_avoids_qb_double_counting(
) -> None:
    """Keep QB injury burden outside generic injury model."""

    injury_candidate = next(
        candidate
        for candidate in GOVERNANCE_CANDIDATES
        if candidate.model_name
        == "logistic_elo_qb_unit_burdens"
    )

    assert (
        "qb_injury_burden_difference"
        not in injury_candidate.feature_columns
    )

    assert (
        "qb_out_count_difference"
        not in injury_candidate.feature_columns
    )


def test_expected_governance_candidates_are_frozen(
) -> None:
    """Keep the intended champion-challenger set."""

    assert {
        candidate.model_name
        for candidate in GOVERNANCE_CANDIDATES
    } == {
        "logistic_elo_plus_qb",
        "logistic_elo_qb_post_bye",
        "logistic_elo_qb_unit_burdens",
        "logistic_full_core",
    }