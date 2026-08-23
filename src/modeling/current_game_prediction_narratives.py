"""
NFL Analytics Platform
Current Game Prediction Narratives

Purpose:
    Create accessible English and Hungarian game
    explanations from production probabilities, routing
    metadata and exact logistic feature contributions.

Terminology policy:
    Stable technical terms such as Elo, logistic, blend,
    fallback, feature and model identifier remain
    untranslated.

Author:
    Ferenc Kaizer

Version:
    0.1.0
"""

from math import isfinite

import pandas as pd

from src.modeling.production_probability_predictions import (
    BLEND_PREDICTION_MODE,
    ELO_FALLBACK_PREDICTION_MODE,
)


NARRATIVE_VERSION = "1.0.0"

REQUIRED_EXPLANATION_COLUMNS = {
    "game_id",
    "home_team",
    "away_team",
    "favorite",
    "underdog",
    "favorite_win_probability",
    "home_win_probability",
    "away_win_probability",
    "published_nfelo_home_probability",
    "primary_logistic_home_win_probability",
    "fallback_logistic_home_win_probability",
    "applied_primary_logistic_weight",
    "applied_published_nfelo_weight",
    "prediction_mode",
    "has_complete_injury_data",
    "both_listed_qb_ratings_available",
    "matchup_label",
    "model_name",
    "model_version",
    "prediction_generated_at",
}

REQUIRED_CONTRIBUTION_COLUMNS = {
    "game_id",
    "feature_name",
    "log_odds_contribution",
    "contribution_rank",
}

NARRATIVE_COLUMNS = (
    "game_id",
    "narrative_version",
    "model_name",
    "model_version",
    "prediction_generated_at",
    "headline_en",
    "headline_hu",
    "summary_en",
    "summary_hu",
    "model_context_en",
    "model_context_hu",
    "top_factor_feature",
    "top_factor_direction",
    "top_factor_en",
    "top_factor_hu",
)

MATCHUP_LABELS = {
    "toss_up": {
        "en": "Toss-up",
        "hu": "Kiélezett meccs",
    },
    "slight_edge": {
        "en": "Slight edge",
        "hu": "Enyhe előny",
    },
    "clear_edge": {
        "en": "Clear edge",
        "hu": "Egyértelmű előny",
    },
    "strong_edge": {
        "en": "Strong edge",
        "hu": "Erős előny",
    },
}


def validate_required_columns(
    data: pd.DataFrame,
    required_columns: set[str],
    data_name: str,
) -> None:
    """Validate required narrative input columns."""

    missing_columns = sorted(
        required_columns
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{data_name} is missing columns: "
            + ", ".join(missing_columns)
        )


def format_percentage(
    probability: float,
    language: str,
) -> str:
    """Format one probability for English or Hungarian."""

    numeric_probability = float(
        probability
    )

    if (
        not isfinite(numeric_probability)
        or not 0.0 <= numeric_probability <= 1.0
    ):
        raise ValueError(
            "Narrative probability must be finite "
            "and between zero and one."
        )

    formatted = (
        f"{numeric_probability * 100.0:.1f}%"
    )

    if language == "en":
        return formatted

    if language == "hu":
        return formatted.replace(
            ".",
            ",",
        )

    raise ValueError(
        f"Unsupported narrative language: {language}"
    )


def get_favorite_probability(
    home_probability: float,
    favorite: str,
    home_team: str,
) -> float:
    """Convert a home probability to favorite perspective."""

    if favorite == home_team:
        return float(home_probability)

    return 1.0 - float(home_probability)


def describe_missing_inputs(
    has_complete_injury_data: bool,
    both_listed_qb_ratings_available: bool,
) -> tuple[str, str]:
    """Describe why the external fallback is active."""

    injury_missing = not bool(
        has_complete_injury_data
    )
    qb_missing = not bool(
        both_listed_qb_ratings_available
    )

    if injury_missing and qb_missing:
        return (
            "complete current injury data and listed QB ratings "
            "are unavailable",
            "nem áll rendelkezésre teljes aktuális injury adat "
            "és mindkét listed QB rating",
        )

    if injury_missing:
        return (
            "complete current injury data is unavailable",
            "nem áll rendelkezésre teljes aktuális injury adat",
        )

    if qb_missing:
        return (
            "both listed QB ratings are unavailable",
            "nem áll rendelkezésre mindkét listed QB rating",
        )

    return (
        "an exact current nfelo game probability is not yet available",
        "még nem áll rendelkezésre exact aktuális nfelo game probability",
    )


def describe_top_factor(
    contribution: pd.Series,
    favorite: str,
    home_team: str,
) -> tuple[str, str, str]:
    """Describe whether the strongest feature supports favorite."""

    feature_name = str(
        contribution["feature_name"]
    )

    contribution_value = float(
        contribution["log_odds_contribution"]
    )

    supports_home = (
        contribution_value > 0.0
    )

    favorite_is_home = (
        favorite == home_team
    )

    supports_favorite = (
        supports_home
        == favorite_is_home
    )

    if supports_favorite:
        return (
            "supports_favorite",
            (
                f"{feature_name} is the strongest logistic "
                f"feature and supports {favorite}."
            ),
            (
                f"A legerősebb logistic feature a "
                f"{feature_name}, amely {favorite} felé "
                "mozdítja a becslést."
            ),
        )

    return (
        "opposes_favorite",
        (
            f"{feature_name} is the strongest logistic "
            f"feature but works against {favorite}."
        ),
        (
            f"A legerősebb logistic feature a "
            f"{feature_name}, de {favorite} ellenében "
            "mozdítja a becslést."
        ),
    )


def create_current_game_prediction_narratives(
    explanations: pd.DataFrame,
    feature_contributions: pd.DataFrame,
) -> pd.DataFrame:
    """Create accessible EN/HU game narratives."""

    validate_required_columns(
        data=explanations,
        required_columns=(
            REQUIRED_EXPLANATION_COLUMNS
        ),
        data_name="Production explanations",
    )

    validate_required_columns(
        data=feature_contributions,
        required_columns=(
            REQUIRED_CONTRIBUTION_COLUMNS
        ),
        data_name="Logistic feature contributions",
    )

    if explanations[
        "game_id"
    ].duplicated().any():
        raise ValueError(
            "Production explanations contain duplicate "
            "game identifiers."
        )

    contribution_games = set(
        feature_contributions["game_id"]
    )

    explanation_games = set(
        explanations["game_id"]
    )

    orphan_games = sorted(
        contribution_games
        - explanation_games
    )

    if orphan_games:
        raise ValueError(
            "Logistic feature contributions contain "
            "unknown game identifiers."
        )

    contribution_groups = {
        game_id: group.sort_values(
            by="contribution_rank",
            kind="stable",
        )
        for game_id, group
        in feature_contributions.groupby(
            "game_id",
            sort=False,
        )
    }

    narrative_rows: list[
        dict[str, object]
    ] = []

    for explanation in explanations.itertuples(
        index=False
    ):
        game_id = str(
            explanation.game_id
        )
        favorite = str(
            explanation.favorite
        )
        underdog = str(
            explanation.underdog
        )
        home_team = str(
            explanation.home_team
        )

        favorite_probability = float(
            explanation.favorite_win_probability
        )

        matchup_label = str(
            explanation.matchup_label
        )

        if matchup_label not in MATCHUP_LABELS:
            raise ValueError(
                "Unknown production matchup label: "
                f"{matchup_label}"
            )

        probability_en = format_percentage(
            favorite_probability,
            language="en",
        )
        probability_hu = format_percentage(
            favorite_probability,
            language="hu",
        )

        label_en = MATCHUP_LABELS[
            matchup_label
        ]["en"]
        label_hu = MATCHUP_LABELS[
            matchup_label
        ]["hu"]

        headline_en = (
            f"{favorite} win probability: "
            f"{probability_en} — {label_en}."
        )
        headline_hu = (
            f"{favorite} győzelmi valószínűsége: "
            f"{probability_hu} – {label_hu}."
        )

        prediction_mode = str(
            explanation.prediction_mode
        )

        top_factor_feature: (
            str | None
        ) = None
        top_factor_direction: (
            str | None
        ) = None
        top_factor_en: (
            str | None
        ) = None
        top_factor_hu: (
            str | None
        ) = None

        if prediction_mode == BLEND_PREDICTION_MODE:
            if game_id not in contribution_groups:
                raise RuntimeError(
                    "Blend prediction is missing logistic "
                    f"feature contributions for {game_id}."
                )

            logistic_home_probability = float(
                explanation
                .primary_logistic_home_win_probability
            )
            published_home_probability = float(
                explanation
                .published_nfelo_home_probability
            )

            logistic_favorite_probability = (
                get_favorite_probability(
                    home_probability=(
                        logistic_home_probability
                    ),
                    favorite=favorite,
                    home_team=home_team,
                )
            )

            published_favorite_probability = (
                get_favorite_probability(
                    home_probability=(
                        published_home_probability
                    ),
                    favorite=favorite,
                    home_team=home_team,
                )
            )

            logistic_en = format_percentage(
                logistic_favorite_probability,
                language="en",
            )
            logistic_hu = format_percentage(
                logistic_favorite_probability,
                language="hu",
            )
            published_en = format_percentage(
                published_favorite_probability,
                language="en",
            )
            published_hu = format_percentage(
                published_favorite_probability,
                language="hu",
            )

            logistic_weight = int(
                round(
                    float(
                        explanation
                        .applied_primary_logistic_weight
                    )
                    * 100.0
                )
            )
            published_weight = int(
                round(
                    float(
                        explanation
                        .applied_published_nfelo_weight
                    )
                    * 100.0
                )
            )

            summary_en = (
                f"{favorite} is favored over {underdog}. "
                f"The logistic component estimates "
                f"{logistic_en}, compared with "
                f"{published_en} from published nfelo."
            )
            summary_hu = (
                f"{favorite} a favorit {underdog} ellen. "
                f"A logistic component becslése "
                f"{logistic_hu}, a published nfelo "
                f"becslése pedig {published_hu}."
            )

            model_context_en = (
                f"The final probability uses a "
                f"{logistic_weight}% logistic and "
                f"{published_weight}% published nfelo blend."
            )
            model_context_hu = (
                f"A végső valószínűség "
                f"{logistic_weight}% logistic és "
                f"{published_weight}% published nfelo "
                "blendből készül."
            )

            top_factor = (
                contribution_groups[
                    game_id
                ].iloc[0]
            )

            (
                top_factor_direction,
                top_factor_en,
                top_factor_hu,
            ) = describe_top_factor(
                contribution=top_factor,
                favorite=favorite,
                home_team=home_team,
            )

            top_factor_feature = str(
                top_factor["feature_name"]
            )

        elif (
            prediction_mode
            == ELO_FALLBACK_PREDICTION_MODE
        ):
            if game_id in contribution_groups:
                raise RuntimeError(
                "External fallback must not have primary "
                    f"feature contributions for {game_id}."
                )

            (
                missing_reason_en,
                missing_reason_hu,
            ) = describe_missing_inputs(
                has_complete_injury_data=bool(
                    explanation
                    .has_complete_injury_data
                ),
                both_listed_qb_ratings_available=bool(
                    explanation
                    .both_listed_qb_ratings_available
                ),
            )

            summary_en = (
                f"{favorite} is favored over {underdog} "
                "using the external Elo-QB logistic "
                "fallback probability."
            )
            summary_hu = (
                f"{favorite} a favorit {underdog} ellen "
                "az external Elo-QB logistic fallback "
                "valószínűsége alapján."
            )

            model_context_en = (
                "External Elo-QB fallback is active because "
                f"{missing_reason_en}."
            )
            model_context_hu = (
                "External Elo-QB fallback aktív, mert "
                f"{missing_reason_hu}."
            )

        else:
            raise ValueError(
                "Unknown production prediction mode: "
                f"{prediction_mode}"
            )

        narrative_rows.append(
            {
                "game_id": game_id,
                "narrative_version": (
                    NARRATIVE_VERSION
                ),
                "model_name": str(
                    explanation.model_name
                ),
                "model_version": str(
                    explanation.model_version
                ),
                "prediction_generated_at": (
                    explanation
                    .prediction_generated_at
                ),
                "headline_en": headline_en,
                "headline_hu": headline_hu,
                "summary_en": summary_en,
                "summary_hu": summary_hu,
                "model_context_en": (
                    model_context_en
                ),
                "model_context_hu": (
                    model_context_hu
                ),
                "top_factor_feature": (
                    top_factor_feature
                ),
                "top_factor_direction": (
                    top_factor_direction
                ),
                "top_factor_en": top_factor_en,
                "top_factor_hu": top_factor_hu,
            }
        )

    return pd.DataFrame(
        narrative_rows,
        columns=NARRATIVE_COLUMNS,
    )
