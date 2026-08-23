# Current Production Predictions

**Model version:** 0.3.0
**Status:** Selected for the 2026 forward test
**Last updated:** 2026-08-09

---

## Purpose

This layer produces one auditable home-win probability for every current NFL game.

## Selected Routing

`EXTERNAL_NFELO_BLEND` is used when the exact external game row, listed-QB ratings and complete injury features are available:

`final probability = 0.70 × primary logistic + 0.30 × published nfelo probability`

The primary logistic model uses:

- external nfelo rating difference;
- listed-QB rating difference;
- external nfelo QB-adjustment difference;
- offense, defense and special-teams injury-burden differences.

`EXTERNAL_ELO_QB_FALLBACK` is used when the primary feature set is incomplete. It is a separately validated logistic model using external nfelo rating and QB-adjustment differences. It is not the former internal-Elo fallback.

During the current preseason build, all 272 games use fallback because complete weekly primary inputs are not yet available. This is expected routing, not imputation.

## External Source Policy

`processed.external_nfelo_game_ratings` is refreshed with retry handling. Historical coverage is 2,227 of 2,227 modeled games. Latest team snapshots cover the full current schedule; exact published game probabilities are used only when an exact external game row exists.

The external candidate beat the previous internal production routing on the locked 2025 holdout:

- Brier score: `0.224984` versus `0.232630`;
- paired Brier delta: `-0.007646`.

## Outputs

Primary tables:

- `analytics.current_game_predictions`;
- `analytics.current_game_prediction_explanations`;
- `analytics.current_game_logistic_feature_contributions`;
- `analytics.current_game_prediction_narratives`;
- `analytics.current_game_prediction_data_science_view`.

The prediction and explanation tables retain transitional audit aliases while exposing the external component probabilities, weights, features, routing mode and reason directly.

## Validation

Run:

```powershell
python -m src.utils.run_sql sql/018_current_game_predictions_quality_checks.sql
python -m src.utils.run_sql sql/019_prediction_explanations_quality_checks.sql
python -m src.utils.run_sql sql/026_logistic_feature_contributions_quality_checks.sql
python -m src.utils.run_sql sql/027_prediction_narratives_quality_checks.sql
python -m src.utils.run_sql sql/028_prediction_data_science_view_quality_checks.sql
```

The production model remains frozen for prospective 2026 monitoring. A replacement requires leakage-safe chronological validation and a protected holdout decision.
