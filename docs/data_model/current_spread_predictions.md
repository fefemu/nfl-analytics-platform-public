# Current Production Spread Predictions

**Model version:** 0.2.0
**Status:** Selected for the 2026 forward test
**Last updated:** 2026-08-09

---

## Purpose

The Spread layer estimates expected final scoring margin from the home-team perspective:

`target_point_differential = home score - away score`

Positive values favor the home team; negative values favor the away team.

## Selected Model

The locked model is `external_nfelo_external_qb_spread`, a standardized Ridge regression with `alpha = 10` and two features:

- external nfelo rating difference;
- external nfelo QB-adjustment difference.

Its production routing mode is `EXTERNAL_NFELO_QB_RIDGE`. Both features are available from the latest external team snapshot for every current game. Listed-QB availability remains audit metadata and no longer changes routing.

## Holdout Evidence

On the corrected locked 2025 holdout of 285 games:

- external model MAE: `9.858143`;
- previous production routing MAE: `10.132231`;
- paired MAE delta: `-0.274088`;
- bootstrap 95% interval: `[-0.509856, -0.037845]`.

## Outputs

Predictions are stored in:

`analytics.current_game_spread_predictions`

The table includes model/version metadata, external features, listed-QB audit fields, training counts, opposite home/away margins, predicted winner and generation timestamp.

Spread cover probabilities and expected values use 1,139 chronological out-of-sample residuals from the same locked external model. Current offers are stored in:

`analytics.current_spread_value`

## Validation

Run:

```powershell
python -m src.utils.run_sql sql/030_current_spread_predictions_quality_checks.sql
python -m src.betting.calibrate_spread_cover_probabilities
```

The model is frozen for prospective 2026 monitoring. Future promotion requires the same chronological and protected-holdout governance.
