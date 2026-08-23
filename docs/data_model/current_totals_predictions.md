# Current Production Totals Predictions

**Status:** Production  
**Model version:** 0.1.0  
**Last Updated:** 2026-08-08

---

## Purpose

The production totals layer estimates the expected combined score for current NFL games.

The output target is:

`target_total_points = home_score + away_score`

The production system uses an explicit primary/fallback routing policy so every upcoming game remains covered without silently imputing unavailable rolling or listed-QB features.

---

## Development Protocol

The model was developed with chronological evaluation rather than random train-test splitting.

Development periods:

- train: 2018–2022;
- validation: 2023–2024;
- final untouched holdout: 2025.

Model selection also used expanding-window validation seasons:

- 2021;
- 2022;
- 2023;
- 2024.

Only information available before each game was used.

---

## Weather and Venue Features

The normalized weather layer is stored in:

`analytics.game_weather_features`

It provides:

- normalized roof and surface values;
- indoor and weather-exposed indicators;
- observed-weather availability;
- temperature and wind;
- freezing, high-wind and extreme-heat indicators;
- continuous cold, heat and wind severity.

Indoor games receive neutral modeled outdoor-weather values while retaining an explicit indoor indicator.

Unknown roof values remain represented rather than causing schedule rows to be dropped.

---

## League Scoring Environment

The leakage-safe scoring-environment layer is stored in:

`analytics.game_scoring_environment_features`

It calculates league scoring history using completed games with:

`history.gameday < target.gameday`

Same-day and future games are excluded.

Available windows:

- previous 32 games;
- previous 64 games;
- previous 128 games.

The 64-game league average was selected for production after validation and expanding-window comparison.

---

## Primary Model

Model name:

`ridge_epa_weather_qb_league_64_totals`

Features:

1. home-plus-away offensive EPA per play over the previous four team games;
2. home-plus-away defensive EPA allowed per play over the previous four team games;
3. indoor indicator;
4. observed-weather availability;
5. cold degrees below 50°F;
6. heat degrees above 80°F;
7. wind mph above 10;
8. home-plus-away listed-QB rating;
9. league average total over the previous 64 completed games.

Model:

- `StandardScaler`;
- Ridge regression;
- `alpha = 100`.

Primary routing requires:

- complete four-game rolling windows for both teams;
- complete listed-QB ratings;
- all locked model features.

Partial one-to-three-game rolling windows are never treated as complete primary inputs.

---

## Primary Validation

Four-fold expanding-window result:

- validation games: 847;
- pooled MAE: `10.557334`;
- pooled RMSE: `13.314582`;
- pooled bias: `0.786514`;
- pooled R-squared: `0.063321`;
- constant-baseline MAE: `10.898283`;
- MAE improvement: approximately `3.13%`.

The 64-game and 128-game scoring windows were compared with paired game-level errors.

The 128-minus-64 MAE delta was:

`0.016874`

Bootstrap 95% interval:

`[-0.024964, 0.058777]`

The interval contains zero, so the windows were practically tied. The 64-game window was selected because it had lower pooled MAE, lower bias, lower fold variance and faster response to current league scoring conditions.

---

## Final 2025 Holdout

The frozen primary model was evaluated once on the untouched 2025 holdout.

Results:

- training games: 1,455;
- holdout games: 215;
- MAE: `10.733582`;
- RMSE: `13.665821`;
- bias: `0.327409`;
- R-squared: `0.034299`;
- constant-baseline MAE: `11.040921`;
- MAE improvement: `0.307339`;
- relative MAE improvement: `2.783633%`.

The model specification was not changed after the holdout result.

An initial implementation-validation run revealed that partial rolling windows were not explicitly excluded. The evaluator was corrected to enforce the already selected complete-four-game protocol. No feature, window or model parameter changed.

---

## Fallback Model

Model name:

`ridge_league_64_indoor_elo_totals`

Features:

1. league average total over the previous 64 games;
2. indoor indicator;
3. current home-plus-away Elo rating.

Model:

- `StandardScaler`;
- Ridge regression;
- `alpha = 1`.

The fallback does not require:

- season rolling history;
- listed-QB ratings;
- observed game-time weather.

This makes it available during preseason and the opening weeks of the season.

---

## Fallback Validation

Four-fold expanding-window result:

- validation games: 1,139;
- pooled MAE: `10.727000`;
- pooled RMSE: `13.436813`;
- pooled bias: `1.000967`;
- pooled R-squared: `0.028590`;
- constant-baseline MAE: `10.956750`.

Alpha values `0`, `0.1` and `1` were practically identical. Ridge `alpha = 1` was selected to preserve regularization and numerical stability.

---

## Current Production Output

Production table:

`analytics.current_game_total_predictions`

Current preseason coverage:

- upcoming games: 272;
- primary predictions: 0;
- fallback predictions: 272.

The current preseason expected-total range is approximately:

- minimum: `40.21`;
- average: `45.36`;
- maximum: `52.41`.

Primary routing can activate during the season when both teams have complete four-game history and listed-QB inputs are available.

---

## Output Fields

The production table includes:

- game and schedule metadata;
- model name and version;
- prediction mode and routing reason;
- Ridge alpha;
- primary and fallback training counts;
- rolling and QB availability flags;
- primary aggregate features;
- fallback Elo aggregate;
- venue and weather context;
- 64-game league scoring environment;
- predicted total points;
- generation timestamp.

---

## Quality Controls

Independent SQL validation:

`sql/031_current_totals_predictions_quality_checks.sql`

The checks cover:

- row-count parity with probability predictions;
- duplicate game identifiers;
- cross-output game coverage;
- schedule metadata consistency;
- finite predictions;
- primary and fallback routing;
- model names, versions and alpha values;
- current Elo-sum reconstruction;
- primary and fallback training counts;
- required production metadata.

The quality-check suite currently contains 13 passing checks.

---

## Pipeline Integration

The complete modeling pipeline contains 30 ordered steps.

Totals dependencies are rebuilt in this order:

1. team-game efficiency;
2. rolling team features;
3. current Elo and QB ratings;
4. weather features;
5. league scoring environment;
6. game modeling dataset;
7. modeling splits;
8. current probability predictions;
9. current spread predictions;
10. current totals predictions;
11. season simulation.

The generated DuckDB file is local and is not committed.