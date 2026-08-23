# Current Model-Implied Team Scores

**Status:** Production  
**Last Updated:** 2026-08-09

## Purpose

`analytics.current_game_score_predictions` converts the production Spread and Totals predictions into an expected score for each team. It is a derived reporting product, not a separately trained model.

## Calculation

For predicted home margin `M` and predicted combined total `T`:

```text
implied_home_score = (T + M) / 2
implied_away_score = (T - M) / 2
```

Therefore the output always reconstructs both source predictions:

```text
implied_home_score + implied_away_score = predicted_total_points
implied_home_score - implied_away_score = predicted_home_margin
```

## Data Lineage

The build joins these production tables by `game_id`:

- `analytics.current_game_spread_predictions`;
- `analytics.current_game_total_predictions`.

Game metadata must match after normalizing database time representations. The output preserves the model name, version, routing mode and generation timestamp from both sources.

## Validation

The builder requires:

- one row per game in each source;
- identical game coverage and metadata;
- finite Spread and Totals predictions;
- nonnegative implied scores;
- exact reconstruction of margin and total within numerical tolerance;
- winner consistency with the predicted margin;
- complete source-model and timestamp metadata.

Run the reproducible checks with:

```powershell
python -m src.utils.run_sql sql/034_current_game_score_predictions_quality_checks.sql
```

## Interpretation

These values are expected scoring levels implied by two regression outputs. They are useful for matchup presentation, comparison and downstream analytics, but they are not claims that an NFL score can finish with fractional points.
