# Current Betting Board

**Status:** Production
**Last updated:** 2026-08-09

---

## Purpose

The betting layer converts production predictions and current bookmaker prices into comparable expected-value offers for:

- Moneyline;
- Spread;
- Totals.

`analytics.current_betting_board` presents all three markets with a common schema so downstream dashboards can rank and filter opportunities without market-specific joins.

## Totals Expected Value

`analytics.current_totals_value` compares the production total with each paired Over/Under market line. Uncertainty comes from chronological out-of-sample residuals generated separately for `RIDGE_TOTALS_PRIMARY` and `RIDGE_TOTALS_FALLBACK`.

For every offer the layer estimates:

- win, push and loss probability;
- no-push win probability;
- model–market probability gap versus equal-weighted consensus no-vig probability;
- fair decimal odds;
- expected value per unit and percent;
- full Kelly fraction;
- positive-EV status.

Only market lines containing both Over and Under outcomes are eligible. Push probability is retained explicitly, and a push contributes zero profit or loss to expected value.

## Combined Board

The combined board standardizes:

- game and market metadata;
- best bookmaker and price;
- model identity and routing mode;
- model, push and loss probabilities;
- model–market probability gap (stored in legacy internal `probability_edge*` fields);
- fair odds;
- EV and Kelly sizing;
- generation timestamps.

It currently contains 1,808 offers across 272 games from the latest odds snapshot.

## Operational Flow

The odds pipeline now contains ten ordered steps. After building the current market board it refreshes Moneyline EV, Spread EV, Totals EV and finally the combined board.

Run:

```powershell
python -m src.pipeline.run_odds_pipeline
```

The command downloads a new odds snapshot and consumes Odds API credits. For local recalculation without downloading odds, run the four betting builders independently.

## Quality Checks

```powershell
python -m src.utils.run_sql sql/032_current_totals_value_quality_checks.sql
python -m src.utils.run_sql sql/033_current_betting_board_quality_checks.sql
```

EV is a model estimate, not a guaranteed return. Prospective ROI and CLV evaluation remain required before operational betting decisions.

## Market Consensus and Score Construction

Equivalent markets and matching lines are paired bookmaker by bookmaker. Margin is removed within each bookmaker first, and the resulting no-vig probabilities are combined as an equal-weighted consensus. The model–market probability gap compares the model with that benchmark. EV is calculated separately from model probability and the best available executable price.

The Total Ridge model directly predicts combined points and the Spread model predicts margin. Displayed team scores are algebraically derived from those two estimates; they are not independently predicted and added.
