# Historical Market Evaluation

**Status:** Production benchmark  
**Last Updated:** 2026-08-14

## Business Purpose

The historical market evaluation tests whether model-preferred selections would have produced useful results against a strong pregame reference: the closing market. It prevents current positive-EV rows from being treated as recommendations without historical evidence.

## Leakage Control

The evaluation uses expanding-window out-of-fold predictions for the 2021–2024 seasons. Every evaluated game is predicted by models trained only on earlier seasons. The locked 2025 holdout remains outside this development report.

Selected routing mirrors the validated production families:

- probability: external nfelo/QB/injury primary plus external Elo/QB fallback;
- Spread: external nfelo plus external QB Ridge;
- Totals: current locked primary plus early-season fallback.

## Outputs

- `analytics.historical_betting_ledger`: one model-preferred selection per game and market;
- `analytics.historical_betting_performance`: flat-stake results by market and edge bucket.

The ledger records selection, model and market values, edge, closing price, win/loss/push settlement, unit profit and opening-to-close movement where available.

## Pricing Limitations

Spread and Totals returns use recorded closing American prices. Historical Moneyline bookmaker prices are incomplete, so Moneyline uses a synthetic no-vig price derived from the closing market probability and is explicitly marked `SYNTHETIC_CLOSE_FAIR`.

The `clv_value` field is a retrospective opening-to-close market-movement diagnostic for the model-selected side. It is not prospective CLV because the selection is benchmarked at closing time. True CLV requires timestamped odds captured when a selection is actually generated; the automated refresh workflow will provide that data.

## Interpretation

ROI is calculated with one unit staked per selection. It does not include stake sizing, limits, transaction costs or execution constraints. Edge buckets are diagnostic and must not be promoted into betting rules merely because one historical bucket is profitable.

Run quality checks with:

```powershell
python -m src.utils.run_sql sql/035_historical_market_evaluation_quality_checks.sql
```
