# In-Season Refresh Runbook

**Status:** Operational first generation  
**Last Updated:** 2026-08-14

## Purpose

The refresh runner rebuilds current models and simulations, processes one explicitly selected odds source, refreshes all EV products and archives the resulting future market board without rewriting prior snapshots.

## Safe Offline Run

Use an existing Odds API JSON snapshot without consuming API credit:

```powershell
python -m src.pipeline.run_in_season_refresh --snapshot data/raw/odds/nfl_odds_YYYYMMDDTHHMMSSZ.json
```

## Explicit Online Run

Download a new snapshot and consume Odds API credit only when intentionally requested:

```powershell
python -m src.pipeline.run_in_season_refresh --online
```

## Scheduled Production Run

`.github/workflows/weekly-production-refresh.yml` runs the complete production
orchestrator at these local times using the `Europe/Budapest` timezone:

- Tuesday 08:00;
- Thursday 15:00;
- Sunday 15:00.

It can also be started manually with `workflow_dispatch`. A fresh hosted runner
first restores `operational.duckdb` from the latest private data Release, then
restores the ignored schedule, player-directory, depth-chart, injury-report and
completed-season snap-count sources required by the loaders. It runs modeling,
downloads one new Moneyline/Spread/Total odds
snapshot, rebuilds EV and simulation outputs, executes the regression suite and
publishes a new atomic private Release.

Required Actions secrets:

- `ODDS_API_KEY`;
- `NFL_ANALYTICS_DASHBOARD_PUBLISH_TOKEN` with Contents read/write access to
  `fefemu/nfl-analytics-platform-data`.

The first scheduled run requires a seeded latest Release containing both
`dashboard.duckdb` and `operational.duckdb`. A failed run uploads its log as a
short-lived Actions artifact and does not replace the previous latest Release.

There is no implicit default mode. One of `--snapshot` or `--online` is required.

## Execution Order

1. rebuild the 33-step modeling and simulation pipeline;
2. process the online or offline odds snapshot;
3. rebuild Moneyline, Spread and Totals EV;
4. rebuild the combined betting board;
5. archive every future board row;
6. mark positive-EV rows as forward tip candidates;
7. refresh prospective CLV comparisons;
8. finalize the refresh audit row.

## Audit and Recovery

`analytics.refresh_run_history` records RUNNING, SUCCESS or FAILED for every attempted refresh. A failed run retains its error message and never receives a success marker. Reusing the same snapshot is idempotent in the forward archive.

Disconnect DBeaver and other DuckDB writers before running. The generated DuckDB remains local and must not be committed.

## Forward-Test Outputs

- `analytics.forward_betting_board_archive`: immutable pregame market snapshots;
- `analytics.forward_tip_clv`: positive-EV candidates with their latest later pregame comparison;
- `analytics.refresh_run_history`: operational run audit.

Only rows whose `commence_time` is later than the odds fetch, prediction generation,
betting-board generation and final archive/lock timestamp are archived. A stale
pregame odds file processed after kickoff is therefore not eligible for the forward
test. Reusing an archive identity is idempotent only when all locked payload values
are identical; conflicting values fail the refresh instead of silently replacing or
ignoring the original observation. Historical OOF selections remain internal and
are not mixed with the forward archive.

Run quality checks with:

```powershell
python -m src.utils.run_sql sql/036_forward_refresh_quality_checks.sql
```
