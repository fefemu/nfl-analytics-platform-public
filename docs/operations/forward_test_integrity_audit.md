# 2026 Forward-Test Integrity Audit

**Audit date:** 2026-09-05

**Scope:** Forward betting-board archive, locked prediction and price fields,
production-state persistence, snapshot timing and post-kickoff boundaries.

## Findings and controls

### Append-only storage

`analytics.forward_betting_board_archive` inserts unseen archive identities and
never updates existing rows. The archive retains the odds snapshot, offered price,
bookmaker, model name/version/mode, probability, market line, prediction timestamp,
refresh run and candidate state used at lock time.

The persistence layer now rejects an existing archive identity when any locked
payload value differs. Exact reprocessing remains idempotent; a changed prediction
or price can no longer be silently ignored under the same identity.

### Pregame lock boundary

The previous eligibility check compared kickoff only with the source odds
`fetched_at` timestamp. That was insufficient because an old pregame snapshot could
theoretically be processed and archived after kickoff.

Eligibility now requires kickoff to be later than all four timestamps:

- odds fetch;
- prediction generation;
- betting-board generation;
- archive/lock time.

The runtime validator and SQL quality checks enforce the same rule.

### Hosted operational continuity

The scheduled runner restores `operational.duckdb` from the latest private release.
The audit verified that the production publisher explicitly uploads the refreshed
operational database alongside the compact public dashboard snapshot. This existing
control preserves forward archive and refresh history across stateless GitHub-hosted
runners.

### Model identity

Every locked row retains `model_name`, `model_version`, `prediction_mode` and
`prediction_generated_at`. Future settlement must read these archived values and
must not recreate historical predictions with the then-current model.

## Remaining work

- `analytics.forward_tip_market_movement` preserves the first positive-EV entry and
  compares it with the latest later pregame observation for the same game, market and
  outcome. It includes both price movement and directionally normalized line movement.
- The comparison is explicitly marked as latest pre-kickoff market movement, with
  `is_closing_snapshot = false` and `is_clv = false`.
- A formal kickoff-near snapshot rule and maximum acceptable distance from kickoff
  remain required before the platform may describe the metric as CLV.
- Forward settlement and performance reporting must join results only after games
  are final and remain physically distinct from historical backtests.

## P0 conclusion

The critical lock-time gap identified by this audit is addressed in code and covered
by regression tests. Hosted-state persistence was verified as already implemented.
Kickoff-near capture (P3b) and postgame settlement (P4) remain separate backlog phases.
