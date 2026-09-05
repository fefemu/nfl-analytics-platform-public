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

- The existing `analytics.forward_tip_clv` view provides a latest-later-pregame
  comparison for the same outcome and exact line. It must not yet be described as a
  complete closing-line benchmark.
- A formal closing-snapshot rule and maximum acceptable distance from kickoff are
  still required.
- Spread and Total CLV must incorporate line movement as well as price movement;
  the current exact-line comparison is only an intermediate implementation.
- Forward settlement and performance reporting must join results only after games
  are final and remain physically distinct from historical backtests.

## P0 conclusion

The critical lock-time gap identified by this audit is addressed in code and covered
by regression tests. Hosted-state persistence was verified as already implemented.
Full CLV methodology and postgame settlement remain the separately scheduled P3 and
P4 backlog phases.
