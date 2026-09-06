# Forward Performance

`analytics.forward_tip_settlement` is the derived settlement ledger for the
immutable positive-EV entry observations stored by the 2026 forward archive.
It never rewrites entry prices, probabilities, model identity or timestamps.

Rows remain `PENDING` until `processed.schedule` contains both final scores and
marks the game complete. Completed Moneyline, Spread and Total selections are
settled as `WIN`, `LOSS` or `PUSH` at the locked entry line and decimal odds.
Flat-stake profit is decimal odds minus one for a win, minus one for a loss and
zero for a push.

Moneyline Brier loss and Log Loss use the locked probability of the selected
outcome. Tied games are pushes and are excluded from these binary probability
scores. Spread and Total rows are not included in probability scoring.

`analytics.forward_performance_summary` provides season, season/market, week
and week/market aggregates: tracked and settled counts, W/L/push, win rate,
average odds, units, ROI, maximum drawdown, Moneyline Brier/Log Loss and
qualifying prospective-CLV coverage. Each aggregate is prepared for all tracked
rows, settled rows only and CLV-eligible rows so the UI never recalculates
performance metrics. These are live-forward results only and
must not be combined with historical backtests.

The first live samples are inherently small and are not evidence of sustainable
profitability.
