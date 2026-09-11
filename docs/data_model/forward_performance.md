# Results methodology

The public Results page separates two different questions. Weekly Overview
describes the selected week's completed and upcoming games, Betting Board shows
current market signals, and Results evaluates locked pregame predictions and
published selections.

## Model Results

Model Results measures prediction quality independently of betting selections.
Moneyline reports accuracy, Brier score and Log Loss. Spread reports error in the
immutable predicted home margin, and Total reports error in the immutable
predicted combined points; both use MAE, RMSE and signed prediction-minus-actual
bias. No sportsbook line is used in these model-quality calculations.
`analytics.completed_game_prediction_results` contains one row per completed game,
selected from the latest valid immutable pre-game state in
`analytics.game_prediction_archive`. It reports the predicted winner, pre-game win
probability, actual winner, correctness, Brier loss and Log Loss.

## Betting Results

Betting Results measures only selections that were actually eligible for the
production **Selected Market Signals** view before kickoff. The source chain is:

`analytics.selected_signal_archive` → `analytics.selected_signal_settlement` →
`analytics.selected_signal_performance_summary`.

The archive stores the first published game-market selection, its price and line,
model and market inputs, publication timestamp, refresh/snapshot identifiers, and
the criteria and guardrail versions used for the decision. It is append-only and a
later refresh cannot replace the entry state. Settlement uses a flat one-unit stake.

Verified forward betting tracking begins only when this immutable selection archive
is deployed. Historical selections are not reconstructed with current rules. A
positive-EV observation is not automatically a published selection.

## Research observations

The legacy `analytics.forward_betting_board_archive`,
`analytics.forward_tip_market_movement`, `analytics.forward_tip_settlement` and
`analytics.forward_performance_summary` remain available for compatibility and
research into market opportunities. They are not read by the user-facing Live
Results page and their W-L-P or ROI aggregates must not be presented as Betting
Results.

In short: **Model Results ≠ Betting Results**.
