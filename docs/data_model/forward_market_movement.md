# Forward Market Movement

**Status:** Production tracking  
**Last updated:** 2026-09-05

## Purpose

`analytics.forward_tip_market_movement` compares the first archived positive-EV observation for a game, market and outcome with the latest later market snapshot available before kickoff.

This is **Market Movement / Latest Pre-Kickoff Value**, not Closing Line Value. The current Tuesday, Thursday and Sunday refreshes are not guaranteed to capture the closing market. Every row therefore has `is_closing_snapshot = false` and `is_clv = false`.

## Entry and comparison rules

- The entry is the chronologically first positive-EV observation for one `game_id`, `market_key` and `outcome_type`.
- Entry price, line, prediction, model identity and timestamps remain immutable in `analytics.forward_betting_board_archive`.
- The comparison is the latest later archived observation for the same game, market and outcome whose fetch time is still before kickoff.
- A missing later observation remains explicit as `NO_LATER_SNAPSHOT`; it is never interpreted as zero movement.

## Movement measures

Moneyline price movement is the later implied probability minus the entry implied probability. A positive value means the entry offered the better price.

Spread and Total comparisons preserve both prices and lines. `entry_line_advantage_points` is oriented so positive always means the entry line was better for the selected outcome:

- Spread: entry line minus latest line;
- Over: latest line minus entry line;
- Under: entry line minus latest line.

Line movement determines the direction when the line changed. If the line is unchanged, implied-probability price movement determines it.

## Future promotion to CLV

P3b will add a lightweight market capture near `kickoff - 60 minutes`. Only observations meeting a documented kickoff-distance tolerance may later be labelled closing snapshots and used for prospective CLV.
