# Glossary

**Project:** NFL Analytics Platform
**Version:** 0.1.0
**Status:** Living Document
**Last Updated:** 2026-08-04

---

## Purpose

This glossary defines the data engineering, football analytics, machine learning, simulation and betting terminology used throughout the NFL Analytics Platform.

It serves as a shared reference for the repository documentation, Data Science Lab and public application.

---

## Terms

| Term | Category | Definition |
|------|----------|------------|
| Accuracy | Model Evaluation | The percentage of games for which the predicted winner matches the actual winner. Accuracy does not measure the quality of the predicted probability. |
| Actual Primary QB | Football Analytics | The quarterback who received the largest share of his team’s qualifying dropbacks in the completed game. This is retrospective audit information and must not be used as a pregame feature. |
| Analytics Layer | Data Engineering | DuckDB tables containing features, ratings, predictions, simulations and reporting-ready outputs derived from processed data. |
| API Credit | Data Engineering | A unit of usage charged by an external API. Odds API refresh frequency must be controlled to remain within the monthly credit limit. |
| Away Win Probability | Modeling | The model probability that the away team wins. For a binary no-tie pregame model it equals one minus the home win probability. |
| Backtest | Model Evaluation | Historical evaluation of a model or betting rule using only information that would have been available at each historical decision time. |
| Best Available Odds | Betting | The highest currently available bookmaker price for one equivalent betting outcome and market line. |
| Model–Market Probability Gap | Betting | Model probability minus the equal-weighted consensus of bookmaker-level no-vig probabilities. It measures disagreement, not expected profit. |
| Brier Score | Model Evaluation | The mean squared error of predicted probabilities against binary outcomes. Lower values are better. |
| Calibration | Model Evaluation | The agreement between predicted probabilities andq observed outcome frequencies. A calibrated 60% prediction should occur successfully about 60% of the time. |
| Closing Line | Betting | The final market price or line available shortly before a game begins. |
| Closing-Line Value | Betting | A comparison between the price obtained at decision time and the closing market price. It measures whether the bettor consistently captured better prices than the final market. |
| Competitive EPA | Football Analytics | EPA calculated only from plays that occur while the game remains competitively meaningful under the project’s score and time rules. |
| Core Model Eligible | Modeling | A game with a valid binary target, complete short rolling history and both listed-QB ratings available. |
| Current Elo Rating | Modeling | A team’s most recent Elo rating after processing all completed games currently available to the platform. |
| Current Market Board | Betting | The latest schedule-linked table containing available bookmaker prices, no-vig probabilities and market summaries. |
| Data Ingestion | Data Engineering | The process of retrieving source data and loading it into the platform’s raw data layer. |
| Data Leakage | Model Evaluation | The use of information that would not have been available when the historical prediction was made. Leakage produces unrealistically optimistic evaluation results. |
| Data Science Lab | Product | The planned technical dashboard area containing model comparisons, calibration, diagnostics, feature analysis and methodology. |
| Decimal Odds | Betting | Betting odds representing the total return per unit staked, including the original stake. |
| Designed Rush | Football Analytics | A planned rushing play, excluding quarterback scrambles when identified separately. |
| Depth Chart | Football Analytics | A pregame roster hierarchy listing players by team, position or role and expected depth rank. It represents listed role importance, not retrospective proof of who actually played. |
| Depth Tier | Feature Engineering | A normalized depth-chart category derived from listed rank: starter, primary backup or reserve. |
| Distribution Shift | Model Evaluation | A change in the statistical distribution of model inputs or outcomes between development and later prediction periods. |
| Dropback | Football Analytics | A quarterback passing opportunity including pass attempts, sacks and qualifying scrambles according to the project’s play definitions. |
| DuckDB | Data Engineering | The embedded analytical database used for raw, processed and analytics tables in the local platform. |
| Dynamic Simulation | Simulation | A simulation in which team ratings and later-game probabilities change in response to earlier simulated results. |
| Elo | Modeling | A sequential rating system that updates team strength after every game based on the difference between actual and expected results. |
| Elo K-Factor | Modeling | The parameter controlling how strongly one game changes the two teams’ Elo ratings. |
| Elo Rating Difference | Modeling | The home team’s pregame Elo rating minus the away team’s pregame Elo rating. |
| Empirical-Bayes Shrinkage | Modeling | A method that pulls uncertain player estimates toward a population mean, with stronger shrinkage applied to smaller samples. |
| Expected Calibration Error | Model Evaluation | A weighted average of the absolute gaps between predicted probability and observed outcome rate across probability bins. |
| Expected Final Elo | Simulation | The average team Elo rating at the end of all Monte Carlo season runs. |
| Expected Value | Betting | The model-estimated average profit or loss per unit staked. For decimal odds it can be calculated as model probability multiplied by decimal odds minus one. |
| Expected Wins | Simulation | The average number of full-season wins produced for a team across all Monte Carlo simulations. |
| Explosive Play Rate | Football Analytics | The share of qualifying plays meeting the project’s explosive-play yardage definition. |
| Extended Rest | Feature Engineering | A pregame indicator showing that a team has at least nine rest days. |
| External Validation | Model Evaluation | A later time period used to compare model candidates after fitting and tuning on earlier seasons. |
| Feature | Modeling | A measurable pregame input supplied to a predictive model. |
| Feature Ablation | Model Evaluation | An experiment comparing models with different feature groups to measure whether each group adds stable predictive value. |
| Feature Drift | Model Evaluation | A change in the distribution of a feature between development and later periods. |
| Game Modeling Dataset | Modeling | The one-row-per-game analytics table combining targets, Elo, rolling team features, QB features and schedule context. |
| Holdout | Model Evaluation | A final time period kept unavailable during model development and opened once after the model specification is frozen. |
| Home-Field Advantage | Modeling | The Elo rating adjustment applied to the home team before calculating win probability. The current production value is 50 Elo points. |
| Home Win Probability | Modeling | The model probability that the designated home team wins the game. |
| Implied Probability | Betting | The probability represented by bookmaker odds before removing the bookmaker margin. |
| Injury Burden | Feature Engineering | A planned aggregate representation of unavailable or limited players, potentially weighted by position, starter status and player importance. |
| Injury Snapshot | Data Engineering | A timestamped record of player injury status as it was known at a specific pregame moment. |
| Isotonic Calibration | Model Evaluation | A non-parametric monotonic probability calibration method. |
| Listed QB | Football Analytics | The quarterback recorded as the expected or listed starter before the game. This is the QB identity eligible for pregame prediction features. |
| Log Loss | Model Evaluation | A probability scoring rule that penalizes confident incorrect predictions strongly. Lower values are better. |
| Log Odds | Modeling | The logarithm of probability divided by one minus probability. Additive logistic and Elo explanation components can be represented in log-odds space. |
| Market-Aware Model | Modeling | A predictive model that includes market information such as an opening line as an input. It must remain distinguishable from an independent football model. |
| Median Wins | Simulation | The middle simulated win total for a team across all season runs. |
| Model Version | Software Engineering | A stable identifier stored with predictions so that outputs can be traced to the exact model specification that generated them. |
| Moneyline | Betting | A market on which team wins the game without applying a point spread. |
| Monte Carlo Simulation | Simulation | Repeated random sampling of possible season outcomes from game-level probability models. |
| Most Likely Wins | Simulation | The win total occurring most frequently across a team’s simulated seasons. |
| No-Vig Probability | Betting | A market probability after normalizing away the bookmaker margin. |
| Opening Line | Betting | The first widely available bookmaker price or line for a game. |
| Opponent Adjustment | Feature Engineering | A transformation accounting for the strength of opponents faced when estimating team or player performance. |
| Overround | Betting | The amount by which the sum of raw bookmaker implied probabilities exceeds one. It represents the bookmaker margin before normalization. |
| P10 Wins | Simulation | The 10th percentile of a team’s simulated win distribution. Approximately 10% of simulations finish at or below this value. |
| P90 Wins | Simulation | The 90th percentile of a team’s simulated win distribution. Approximately 90% of simulations finish at or below this value. |
| Parquet | Data Engineering | A compressed columnar file format used for historical schedule and play-by-play data. |
| Player-Game-Role Grain | Data Engineering | A table grain allowing one player to have multiple valid records in one game when the player holds multiple roles, such as offense and special teams. |
| Post-Bye | Feature Engineering | A pregame indicator showing that a team has at least 13 rest days under the current project definition. |
| Pregame Feature | Modeling | A model input constructed only from information available before kickoff. |
| Processed Layer | Data Engineering | Cleaned, typed and normalized DuckDB tables derived from raw source records. |
| Production Model | Modeling | The model currently approved for generating live platform predictions. The present production baseline is Elo. |
| Prospective Evaluation | Model Evaluation | Evaluation on future games whose outcomes were unknown when the model and decision rules were finalized. |
| QB Rating Difference | Modeling | The listed home quarterback’s pregame project rating minus the listed away quarterback’s rating. |
| Raw Layer | Data Engineering | Source-aligned DuckDB tables preserving ingested records with minimal business transformation. |
| Regularization | Modeling | A constraint that reduces model coefficient magnitude to limit overfitting. |
| Rolling Feature | Feature Engineering | A statistic calculated from a fixed number of earlier games and shifted so that the current game is excluded. |
| Season Retention | Modeling | The fraction of a team’s prior-season Elo deviation from the league mean retained after offseason regression. The current production value is 60%. |
| SHAP | Explainability | A feature-attribution framework commonly used to explain nonlinear model predictions. SHAP describes model behavior and does not establish causality. |
| Short Week | Feature Engineering | A pregame indicator showing that a team has no more than six rest days. |
| Sigmoid Calibration | Model Evaluation | A parametric logistic transformation fitted to recalibrate model probabilities. |
| Snapshot | Data Engineering | An immutable, timestamped representation of external information as it was available at one moment. |
| Snap Share | Feature Engineering | The fraction of a team’s offensive, defensive or special-teams snaps played by one player. Pregame rolling snap share must use completed prior games only. |
| Source Generation | Data Engineering | A materially different version of an external dataset with its own schema and timing behavior, such as weekly legacy NFL depth charts and timestamped ESPN depth charts. |
| Spread | Betting | A market applying a handicap expressed in points to the game result. |
| Standardized Mean Difference | Model Evaluation | The difference between two feature means divided by a reference standard deviation, used as a simple feature-drift measure. |
| Starter Snapshot | Data Engineering | A timestamped record of the players expected to start a game at a specific pregame time. |
| Starter Role | Football Analytics | A rank-one depth-chart role indicating that a player was listed first at a position before the game. It is not the same as confirmed actual participation. |
| Success Rate | Football Analytics | The share of plays producing positive expected points added under the project’s current definition. |
| Target | Modeling | The outcome a model is trained to predict. The current Moneyline target is a binary home-team win indicator. |
| Time-Based Split | Model Evaluation | A train, validation or test assignment based on chronological order rather than random sampling. |
| Time-CV | Model Evaluation | Expanding-window cross-validation in which each validation season occurs after every season used for training. |
| Totals | Betting | A market on whether the game’s combined score will finish above or below a bookmaker line. |
| Turnover Rate | Football Analytics | The share of qualifying plays ending in a turnover under the project’s play definitions. |
| Value Bet | Betting | A potential wager for which the model probability and available odds imply positive expected value. Positive estimated value does not guarantee profit on an individual wager. |
| Vig | Betting | The bookmaker margin embedded in quoted odds. |
| Win Distribution | Simulation | The probability assigned to every possible regular-season win total for one team. |
| XGBoost | Modeling | A gradient-boosted decision-tree algorithm evaluated as a nonlinear model candi
