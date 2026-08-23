# Model Governance and Probability Blending

**Project:** NFL Analytics Platform  
**Version:** 0.1.0  
**Status:** Active  
**Last Updated:** 2026-08-05

---

## Purpose

This document defines the model-governance framework used by the NFL Analytics Platform.

It describes:

- how candidate models are compared;
- how time-based validation prevents leakage;
- why probability metrics are prioritized over accuracy;
- how the 2025 historical audit is interpreted;
- how the Elo and injury-enhanced models are blended;
- how the production candidate is selected;
- what will be monitored during the 2026 forward test;
- which results will be exposed in the Streamlit Data Science Lab.

The objective is not to select whichever model performed best in one isolated season. The objective is to identify a probability model that is accurate, well calibrated, stable across seasons and suitable for betting-value analysis.

---

## Governance Principles

Model selection follows these principles:

1. All candidates are evaluated on the same eligible games.
2. Training data must occur strictly before the evaluation season.
3. Candidate definitions are frozen before seasonal scoring.
4. Brier score and log loss are the primary selection metrics.
5. Accuracy is reported as an important secondary metric.
6. Aggregate performance and season-level stability are both required.
7. One unusually strong or weak season does not decide production selection.
8. Every production model must have a documented fallback.
9. Historical evaluation and future forward testing are reported separately.
10. Generated governance results must be reproducible from the modeling pipeline.

---

## Why Accuracy Is Not Enough

Accuracy evaluates whether the predicted winner is correct after converting a probability into a binary decision.

For example:

- a 51% home-win prediction;
- a 70% home-win prediction;
- a 95% home-win prediction

all count as the same correct prediction when the home team wins.

Accuracy therefore does not distinguish between cautious, well-calibrated probabilities and unjustifiably confident probabilities.

This is especially important for betting applications.

The business question is not only:

> Which team is more likely to win?

It is also:

> Is the model probability sufficiently different from the market probability to justify a wager?

A model that predicts the correct winner slightly less often can still be more valuable if its probabilities are better calibrated.

---

## Primary Metrics

### Accuracy

Accuracy is the percentage of games where the predicted winner matches the actual winner.

It is intuitive and useful for communication, but it ignores probability quality.

Higher values are better.

### Brier Score

The Brier score is the mean squared error between predicted probabilities and binary outcomes.

Conceptually:

`Brier = mean((predicted probability - actual outcome)²)`

It rewards probabilities that are close to the observed outcomes and penalizes unjustified confidence.

Lower values are better.

The Brier score is the primary model-selection metric in this project.

### Log Loss

Log loss measures the probability assigned to the observed outcome.

It penalizes confident incorrect predictions more heavily than the Brier score.

Lower values are better.

Log loss acts as an important safeguard against models that appear strong on average but occasionally generate dangerously overconfident probabilities.

### Calibration

Calibration measures whether predicted probabilities correspond to observed frequencies.

If a model assigns 60% win probability to a large collection of games, approximately 60% of those teams should win.

Calibration is especially important for:

- market comparisons;
- expected-value calculations;
- bankroll management;
- public probability communication.

---

## Time-Based Evaluation

Random train/test splitting is not used for final model governance.

NFL games form a time series. A production model can only use information available before the game being predicted.

The governance framework therefore uses expanding-window validation.

For an evaluation season:

- all eligible earlier seasons are used for training;
- the selected season is used for evaluation;
- future seasons are never included in training.

The governance folds are:

| Evaluation season | Training seasons |
|---|---|
| 2020 | 2018–2019 |
| 2021 | 2018–2020 |
| 2022 | 2018–2021 |
| 2023 | 2018–2022 |
| 2024 | 2018–2023 |
| 2025 | 2018–2024 |

This reproduces how the model would have behaved at the beginning of each historical season.

The common governance sample contains:

- 1,254 evaluation games;
- six evaluation seasons;
- identical eligibility rules for every candidate.

---

## Common Eligibility Rules

Candidate models are evaluated only on games where all required comparison features are available.

The common sample requires:

- a valid binary game outcome;
- a pregame Elo probability;
- both listed-QB ratings;
- complete injury-feature coverage;
- all required core model features.

Using a common sample prevents one model from receiving an unfair advantage because it was evaluated on an easier or different collection of games.

The eligibility rule is deliberately stricter than live production.

In live production, games with incomplete injury data use Elo fallback rather than disappearing from the prediction table.

---

## Frozen Candidate Models

Candidate definitions are stored centrally in:

`src/modeling/model_governance_candidates.py`

### Elo

The standalone leakage-safe Elo model.

It represents the stable rating baseline and provides:

- team-strength information;
- home-field adjustment;
- season-to-season regression;
- a complete fallback probability.

### Logistic Elo Plus QB

Features:

1. Elo rating difference;
2. listed-QB rating difference.

Regularization:

`C = 1.0`

This is the strongest simple logistic baseline.

### Logistic Elo, QB and Post-Bye

Features:

1. Elo rating difference;
2. listed-QB rating difference;
3. post-bye difference.

Regularization:

`C = 1.0`

This model performed strongly during initial 2023–2024 validation but failed to generalize to the 2025 historical holdout.

It remains in governance as a documented historical candidate rather than being silently removed.

### Logistic Elo, QB and Unit Injury Burdens

Features:

1. Elo rating difference;
2. listed-QB rating difference;
3. non-QB offensive injury-burden difference;
4. defensive injury-burden difference;
5. special-teams injury-burden difference.

Regularization:

`C = 0.1`

This is the leading injury-enhanced logistic model.

Quarterback injury burden is not included as another generic feature because listed-QB rating already represents the expected starting quarterback.

### Logistic Full Core

The full 17-feature logistic model.

It contains:

- Elo difference;
- listed-QB difference;
- 15 rolling last-four team-efficiency differences.

Regularization:

`C = 0.01`

Strong regularization is required because the rolling efficiency variables are correlated.

---

## Aggregate Governance Results

The frozen candidates were evaluated using expanding-window predictions across 2020–2025.

| Model | Games | Accuracy | Brier | Log loss |
|---|---:|---:|---:|---:|
| Logistic Elo + QB + unit burdens | 1,254 | 64.91% | 0.220788 | 0.632027 |
| Logistic Elo + QB | 1,254 | 64.35% | 0.221349 | 0.633424 |
| Logistic Elo + QB + post-bye | 1,254 | 64.27% | 0.221378 | 0.633254 |
| Logistic full core | 1,254 | 63.56% | 0.221787 | 0.633981 |
| Elo | 1,254 | 63.64% | 0.223252 | 0.638115 |

Across the full governance period, the injury-enhanced logistic candidate is the strongest standalone model.

Compared with Elo, it improves:

- accuracy by 1.27 percentage points;
- Brier score by 0.002464;
- log loss by 0.006088.

The improvement is real but modest, which supports using a blended production approach rather than completely replacing Elo.

---

## Season-Level Stability

Aggregate metrics alone can hide instability.

Each candidate is therefore evaluated separately for every season.

The season-level results show that:

- no single model wins every season;
- Elo remains competitive and occasionally superior;
- injury information improves the multi-season average;
- the full-core model is not consistently better despite using more features;
- 2025 differs materially from the preceding aggregate pattern.

This variation is expected in NFL modeling because each season contains only a few hundred eligible games and can be influenced by:

- quarterback changes;
- injury-report behavior;
- schedule composition;
- close-game randomness;
- turnover variance;
- structural changes in team strength;
- feature distribution changes.

Governance must distinguish normal seasonal variance from a persistent model failure.

---

## The 2025 Historical Audit

The 2025 season was originally protected as an untouched holdout.

After the first selected logistic model was frozen, the holdout was opened and became a historical audit season.

The initial post-bye candidate performed poorly:

| Model | Games | Accuracy | Brier | Log loss |
|---|---:|---:|---:|---:|
| Logistic Elo + QB + post-bye | 215 | 60.00% | 0.236957 | 0.667263 |
| Elo | 215 | 64.19% | 0.230111 | 0.652224 |

Diagnostics showed that the logistic model’s largest weakness occurred during the late regular season.

This result invalidated the post-bye model as the production winner.

The governance framework then evaluated all frozen candidates under identical expanding-window rules.

For 2025:

- the full-core logistic model produced the best probability metrics;
- Elo remained stronger than the injury-enhanced logistic model;
- the injury model’s 2025 weakness was material;
- the six-season aggregate still favored injury information.

The correct conclusion is not that injuries are useless.

The correct conclusion is:

- injury features add value across the complete historical sample;
- their value is not equally strong every season;
- Elo provides useful stability when the injury component weakens.

This directly motivates probability blending.

---

## Probability Blending

Blending combines predictions from models with different strengths.

The selected components are:

- the stable Elo baseline;
- the higher-performing injury-enhanced logistic model.

For an injury weight `w`:

`blended probability = w × injury probability + (1 - w) × Elo probability`

A weight of 0.70 therefore means:

- 70% injury-enhanced logistic probability;
- 30% Elo probability.

Blending is performed on probabilities, not predicted winners.

---

## Historical Blend Audit

To measure whether blending would have helped before seeing 2025:

1. out-of-fold predictions from 2020–2024 were used to select a weight;
2. the selected weight was frozen;
3. it was applied unchanged to 2025.

The 2020–2024 selection chose:

- 90% injury model;
- 10% Elo.

Selection-period results:

| Model | Games | Accuracy | Brier | Log loss |
|---|---:|---:|---:|---:|
| 90/10 blend | 1,039 | 66.51% | 0.217585 | 0.625219 |
| Injury logistic | 1,039 | 66.22% | 0.217627 | 0.625332 |
| Elo | 1,039 | 63.52% | 0.221832 | 0.635196 |

Applied unchanged to 2025:

| Model | Games | Accuracy | Brier | Log loss |
|---|---:|---:|---:|---:|
| Elo | 215 | 64.19% | 0.230111 | 0.652224 |
| 90/10 blend | 215 | 59.53% | 0.234762 | 0.661457 |
| Injury logistic | 215 | 58.60% | 0.236063 | 0.664384 |

The blend did not beat Elo in 2025, but it reduced the injury model’s loss.

This is evidence that Elo provides diversification and stabilizes the combined probability.

---

## Production Blend Selection

After the 2025 audit was completed, 2025 became part of the historical governance dataset.

Weights were therefore re-evaluated across the complete 2020–2025 period to select the candidate for the 2026 forward test.

The selected production weight is:

- 70% injury-enhanced logistic;
- 30% Elo.

Aggregate results:

| Model | Games | Accuracy | Brier | Log loss |
|---|---:|---:|---:|---:|
| 70/30 blend | 1,254 | 65.23% | 0.220351 | 0.631040 |
| Injury logistic | 1,254 | 64.91% | 0.220788 | 0.632027 |
| Elo | 1,254 | 63.64% | 0.223252 | 0.638115 |

The blend has:

- the highest aggregate accuracy;
- the best aggregate Brier score;
- the best aggregate log loss.

---

## Weight Robustness

The best production weights form a relatively flat region.

| Injury weight | Elo weight | Accuracy | Brier | Log loss |
|---:|---:|---:|---:|---:|
| 0.70 | 0.30 | 65.23% | 0.220351 | 0.631040 |
| 0.75 | 0.25 | 64.83% | 0.220353 | 0.631039 |
| 0.65 | 0.35 | 65.23% | 0.220376 | 0.631106 |
| 0.80 | 0.20 | 65.23% | 0.220384 | 0.631104 |
| 0.60 | 0.40 | 64.99% | 0.220429 | 0.631237 |

The differences are small.

This means the selected 70/30 blend is not based on a fragile, isolated optimum.

The weight is preferred because it:

- has the lowest Brier score;
- ties for the best accuracy among the strongest nearby weights;
- retains a meaningful 30% stable Elo contribution;
- is simple to communicate.

---

## Selected Production Candidate

The formal production candidate is defined in:

`src/modeling/production_probability_model.py`

Configuration:

| Property | Value |
|---|---|
| Model name | `elo_injury_logistic_blend` |
| Version | `0.2.0` |
| Logistic component | `logistic_elo_qb_unit_burdens` |
| Logistic weight | 0.70 |
| Elo weight | 0.30 |
| Logistic regularization | 0.1 |
| Forward season | 2026 |
| Status | `selected_for_2026_forward_test` |

The previous selected-model configuration remains in the repository as a historical rejected candidate.

It is not overwritten because its diagnostics provide an auditable record of why the governance process changed.

---

## Production Fallback Policy

The injury-enhanced component requires complete pregame injury features.

The live prediction policy is:

### Complete Injury Coverage

Use:

`70% injury logistic + 30% Elo`

Prediction mode:

`BLEND`

### Incomplete Injury Coverage

Use:

`100% Elo`

Prediction mode:

`ELO_FALLBACK`

A missing injury report must never be interpreted as confirmation that the team is healthy.

The active prediction mode should be visible in:

- current prediction tables;
- market-edge tables;
- Streamlit game cards;
- Data Science Lab diagnostics.

---

## 2026 Forward Test

The next untouched evaluation period is the 2026 season.

The 70/30 blend must remain frozen during the formal forward test unless a documented model-governance event authorizes a replacement.

The forward test should track:

- total eligible games;
- blend and fallback game counts;
- accuracy;
- Brier score;
- log loss;
- expected calibration error;
- season-to-date results;
- rolling results;
- Elo comparison;
- market comparison;
- closing-line value;
- betting results by edge threshold.

A weak short-term run alone is not sufficient reason to replace the model.

Changes should require evidence such as:

- sustained calibration deterioration;
- persistent loss against Elo;
- data-quality failure;
- feature-definition change;
- source-generation change;
- statistically meaningful challenger improvement.

---

## Champion–Challenger Structure

The selected 70/30 blend is the production champion for the 2026 forward test.

Potential challengers may include:

- alternative blend weights;
- weather-enhanced models;
- starter-confirmation models;
- improved player-value models;
- spread-derived priors;
- totals-specific models;
- calibrated boosting models;
- future external rating sources.

Challengers must be evaluated without altering the historical champion results.

A challenger becomes eligible for promotion only after:

1. frozen feature definitions;
2. leakage review;
3. time-based validation;
4. comparison on the same sample;
5. calibration analysis;
6. stability analysis;
7. documented governance approval.

---

## Reporting Tables

The model-governance reporting layer creates five DuckDB tables.

### `analytics.model_governance_scorecard`

Contains one aggregate row for each frozen candidate.

Current row count:

`5`

Intended uses:

- aggregate model ranking;
- KPI cards;
- accuracy versus probability-metric comparison;
- production candidate context.

### `analytics.model_governance_season_results`

Contains one row per model and evaluation season.

Current row count:

`30`

Coverage:

- five candidate models;
- six seasons;
- 2020–2025.

Intended uses:

- season trend charts;
- model stability analysis;
- weak-season identification;
- champion–challenger comparisons.

### `analytics.model_blend_weight_grid`

Contains every tested Elo/injury blend weight.

Current row count:

`42`

It includes:

- the 2020–2024 historical-audit selection scope;
- the 2020–2025 production-selection scope;
- 21 weights per scope.

Intended uses:

- weight sensitivity curves;
- robustness analysis;
- selected-weight highlighting.

### `analytics.model_blend_scorecard`

Contains the three comparison models for each governance period:

- Elo;
- standalone injury logistic;
- Elo/injury blend.

Current row count:

`9`

The three periods are:

- `selection_2020_2024`;
- `historical_audit_2025`;
- `production_selection_2020_2025`.

### `analytics.production_model_registry`

Contains the formal selected production configuration.

Current row count:

`1`

It records:

- model name;
- model version;
- status;
- forward season;
- component weights;
- feature group;
- regularization;
- fallback policy;
- selection metrics.

This table is the primary reporting source for the active-model card.

---

## Streamlit Data Science Lab

The governance data is designed to support a dedicated Data Science Lab page.

### Production Model Card

Display:

- model name and version;
- selection status;
- forward-test season;
- injury and Elo weights;
- fallback policy;
- selected features;
- aggregate metrics.

### Candidate Comparison Table

Display:

- model;
- accuracy;
- Brier score;
- log loss;
- improvement versus Elo;
- ranking by primary metric.

The table should explain that lower Brier and log loss values are better.

### Season Performance Chart

A line chart should compare candidates by season.

Metric selector:

- accuracy;
- Brier score;
- log loss.

The chart should make the 2025 divergence visible without presenting it as the only relevant period.

### Accuracy Versus Probability Quality

A scatter plot can use:

- x-axis: accuracy;
- y-axis: Brier score;
- point size or color: log loss;
- label: model name.

This illustrates why the most accurate model is not automatically the best probability model.

### Blend Weight Curve

Plot:

- x-axis: injury-model weight;
- y-axis: Brier score or log loss;
- separate lines for historical and production selection scopes;
- highlight 90/10 and 70/30 selections.

This chart demonstrates that the selected production weight lies inside a stable region.

### Historical Audit Panel

Display the 2025 results for:

- Elo;
- frozen 90/10 blend;
- standalone injury model.

The panel should explain that the blend reduced the injury model’s loss but did not outperform Elo during that season.

### Champion–Challenger Panel

Display:

- active champion;
- candidate challengers;
- evaluation status;
- forward-test sample size;
- promotion criteria.

### Calibration Panel

Future additions should include:

- reliability curve;
- calibration bins;
- expected calibration error;
- prediction distribution;
- confidence-band sample sizes.

### Active Prediction Mode

For live games, show whether the prediction uses:

- `BLEND`;
- `ELO_FALLBACK`.

The reason for fallback should also be visible.

---

## Data Quality

Governance tables are validated by:

`sql/025_model_governance_quality_checks.sql`

The suite contains 19 checks covering:

- expected table row counts;
- duplicate model records;
- valid season coverage;
- valid metric ranges;
- five candidates per season;
- valid blend weights;
- two weight-selection scopes;
- three models per blend period;
- one production registry row;
- agreement between the registry and the selected weight grid.

All 19 checks currently pass.

---

## Reproducibility

The governance workflow is reproducible through the local modeling pipeline.

The pipeline:

1. rebuilds source feature tables;
2. rebuilds the modeling dataset;
3. rebuilds train and evaluation splits;
4. runs governance candidates;
5. runs the blend analysis;
6. persists reporting tables;
7. rebuilds current predictions;
8. rebuilds the current season simulation.

The pipeline does not call external download APIs.

This protects:

- API quotas;
- historical source snapshots;
- reproducibility;
- separation between ingestion and model computation.

---

## Known Limitations

1. Six historical evaluation seasons still represent a limited sample.
2. NFL outcomes contain substantial irreducible randomness.
3. Injury report conventions can change between seasons.
4. Injury burden estimates usage importance rather than pure player talent.
5. The selected blend has not yet completed a true forward season.
6. Current live predictions still need explicit blend and fallback integration.
7. Weather, final inactives and external ratings are not yet part of the production model.
8. Betting profitability cannot be inferred from accuracy alone.
9. Market-edge performance requires timestamped odds and closing-line evaluation.
10. Model monitoring thresholds still need to be finalized.

---

## Planned Extensions

Planned governance improvements include:

- current prediction blend integration;
- active fallback-mode reporting;
- 2026 forward-test monitoring;
- calibration dashboards;
- bootstrap confidence intervals;
- rolling Brier and log-loss charts;
- market and closing-line comparison;
- spread and totals governance;
- injury-source freshness monitoring;
- feature drift alerts;
- documented challenger promotion rules;
- Streamlit Data Science Lab implementation.

---

## Summary

The NFL Analytics Platform evaluates models with leakage-safe expanding-window validation across complete seasons.

Accuracy remains an important communication metric, but Brier score and log loss are the primary model-selection criteria because the platform ultimately compares model probabilities with market probabilities.

Across 2020–2025, the injury-enhanced logistic model is the strongest standalone candidate. Its weakness during 2025 shows that it should not completely replace the stable Elo baseline.

The selected production candidate therefore blends:

- 70% injury-enhanced logistic probability;
- 30% Elo probability.

The blend achieves the strongest aggregate accuracy, Brier score and log loss across the complete governance sample.

It is frozen as version `0.2.0` for the 2026 forward test, with Elo used as an explicit fallback whenever current injury coverage is incomplete.