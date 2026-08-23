# Project Charter

**Project:** NFL Analytics Platform

**Version:** 0.1.0

**Status:** Draft

**Phase:** Foundation

**Sprint:** Sprint 0

**Author:** Ferenc Kaizer

**Last Updated:** 2026-07-12

---

# Document Purpose

This document defines the vision, objectives, scope, guiding principles and long-term roadmap of the NFL Analytics Platform.

It serves as the primary reference for all architectural, technical and business decisions made throughout the project.

The document is intended to evolve together with the project and will be updated as new requirements, decisions and lessons learned emerge.

---

# Table of Contents

1. Executive Summary
2. Project Vision
3. Project Objectives
4. Project Scope
5. Non-Goals
6. Success Criteria
7. Development Principles
8. Stakeholders
9. High-Level Architecture
10. Key Risks
11. Deliverables
12. High-Level Roadmap

# Executive Summary

The NFL Analytics Platform is a long-term portfolio project designed to demonstrate the complete lifecycle of a modern Data Science solution using professional American football as its business domain.

The platform combines data engineering, analytics, machine learning, software engineering and business intelligence into a single reproducible system.

Unlike many hobby prediction projects, the primary objective is not simply to predict NFL games. The goal is to design, build and document a complete analytics platform that follows professional software development and Data Science practices.

The project was initiated to bridge the gap between existing professional experience and future career goals. While my current role provides extensive experience in SQL, database design and Power BI, opportunities to work on advanced Machine Learning and Data Science projects are limited.

By combining three personal interests—data, NFL and sports betting—the project provides an engaging environment to develop practical skills while creating a portfolio-quality system suitable for technical interviews and future career opportunities.

The first production release (Version 1.0) will focus on predicting Moneyline outcomes, evaluating betting value, tracking betting performance and providing business intelligence dashboards. Additional betting markets such as Spread and Totals are intentionally planned for future releases after the core platform has been completed and validated.

# Project Vision

The vision of the NFL Analytics Platform is to become a complete, production-quality Data Science portfolio project that demonstrates the end-to-end development of an analytics solution.

The platform will serve as a practical environment for learning and applying modern Data Science, Machine Learning and Software Engineering practices while solving real-world analytical problems within the NFL domain.

Rather than focusing solely on prediction accuracy, the project aims to build a transparent, explainable and reproducible analytics platform where every decision can be traced back to data.

The long-term objective is to continuously improve the platform throughout multiple NFL seasons by introducing new analytical capabilities, validating hypotheses with historical data and measuring every improvement through objective metrics.

Ultimately, the project should represent the level of technical quality, documentation and engineering discipline expected from a professional Data Scientist or Machine Learning Engineer.

The platform should also serve as a personal laboratory for experimenting with new analytical techniques, allowing future models, betting strategies and visualization methods to be evaluated within a consistent and well-documented framework.

# Project Objectives

The project has five primary objectives.

## 1. Build an End-to-End Data Science Platform

Design and implement a complete analytics platform covering the entire Data Science lifecycle, including data ingestion, data storage, feature engineering, machine learning, business intelligence and performance evaluation.

---

## 2. Develop Practical Data Science Skills

Gain hands-on experience with technologies and methodologies that are not part of my current professional responsibilities, including Python, Machine Learning, feature engineering, model evaluation and software engineering best practices.

---

## 3. Build a Portfolio-Quality Project

Create a well-documented, production-style project that demonstrates technical skills, engineering discipline and analytical thinking for future Data Science and Machine Learning career opportunities.

---

## 4. Evaluate Betting Strategies

Develop a statistically driven betting framework capable of evaluating whether model predictions can identify value betting opportunities over the course of multiple NFL seasons.

The objective is not to maximise short-term profit but to measure long-term performance using objective metrics such as ROI, Closing Line Value (CLV), calibration and prediction accuracy.

## 5. Build a Repeatable Analytics Workflow

Design the platform as a repeatable and maintainable analytics solution capable of incorporating new weekly NFL data with minimal manual effort.

The workflow should support continuous feature generation, model retraining, prediction generation and betting performance tracking throughout each NFL season.

# Project Scope

The initial version of the NFL Analytics Platform (Version 1.0) will focus on delivering a complete, reproducible and well-documented analytics workflow for predicting NFL Moneyline outcomes.

The scope of Version 1.0 includes:

- Historical NFL data collection
- Automated weekly data ingestion
- Data validation and cleaning
- Centralised analytical database
- Feature engineering
- Machine Learning model development
- Weekly prediction generation
- Betting value evaluation
- Bet tracking and performance analysis
- Interactive Power BI dashboards
- Complete technical documentation

The platform is designed to be modular, allowing additional betting markets and analytical capabilities to be introduced in future versions without major architectural changes.


# Non-Goals

The following features are intentionally excluded from Version 1.0.

- Live betting
- Player prop betting
- Fantasy football analytics
- Injury prediction models
- Weather prediction models
- Cloud deployment
- Mobile application
- Public API
- Deep learning models
- Real-time streaming architecture

These features may be considered in future releases after the core platform has been completed and validated.

# Success Criteria

The project will be considered successful if it achieves the following objectives.

## Technical

- A fully automated weekly data ingestion workflow
- A reproducible data pipeline
- A documented feature engineering process
- A working Machine Learning prediction pipeline
- A complete GitHub repository with documentation

## Analytical

- Reliable historical backtesting
- Well-calibrated probability predictions
- Explainable model outputs
- Objective model evaluation using appropriate performance metrics

## Betting

- Consistent bet tracking
- ROI calculation
- Closing Line Value (CLV) tracking
- Long-term betting performance evaluation

## Business Intelligence

- Interactive Power BI dashboards
- Weekly performance reporting
- Historical trend analysis

## Professional Development

- Demonstrate practical Data Science skills
- Improve technical English
- Build a portfolio suitable for Data Science interviews

# Development Principles

The following principles guide every technical and business decision made throughout the project.

1. Build for learning before optimisation.

2. Every assumption must earn its place in the model.

3. If a decision cannot be explained, it should not be implemented.

4. Prefer simple solutions over unnecessary complexity.

5. Every experiment must be measurable.

6. Every important decision must be documented.

7. Historical data must never leak future information into the model.

8. Reproducibility is more important than convenience.

9. Documentation is part of the product.

10. Continuous improvement is preferred over premature perfection.


# Stakeholders

The project has the following primary stakeholders.

## Project Owner

Responsible for defining the project direction, making final decisions and maintaining the long-term vision of the platform.

## End User

Uses the weekly predictions, betting analytics and Power BI dashboards to support data-driven NFL betting decisions.

## Data Science Learner

Uses the project as a practical environment for developing Python, Machine Learning, data engineering and software engineering skills.

## Portfolio Reviewer

Evaluates the project from a technical and professional perspective, including potential employers, recruiters and Data Science interviewers.

## Future Maintainer

May extend or modify the platform in future NFL seasons. For this reason, the project must remain understandable, reproducible and well documented.


# High-Level Architecture

The NFL Analytics Platform follows a modular layered architecture that separates data collection, data processing, machine learning, betting analytics and business intelligence.

Version 1.0 is intentionally designed as a lightweight local solution that prioritises simplicity, reproducibility and maintainability while providing a clear upgrade path for future cloud automation.

---

## External Data Sources

The platform initially integrates two primary data sources.

### nflverse

Historical NFL schedules, results, play-by-play data and derived football statistics.

### The Odds API

Current bookmaker odds for upcoming NFL games.

Version 1.0 will initially support Moneyline markets only.

---

## High-Level Data Flow

```text
nflverse + The Odds API
          │
          ▼
   Python Data Ingestion
          │
          ▼
      Raw Parquet Layer
          │
          ▼
 Validation & Cleaning
          │
          ▼
      Analytical Database
      (DuckDB)
          │
          ▼
   Feature Engineering
          │
          ▼
 Machine Learning Models
          │
          ▼
 Betting Analytics
          │
          ▼
    Analytics Mart
          │
          ▼
   Power BI Dashboard
```

---

## Architectural Principles

- Modular architecture
- Layered data processing
- Reproducible workflows
- Clear separation between data engineering, machine learning and reporting
- Time-aware processing to prevent data leakage
- Designed for future automation and cloud deployment

Detailed technical architecture is documented separately in **System_Architecture.md**.

# Key Risks

The project has the following key risks.

## Data Availability

Historical odds, injury information and other external datasets may be incomplete, inconsistent or expensive.

## Data Leakage

Future information may accidentally be included in historical model features, producing unrealistic backtest results.

## Overfitting

The model may learn noise or historical patterns that do not generalise to future NFL games.

## Limited Sample Size

The NFL has relatively few games per season, which limits the amount of training data available.

## Market Efficiency

Sports betting markets are highly competitive, and profitable opportunities may be small or inconsistent.

## Scope Creep

Additional ideas such as Spread, Totals, player props, weather and injury models may delay the completion of Version 1.0.

## Operational Reliability

Weekly data updates, API availability or pipeline errors may prevent predictions from being generated on time.

## Financial Risk

Model predictions may be incorrect, and real-money betting may result in financial losses. Betting performance must therefore be tracked separately from model quality.


# Deliverables

Version 1.0 will deliver the following components.

## Data Engineering

- Automated NFL data ingestion
- Historical analytical database
- Data validation pipeline
- Feature engineering pipeline

## Machine Learning

- Baseline Moneyline prediction model
- Historical backtesting framework
- Probability calibration
- Weekly prediction generation

## Betting Analytics

- Betting recommendation engine
- Bet tracking
- ROI analysis
- Closing Line Value (CLV) tracking
- Bankroll tracking

## Business Intelligence

- Interactive Power BI dashboards
- Historical reporting
- Weekly performance reporting

## Documentation

- Complete technical documentation
- GitHub repository
- Architecture documentation
- Data dictionary
- Glossary
- Decision Log
- Experiment Log


# High-Level Roadmap

The project will be developed iteratively through multiple phases.

| Phase | Objective |
|--------|-----------|
| Phase 0 | Foundation and project setup |
| Phase 1 | Data Engineering |
| Phase 2 | Feature Engineering |
| Phase 3 | Machine Learning |
| Phase 4 | Betting Analytics |
| Phase 5 | Business Intelligence |
| Version 1.0 | Complete NFL Analytics Platform |

A detailed implementation plan is maintained separately in **Project_Roadmap.md**.


