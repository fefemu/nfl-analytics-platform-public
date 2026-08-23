-- =====================================================
-- NFL Analytics Platform
-- File: 001_data_quality_checks.sql
--
-- Purpose:
--     Execute basic data quality checks against
--     the raw schedule table stored in DuckDB.
--
-- Author:
--     Ferenc Kaizer
--
-- Version:
--     0.1.0
-- =====================================================


-- -----------------------------------------------------
-- 1. Total record count
-- -----------------------------------------------------

SELECT
    COUNT(*) AS total_records
FROM raw.schedule;


-- -----------------------------------------------------
-- 2. Season range
-- -----------------------------------------------------

SELECT
    MIN(season) AS first_season,
    MAX(season) AS latest_season
FROM raw.schedule;


-- -----------------------------------------------------
-- 3. Records by game type
-- -----------------------------------------------------

SELECT
    game_type,
    COUNT(*) AS game_count
FROM raw.schedule
GROUP BY game_type
ORDER BY game_type;


-- -----------------------------------------------------
-- 4. Records by season
-- -----------------------------------------------------

SELECT
    season,
    COUNT(*) AS game_count
FROM raw.schedule
GROUP BY season
ORDER BY season;


-- -----------------------------------------------------
-- 5. Missing game identifiers
-- -----------------------------------------------------

SELECT
    COUNT(*) AS missing_game_ids
FROM raw.schedule
WHERE game_id IS NULL;


-- -----------------------------------------------------
-- 6. Duplicate game identifiers
-- Expected result: zero rows
-- -----------------------------------------------------

SELECT
    game_id,
    COUNT(*) AS duplicate_count
FROM raw.schedule
WHERE game_id IS NOT NULL
GROUP BY game_id
HAVING COUNT(*) > 1
ORDER BY duplicate_count DESC, game_id;


-- -----------------------------------------------------
-- 7. Games without final scores
-- These are expected for future scheduled games.
-- -----------------------------------------------------

SELECT
    COUNT(*) AS games_without_final_scores
FROM raw.schedule
WHERE home_score IS NULL
  AND away_score IS NULL;


-- -----------------------------------------------------
-- 8. Records where only one score is missing
-- Expected result: zero
-- -----------------------------------------------------

SELECT
    COUNT(*) AS inconsistent_score_records
FROM raw.schedule
WHERE (home_score IS NULL AND away_score IS NOT NULL)
   OR (home_score IS NOT NULL AND away_score IS NULL);


-- -----------------------------------------------------
-- 9. Invalid team assignments
-- Expected result: zero
-- -----------------------------------------------------

SELECT
    COUNT(*) AS invalid_team_records
FROM raw.schedule
WHERE home_team IS NULL
   OR away_team IS NULL
   OR home_team = away_team;


-- -----------------------------------------------------
-- 10. Invalid regular-season week values
-- Expected result: zero
-- -----------------------------------------------------

SELECT
    COUNT(*) AS invalid_regular_season_weeks
FROM raw.schedule
WHERE game_type = 'REG'
  AND (week < 1 OR week > 18);


-- -----------------------------------------------------
-- 11. Negative scores
-- Expected result: zero
-- -----------------------------------------------------

SELECT
    COUNT(*) AS negative_score_records
FROM raw.schedule
WHERE home_score < 0
   OR away_score < 0;


-- -----------------------------------------------------
-- 12. Missing values in selected analytical fields
-- -----------------------------------------------------

SELECT
    COUNT(*) FILTER (WHERE home_moneyline IS NULL)
        AS missing_home_moneyline,
    COUNT(*) FILTER (WHERE away_moneyline IS NULL)
        AS missing_away_moneyline,
    COUNT(*) FILTER (WHERE home_rest IS NULL)
        AS missing_home_rest,
    COUNT(*) FILTER (WHERE away_rest IS NULL)
        AS missing_away_rest,
    COUNT(*) FILTER (WHERE roof IS NULL)
        AS missing_roof,
    COUNT(*) FILTER (WHERE surface IS NULL)
        AS missing_surface,
    COUNT(*) FILTER (WHERE temp IS NULL)
        AS missing_temperature,
    COUNT(*) FILTER (WHERE wind IS NULL)
        AS missing_wind
FROM raw.schedule;
