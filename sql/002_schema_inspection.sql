-- =====================================================
-- NFL Analytics Platform
-- File: 002_schema_inspection.sql
--
-- Purpose:
--     Explore the structure of the raw schedule table
--     stored in DuckDB.
--
-- Author:
--     Ferenc Kaizer
--
-- Version:
--     0.1.0
-- =====================================================


-- -----------------------------------------------------
-- List all columns of the raw schedule table
-- -----------------------------------------------------

SELECT
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns
WHERE table_schema = 'raw'
  AND table_name = 'schedule'
ORDER BY ordinal_position;
