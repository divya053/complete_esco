-- Migration script to add team_archive column to bid_checklists table
-- Run this in your MySQL database

-- Check if team_archive column exists, if not add it
SET @column_exists = (
    SELECT COUNT(*)
    FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_SCHEMA = DATABASE()
    AND TABLE_NAME = 'bid_checklists' 
    AND COLUMN_NAME = 'team_archive'
);

-- Add team_archive column if it doesn't exist
SET @sql = IF(@column_exists = 0, 
    'ALTER TABLE bid_checklists ADD COLUMN team_archive VARCHAR(50) DEFAULT NULL',
    'SELECT "team_archive column already exists" as message'
);

PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Verify the changes
SELECT id, task_name, team_archive FROM bid_checklists LIMIT 5;
