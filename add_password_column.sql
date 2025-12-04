-- Migration script to add password column to employees table
-- Run this in your MySQL database

-- Check if password column exists, if not add it
SET @column_exists = (
    SELECT COUNT(*)
    FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_SCHEMA = DATABASE()
    AND TABLE_NAME = 'employees' 
    AND COLUMN_NAME = 'password'
);

-- Add password column if it doesn't exist
SET @sql = IF(@column_exists = 0, 
    'ALTER TABLE employees ADD COLUMN password VARCHAR(255) NOT NULL DEFAULT "default_password"',
    'SELECT "Password column already exists" as message'
);

PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Update existing employees with generated passwords
UPDATE employees 
SET password = CONCAT('emp_', id, '_', SUBSTRING(MD5(email), 1, 8))
WHERE password = 'default_password';

-- Verify the changes
SELECT id, name, email, password FROM employees LIMIT 5;
