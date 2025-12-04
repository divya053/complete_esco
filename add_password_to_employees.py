#!/usr/bin/env python3
"""
Migration script to add password column to employees table
"""

import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def add_password_column():
    """Add password column to employees table"""
    try:
        # Database connection parameters
        connection = mysql.connector.connect(
            host=os.getenv('MYSQL_HOST', 'localhost'),
            database=os.getenv('MYSQL_DB', 'esco_suite'),
            user=os.getenv('MYSQL_USER', 'root'),
            password=os.getenv('MYSQL_PASSWORD', '')
        )
        
        if connection.is_connected():
            cursor = connection.cursor()
            
            # Check if password column already exists
            cursor.execute("""
                SELECT COLUMN_NAME 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = %s 
                AND TABLE_NAME = 'employees' 
                AND COLUMN_NAME = 'password'
            """, (os.getenv('MYSQL_DB', 'esco_suite'),))
            
            if cursor.fetchone():
                print("Password column already exists in employees table")
                return True
            
            # Add password column
            cursor.execute("""
                ALTER TABLE employees 
                ADD COLUMN password VARCHAR(255) NOT NULL DEFAULT 'default_password'
            """)
            
            connection.commit()
            print("Successfully added password column to employees table")
            
            # Update existing employees with default password
            cursor.execute("""
                UPDATE employees 
                SET password = CONCAT('emp_', id, '_', SUBSTRING(MD5(email), 1, 8))
                WHERE password = 'default_password'
            """)
            
            connection.commit()
            print("Updated existing employees with generated passwords")
            
            return True
            
    except Error as e:
        print(f"Error adding password column: {e}")
        return False
        
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

if __name__ == "__main__":
    print("Adding password column to employees table...")
    if add_password_column():
        print("Migration completed successfully!")
    else:
        print("Migration failed!")
