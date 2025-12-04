#!/usr/bin/env python3
"""
Script to add team_archive column to bid_checklists table
"""

import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def add_team_archive_column():
    """Add team_archive column to bid_checklists table"""
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
            
            # Check if team_archive column already exists
            cursor.execute("""
                SELECT COLUMN_NAME 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = %s 
                AND TABLE_NAME = 'bid_checklists' 
                AND COLUMN_NAME = 'team_archive'
            """, (os.getenv('MYSQL_DB', 'esco_suite'),))
            
            if cursor.fetchone():
                print("team_archive column already exists in bid_checklists table")
                return True
            
            # Add team_archive column
            cursor.execute("""
                ALTER TABLE bid_checklists 
                ADD COLUMN team_archive VARCHAR(50) DEFAULT NULL
            """)
            
            connection.commit()
            print("Successfully added team_archive column to bid_checklists table")
            return True
            
    except Error as e:
        print(f"Error adding team_archive column: {e}")
        return False
        
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

if __name__ == "__main__":
    print("Adding team_archive column to bid_checklists table...")
    if add_team_archive_column():
        print("Migration completed successfully!")
    else:
        print("Migration failed!")
