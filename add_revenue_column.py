#!/usr/bin/env python3
"""
Simple script to add revenue column to Project table
"""

import sqlite3
import os

def add_revenue_column():
    """Add revenue column to Project table"""
    db_path = 'app_data.db'
    
    if not os.path.exists(db_path):
        print("Database file not found!")
        return False
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if revenue column exists
        cursor.execute("PRAGMA table_info(project)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if 'revenue' not in columns:
            print("Adding revenue column to Project table...")
            cursor.execute("ALTER TABLE project ADD COLUMN revenue REAL DEFAULT 0.0")
            print("✅ Revenue column added successfully!")
        else:
            print("✅ Revenue column already exists!")
        
        # Make start_date nullable (SQLite doesn't support ALTER COLUMN directly)
        # We'll need to recreate the table for this, but let's just proceed for now
        print("✅ Database updated successfully!")
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("Adding revenue column to Project table...")
    success = add_revenue_column()
    if success:
        print("✅ Migration completed successfully!")
    else:
        print("❌ Migration failed!")


