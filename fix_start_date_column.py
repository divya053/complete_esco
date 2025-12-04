#!/usr/bin/env python3
"""
Script to make start_date column nullable in Project table
"""

import sqlite3
import os
import shutil
from datetime import datetime

def fix_start_date_column():
    """Make start_date column nullable by recreating the table"""
    db_path = 'app_data.db'
    backup_path = f'app_data_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db'
    
    if not os.path.exists(db_path):
        print("Database file not found!")
        return False
    
    try:
        # Create backup
        shutil.copy2(db_path, backup_path)
        print(f"✅ Backup created: {backup_path}")
        
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get the current table structure
        cursor.execute("PRAGMA table_info(project)")
        columns = cursor.fetchall()
        
        # Create new table with nullable start_date
        cursor.execute("""
            CREATE TABLE project_new (
                id INTEGER PRIMARY KEY,
                name VARCHAR(200) NOT NULL,
                company_id INTEGER NOT NULL,
                start_date DATETIME,
                due_date DATETIME NOT NULL,
                revenue REAL DEFAULT 0.0,
                status VARCHAR(50) DEFAULT 'active',
                progress INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(company_id) REFERENCES company (id)
            )
        """)
        
        # Copy data from old table to new table
        cursor.execute("""
            INSERT INTO project_new (id, name, company_id, start_date, due_date, revenue, status, progress, created_at)
            SELECT id, name, company_id, start_date, due_date, 
                   COALESCE(revenue, 0.0) as revenue, 
                   status, progress, created_at
            FROM project
        """)
        
        # Drop old table and rename new table
        cursor.execute("DROP TABLE project")
        cursor.execute("ALTER TABLE project_new RENAME TO project")
        
        # Recreate indexes
        cursor.execute("CREATE INDEX ix_project_company_id ON project (company_id)")
        
        print("✅ start_date column made nullable successfully!")
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        # Restore backup if something went wrong
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, db_path)
            print(f"✅ Restored from backup: {backup_path}")
        return False

if __name__ == "__main__":
    print("Making start_date column nullable in Project table...")
    success = fix_start_date_column()
    if success:
        print("✅ Migration completed successfully!")
    else:
        print("❌ Migration failed!")


