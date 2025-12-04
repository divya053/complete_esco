#!/usr/bin/env python3
"""
Database migration script to add revenue field to Project model
"""

import os
import sys
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app_v2 import app
from database import db

def migrate_database():
    """Add revenue field to Project table if it doesn't exist"""
    with app.app_context():
        try:
            # Check if revenue column exists
            inspector = db.inspect(db.engine)
            columns = [col['name'] for col in inspector.get_columns('project')]
            
            if 'revenue' not in columns:
                print("Adding revenue column to Project table...")
                db.engine.execute('ALTER TABLE project ADD COLUMN revenue FLOAT DEFAULT 0.0')
                print("✅ Revenue column added successfully!")
            else:
                print("✅ Revenue column already exists!")
                
            # Check if start_date is nullable
            start_date_col = next((col for col in inspector.get_columns('project') if col['name'] == 'start_date'), None)
            if start_date_col and not start_date_col['nullable']:
                print("Making start_date column nullable...")
                db.engine.execute('ALTER TABLE project ALTER COLUMN start_date DROP NOT NULL')
                print("✅ start_date column made nullable!")
            else:
                print("✅ start_date column is already nullable!")
                
            print("🎉 Database migration completed successfully!")
            
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            return False
            
    return True

if __name__ == "__main__":
    print("Starting database migration...")
    success = migrate_database()
    if success:
        print("Migration completed successfully!")
    else:
        print("Migration failed!")
        sys.exit(1)


