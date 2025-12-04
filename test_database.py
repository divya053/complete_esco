#!/usr/bin/env python3
"""
Test script to verify database structure
"""

import sqlite3
import os

def test_database():
    """Test the database structure"""
    db_path = 'app_data.db'
    
    if not os.path.exists(db_path):
        print("Database file not found!")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check project table structure
        cursor.execute("PRAGMA table_info(project)")
        columns = cursor.fetchall()
        
        print("Project table columns:")
        for col in columns:
            print(f"  {col[1]} - {col[2]} - Nullable: {not col[3]}")
        
        # Check if we can insert a project with NULL start_date
        try:
            cursor.execute("""
                INSERT INTO project (name, company_id, due_date, revenue, status, progress)
                VALUES (?, ?, ?, ?, ?, ?)
            """, ('Test Project', 1, '2024-12-31 00:00:00', 1000.0, 'active', 0))
            
            cursor.execute("DELETE FROM project WHERE name = 'Test Project'")
            print("✅ Database structure is correct - start_date can be NULL")
            
        except Exception as e:
            print(f"❌ Database structure issue: {e}")
            return False
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing database structure...")
    success = test_database()
    if success:
        print("✅ Database test passed!")
    else:
        print("❌ Database test failed!")


