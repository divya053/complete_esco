#!/usr/bin/env python3
"""
Database migration script to add assignee_email column to bid_assign table
"""

import os
from flask import Flask
from flask_mysqldb import MySQL
from MySQLdb.cursors import DictCursor

# Initialize Flask app for database connection
app = Flask(__name__)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'esco'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app)

def migrate_assignee_email():
    """Add assignee_email column to bid_assign table if it doesn't exist"""
    try:
        cur = mysql.connection.cursor()
        
        # Check if assignee_email column exists
        cur.execute("""
            SELECT COLUMN_NAME 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = 'esco' 
            AND TABLE_NAME = 'bid_assign' 
            AND COLUMN_NAME = 'assignee_email'
        """)
        
        column_exists = cur.fetchone()
        
        if not column_exists:
            print("Adding assignee_email column to bid_assign table...")
            cur.execute("""
                ALTER TABLE bid_assign 
                ADD COLUMN assignee_email VARCHAR(100) AFTER person_name
            """)
            mysql.connection.commit()
            print("✅ assignee_email column added successfully!")
        else:
            print("✅ assignee_email column already exists!")
        
        # Verify the column was added
        cur.execute("DESCRIBE bid_assign")
        columns = cur.fetchall()
        print("\nCurrent bid_assign table structure:")
        for col in columns:
            print(f"  - {col['Field']} ({col['Type']})")
        
        cur.close()
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        if 'cur' in locals():
            cur.close()
        return False

def main():
    """Run the migration"""
    print("=" * 60)
    print("DATABASE MIGRATION: Add assignee_email column")
    print("=" * 60)
    
    with app.app_context():
        success = migrate_assignee_email()
        
        if success:
            print("\n✅ Migration completed successfully!")
            print("You can now run the Flask app without the assignee_email error.")
        else:
            print("\n❌ Migration failed!")
            print("Please check the error message above and try again.")

if __name__ == "__main__":
    main()
