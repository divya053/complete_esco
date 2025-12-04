#!/usr/bin/env python3
"""
Add revenue column to go_bids and bid_assign (assigned_bids) and backfill values.

This script connects to the local MySQL database configured for the ESCO app
and performs a safe migration:
  - Adds revenue DECIMAL(15,2) DEFAULT 0.00 to go_bids (if missing)
  - Adds revenue DECIMAL(15,2) DEFAULT 0.00 to bid_assign (if missing)
  - Backfills bid_assign.revenue from existing bid_assign.value
  - Backfills go_bids.revenue from bid_assign.value by g_id
"""

from flask import Flask
from flask_mysqldb import MySQL


def get_mysql():
    app = Flask(__name__)
    app.config['MYSQL_HOST'] = 'localhost'
    app.config['MYSQL_USER'] = 'root'
    app.config['MYSQL_PASSWORD'] = ''
    app.config['MYSQL_DB'] = 'esco'
    app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
    return MySQL(app), app


def column_exists(cur, table: str, column: str) -> bool:
    cur.execute(
        """
        SELECT 1
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = %s
          AND COLUMN_NAME = %s
        """,
        (table, column),
    )
    return cur.fetchone() is not None


def add_column_if_missing(cur, table: str, ddl: str):
    # ddl example: "ALTER TABLE go_bids ADD COLUMN revenue DECIMAL(15,2) DEFAULT 0.00"
    cur.execute(ddl)


def migrate():
    mysql, app = get_mysql()
    with app.app_context():
        cur = mysql.connection.cursor()
        try:
            # go_bids.revenue
            if not column_exists(cur, 'go_bids', 'revenue'):
                print('Adding go_bids.revenue ...')
                add_column_if_missing(cur, 'go_bids',
                    'ALTER TABLE go_bids ADD COLUMN revenue DECIMAL(15,2) DEFAULT 0.00')

            # bid_assign.revenue
            if not column_exists(cur, 'bid_assign', 'revenue'):
                print('Adding bid_assign.revenue ...')
                add_column_if_missing(cur, 'bid_assign',
                    'ALTER TABLE bid_assign ADD COLUMN revenue DECIMAL(15,2) DEFAULT 0.00')

            # Backfill bid_assign.revenue from bid_assign.value
            print('Backfilling bid_assign.revenue from bid_assign.value ...')
            cur.execute(
                """
                UPDATE bid_assign
                SET revenue = COALESCE(value, 0)
                WHERE revenue IS NULL OR revenue = 0
                """
            )

            # Backfill go_bids.revenue from bid_assign.value via g_id
            print('Backfilling go_bids.revenue from bid_assign.value by g_id ...')
            cur.execute(
                """
                UPDATE go_bids gb
                JOIN (
                    SELECT g_id, MAX(COALESCE(value,0)) AS v
                    FROM bid_assign
                    GROUP BY g_id
                ) ba ON ba.g_id = gb.g_id
                SET gb.revenue = ba.v
                WHERE gb.revenue IS NULL OR gb.revenue = 0
                """
            )

            mysql.connection.commit()
            print('✅ Migration completed successfully.')
        except Exception as e:
            mysql.connection.rollback()
            print(f'❌ Migration failed: {e}')
        finally:
            cur.close()


if __name__ == '__main__':
    migrate()


