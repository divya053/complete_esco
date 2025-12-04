#!/usr/bin/env python3
"""
Auto-assign go_bids into bid_assign based on revenue ranking:
 - Highest revenue -> Chris
 - Lowest revenue -> Gurki
 - All remaining -> Inder

This script is idempotent; it upserts into bid_assign and updates existing rows.
"""

from flask import Flask
from flask_mysqldb import MySQL


ASSIGNEES = {
    'TOP': {'name': 'Chris', 'email': ''},
    'MID': {'name': 'Inder', 'email': ''},
    'LOW': {'name': 'Gurki', 'email': ''},
}


def get_mysql():
    app = Flask(__name__)
    app.config['MYSQL_HOST'] = 'localhost'
    app.config['MYSQL_USER'] = 'root'
    app.config['MYSQL_PASSWORD'] = ''
    app.config['MYSQL_DB'] = 'esco'
    app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
    return MySQL(app), app


def main():
    mysql, app = get_mysql()
    with app.app_context():
        cur = mysql.connection.cursor()
        try:
            # Pull all go_bids with revenue, sort desc
            cur.execute("""
                SELECT g_id, b_name, in_date, due_date, state, scope, type, company,
                       COALESCE(scoring, 0) AS value, COALESCE(revenue, 0) AS revenue
                FROM go_bids
                WHERE COALESCE(revenue, 0) >= 0
                ORDER BY revenue DESC, g_id ASC
            """)
            rows = cur.fetchall()

            if not rows:
                print('No go_bids found to assign.')
                return

            top_idx = 0
            low_idx = len(rows) - 1 if rows else 0

            for idx, r in enumerate(rows):
                if idx == top_idx:
                    assignee = ASSIGNEES['TOP']
                elif idx == low_idx:
                    assignee = ASSIGNEES['LOW']
                else:
                    assignee = ASSIGNEES['MID']

                # Upsert into bid_assign
                # Try update first
                cur.execute("SELECT a_id FROM bid_assign WHERE g_id=%s", (r['g_id'],))
                existing = cur.fetchone()
                if existing:
                    cur.execute(
                        """
                        UPDATE bid_assign
                        SET depart=%s, person_name=%s, assignee_email=%s, state=%s,
                            status='assigned', value=%s, revenue=%s
                        WHERE g_id=%s
                        """,
                        ('business', assignee['name'], assignee['email'], r['state'] or 'business',
                         r['value'], r['revenue'], r['g_id'])
                    )
                else:
                    # Insert
                    try:
                        cur.execute(
                            """
                            INSERT INTO bid_assign (
                                g_id, b_name, in_date, due_date, state, scope, type, company,
                                depart, person_name, assignee_email, status, value, revenue
                            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'assigned',%s,%s)
                            """,
                            (
                                r['g_id'], r['b_name'], r['in_date'], r['due_date'], r['state'], r['scope'], r['type'], r['company'],
                                'business', assignee['name'], assignee['email'], r['value'], r['revenue']
                            )
                        )
                    except Exception as e:
                        # Fallback insert if assignee_email missing in schema
                        if "Unknown column 'assignee_email'" in str(e):
                            cur.execute(
                                """
                                INSERT INTO bid_assign (
                                    g_id, b_name, in_date, due_date, state, scope, type, company,
                                    depart, person_name, status, value, revenue
                                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'assigned',%s,%s)
                                """,
                                (
                                    r['g_id'], r['b_name'], r['in_date'], r['due_date'], r['state'], r['scope'], r['type'], r['company'],
                                    'business', assignee['name'], r['value'], r['revenue']
                                )
                            )
                        else:
                            raise e

            mysql.connection.commit()
            print('✅ Auto-assignment completed.')
        except Exception as e:
            mysql.connection.rollback()
            print(f'❌ Auto-assignment failed: {e}')
        finally:
            cur.close()


if __name__ == '__main__':
    main()


