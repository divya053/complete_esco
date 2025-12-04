import os
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_socketio import SocketIO
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
from flask_mysqldb import MySQL
from MySQLdb.cursors import DictCursor

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

# MySQL Config
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'flaskuser'
app.config['MYSQL_PASSWORD'] = 'flaskpass123'
app.config['MYSQL_DB'] = 'flask_app'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app)
socketio = SocketIO(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Custom User class
class User(UserMixin):
    def __init__(self, id, email, password, is_admin, role):
        self.id = id
        self.email = email
        self.password = password
        self.is_admin = bool(is_admin)
        self.role = role

@login_manager.user_loader
def load_user(user_id):
    cur = mysql.connection.cursor(DictCursor)
    cur.execute("SELECT * FROM users WHERE id=%s", (user_id,))
    data = cur.fetchone()
    cur.close()
    if data:
        return User(data['id'], data['email'], data['password'], data['is_admin'], data['role'])
    return None

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        cur = mysql.connection.cursor(DictCursor)
        cur.execute("SELECT * FROM users WHERE email=%s AND password=%s", (email, password))
        user = cur.fetchone()
        cur.close()
        if user:
            user_obj = User(user['id'], user['email'], user['password'], user['is_admin'], user['role'])
            login_user(user_obj)
            if user['is_admin']:
                return redirect(url_for('master_dashboard'))
            return redirect(url_for('dashboard'))
        return "Invalid credentials", 401
    return render_template('login.html')

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/logs')
@login_required
def logs_page():
    cur = mysql.connection.cursor(DictCursor)
    cur.execute("SELECT * FROM logs ORDER BY timestamp DESC")
    logs = cur.fetchall()
    cur.close()
    return render_template("logs.html", logs=logs)

@app.route('/admin/users', methods=['GET', 'POST'])
@login_required
def users_admin():
    if not current_user.is_admin:
        return "Access Denied", 403
    cur = mysql.connection.cursor(DictCursor)
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        role = request.form.get('role', 'member')
        is_admin = 1 if role.lower() == "admin" else 0
        cur.execute("INSERT INTO users (email, password, role, is_admin) VALUES (%s, %s, %s, %s)",
                    (email, password, role, is_admin))
        mysql.connection.commit()
        return redirect(url_for('users_admin'))
    cur.execute("SELECT * FROM users")
    users = cur.fetchall()
    cur.close()
    return render_template('users.html', users=users)

@app.route('/')
def index():
    if current_user.is_authenticated:
        if current_user.is_admin:
            return redirect(url_for('master_dashboard'))
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.is_admin:
        return redirect(url_for('master_dashboard'))
    return redirect(url_for('role_dashboard'))

@app.route('/dashboard/role')
@login_required
def role_dashboard():
    role = current_user.role or "member"
    cur = mysql.connection.cursor(DictCursor)
    cur.execute("""
        SELECT b.id, b.name, b.current_stage, u.email AS user_email, u.role AS user_role
        FROM bids b JOIN users u ON b.user_id = u.id
        WHERE LOWER(u.role) = %s
    """, (role.lower(),))
    bids = cur.fetchall()
    cur.close()
    return render_template("dashboard.html", bids=bids, user=current_user, role=role)

@app.route('/master-dashboard')
@login_required
def master_dashboard():
    if not current_user.is_admin:
        return "Access Denied", 403
    cur = mysql.connection.cursor(DictCursor)
    cur.execute("""
        SELECT b.id, b.name, b.current_stage, u.email AS user_email, u.role AS user_role
        FROM bids b JOIN users u ON b.user_id = u.id
    """)
    bids = cur.fetchall()

    # stage counts
    cur.execute("SELECT current_stage, COUNT(*) as total FROM bids GROUP BY current_stage")
    stage_counts = {row['current_stage']: row['total'] for row in cur.fetchall()}
    cur.close()

    stats = {"total_bids": len(bids)}
    return render_template("master_dashboard.html", bids=bids, data=stats, stage_counts=stage_counts)
# @app.route('/admin/users/edit/<int:user_id>', methods=['GET', 'POST'])
# @login_required
# def edit_user(user_id):
#     if not current_user.is_admin:
#         return "Access Denied", 403

#     cur = mysql.connection.cursor(DictCursor)
#     if request.method == 'POST':
#         email = request.form['email']
#         role = request.form['role']
#         is_admin = 1 if role.lower() == "admin" else 0
#         cur.execute("UPDATE users SET email=%s, role=%s, is_admin=%s WHERE id=%s",
#                     (email, role, is_admin, user_id))
#         mysql.connection.commit()
#         cur.close()
#         return redirect(url_for('users_admin'))

#     cur.execute("SELECT * FROM users WHERE id=%s", (user_id,))
#     user = cur.fetchone()
#     cur.close()
#     return render_template('edit_user.html', user=user)

# @app.route('/admin/users/delete/<int:user_id>', methods=['POST'])
# @login_required
# def delete_user(user_id):
#     if not current_user.is_admin:
#         return "Access Denied", 403
#     cur = mysql.connection.cursor()
#     cur.execute("DELETE FROM users WHERE id=%s", (user_id,))
#     mysql.connection.commit()
#     cur.close()
#     return redirect(url_for('users_admin'))

@app.route('/admin/users/delete/<int:user_id>', methods=['POST'])
@login_required
def delete_user(user_id):
    if not current_user.is_admin:
        return "Access Denied", 403

    try:
        cur = mysql.connection.cursor()
        # Step 1: Delete related logs
        cur.execute("DELETE FROM logs WHERE user_id=%s", (user_id,))
        # Step 2: Delete related bids
        cur.execute("DELETE FROM bids WHERE user_id=%s", (user_id,))
        # Step 3: Delete the user
        cur.execute("DELETE FROM users WHERE id=%s", (user_id,))
        mysql.connection.commit()
        cur.close()
        return redirect(url_for('users_admin'))
    except Exception as e:
        mysql.connection.rollback()
        return f"Error deleting user: {str(e)}", 500




@app.route('/admin/users/edit/<int:user_id>', methods=['GET', 'POST'])
@login_required
def edit_user(user_id):
    if not current_user.is_admin:
        return "Access Denied", 403

    cur = mysql.connection.cursor(DictCursor)
    if request.method == 'POST':
        email = request.form['email']
        role = request.form['role']
        is_admin = 1 if role.lower() == "admin" else 0
        cur.execute("UPDATE users SET email=%s, role=%s, is_admin=%s WHERE id=%s",
                    (email, role, is_admin, user_id))
        mysql.connection.commit()
        cur.close()
        return redirect(url_for('users_admin'))

    cur.execute("SELECT * FROM users WHERE id=%s", (user_id,))
    user = cur.fetchone()
    cur.close()
    return render_template('edit_user.html', user=user)

@app.route('/api/update_stage/<int:bid_id>', methods=['POST'])
@login_required
def update_stage(bid_id):
    data = request.get_json()
    new_stage = data.get("stage")
    cur = mysql.connection.cursor()
    cur.execute("UPDATE bids SET current_stage=%s WHERE id=%s", (new_stage, bid_id))
    mysql.connection.commit()
    log_msg = f"User {current_user.email} updated Bid {bid_id} to stage {new_stage}"
    cur.execute("INSERT INTO logs (action, user_id) VALUES (%s, %s)", (log_msg, current_user.id))
    mysql.connection.commit()
    cur.close()
    socketio.emit('master_update', {"bid_id": bid_id, "stage": new_stage})
    return jsonify({"success": f"Bid {bid_id} updated"})

if __name__ == "__main__":
    socketio.run(app, debug=True, port=5001)
