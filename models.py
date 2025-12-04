# In a file named models.py

# Use the shared db instance to avoid circular imports
from database import db
from datetime import datetime
from flask_login import UserMixin  # <-- ADD THIS LINE
from datetime import datetime

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    role = db.Column(db.String(50), default='member')

class Bid(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    current_stage = db.Column(db.String(50), default='analyzer')
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    user = db.relationship('User', backref='bids')

class Company(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Company {self.name}>'

class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    company_id = db.Column(db.Integer, db.ForeignKey('company.id'), nullable=False)
    company = db.relationship('Company', backref='projects')
    start_date = db.Column(db.DateTime, nullable=True)  # Made nullable and optional
    due_date = db.Column(db.DateTime, nullable=False)
    revenue = db.Column(db.Float, default=0.0)  # Added revenue field
    status = db.Column(db.String(50), default='active')  # active, completed, on_hold, cancelled
    progress = db.Column(db.Integer, default=0)  # 0-100 percentage
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Project {self.name} - {self.company.name}>'

class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    project = db.relationship('Project', backref='tasks')
    assigned_user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    assigned_user = db.relationship('User', backref='assigned_tasks')
    due_date = db.Column(db.DateTime, nullable=False)
    status = db.Column(db.String(50), default='pending')  # pending, in_progress, completed, overdue
    priority = db.Column(db.String(20), default='medium')  # low, medium, high, urgent
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Task {self.name} - {self.project.name}>'

class Log(db.Model):
    __tablename__ = 'logs'
    id = db.Column(db.Integer, primary_key=True)
    action = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    user = db.relationship('User', backref='logs')

    # This helps in debugging by showing a readable representation of the object
    def __repr__(self):
        return f'<Log {self.id}: {self.action[:50]}>'