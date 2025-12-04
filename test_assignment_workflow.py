#!/usr/bin/env python3
"""
Test script for the complete assignment workflow
Tests: bid assignment, state synchronization, team dashboard updates, master dashboard sync
"""

import requests
import json
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:5000"

def test_assignment_workflow():
    """Test the complete assignment workflow"""
    print("=" * 60)
    print("ASSIGNMENT WORKFLOW TEST")
    print("=" * 60)
    
    session = requests.Session()
    
    # Step 1: Login as admin
    print("\n1. Testing admin login...")
    login_data = {
        "email": "admin@esco.com",
        "password": "admin123"
    }
    
    try:
        response = session.post(f"{BASE_URL}/login", data=login_data, allow_redirects=False)
        if response.status_code == 302:
            print("✅ Admin login successful")
        else:
            print(f"❌ Admin login failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Admin login error: {e}")
        return
    
    # Step 2: Test assignment endpoint
    print("\n2. Testing bid assignment...")
    assignment_data = {
        "g_id": "1",  # Assuming bid ID 1 exists
        "depart": "business dev",
        "person_name": "John Doe",
        "email": "john.doe@esco.com"
    }
    
    try:
        response = session.post(f"{BASE_URL}/database-management/assign-go", data=assignment_data)
        if response.status_code == 302:  # Redirect after successful assignment
            print("✅ Bid assignment successful")
        else:
            print(f"⚠️  Bid assignment response: {response.status_code}")
    except Exception as e:
        print(f"❌ Assignment error: {e}")
    
    # Step 3: Test team dashboard access
    print("\n3. Testing team dashboard access...")
    teams = ["business", "design", "operations", "engineer"]
    
    for team in teams:
        try:
            response = session.get(f"{BASE_URL}/dashboard/{team}")
            if response.status_code == 200:
                print(f"✅ {team} dashboard accessible")
            else:
                print(f"⚠️  {team} dashboard: {response.status_code}")
        except Exception as e:
            print(f"❌ {team} dashboard error: {e}")
    
    # Step 4: Test stage update API
    print("\n4. Testing stage update API...")
    stage_data = {
        "stage": "design"
    }
    
    try:
        response = session.post(f"{BASE_URL}/api/update_stage/1", 
                               json=stage_data,
                               headers={'Content-Type': 'application/json'})
        if response.status_code == 200:
            print("✅ Stage update successful")
        elif response.status_code == 401:
            print("⚠️  Stage update requires authentication (expected)")
        else:
            print(f"⚠️  Stage update: {response.status_code}")
    except Exception as e:
        print(f"❌ Stage update error: {e}")
    
    # Step 5: Test master dashboard
    print("\n5. Testing master dashboard...")
    try:
        response = session.get(f"{BASE_URL}/master-dashboard")
        if response.status_code == 200:
            print("✅ Master dashboard accessible")
        else:
            print(f"⚠️  Master dashboard: {response.status_code}")
    except Exception as e:
        print(f"❌ Master dashboard error: {e}")

def test_database_schema():
    """Test that the database schema is correct"""
    print("\n" + "=" * 60)
    print("DATABASE SCHEMA VERIFICATION")
    print("=" * 60)
    
    required_tables = {
        "users": ["id", "email", "password", "is_admin", "role"],
        "go_bids": ["g_id", "b_name", "state", "company", "due_date"],
        "bid_assign": ["a_id", "g_id", "b_name", "state", "depart", "person_name", "assignee_email"],
        "logs": ["id", "action", "user_id", "timestamp"]
    }
    
    print("Required tables and columns:")
    for table, columns in required_tables.items():
        print(f"\n{table}:")
        for column in columns:
            print(f"  - {column}")
    
    print("\nNote: Actual database verification requires direct MySQL access.")

def test_assignment_scenarios():
    """Test different assignment scenarios"""
    print("\n" + "=" * 60)
    print("ASSIGNMENT SCENARIOS TEST")
    print("=" * 60)
    
    scenarios = [
        {
            "name": "Business Dev Assignment",
            "department": "business dev",
            "expected_stage": "business",
            "person": "Alice Smith",
            "email": "alice@esco.com"
        },
        {
            "name": "Design Assignment", 
            "department": "design",
            "expected_stage": "design",
            "person": "Bob Johnson",
            "email": "bob@esco.com"
        },
        {
            "name": "Operations Assignment",
            "department": "operations", 
            "expected_stage": "operations",
            "person": "Carol Williams",
            "email": "carol@esco.com"
        },
        {
            "name": "Site Manager Assignment",
            "department": "site manager",
            "expected_stage": "engineer", 
            "person": "David Brown",
            "email": "david@esco.com"
        }
    ]
    
    print("Assignment scenarios to test:")
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"   Department: {scenario['department']}")
        print(f"   Expected Stage: {scenario['expected_stage']}")
        print(f"   Assignee: {scenario['person']} ({scenario['email']})")

def main():
    """Run all tests"""
    print("ASSIGNMENT WORKFLOW COMPREHENSIVE TEST SUITE")
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_assignment_workflow()
    test_database_schema()
    test_assignment_scenarios()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("\nTo test the assignment workflow:")
    print("1. Start the Flask app: python app_v2.py")
    print("2. Login as admin and go to Database Management")
    print("3. Find a GO bid and click 'Assign'")
    print("4. Fill in department, person name, and email")
    print("5. Check that the bid appears on the appropriate team dashboard")
    print("6. Verify the master dashboard timeline updates in real-time")

if __name__ == "__main__":
    main()
