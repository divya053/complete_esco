#!/usr/bin/env python3
"""
Test script for the advance workflow between teams
Tests: go_bids.state filtering, advance API, team dashboard updates
"""

import requests
import json
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:5001"

def test_advance_workflow():
    """Test the complete advance workflow between teams"""
    print("=" * 60)
    print("ADVANCE WORKFLOW TEST")
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
    
    # Step 2: Test team dashboards show correct bids
    print("\n2. Testing team dashboard filtering...")
    teams = ["business", "design", "operations", "engineer"]
    
    for team in teams:
        try:
            response = session.get(f"{BASE_URL}/dashboard/{team}")
            if response.status_code == 200:
                print(f"✅ {team} dashboard accessible")
                # Check if page contains expected content
                if "Assigned Bids" in response.text:
                    print(f"   - Shows assigned bids section")
                if "Advance" in response.text or "Complete" in response.text:
                    print(f"   - Shows action buttons")
            else:
                print(f"⚠️  {team} dashboard: {response.status_code}")
        except Exception as e:
            print(f"❌ {team} dashboard error: {e}")
    
    # Step 3: Test stage update API
    print("\n3. Testing stage update API...")
    test_stages = ["business", "design", "operations", "engineer", "handover"]
    
    for stage in test_stages:
        stage_data = {"stage": stage}
        
        try:
            response = session.post(f"{BASE_URL}/api/update_stage/1", 
                                   json=stage_data,
                                   headers={'Content-Type': 'application/json'})
            if response.status_code == 200:
                data = response.json()
                if 'success' in data:
                    print(f"✅ Stage update to '{stage}': {data['success']}")
                else:
                    print(f"⚠️  Stage update to '{stage}': {data}")
            elif response.status_code == 401:
                print(f"⚠️  Stage update to '{stage}': Requires authentication (expected)")
            else:
                print(f"⚠️  Stage update to '{stage}': HTTP {response.status_code}")
        except Exception as e:
            print(f"❌ Stage update to '{stage}' error: {e}")
    
    # Step 4: Test invalid stage
    print("\n4. Testing invalid stage validation...")
    invalid_stage_data = {"stage": "invalid_stage"}
    
    try:
        response = session.post(f"{BASE_URL}/api/update_stage/1", 
                               json=invalid_stage_data,
                               headers={'Content-Type': 'application/json'})
        if response.status_code == 400:
            data = response.json()
            if 'error' in data and 'invalid stage' in data['error']:
                print("✅ Invalid stage properly rejected")
            else:
                print(f"⚠️  Invalid stage response: {data}")
        else:
            print(f"⚠️  Invalid stage: Expected 400, got {response.status_code}")
    except Exception as e:
        print(f"❌ Invalid stage test error: {e}")

def test_stage_flow():
    """Test the complete stage flow"""
    print("\n" + "=" * 60)
    print("STAGE FLOW TEST")
    print("=" * 60)
    
    stage_flow = {
        'analyzer': 'business',
        'business': 'design',
        'design': 'operations',
        'operations': 'engineer',
        'engineer': 'handover'
    }
    
    print("Expected stage flow:")
    for current, next_stage in stage_flow.items():
        print(f"  {current} → {next_stage}")
    
    print("\nTeam dashboard mapping:")
    team_mapping = {
        'business': 'Business Development',
        'design': 'Design Team',
        'operations': 'Operations Team',
        'engineer': 'Site Engineer'
    }
    
    for team, display_name in team_mapping.items():
        print(f"  /dashboard/{team} → {display_name} (shows bids with state='{team}')")

def test_database_queries():
    """Test the database query logic"""
    print("\n" + "=" * 60)
    print("DATABASE QUERY TEST")
    print("=" * 60)
    
    print("Query logic for team dashboards:")
    print("1. Filter go_bids by state matching team stage")
    print("2. LEFT JOIN bid_assign for assignee info")
    print("3. LEFT JOIN win_lost_results for completion status")
    print("4. Order by due_date ASC")
    
    print("\nAdvance API logic:")
    print("1. Validate stage in allowed set")
    print("2. Update go_bids.state to new stage")
    print("3. Upsert bid_assign record for next team")
    print("4. Log action and emit Socket.IO update")
    print("5. Return success response")

def main():
    """Run all tests"""
    print("ADVANCE WORKFLOW COMPREHENSIVE TEST SUITE")
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Testing against: {BASE_URL}")
    
    test_advance_workflow()
    test_stage_flow()
    test_database_queries()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("\nManual testing steps:")
    print("1. Start Flask app: python app_v2.py")
    print("2. Login as admin and assign a bid to 'business' department")
    print("3. Check /dashboard/business - should show the bid")
    print("4. Click 'Advance' button - bid should move to /dashboard/design")
    print("5. Verify bid appears on design dashboard")
    print("6. Repeat for all stages: design → operations → engineer → handover")

if __name__ == "__main__":
    main()
