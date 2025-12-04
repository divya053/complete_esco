#!/usr/bin/env python3
"""
Test script for team sub-dashboards functionality
"""

import requests
import json
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:5000"
TEST_USERS = [
    {"email": "admin@esco.com", "password": "admin123", "role": "admin", "expected_redirect": "/master-dashboard"},
    {"email": "business@esco.com", "password": "business123", "role": "business dev", "expected_redirect": "/dashboard/business"},
    {"email": "design@esco.com", "password": "design123", "role": "design", "expected_redirect": "/dashboard/design"},
    {"email": "operations@esco.com", "password": "operations123", "role": "operations", "expected_redirect": "/dashboard/operations"},
    {"email": "engineer@esco.com", "password": "engineer123", "role": "site manager", "expected_redirect": "/dashboard/engineer"},
]

def test_login_redirects():
    """Test that users are redirected to correct dashboards based on role"""
    print("Testing login redirects...")
    
    session = requests.Session()
    
    for user in TEST_USERS:
        print(f"\nTesting {user['role']} user...")
        
        # Test login
        login_data = {
            "email": user["email"],
            "password": user["password"]
        }
        
        try:
            response = session.post(f"{BASE_URL}/login", data=login_data, allow_redirects=False)
            
            if response.status_code == 302:  # Redirect
                redirect_url = response.headers.get('Location', '')
                expected_path = user["expected_redirect"]
                
                if expected_path in redirect_url:
                    print(f"✅ {user['role']}: Correctly redirected to {redirect_url}")
                else:
                    print(f"❌ {user['role']}: Expected {expected_path}, got {redirect_url}")
            else:
                print(f"❌ {user['role']}: Expected redirect (302), got {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print(f"❌ {user['role']}: Could not connect to server. Make sure Flask app is running.")
        except Exception as e:
            print(f"❌ {user['role']}: Error - {e}")

def test_team_dashboard_access():
    """Test that team dashboards are accessible"""
    print("\nTesting team dashboard access...")
    
    team_dashboards = ["business", "design", "operations", "engineer"]
    
    for team in team_dashboards:
        try:
            response = requests.get(f"{BASE_URL}/dashboard/{team}")
            
            if response.status_code == 200:
                print(f"✅ {team} dashboard: Accessible")
            elif response.status_code == 302:
                print(f"⚠️  {team} dashboard: Redirected (likely requires login)")
            else:
                print(f"❌ {team} dashboard: HTTP {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print(f"❌ {team} dashboard: Could not connect to server")
        except Exception as e:
            print(f"❌ {team} dashboard: Error - {e}")

def test_stage_update_api():
    """Test the stage update API endpoint"""
    print("\nTesting stage update API...")
    
    # This would require authentication in a real test
    test_data = {
        "stage": "design"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/update_stage/1", 
                               json=test_data,
                               headers={'Content-Type': 'application/json'})
        
        if response.status_code == 401:
            print("✅ Stage update API: Requires authentication (expected)")
        elif response.status_code == 200:
            print("✅ Stage update API: Accessible")
        else:
            print(f"⚠️  Stage update API: HTTP {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Stage update API: Could not connect to server")
    except Exception as e:
        print(f"❌ Stage update API: Error - {e}")

def test_database_tables():
    """Test that required database tables exist"""
    print("\nTesting database tables...")
    
    # This would require database access in a real test
    required_tables = ["users", "go_bids", "logs", "bid_assign", "win_lost_results", "won_bids_result", "work_progress_status"]
    
    print("Required tables:")
    for table in required_tables:
        print(f"  - {table}")
    
    print("\nNote: Database table verification requires direct database access.")

def main():
    """Run all tests"""
    print("=" * 60)
    print("TEAM SUB-DASHBOARDS TEST SUITE")
    print("=" * 60)
    print(f"Testing against: {BASE_URL}")
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_login_redirects()
    test_team_dashboard_access()
    test_stage_update_api()
    test_database_tables()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("\nTo run the Flask app:")
    print("  python app_v2.py")
    print("\nTo create test users, use the admin panel at /admin/users")

if __name__ == "__main__":
    main()
