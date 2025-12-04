#!/usr/bin/env python3
"""
Quick status check for RFP Analyzer
Checks if all dependencies are ready
"""

import os
import json
import sys

def check_company_db():
    """Check if company knowledge bases exist"""
    print("\n=== Company Knowledge Bases ===")
    companies = ["IKIO", "METCO", "SUNSPRINT"]
    all_ok = True
    
    for company in companies:
        path = f"company_db/{company}.json"
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    num_texts = len(data.get('texts', []))
                    print(f"[OK] {company}: {num_texts} knowledge chunks")
            except Exception as e:
                print(f"[ERROR] {company}: Failed to read - {e}")
                all_ok = False
        else:
            print(f"[MISSING] {company}: File not found at {path}")
            all_ok = False
    
    return all_ok

def check_ollama():
    """Check if Ollama is running and model is available"""
    print("\n=== Ollama Service ===")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        if response.status_code == 200:
            version = response.json().get('version', 'unknown')
            print(f"[OK] Ollama is running (version: {version})")
            
            # Check for models
            try:
                import ollama as ollama_client
                models = ollama_client.list()
                if models and hasattr(models, 'models'):
                    model_names = [m.model for m in models.models]
                    print(f"[OK] Available models: {', '.join(model_names)}")
                    if 'llama3' in str(model_names).lower():
                        print(f"[OK] llama3 model found")
                        return True
                    else:
                        print(f"[WARNING] llama3 model not found. Run: ollama pull llama3")
                        return False
                else:
                    print(f"[WARNING] Could not list models")
                    return False
            except Exception as e:
                print(f"[WARNING] Could not check models: {e}")
                return True  # Assume OK if version check passed
        else:
            print(f"[ERROR] Ollama returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] Ollama not running or not accessible: {e}")
        print("       Run: ollama serve")
        return False

def check_python_deps():
    """Check if required Python packages are installed"""
    print("\n=== Python Dependencies ===")
    required = [
        'flask',
        'flask_login',
        'torch',
        'sentence_transformers',
        'pandas',
        'numpy',
        'fitz',  # PyMuPDF
        'doctr',
        'docx',
        'ollama'
    ]
    
    all_ok = True
    for pkg in required:
        try:
            if pkg == 'fitz':
                import fitz
                pkg_name = 'PyMuPDF'
            elif pkg == 'docx':
                from docx import Document
                pkg_name = 'python-docx'
            else:
                __import__(pkg)
                pkg_name = pkg
            print(f"[OK] {pkg_name}")
        except ImportError:
            print(f"[MISSING] {pkg_name} - Run: pip install {pkg_name}")
            all_ok = False
    
    return all_ok

def check_directory_structure():
    """Check if required directories exist"""
    print("\n=== Directory Structure ===")
    dirs = ['company_db', 'templates']
    all_ok = True
    
    for d in dirs:
        if os.path.exists(d) and os.path.isdir(d):
            print(f"[OK] {d}/ exists")
        else:
            print(f"[MISSING] {d}/ directory")
            all_ok = False
    
    return all_ok

def check_templates():
    """Check if required templates exist"""
    print("\n=== Flask Templates ===")
    templates = ['rfp_analyzer_main.html', 'bid_analyzer_landing.html']
    all_ok = True
    
    for t in templates:
        path = f"templates/{t}"
        if os.path.exists(path):
            print(f"[OK] {t}")
        else:
            print(f"[MISSING] {t}")
            all_ok = False
    
    return all_ok

def main():
    """Run all checks"""
    print("=" * 60)
    print("RFP ANALYZER STATUS CHECK")
    print("=" * 60)
    
    checks = [
        ("Directory Structure", check_directory_structure()),
        ("Python Dependencies", check_python_deps()),
        ("Company Knowledge Bases", check_company_db()),
        ("Ollama Service", check_ollama()),
        ("Flask Templates", check_templates()),
    ]
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, result in checks:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {name}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n[SUCCESS] All checks passed! RFP Analyzer should work.")
        print("\nNext steps:")
        print("1. Start Flask app: python app_v2.py")
        print("2. Navigate to: http://localhost:5001/rfp-analyzer/")
        print("3. Upload RFP PDF and analyze")
        return 0
    else:
        print("\n[ACTION REQUIRED] Some checks failed. Please fix issues above.")
        print("\nQuick fixes:")
        print("- Missing company DB: python setup_company_kb.py")
        print("- Missing packages: pip install -r requirements.txt")
        print("- Ollama not running: ollama serve")
        print("- Missing llama3: ollama pull llama3")
        return 1

if __name__ == "__main__":
    sys.exit(main())

