#!/usr/bin/env python3
"""
Test script to verify RFP analyzer components work
"""

import os
import json

def test_company_databases():
    """Test if company databases exist and are valid"""
    print("\n=== Testing Company Databases ===")
    companies = ["IKIO", "METCO", "SUNSPRINT"]
    
    for company in companies:
        path = f"company_db/{company}.json"
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    texts = data.get('texts', [])
                    if texts and len(texts) > 0 and isinstance(texts[0], str) and len(texts[0]) < 1000:
                        print(f"[OK] {company}: {len(texts)} texts, first text length: {len(texts[0])}")
                    else:
                        print(f"[ERROR] {company}: Invalid format or binary data")
            except Exception as e:
                print(f"[ERROR] {company}: {e}")
        else:
            print(f"[MISSING] {company}")

def test_ollama():
    """Test Ollama connection"""
    print("\n=== Testing Ollama ===")
    try:
        import ollama
        client = ollama.Client(host="http://localhost:11434")
        
        # Try a simple generation
        response = client.generate(
            model="llama3",
            prompt="Say 'test successful' and nothing else.",
            options={"temperature": 0, "num_ctx": 512}
        )
        
        if response and 'response' in response:
            print(f"[OK] Ollama responding: {response['response'][:50]}")
        else:
            print(f"[ERROR] Ollama response format unexpected: {response}")
    except Exception as e:
        print(f"[ERROR] Ollama: {e}")

def test_imports():
    """Test if all required imports work"""
    print("\n=== Testing Imports ===")
    required_modules = [
        'flask',
        'flask_login',
        'torch',
        'sentence_transformers',
        'pandas',
        'numpy',
        'fitz',
        'doctr',
        'docx',
        'ollama'
    ]
    
    for module in required_modules:
        try:
            if module == 'fitz':
                import fitz
                print(f"[OK] PyMuPDF (fitz)")
            elif module == 'docx':
                from docx import Document
                print(f"[OK] python-docx")
            else:
                __import__(module)
                print(f"[OK] {module}")
        except ImportError as e:
            print(f"[ERROR] {module}: {e}")

def test_route_file():
    """Test if rfp_analyzer_routes.py can be imported"""
    print("\n=== Testing Route File ===")
    try:
        import sys
        sys.path.insert(0, os.getcwd())
        import rfp_analyzer_routes
        print(f"[OK] rfp_analyzer_routes.py imported successfully")
        
        # Check if key functions exist
        functions = ['extract_page_texts', 'summarize_batch_with_llama', 'build_master_summary']
        for func in functions:
            if hasattr(rfp_analyzer_routes, func):
                print(f"[OK] Function {func} exists")
            else:
                print(f"[ERROR] Function {func} missing")
        
        # Check if scorer exists
        if hasattr(rfp_analyzer_routes, 'scorer'):
            print(f"[OK] Scorer initialized")
        else:
            print(f"[ERROR] Scorer not initialized")
            
    except Exception as e:
        print(f"[ERROR] Cannot import rfp_analyzer_routes.py: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=" * 60)
    print("RFP ANALYZER COMPONENT TEST")
    print("=" * 60)
    
    test_imports()
    test_company_databases()
    test_ollama()
    test_route_file()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

