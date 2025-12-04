#!/usr/bin/env python3
"""
Test script to verify Ollama timeout and retry fixes
"""
import requests
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def get_ollama_session():
    """Create a session with connection pooling like in app_v2.py"""
    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=10,
        pool_maxsize=20,
        max_retries=Retry(total=0),
        pool_block=False
    )
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def test_ollama_with_retries():
    """Test Ollama request with the improved retry logic"""
    session = get_ollama_session()
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Say hello in one word.'}
    ]

    body = {
        'model': 'llama3',
        'messages': messages,
        'stream': False,
        'options': {'temperature': 0.2}
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}")
            start_time = time.time()
            resp = session.post(
                'http://127.0.0.1:11434/api/chat',
                headers={'Content-Type': 'application/json'},
                json=body,
                timeout=(10, 60)  # 10s connect, 60s read
            )
            elapsed = time.time() - start_time
            print(f"Request completed in {elapsed:.2f} seconds")

            if resp.status_code >= 400:
                print(f"HTTP {resp.status_code}: {resp.text}")
                if attempt < max_retries - 1:
                    delay = 2 ** attempt
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                return False

            data = resp.json()
            if 'message' in data and 'content' in data['message']:
                content = data['message']['content']
                print(f"Success! Response: {content}")
                return True
            else:
                print(f"Unexpected response format: {data}")
                return False

        except requests.Timeout as e:
            print(f"Timeout error: {e}")
            if attempt < max_retries - 1:
                delay = 2 ** attempt
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                continue
            return False
        except requests.RequestException as e:
            print(f"Request error: {e}")
            if attempt < max_retries - 1:
                delay = 2 ** attempt
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                continue
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False

    return False

if __name__ == "__main__":
    print("Testing Ollama fixes...")
    success = test_ollama_with_retries()
    if success:
        print("✅ Ollama test passed!")
    else:
        print("❌ Ollama test failed!")
