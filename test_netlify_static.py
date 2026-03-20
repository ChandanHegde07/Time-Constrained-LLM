import os
import sys
import json

# Add the project root to Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Test Netlify handler for static files
from netlify.functions.api import handler

def test_static_handler():
    print("Testing Netlify handler for static files...")
    
    # Test styles.css
    event = {
        'httpMethod': 'GET',
        'path': '/static/styles.css',
        'headers': {},
        'queryStringParameters': {},
        'body': ''
    }
    
    print(f"Test event: {json.dumps(event, indent=2)}")
    
    try:
        response = handler(event, None)
        print(f"Response status code: {response['statusCode']}")
        print(f"Response headers: {json.dumps(response['headers'], indent=2)}")
        print(f"Response body (first 200 chars): {response['body'][:200]}...")
        
        if response['statusCode'] == 200:
            print("\033[92m✓ Static file served successfully\033[0m")
        else:
            print("\033[91m✗ Static file failed to serve\033[0m")
            
    except Exception as e:
        print(f"\033[91m✗ Error calling handler: {str(e)}\033[0m")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    test_static_handler()
