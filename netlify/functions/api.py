import os
import sys
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Set environment variables for Netlify
os.environ.setdefault('ENVIRONMENT', 'production')
os.environ.setdefault('PYTHONPATH', project_root)

# Import the Flask app
from app import app

# Netlify Functions handler
def handler(event, context):
    """
    Netlify Functions handler for Flask app
    """
    try:
        # Parse the incoming event
        http_method = event.get('httpMethod', 'GET')
        path = event.get('path', '/')
        headers = event.get('headers', {})
        query_string = event.get('queryStringParameters', {}) or {}
        body = event.get('body', '')
        
        # Create a test client
        with app.test_client() as client:
            # Prepare the request
            if http_method == 'GET':
                response = client.get(path, headers=headers, query_string=query_string)
            elif http_method == 'POST':
                response = client.post(
                    path,
                    headers=headers,
                    data=body,
                    query_string=query_string,
                    content_type=headers.get('content-type', 'application/json')
                )
            elif http_method == 'PUT':
                response = client.put(
                    path,
                    headers=headers,
                    data=body,
                    query_string=query_string,
                    content_type=headers.get('content-type', 'application/json')
                )
            elif http_method == 'DELETE':
                response = client.delete(path, headers=headers, query_string=query_string)
            else:
                response = client.get(path, headers=headers, query_string=query_string)
            
            # Return the response
            return {
                'statusCode': response.status_code,
                'headers': dict(response.headers),
                'body': response.get_data(as_text=True)
            }
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': jsonify({'error': str(e)}).get_data(as_text=True)
        }
