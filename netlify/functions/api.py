import os
import sys
import json
from flask import Flask, request, jsonify

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

os.environ.setdefault('ENVIRONMENT', 'production')
os.environ.setdefault('PYTHONPATH', project_root)

from app import app

def handler(event, context):

    try:
        http_method = event.get('httpMethod', 'GET')
        path = event.get('path', '/')
        prefix = '/.netlify/functions/api/'
        if path.startswith(prefix):
            path = path[len(prefix):]
            if path == '':
                path = '/'
            elif not path.startswith('/'):
                path = '/' + path
        headers = event.get('headers', {})
        query_string = event.get('queryStringParameters', {}) or {}
        body = event.get('body', '')

        with app.test_client() as client:
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

            # If we get a 404, return a plain text message with the path for debugging
            if response.status_code == 404:
                return {
                    'statusCode': 404,
                    'headers': {'Content-Type': 'text/plain'},
                    'body': f'Not found: {path}'
                }

            return {
                'statusCode': response.status_code,
                'headers': dict(response.headers),
                'body': response.get_data(as_text=True)
            }
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': str(e)})
        }
