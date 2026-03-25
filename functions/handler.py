import io
import json
from app import app

def handler(event, context):
    path = event.get('path', '/')
    method = event.get('httpMethod', 'GET')
    headers = event.get('headers', {}) or {}
    body = event.get('body', '') or ''
    
    environ = {
        'REQUEST_METHOD': method,
        'PATH_INFO': path,
        'QUERY_STRING': event.get('querystring', ''),
        'SERVER_NAME': 'localhost',
        'SERVER_PORT': '443',
        'wsgi.url_scheme': 'https',
        'wsgi.input': io.StringIO(body),
        'wsgi.errors': io.StringIO(),
    }
    
    for key, value in headers.items():
        key = key.upper().replace('-', '_')
        if key not in ('CONTENT_TYPE', 'CONTENT_LENGTH'):
            environ[f'HTTP_{key}'] = value
    
    if method in ('POST', 'PUT', 'PATCH'):
        environ['CONTENT_TYPE'] = headers.get('content-type', headers.get('Content-Type', ''))
        environ['CONTENT_LENGTH'] = str(len(body))
    
    response_status = [None]
    response_headers = [None]
    response_body = []
    
    def start_response(status, headers, exc_info=None):
        response_status[0] = status
        response_headers[0] = headers
        return lambda x: response_body.append(x)
    
    result = app(environ, start_response)
    
    body_bytes = b''.join(response_body)
    
    return {
        'statusCode': int(response_status[0].split()[0]) if response_status[0] else 404,
        'headers': dict(response_headers[0]) if response_headers[0] else {},
        'body': body_bytes.decode('utf-8')
    }
