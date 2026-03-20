import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

os.environ.setdefault('ENVIRONMENT', 'production')
os.environ.setdefault('PYTHONPATH', project_root)

from app import app

if __name__ == '__main__':
    app.run()
