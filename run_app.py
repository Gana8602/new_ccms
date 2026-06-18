import os
import sys
import webbrowser
import threading
import time
from django.core.management import execute_from_command_line

# Add the project directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ccms_project.settings')

def open_browser():
    # Wait a moment for the server to spin up
    time.sleep(2)
    webbrowser.open('http://127.0.0.1:8000')

if __name__ == '__main__':
    # Start browser thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Run the Django server on port 8000
    try:
        execute_from_command_line([sys.argv[0], 'runserver', '127.0.0.1:8000', '--noreload'])
    except KeyboardInterrupt:
        print("\nStopping CCMS Server...")
        sys.exit(0)
