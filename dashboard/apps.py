import os
import subprocess
import threading
from django.apps import AppConfig

class DashboardConfig(AppConfig):
    name = 'dashboard'

    def ready(self):
        # We start the stream only if RUN_MAIN is true to prevent duplicate streams during local dev reloading.
        # Also check if it's running via gunicorn for production.
        if os.environ.get('RUN_MAIN', None) == 'true' or 'gunicorn' in os.environ.get('SERVER_SOFTWARE', ''):
            def start_mock_rtsp():
                import time
                time.sleep(5) # Let the RTSP server (like MediaMTX) start up first
                
                # Use a synthetic test pattern to guarantee streaming works out of the box in deployment
                # Default to the docker network mediamtx hostname, fallback to localhost
                rtsp_host = os.environ.get('RTSP_HOST', '127.0.0.1')
                rtsp_url = f"rtsp://{rtsp_host}:8554/mystream1"
                
                cmd = [
                    'ffmpeg', '-hide_banner', '-loglevel', 'warning',
                    '-re', '-f', 'lavfi', '-i', 'testsrc=size=1280x720:rate=20',
                    '-c:v', 'libx264', '-preset', 'ultrafast', '-tune', 'zerolatency',
                    '-f', 'rtsp', '-rtsp_transport', 'tcp',
                    rtsp_url
                ]
                try:
                    # Run in the background
                    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    print(f"Started automated mock RTSP stream at {rtsp_url}")
                except Exception as e:
                    print(f"Failed to start mock RTSP stream: {e}")

            threading.Thread(target=start_mock_rtsp, daemon=True).start()
