from collections import deque

class SmoothingService:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.history = {}

    def get_smoothed_count(self, camera_id, current_count):
        cam_id_str = str(camera_id)
        if cam_id_str not in self.history:
            self.history[cam_id_str] = deque(maxlen=self.window_size)
        
        self.history[cam_id_str].append(current_count)
        
        # Calculate moving average
        avg = sum(self.history[cam_id_str]) / len(self.history[cam_id_str])
        return int(round(avg))
