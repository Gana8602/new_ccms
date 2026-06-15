import cv2
import numpy as np
import time

class HeatmapGenerator:
    def __init__(self):
        self.accumulated_density = None
        self.decay_rate = 0.95 # How fast older heat fades
        
    def generate(self, frame, head_centers):
        """Generates heatmap overlay on frame based on head centers."""
        h, w = frame.shape[:2]
        
        if self.accumulated_density is None or self.accumulated_density.shape != (h, w):
            self.accumulated_density = np.zeros((h, w), dtype=np.float32)
            
        # 1. Decay existing heatmap
        self.accumulated_density *= self.decay_rate
        
        # 2. Add new heat from current heads
        current_density = np.zeros((h, w), dtype=np.float32)
        
        for cx, cy in head_centers:
            # Bounds check
            if 0 <= cx < w and 0 <= cy < h:
                # Add a gaussian dot. Using circle and blur is fast enough for real-time
                cv2.circle(current_density, (int(cx), int(cy)), 15, 1.0, -1)
                
        # Blur the new points to create a smooth heatmap
        if len(head_centers) > 0:
            current_density = cv2.GaussianBlur(current_density, (31, 31), 0)
            
        # 3. Add to accumulator
        self.accumulated_density += current_density
        
        # 4. Normalize and colorize
        max_val = np.max(self.accumulated_density)
        if max_val > 0:
            normalized = (self.accumulated_density / max_val * 255).astype(np.uint8)
        else:
            normalized = np.zeros((h, w), dtype=np.uint8)
            
        # Apply JET colormap
        heatmap_color = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        
        # Mask out cold areas (where density is near 0)
        mask = normalized > 10
        
        # Overlay on original frame
        overlay = frame.copy()
        overlay[mask] = cv2.addWeighted(frame[mask], 0.5, heatmap_color[mask], 0.5, 0)
        
        return overlay
