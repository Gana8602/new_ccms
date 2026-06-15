class DensityEstimator:
    def __init__(self, roi_area_pixels=None):
        self.roi_area_pixels = roi_area_pixels or (640 * 360) # Default full frame

    def estimate(self, head_count):
        """
        Mock density estimation based on head count.
        Phase 2 will replace this with CSRNet/P2PNet tensor evaluation.
        """
        # Roughly speaking, if 100 people are in 640x360, it's very dense.
        density_val = head_count / 100.0
        
        level = "LOW"
        if head_count > 15:
            level = "MEDIUM"
        if head_count > 40:
            level = "HIGH"
        if head_count > 70:
            level = "CRITICAL"
            
        return {
            "estimated_people": head_count, # Mocked
            "density_level": level,
            "density_percentage": min(int(density_val * 100), 100)
        }
