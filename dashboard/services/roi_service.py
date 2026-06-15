import cv2
import numpy as np

class ROIService:
    def __init__(self):
        # camera_id -> list of points
        self.rois = {}

    def set_roi(self, camera_id, points):
        """points is a list of [x, y] coordinates"""
        if camera_id == "all":
            # Just a convention if we want to set it for all currently known
            pass
            
        if points:
            self.rois[str(camera_id)] = np.array(points, np.int32)
        else:
            if str(camera_id) in self.rois:
                del self.rois[str(camera_id)]

    def is_inside(self, camera_id, point):
        cam_id_str = str(camera_id)
        # Check specific camera ROI, fallback to "all" ROI if exists
        polygon = self.rois.get(cam_id_str, self.rois.get("all"))
        
        if polygon is None:
            return True # If no ROI, everything is inside
        
        result = cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False)
        return result >= 0

    def draw_roi(self, frame, camera_id):
        cam_id_str = str(camera_id)
        polygon = self.rois.get(cam_id_str, self.rois.get("all"))
        if polygon is not None:
            cv2.polylines(frame, [polygon], isClosed=True, color=(0, 255, 255), thickness=2)
