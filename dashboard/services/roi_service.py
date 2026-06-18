import cv2
import numpy as np

class ROIService:
    def __init__(self):
        # camera_id -> list of dicts: {"name": str, "points": np.array, "color_hex": str, "color_bgr": tuple, "count": int}
        self.camera_zones = {}
        self.has_custom_zones = False

    def set_zones(self, camera_id, zones_list):
        """
        zones_list is a list of dicts:
        [
            {
                "name": "Zone 1",
                "color_hex": "#ef4444",
                "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            },
            ...
        ]
        """
        cam_id_str = str(camera_id)
        zones = []
        for z in zones_list:
            pts = np.array(z["points"], np.int32)
            hex_color = z.get("color_hex", "#eab308")
            
            # Convert hex to BGR
            h = hex_color.lstrip('#')
            if len(h) == 6:
                r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                color_bgr = (b, g, r)
            else:
                color_bgr = (0, 255, 255) # Yellow default
                
            zones.append({
                "name": z["name"],
                "points": pts,
                "color_hex": hex_color,
                "color_bgr": color_bgr,
                "count": 0
            })
        self.camera_zones[cam_id_str] = zones

    def get_zones(self, camera_id):
        cam_id_str = str(camera_id)
        return self.camera_zones.get(cam_id_str, [])

    def is_inside_zone(self, zone, point):
        polygon = zone["points"]
        result = cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False)
        return result >= 0

    # --- Legacy Compatibility Methods ---
    def set_roi(self, camera_id, points):
        """Legacy compatibility method to set a single default zone"""
        if points:
            self.set_zones(camera_id, [{
                "name": "Zone 1",
                "color_hex": "#eab308",
                "points": points
            }])
        else:
            self.set_zones(camera_id, [])

    def is_inside(self, camera_id, point):
        """Legacy compatibility method to check if a point is inside any zone"""
        zones = self.get_zones(camera_id)
        if not zones:
            return True
        return any(self.is_inside_zone(z, point) for z in zones)

    def draw_roi(self, frame, camera_id):
        cam_id_str = str(camera_id)
        zones = self.get_zones(cam_id_str)
        for zone in zones:
            polygon = zone["points"]
            color = zone["color_bgr"]
            
            # Draw polygon box
            cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=2)
            
            # Draw semi-transparent background for the zone label text
            if len(polygon) > 0:
                x, y = polygon[0][0], polygon[0][1]
                label = zone["name"]
                
                # Dynamic text label placement (keep within frame)
                text_y = max(15, y - 8)
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                
                # Dark background banner for the text label
                cv2.rectangle(
                    frame,
                    (x, text_y - 12),
                    (x + text_size[0] + 6, text_y + 4),
                    (20, 20, 20),
                    -1
                )
                # Outer glow border for text banner
                cv2.rectangle(
                    frame,
                    (x, text_y - 12),
                    (x + text_size[0] + 6, text_y + 4),
                    color,
                    1
                )
                # Text label drawing
                cv2.putText(
                    frame,
                    label,
                    (x + 3, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

