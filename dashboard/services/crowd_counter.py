import cv2
import time
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from .roi_service import ROIService
from .smoothing_service import SmoothingService
from .heatmap_generator import HeatmapGenerator
from .density_estimator import DensityEstimator
from .alert_engine import AlertEngine

def draw_hud_head_marker(
    frame,
    bbox,
    track_id,
    confidence,
    identity_label="UNKNOWN",
    identity_confidence=None,
):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    r = max((x2 - x1) // 2, 12)
    
    h, w = frame.shape[:2]
    
    # Create a small ROI patch to do overlay blending efficiently
    patch_size = int(r * 3)
    px1 = max(0, cx - patch_size)
    py1 = max(0, cy - patch_size)
    px2 = min(w, cx + patch_size)
    py2 = min(h, cy + patch_size)
    
    if px2 <= px1 or py2 <= py1:
        return # Out of bounds
        
    roi = frame[py1:py2, px1:px2]
    overlay = roi.copy()
    
    # Coordinates relative to the patch
    pcx = cx - px1
    pcy = cy - py1
    
    color = (255, 255, 0) # Cyan in BGR
    glow = (100, 100, 0) # Dark Cyan
    
    # 1. Glow ring and Core ring
    cv2.circle(overlay, (pcx, pcy), r, glow, 4)
    cv2.circle(overlay, (pcx, pcy), r, color, 1)
    
    # 2. Inner dot
    cv2.circle(overlay, (pcx, pcy), 1, color, -1)
    
    # 3. Four corner ticks
    tl = int(r * 0.4)
    tr = r + 4
    # TL
    cv2.line(overlay, (pcx - tr, pcy - tr), (pcx - tr + tl, pcy - tr), color, 1)
    cv2.line(overlay, (pcx - tr, pcy - tr), (pcx - tr, pcy - tr + tl), color, 1)
    # TR
    cv2.line(overlay, (pcx + tr, pcy - tr), (pcx + tr - tl, pcy - tr), color, 1)
    cv2.line(overlay, (pcx + tr, pcy - tr), (pcx + tr, pcy - tr + tl), color, 1)
    # BL
    cv2.line(overlay, (pcx - tr, pcy + tr), (pcx - tr + tl, pcy + tr), color, 1)
    cv2.line(overlay, (pcx - tr, pcy + tr), (pcx - tr, pcy + tr - tl), color, 1)
    # BR
    cv2.line(overlay, (pcx + tr, pcy + tr), (pcx + tr - tl, pcy + tr), color, 1)
    cv2.line(overlay, (pcx + tr, pcy + tr), (pcx + tr, pcy + tr - tl), color, 1)
    
    # 4. Rotating arcs
    angle = int(time.time() * 120) % 360
    cv2.ellipse(overlay, (pcx, pcy), (r + 8, r + 8), angle, 0, 45, color, 2)
    cv2.ellipse(overlay, (pcx, pcy), (r + 8, r + 8), angle, 120, 165, color, 2)
    cv2.ellipse(overlay, (pcx, pcy), (r + 8, r + 8), angle, 240, 285, color, 2)
    
    # Blend overlay patch back into the main frame
    cv2.addWeighted(overlay, 0.7, roi, 0.3, 0, roi)
    
    # 5. Text Labels (Drawn directly on frame so they don't fade)
    label = identity_label
    confidence_value = (
        identity_confidence if identity_confidence is not None else confidence
    )
    conf_str = f"{int(confidence_value * 100)}%"
    
    # Add a tiny background rect for the label
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
    label_y = max(15, cy - r - 10)
    cv2.rectangle(
        frame,
        (cx - r, label_y - 13),
        (cx - r + text_size[0] + 4, label_y + 3),
        glow,
        -1,
    )
    
    cv2.putText(
        frame,
        label,
        (cx - r + 2, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
    )
    cv2.putText(frame, conf_str, (cx + r + 4, cy - r), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    cv2.putText(
        frame,
        str(track_id),
        (cx - r, cy + r + 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        color,
        1,
    )

class CrowdCounterService:
    FACE_FRAME_INTERVAL = 5
    FACE_LABEL_TTL_SECONDS = 5

    def __init__(self, head_detector, person_detector, face_identity_manager=None):
        self.head_detector = head_detector
        self.person_detector = person_detector
        self.face_identity_manager = face_identity_manager
        
        self.roi_service = ROIService()
        self.smoothing_service = SmoothingService()
        self.heatmap_gen = HeatmapGenerator()
        self.density_estimator = DensityEstimator()
        self.alert_engine = AlertEngine(capacity=200)
        
        self.lock = threading.Lock()
        
        # Dashboard Global Analytics State
        self.stats = {
            "inside": 0,
            "outside": 0,
            "total": 0, # Total heads
            "forward": 0,
            "backward": 0,
            "mode": "head_priority",
            "fps": 0.0,
            "density_level": "LOW",
            "density_percentage": 0,
            "occupancy": 0,
            "available_capacity": 200,
            "alerts": []
        }
        
        self.track_history = {}
        self.crossed_ids = set()
        self.show_heatmap = False # Hidden as per user request
        self.frame_number = 0
        self.face_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="face-recognition"
        )
        self.face_future = None
        self.face_frame_interval = self.FACE_FRAME_INTERVAL
        self.head_identity_cache = {}

    def _collect_face_results(self):
        if self.face_future is None or not self.face_future.done():
            return
        try:
            results = self.face_future.result()
        except Exception as exc:
            with open("rejections.log", "a") as f: f.write(f"CRASH: {exc}\n")
            results = []
        finally:
            self.face_future = None

        expires_at = time.monotonic() + self.FACE_LABEL_TTL_SECONDS
        for result in results:
            head_track_id = result.get("head_track_id")
            if not head_track_id:
                continue
            self.head_identity_cache[head_track_id] = {
                "label": result["label"],
                "confidence": result.get("similarity")
                or result.get("detector_confidence"),
                "expires_at": expires_at,
            }

    def _submit_face_recognition(self, frame, head_detections):
        if self.face_identity_manager is None:
            return
        if self.frame_number % self.face_frame_interval != 0:
            return
        if self.face_future is not None:
            self.face_frame_interval = 10
            return
        self.face_future = self.face_executor.submit(
            self.face_identity_manager.process_frame,
            frame.copy(),
            [dict(detection) for detection in head_detections],
        )

    def _identity_for_head(self, track_id):
        cached = self.head_identity_cache.get(track_id)
        if not cached:
            return "UNKNOWN", None
        if cached["expires_at"] < time.monotonic():
            self.head_identity_cache.pop(track_id, None)
            return "UNKNOWN", None
        return cached["label"], cached["confidence"]
        
    def process_frame(self, frame, camera_id=0):
        if frame is None:
            return frame

        with self.lock:
            start_time = time.time()
            cam_id_str = str(camera_id)
            self.frame_number += 1
            self._collect_face_results()
            
            # --- 1. Detect Heads (PRIMARY) ---
            head_detections = self.head_detector.detect(frame)
            
            # --- 2. Detect Bodies (SECONDARY) ---
            body_detections = self.person_detector.detect(frame)
            self._submit_face_recognition(frame, head_detections)
            
            # --- 3. Compute Stats from Heads and Line Crossing ---
            head_centers = []
            zones = self.roi_service.get_zones(cam_id_str)
            
            # Reset count for each zone in this frame
            for z in zones:
                z["count"] = 0
                
            inside_any_zone = set()
            h, w = frame.shape[:2]
            line_y = h // 2
            
            for det in head_detections:
                cx, cy = det["center"]
                head_centers.append((cx, cy))
                det_id = det["id"]
                
                if not zones:
                    in_some_zone = True
                else:
                    in_some_zone = False
                    for z in zones:
                        if self.roi_service.is_inside_zone(z, (cx, cy)):
                            z["count"] += 1
                            in_some_zone = True
                        
                if in_some_zone:
                    inside_any_zone.add(det_id)

                # Line crossing checks
                if det_id in self.track_history:
                    prev_cy = self.track_history[det_id]
                    if det_id not in self.crossed_ids:
                        if prev_cy < line_y and cy >= line_y:
                            # Moving from top to bottom is backward
                            self.stats["backward"] += 1
                            self.crossed_ids.add(det_id)
                        elif prev_cy > line_y and cy <= line_y:
                            # Moving from bottom to top is forward
                            self.stats["forward"] += 1
                            self.crossed_ids.add(det_id)
                self.track_history[det_id] = cy
                    
            # Clean up history for deregistered track IDs to prevent memory leaks
            if hasattr(self.head_detector, "tracker"):
                active_tracker_ids = {f"H_{tid}" for tid in self.head_detector.tracker.objects.keys()}
            else:
                active_tracker_ids = {det["id"] for det in head_detections}
            for tid in list(self.track_history.keys()):
                if tid not in active_tracker_ids:
                    self.track_history.pop(tid, None)
                    self.crossed_ids.discard(tid)

            current_inside = len(inside_any_zone)
            current_outside = len(head_detections) - current_inside
            
            head_count = len(head_detections)
            smoothed_total = self.smoothing_service.get_smoothed_count(cam_id_str, head_count)
            
            # --- 4. Density & Alerts ---
            density_info = self.density_estimator.estimate(smoothed_total)
            alerts = self.alert_engine.evaluate(current_inside, density_info["density_percentage"], density_info["density_level"])
            
            occupancy = min(100, int((current_inside / self.alert_engine.capacity) * 100))
            available = max(0, self.alert_engine.capacity - current_inside)

            # --- 5. Drawing HUD / Annotations ---
            self.roi_service.draw_roi(frame, cam_id_str)
            
            # Draw Horizontal Counting Line (with a clean, neon purple-pink color and label)
            cv2.line(frame, (0, line_y), (w, line_y), (180, 100, 255), 2)
            cv2.putText(
                frame,
                "COUNTING LINE (CROSS Y-MIDPOINT)",
                (10, line_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (180, 100, 255),
                1,
                cv2.LINE_AA,
            )

            # Draw Heads (Iron Man HUD Marker)
            for det in head_detections:
                identity_label, identity_confidence = self._identity_for_head(det["id"])
                draw_hud_head_marker(
                    frame,
                    det["bbox"],
                    det["id"],
                    det["conf"],
                    identity_label,
                    identity_confidence,
                )

            # Draw Heatmap Overlay
            if self.show_heatmap:
                frame = self.heatmap_gen.generate(frame, head_centers)

            # --- 6. Calculate FPS ---
            elapsed = time.time() - start_time
            fps = 1.0 / elapsed if elapsed > 0 else 0

            # --- 7. Update Stats ---
            self.stats.update({
                "inside": current_inside,
                "outside": current_outside,
                "total": smoothed_total,
                "density_level": density_info["density_level"],
                "density_percentage": density_info["density_percentage"],
                "occupancy": occupancy,
                "available_capacity": available,
                "fps": round(fps, 1),
                "alerts": alerts,
                "zones": [
                    {
                        "name": z["name"],
                        "color_hex": z["color_hex"],
                        "count": z["count"]
                    } for z in zones
                ]
            })
                
            return frame

    def get_global_stats(self):
        return self.stats
