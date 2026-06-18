import os
import cv2
import time
import torch
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from scipy.spatial import distance
from collections import OrderedDict
from dashboard.utils.nms import merge_detections_nms

# --- CONFIGURATION ---
HEAD_INFERENCE_MODE = "hybrid" # "full", "tiled", "hybrid"
HEAD_CONFIDENCE = 0.10
HEAD_IOU = 0.45
HEAD_TILE_SIZE = 768
HEAD_TILE_OVERLAP = 0.25
HEAD_MAX_DET = 2000
# ---------------------

class CentroidTracker:
    def __init__(self, max_disappeared=10, max_distance=100):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = max_disappeared
        self.maxDistance = max_distance

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
        return self.nextObjectID - 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return []

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        assigned_ids = [-1] * len(rects)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                assigned_ids[i] = self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = distance.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                if D[row, col] > self.maxDistance:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                assigned_ids[col] = objectID

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            for col in unusedCols:
                assigned_ids[col] = self.register(inputCentroids[col])

        return assigned_ids

class HeadDetector:
    def __init__(self, model_path="/home/sirisha/Ganapathi/new_ccms/models/best_head.pt"):
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        
        self.tracker = CentroidTracker()
        self.frame_counter = 0
        
        if os.path.exists(model_path):
            try:
                self.model = YOLO(model_path)
                self.model.to(self.device)
                self.is_loaded = True
                print(f"Loaded HeadDetector ({model_path}) on {self.device}")
                
                try:
                    dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)
                    self.model.predict(dummy_frame, verbose=False)
                except Exception as e:
                    print(f"Pre-warm warning: {e}")
            except Exception as e:
                print(f"Failed to load HeadDetector: {e}")
        else:
            print(f"Head model not found at {model_path}.")

    def _get_tiles(self, h, w, tile_size, overlap):
        stride = int(tile_size * (1 - overlap))
        tiles = []
        y_steps = max(1, int(np.ceil((h - tile_size) / stride)) + 1)
        x_steps = max(1, int(np.ceil((w - tile_size) / stride)) + 1)
        
        for y in range(y_steps):
            for x in range(x_steps):
                y1 = y * stride
                x1 = x * stride
                y2 = min(h, y1 + tile_size)
                x2 = min(w, x1 + tile_size)
                
                if y2 == h:
                    y1 = max(0, h - tile_size)
                if x2 == w:
                    x1 = max(0, w - tile_size)
                    
                tiles.append((x1, y1, x2, y2))
        return list(set(tiles))

    def detect(self, frame):
        if not self.is_loaded or frame is None:
            return []
            
        self.frame_counter += 1
        h, w = frame.shape[:2]
        
        all_detections = []
        full_det_count = 0
        tile_det_count = 0
        tiles_count = 0
        
        # FULL FRAME INFERENCE
        if HEAD_INFERENCE_MODE in ["full", "hybrid"]:
            results = self.model.predict(
                frame, 
                conf=HEAD_CONFIDENCE, 
                iou=HEAD_IOU, 
                imgsz=1280, 
                max_det=HEAD_MAX_DET, 
                verbose=False
            )
            if len(results) > 0 and results[0].boxes:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                confs = results[0].boxes.conf.cpu().numpy()
                for box, conf in zip(boxes, confs):
                    all_detections.append({"bbox": box.tolist(), "conf": float(conf)})
                full_det_count = len(boxes)

        # TILED INFERENCE
        if HEAD_INFERENCE_MODE in ["tiled", "hybrid"]:
            tiles = self._get_tiles(h, w, HEAD_TILE_SIZE, HEAD_TILE_OVERLAP)
            tiles_count = len(tiles)
            for (x1, y1, x2, y2) in tiles:
                tile_img = frame[y1:y2, x1:x2]
                results = self.model.predict(
                    tile_img, 
                    conf=HEAD_CONFIDENCE, 
                    iou=HEAD_IOU, 
                    imgsz=640, 
                    max_det=HEAD_MAX_DET, 
                    verbose=False
                )
                if len(results) > 0 and results[0].boxes:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    confs = results[0].boxes.conf.cpu().numpy()
                    for box, conf in zip(boxes, confs):
                        tx1, ty1, tx2, ty2 = box
                        all_detections.append({
                            "bbox": [tx1 + x1, ty1 + y1, tx2 + x1, ty2 + y1], 
                            "conf": float(conf)
                        })
                        tile_det_count += 1
                        
        # NMS MERGE
        merged_detections = merge_detections_nms(all_detections, iou_threshold=HEAD_IOU)
        
        # APPLY TRACKING
        rects = [d["bbox"] for d in merged_detections]
        assigned_ids = self.tracker.update(rects)
        
        final_boxes_info = []
        for det, track_id in zip(merged_detections, assigned_ids):
            bx1, by1, bx2, by2 = det["bbox"]
            final_boxes_info.append({
                "id": f"H_{track_id}",
                "bbox": (bx1, by1, bx2, by2),
                "conf": det["conf"],
                "center": ((bx1 + bx2) // 2, (by1 + by2) // 2)
            })

        # DEBUG LOGGING
        if self.frame_counter % 30 == 0:
            avg_conf = sum(d["conf"] for d in merged_detections) / max(1, len(merged_detections)) if merged_detections else 0.0
            print("\n[HEAD DEBUG]")
            print(f"frame={w}x{h}")
            print(f"mode={HEAD_INFERENCE_MODE}")
            print(f"full={full_det_count}")
            print(f"tiles_raw={tile_det_count}")
            print(f"merged={len(merged_detections)}")
            print(f"tiles={tiles_count}")
            print(f"conf={HEAD_CONFIDENCE}")
            print(f"iou={HEAD_IOU}")
            print(f"avg_conf={avg_conf:.3f}\n")

        # DEBUG IMAGE SAVING
        if self.frame_counter % 100 == 0:
            self._save_debug_image(frame, final_boxes_info, full_det_count, tile_det_count)

        return final_boxes_info

    def _save_debug_image(self, frame, boxes_info, full_count, tile_count):
        debug_dir = os.path.join("media", "debug", "head_detection")
        os.makedirs(debug_dir, exist_ok=True)
        
        debug_frame = frame.copy()
        
        for det in boxes_info:
            cx, cy = det["center"]
            bx1, by1, bx2, by2 = det["bbox"]
            r = max((bx2 - bx1) // 2, 5)
            cv2.circle(debug_frame, (cx, cy), r, (0, 255, 0), 1)
            cv2.putText(debug_frame, det["id"], (bx1, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
        cv2.putText(debug_frame, f"Mode: {HEAD_INFERENCE_MODE}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(debug_frame, f"Full: {full_count} | Tiles: {tile_count} | Merged: {len(boxes_info)}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"head_debug_{timestamp}.jpg"
        filepath = os.path.join(debug_dir, filename)
        
        cv2.imwrite(filepath, debug_frame)
