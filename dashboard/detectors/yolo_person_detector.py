import torch
from ultralytics import YOLO

class YOLOPersonDetector:
    def __init__(self, model_path="yolo11m.pt"):
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.is_loaded = True
        print(f"Loaded YOLOPersonDetector ({model_path}) on {self.device}")
        
        try:
            import numpy as np
            dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)
            self.model.predict(dummy_frame, verbose=False)
        except Exception as e:
            print(f"Pre-warm warning: {e}")

    def detect(self, frame):
        if not self.is_loaded or frame is None:
            return []
            
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], conf=0.10, verbose=False)
        boxes_info = []
        
        if results and len(results) > 0 and results[0].boxes:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            
            if results[0].boxes.id is not None:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
            else:
                ids = [-1] * len(boxes)
                
            for box, conf, track_id in zip(boxes, confs, ids):
                x1, y1, x2, y2 = box
                boxes_info.append({
                    "id": f"B_{track_id}" if track_id != -1 else f"B_u",
                    "bbox": (x1, y1, x2, y2),
                    "conf": float(conf),
                    "center": ((x1 + x2) // 2, (y1 + y2) // 2)
                })
                
        return boxes_info
