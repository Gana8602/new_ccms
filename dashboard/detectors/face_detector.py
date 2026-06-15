import cv2
import os

class FaceDetector:
    def __init__(self, prototxt_path="models/deploy_face.prototxt", model_path="models/res10_300x300_ssd_iter_140000.caffemodel"):
        self.is_loaded = False
        if os.path.exists(prototxt_path) and os.path.exists(model_path):
            try:
                self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
                self.is_loaded = True
                print("Loaded FaceDetector (OpenCV DNN)")
            except Exception as e:
                print(f"Failed to load FaceDetector: {e}")
        else:
            print("FaceDetector models not found.")

    def detect(self, frame):
        if not self.is_loaded or frame is None:
            return []
            
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5: # 50% confidence threshold
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure bounds
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                
                faces.append({
                    "id": f"F_{i}",
                    "bbox": (startX, startY, endX, endY),
                    "conf": float(confidence),
                    "center": ((startX + endX) // 2, (startY + endY) // 2)
                })
        return faces
