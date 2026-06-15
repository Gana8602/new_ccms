import cv2
import os

class GenderDetector:
    def __init__(self, prototxt_path="models/deploy_gender.prototxt", model_path="models/gender_net.caffemodel"):
        self.is_loaded = False
        self.gender_list = ['Male', 'Female']
        if os.path.exists(prototxt_path) and os.path.exists(model_path):
            try:
                self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
                self.is_loaded = True
                print("Loaded GenderDetector")
            except Exception as e:
                print(f"Failed to load GenderDetector: {e}")
        else:
            print("Gender models not found. Gender detection disabled.")

    def detect(self, face_img):
        if not self.is_loaded or face_img is None or face_img.size == 0:
            return "Male" # Default
            
        try:
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
            self.net.setInput(blob)
            gender_preds = self.net.forward()
            return self.gender_list[gender_preds[0].argmax()]
        except Exception:
            return "Male"
