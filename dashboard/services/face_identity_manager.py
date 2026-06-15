import logging
import threading
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
from django.conf import settings
from django.db import (
    OperationalError,
    ProgrammingError,
    close_old_connections,
    transaction,
)
from django.db.models import F
from django.utils import timezone

from dashboard.models import FaceIdentity


logger = logging.getLogger(__name__)


class FaceIdentityManager:
    SAME_PERSON_THRESHOLD = 0.60
    UNCERTAIN_THRESHOLD = 0.45
    MIN_FACE_SIZE = 50
    MIN_BLUR_VARIANCE = 60.0
    MIN_BRIGHTNESS = 45.0
    MIN_FACE_CONFIDENCE = 0.60
    YOLO_FACE_CONFIDENCE = 0.35
    YOLO_MIN_FACE_SIZE = 24
    YOLO_MIN_BLUR_VARIANCE = 25.0
    NEW_ID_MIN_FACE_CONFIDENCE = 0.75
    SAVE_INTERVAL_SECONDS = 5
    FACE_SIZE = 224
    CROP_PADDING = 0.25
    FALLBACK_SAME_PERSON_THRESHOLD = 0.82
    TRACK_IDENTITY_TTL_SECONDS = 30

    def __init__(self):
        self.media_root = Path(settings.MEDIA_ROOT)
        self.known_dir = self.media_root / "faces" / "known"
        self.unknown_dir = self.media_root / "faces" / "unknown"
        self.masked_dir = self.media_root / "faces" / "masked_unknown"
        for directory in (self.known_dir, self.unknown_dir, self.masked_dir):
            directory.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._last_saved = {}
        self._face_app = None
        self._fallback_yolo = None
        self._fallback_net = None
        self._fallback_detector = None
        self._track_identities = {}
        self.backend_name = "opencv-dnn-appearance"
        self._load_backend()

    @property
    def recognition_available(self):
        return (
            self._face_app is not None
            or self._fallback_yolo is not None
            or self._fallback_net is not None
            or self._fallback_detector is not None
        )

    def _load_backend(self):
        try:
            from insightface.app import FaceAnalysis

            providers = ["CPUExecutionProvider"]
            try:
                import onnxruntime as ort

                available = ort.get_available_providers()
                preferred = [
                    provider
                    for provider in ("CUDAExecutionProvider", "CoreMLExecutionProvider")
                    if provider in available
                ]
                providers = preferred + providers
            except ImportError:
                pass

            self._face_app = FaceAnalysis(name="buffalo_l", providers=providers)
            self._face_app.prepare(ctx_id=0 if providers[0] == "CUDAExecutionProvider" else -1)
            self.backend_name = "insightface-arcface"
            logger.info("Face recognition backend ready: %s", self.backend_name)
            return
        except Exception as exc:
            logger.warning(
                "InsightFace is unavailable; using conservative OpenCV "
                "appearance recognition. See FACE_RECOGNITION.md. Cause: %s",
                exc,
            )

        yolo_candidates = (
            Path(settings.BASE_DIR) / "models" / "yolov8n-face.pt",
            Path(settings.BASE_DIR).parent / "ccms" / "yolov8n-face.pt",
        )
        yolo_path = next((path for path in yolo_candidates if path.exists()), None)
        if yolo_path is not None:
            try:
                import torch
                from ultralytics import YOLO

                device = (
                    "mps"
                    if torch.backends.mps.is_available()
                    else "cuda"
                    if torch.cuda.is_available()
                    else "cpu"
                )
                self._fallback_yolo = YOLO(str(yolo_path))
                self._fallback_yolo.to(device)
                self.backend_name = "yolo-face-opencv-appearance"
                logger.info("Face recognition backend ready: %s", self.backend_name)
                return
            except Exception as exc:
                logger.warning("Could not load local YOLO face detector: %s", exc)

        prototxt_path = Path(settings.BASE_DIR) / "models" / "deploy_face.prototxt"
        model_path = (
            Path(settings.BASE_DIR)
            / "models"
            / "res10_300x300_ssd_iter_140000.caffemodel"
        )
        if prototxt_path.exists() and model_path.exists():
            try:
                self._fallback_net = cv2.dnn.readNetFromCaffe(
                    str(prototxt_path), str(model_path)
                )
                logger.info("Face recognition backend ready: %s", self.backend_name)
                return
            except cv2.error as exc:
                logger.warning("Could not load OpenCV DNN face detector: %s", exc)

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascade_path)
        if not detector.empty():
            self._fallback_detector = detector
            self.backend_name = "opencv-haar-appearance"
            logger.info("Face recognition backend ready: %s", self.backend_name)

    def process_frame(self, frame, head_detections=None):
        close_old_connections()
        try:
            return self._process_frame(frame, head_detections)
        finally:
            close_old_connections()

    def _process_frame(self, frame, head_detections=None):
        if frame is None:
            return []

        results = []
        for detected in self._detect_faces(frame):
            bbox = detected["bbox"]
            confidence = detected["confidence"]
            crop = self._square_face_crop(frame, bbox)
            quality = self._quality_check(
                frame,
                bbox,
                crop,
                confidence,
                detected.get("backend"),
            )
            if not quality["accepted"]:
                continue

            head_track_id = self._match_head_track(bbox, head_detections or [])
            masked = self._is_probably_masked(crop)
            embedding = detected.get("embedding")

            if masked:
                image_path = self._save_observation(
                    crop, "MASKED", head_track_id or self._bbox_key(bbox)
                )
                results.append(
                    self._result(
                        bbox, "MASKED", "masked", confidence, None,
                        head_track_id, image_path
                    )
                )
                continue

            if embedding is None:
                embedding = self._fallback_embedding(crop)

            embedding = self._normalize_embedding(embedding)
            identity, similarity = self._identity_for_track(
                head_track_id, embedding
            )
            if identity is None:
                identity, similarity = self._find_best_match(embedding)

            same_person_threshold = (
                self.SAME_PERSON_THRESHOLD
                if self._face_app is not None
                else self.FALLBACK_SAME_PERSON_THRESHOLD
            )
            if identity is not None and similarity >= same_person_threshold:
                self._touch_identity(identity, embedding, crop)
                self._remember_track_identity(head_track_id, identity)
                results.append(
                    self._result(
                        bbox, identity.person_id, "known", confidence,
                        similarity, head_track_id, identity.image_path
                    )
                )
                continue

            # A clear, quality-approved face must be represented in the known
            # gallery. Never force a weak match: create a new ID instead.
            identity = self._create_identity(embedding, crop)
            if identity is not None:
                self._remember_track_identity(head_track_id, identity)
                results.append(
                    self._result(
                        bbox, identity.person_id, "known", confidence,
                        None, head_track_id, identity.image_path
                    )
                )
                continue
        return results

    def _detect_faces(self, frame):
        if self._face_app is not None:
            detections = []
            for face in self._face_app.get(frame):
                bbox = tuple(int(value) for value in face.bbox)
                embedding = getattr(face, "normed_embedding", None)
                if embedding is None:
                    embedding = getattr(face, "embedding", None)
                detections.append(
                    {
                        "bbox": bbox,
                        "confidence": float(getattr(face, "det_score", 0.0)),
                        "embedding": embedding,
                        "backend": "insightface",
                    }
                )
            return detections

        if self._fallback_yolo is not None:
            predictions = self._fallback_yolo.predict(
                frame,
                conf=self.YOLO_FACE_CONFIDENCE,
                imgsz=960,
                verbose=False,
            )
            faces = []
            if predictions and predictions[0].boxes is not None:
                boxes = predictions[0].boxes.xyxy.cpu().numpy().astype(int)
                confidences = predictions[0].boxes.conf.cpu().numpy()
                for box, confidence in zip(boxes, confidences):
                    x1, y1, x2, y2 = box
                    bbox = (int(x1), int(y1), int(x2), int(y2))
                    crop = self._square_face_crop(frame, bbox)
                    faces.append(
                        {
                            "bbox": bbox,
                            "confidence": float(confidence),
                            "embedding": self._fallback_embedding(crop),
                            "backend": "yolo",
                        }
                    )
            return faces

        if self._fallback_net is not None:
            height, width = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)),
                1.0,
                (300, 300),
                (104.0, 177.0, 123.0),
            )
            self._fallback_net.setInput(blob)
            detections = self._fallback_net.forward()
            faces = []
            for index in range(detections.shape[2]):
                confidence = float(detections[0, 0, index, 2])
                if confidence < self.MIN_FACE_CONFIDENCE:
                    continue
                box = detections[0, 0, index, 3:7] * np.array(
                    [width, height, width, height]
                )
                x1, y1, x2, y2 = box.astype(int)
                bbox = (
                    max(0, x1),
                    max(0, y1),
                    min(width, x2),
                    min(height, y2),
                )
                crop = self._square_face_crop(frame, bbox)
                faces.append(
                    {
                        "bbox": bbox,
                        "confidence": confidence,
                        "embedding": self._fallback_embedding(crop),
                        "backend": "opencv-dnn",
                    }
                )
            return faces

        if self._fallback_detector is None:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._fallback_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(self.MIN_FACE_SIZE, self.MIN_FACE_SIZE),
        )
        return [
            {
                "bbox": (int(x), int(y), int(x + w), int(y + h)),
                "confidence": 1.0,
                "embedding": self._fallback_embedding(
                    self._square_face_crop(
                        frame,
                        (int(x), int(y), int(x + w), int(y + h)),
                    )
                ),
                "backend": "haar",
            }
            for x, y, w, h in faces
        ]

    def _fallback_embedding(self, crop):
        if crop is None or crop.size == 0:
            return np.zeros(495, dtype=np.float32)

        face = cv2.resize(crop, (96, 96))
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        texture = cv2.resize(gray, (32, 32)).astype(np.float32) / 255.0
        texture = cv2.dct(texture)[:16, :16].flatten()
        texture = texture[1:]

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        gradient_parts = []
        for row in range(0, 96, 24):
            for column in range(0, 96, 24):
                cell_angle = angle[row:row + 24, column:column + 24]
                cell_magnitude = magnitude[row:row + 24, column:column + 24]
                histogram, _ = np.histogram(
                    cell_angle,
                    bins=9,
                    range=(0, 360),
                    weights=cell_magnitude,
                )
                gradient_parts.extend(histogram.tolist())

        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        color_hist = cv2.calcHist(
            [hsv], [0, 1], None, [12, 8], [0, 180, 0, 256]
        ).flatten()

        descriptor = np.concatenate(
            [
                self._normalize_embedding(texture),
                self._normalize_embedding(np.asarray(gradient_parts)),
                self._normalize_embedding(color_hist),
            ]
        )
        return self._normalize_embedding(descriptor)

    def _square_face_crop(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        height, width = frame.shape[:2]
        face_w = max(1, x2 - x1)
        face_h = max(1, y2 - y1)
        side = int(max(face_w, face_h) * (1.0 + 2.0 * self.CROP_PADDING))
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        sx1 = max(0, cx - side // 2)
        sy1 = max(0, cy - side // 2)
        sx2 = min(width, sx1 + side)
        sy2 = min(height, sy1 + side)
        sx1 = max(0, sx2 - side)
        sy1 = max(0, sy2 - side)
        crop = frame[sy1:sy2, sx1:sx2]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (self.FACE_SIZE, self.FACE_SIZE))

    def _quality_check(self, frame, bbox, crop, confidence, backend=None):
        x1, y1, x2, y2 = bbox
        min_face_size = (
            self.YOLO_MIN_FACE_SIZE if backend == "yolo" else self.MIN_FACE_SIZE
        )
        min_confidence = (
            self.YOLO_FACE_CONFIDENCE
            if backend == "yolo"
            else self.MIN_FACE_CONFIDENCE
        )
        min_blur = (
            self.YOLO_MIN_BLUR_VARIANCE
            if backend == "yolo"
            else self.MIN_BLUR_VARIANCE
        )
        if crop is None:
            return {"accepted": False, "reason": "empty"}
        if x2 - x1 < min_face_size or y2 - y1 < min_face_size:
            return {"accepted": False, "reason": "too_small"}
        if confidence < min_confidence:
            return {"accepted": False, "reason": "low_confidence"}

        raw = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if raw.size == 0:
            return {"accepted": False, "reason": "empty"}
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        if blur_variance < min_blur:
            return {"accepted": False, "reason": "too_blurry"}
        if brightness < self.MIN_BRIGHTNESS:
            return {"accepted": False, "reason": "too_dark"}
        return {
            "accepted": True,
            "reason": "ok",
            "blur_variance": blur_variance,
            "brightness": brightness,
        }

    def _is_probably_masked(self, crop):
        # Conservative heuristic: only flag a mask when upper-face skin is visible
        # and the lower face has a pronounced loss of skin pixels.
        ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
        skin = cv2.inRange(
            ycrcb,
            np.array([0, 133, 77], dtype=np.uint8),
            np.array([255, 173, 127], dtype=np.uint8),
        )
        upper_ratio = float(np.mean(skin[35:105] > 0))
        lower_ratio = float(np.mean(skin[125:205] > 0))
        return upper_ratio >= 0.18 and lower_ratio <= 0.035

    def _find_best_match(self, embedding):
        try:
            identities = list(FaceIdentity.objects.filter(is_active=True))
        except (OperationalError, ProgrammingError):
            logger.warning("FaceIdentity table is not ready; run migrations.")
            return None, None

        best_identity = None
        best_similarity = None
        for identity in identities:
            stored = np.asarray(identity.embedding, dtype=np.float32)
            if stored.size != embedding.size:
                continue
            similarity = self.cosine_similarity(embedding, stored)
            if best_similarity is None or similarity > best_similarity:
                best_identity = identity
                best_similarity = similarity
        return best_identity, best_similarity

    def _identity_for_track(self, head_track_id, embedding):
        if not head_track_id:
            return None, None
        cached = self._track_identities.get(head_track_id)
        if not cached:
            return None, None
        if cached["expires_at"] < timezone.now():
            self._track_identities.pop(head_track_id, None)
            return None, None
        try:
            identity = FaceIdentity.objects.filter(
                pk=cached["identity_id"], is_active=True
            ).first()
        except (OperationalError, ProgrammingError):
            return None, None
        if identity is None:
            return None, None
        stored = np.asarray(identity.embedding, dtype=np.float32)
        if stored.size != embedding.size:
            return None, None
        return identity, self.cosine_similarity(embedding, stored)

    def _remember_track_identity(self, head_track_id, identity):
        if not head_track_id or identity is None:
            return
        self._track_identities[head_track_id] = {
            "identity_id": identity.pk,
            "expires_at": timezone.now()
            + timedelta(seconds=self.TRACK_IDENTITY_TTL_SECONDS),
        }

    @staticmethod
    def cosine_similarity(left, right):
        left = np.asarray(left, dtype=np.float32)
        right = np.asarray(right, dtype=np.float32)
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator == 0:
            return 0.0
        return float(np.dot(left, right) / denominator)

    @staticmethod
    def _normalize_embedding(embedding):
        embedding = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm else embedding

    def _create_identity(self, embedding, crop):
        try:
            with transaction.atomic():
                latest = (
                    FaceIdentity.objects.select_for_update()
                    .order_by("-person_id")
                    .first()
                )
                next_number = 1
                if latest is not None:
                    try:
                        next_number = int(latest.person_id.split("-")[-1]) + 1
                    except (TypeError, ValueError):
                        next_number = FaceIdentity.objects.count() + 1
                person_id = f"PERSON-{next_number:04d}"
                image_path = self._save_known_image(person_id, crop)
                return FaceIdentity.objects.create(
                    person_id=person_id,
                    embedding=embedding.tolist(),
                    image_path=image_path,
                    is_masked=False,
                )
        except (OperationalError, ProgrammingError):
            logger.warning("Could not create face identity; run migrations.")
            return None

    def _touch_identity(self, identity, embedding, crop):
        now = timezone.now()
        key = f"known:{identity.person_id}"
        if not self._should_save(key, now):
            return

        old_embedding = np.asarray(identity.embedding, dtype=np.float32)
        blended = self._normalize_embedding((old_embedding * 0.8) + (embedding * 0.2))
        image_path = self._save_known_image(identity.person_id, crop)
        FaceIdentity.objects.filter(pk=identity.pk).update(
            embedding=blended.tolist(),
            image_path=image_path,
            last_seen=now,
            seen_count=F("seen_count") + 1,
        )
        identity.embedding = blended.tolist()
        identity.image_path = image_path

    def _save_known_image(self, person_id, crop):
        person_dir = self.known_dir / person_id
        person_dir.mkdir(parents=True, exist_ok=True)
        stamp = timezone.localtime().strftime("%Y%m%d_%H%M%S")
        archive = person_dir / f"{stamp}.jpg"
        latest = person_dir / "latest.jpg"
        cv2.imwrite(str(archive), crop)
        cv2.imwrite(str(latest), crop)
        return latest.relative_to(self.media_root).as_posix()

    def _save_observation(self, crop, label, throttle_key):
        now = timezone.now()
        key = f"{label.lower()}:{throttle_key}"
        if not self._should_save(key, now):
            return ""
        directory = self.masked_dir if label == "MASKED" else self.unknown_dir
        stamp = timezone.localtime(now).strftime("%Y%m%d_%H%M%S_%f")
        path = directory / f"{stamp}.jpg"
        cv2.imwrite(str(path), crop)
        return path.relative_to(self.media_root).as_posix()

    def _should_save(self, key, now):
        with self._lock:
            last_saved = self._last_saved.get(key)
            if last_saved and now - last_saved < timedelta(
                seconds=self.SAVE_INTERVAL_SECONDS
            ):
                return False
            self._last_saved[key] = now
            return True

    @staticmethod
    def _bbox_key(bbox):
        x1, y1, x2, y2 = bbox
        return f"{x1 // 50}:{y1 // 50}:{x2 // 50}:{y2 // 50}"

    @staticmethod
    def _match_head_track(face_bbox, head_detections):
        fx1, fy1, fx2, fy2 = face_bbox
        face_area = max(1, (fx2 - fx1) * (fy2 - fy1))
        face_cx = (fx1 + fx2) / 2.0
        face_cy = (fy1 + fy2) / 2.0
        best_id = None
        best_score = 0.0
        for head in head_detections:
            hx1, hy1, hx2, hy2 = head["bbox"]
            ix1 = max(fx1, hx1)
            iy1 = max(fy1, hy1)
            ix2 = min(fx2, hx2)
            iy2 = min(fy2, hy2)
            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            overlap = intersection / face_area
            head_cx = (hx1 + hx2) / 2.0
            head_cy = (hy1 + hy2) / 2.0
            head_size = max(1.0, hx2 - hx1, hy2 - hy1)
            distance = np.hypot(face_cx - head_cx, face_cy - head_cy)
            proximity = max(0.0, 1.0 - (distance / (head_size * 1.5)))
            score = max(overlap, proximity)
            if score > best_score:
                best_score = score
                best_id = head["id"]
        return best_id if best_score >= 0.25 else None

    @staticmethod
    def _result(
        bbox,
        label,
        status,
        detector_confidence,
        similarity,
        head_track_id,
        image_path,
    ):
        return {
            "bbox": bbox,
            "label": label,
            "status": status,
            "detector_confidence": round(float(detector_confidence), 3),
            "similarity": (
                round(float(similarity), 3) if similarity is not None else None
            ),
            "head_track_id": head_track_id,
            "image_path": image_path,
        }
