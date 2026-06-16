import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
import numpy as np
import threading
import json
import time
from pathlib import Path
from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.db.models import Sum

from .detectors.yolo_person_detector import YOLOPersonDetector
from .detectors.head_detector import HeadDetector
from .models import FaceIdentity
from .services.crowd_counter import CrowdCounterService
from .services.face_identity_manager import FaceIdentityManager

head_detector = None
person_detector = None
face_identity_manager = None
crowd_counter = None
stream_reader = None
inference_engine = None
pipeline_lock = threading.Lock()


class FrameQualityGuard:
    PREVIEW_SIZE = (96, 54)
    SPATIAL_PREVIEW_SIZE = (320, 180)
    MAX_TEMPORAL_DIFFERENCE = 32.0
    MAX_CORRUPTION_EDGE_RATIO = 0.42
    MAX_VERTICAL_SMEAR_RATIO = 1.30
    MIN_VERTICAL_FLAT_RATIO = 0.45
    SCENE_CHANGE_CONFIRM_FRAMES = 3

    def __init__(self):
        self.last_good_preview = None
        self.pending_preview = None
        self.pending_count = 0

    def reset(self):
        self.pending_preview = None
        self.pending_count = 0

    def accept(self, frame):
        if (
            frame is None
            or frame.ndim != 3
            or frame.shape[2] != 3
            or frame.size == 0
            or frame.dtype != np.uint8
        ):
            return False

        preview = cv2.resize(
            frame,
            self.PREVIEW_SIZE,
            interpolation=cv2.INTER_AREA,
        )
        spatial_preview = cv2.resize(
            frame,
            self.SPATIAL_PREVIEW_SIZE,
            interpolation=cv2.INTER_AREA,
        )
        spatial_gray = cv2.cvtColor(spatial_preview, cv2.COLOR_BGR2GRAY)
        lower_region = spatial_gray[int(spatial_gray.shape[0] * 0.50):]
        horizontal_change = float(
            np.mean(
                np.abs(
                    np.diff(lower_region.astype(np.float32), axis=1)
                )
            )
        )
        vertical_differences = np.abs(
            np.diff(lower_region.astype(np.float32), axis=0)
        )
        vertical_change = float(np.mean(vertical_differences))
        vertical_flat_ratio = float(np.mean(vertical_differences < 2.0))
        vertical_smear_ratio = horizontal_change / (vertical_change + 1e-6)
        if (
            vertical_smear_ratio > self.MAX_VERTICAL_SMEAR_RATIO
            and vertical_flat_ratio > self.MIN_VERTICAL_FLAT_RATIO
        ):
            self.reset()
            return False

        gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 180)
        edge_ratio = float(np.mean(edges > 0))
        if edge_ratio > self.MAX_CORRUPTION_EDGE_RATIO:
            self.reset()
            return False

        if self.last_good_preview is None:
            self.last_good_preview = preview
            return True

        difference = float(
            np.mean(
                cv2.absdiff(preview, self.last_good_preview).astype(np.float32)
            )
        )
        if difference <= self.MAX_TEMPORAL_DIFFERENCE:
            self.last_good_preview = preview
            self.reset()
            return True

        # A real camera cut remains visually stable for subsequent frames.
        # Decoder corruption usually changes unpredictably on every frame.
        if self.pending_preview is None:
            self.pending_preview = preview
            self.pending_count = 1
            return False

        pending_difference = float(
            np.mean(
                cv2.absdiff(preview, self.pending_preview).astype(np.float32)
            )
        )
        if pending_difference <= self.MAX_TEMPORAL_DIFFERENCE:
            self.pending_count += 1
            self.pending_preview = preview
            if self.pending_count >= self.SCENE_CHANGE_CONFIRM_FRAMES:
                self.last_good_preview = preview
                self.reset()
                return True
        else:
            self.pending_preview = preview
            self.pending_count = 1
        return False


class RTSPVideoStream:
    MAX_REJECTED_FRAMES = 8
    RECONNECT_DELAY_SECONDS = 0.5

    def __init__(self, src):
        self.src = src
        self.stream = None
        self.grabbed = False
        self.frame = None
        self.frame_id = 0
        self.lock = threading.Lock()
        self.stopped = False
        self.quality_guard = FrameQualityGuard()
        self.rejected_frames = 0

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def _connect(self):
        self.stream = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return self.stream.isOpened()

    def _disconnect(self):
        if self.stream is not None:
            self.stream.release()
        self.stream = None
        self.rejected_frames = 0
        self.quality_guard.reset()

    def update(self):
        while not self.stopped:
            if self.stream is None:
                if not self._connect():
                    self._disconnect()
                    time.sleep(1)
                    continue

            grabbed, frame = self.stream.read()

            if not grabbed or frame is None:
                self._disconnect()
                time.sleep(1)
                continue

            if not self.quality_guard.accept(frame):
                self.rejected_frames += 1
                if self.rejected_frames >= self.MAX_REJECTED_FRAMES:
                    self._disconnect()
                    time.sleep(self.RECONNECT_DELAY_SECONDS)
                continue

            self.rejected_frames = 0
            with self.lock:
                self.grabbed = True
                self.frame = frame
                self.frame_id += 1

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None, self.frame_id
            return self.grabbed, self.frame.copy(), self.frame_id

    def stop(self):
        self.stopped = True
        self._disconnect()

# --- Async Inference Engine ---
# This engine continuously processes the *latest* available frame
# and skips any intermediate frames if inference takes too long.
class InferenceEngine:
    def __init__(self):
        self.latest_processed_frame = None
        self.latest_raw_frame = None
        self.lock = threading.Lock()
        self.stopped = False
        
    def start(self):
        threading.Thread(target=self.run, daemon=True).start()
        return self
        
    def run(self):
        last_frame_id = -1
        
        while not self.stopped:
            success, frame, frame_id = stream_reader.read()
            if not success or frame is None:
                time.sleep(0.1)
                continue
                
            if frame_id == last_frame_id:
                time.sleep(0.01)
                continue
                
            last_frame_id = frame_id
            
            # Dynamic Fencing (Centered)
            h, w = frame.shape[:2]
            fence_w = int(w * 0.35)
            fence_h = int(h * 0.35)
            cx, cy = w // 2, h // 2
            fence_pts = [[cx - fence_w, cy - fence_h], [cx + fence_w, cy - fence_h], [cx + fence_w, cy + fence_h], [cx - fence_w, cy + fence_h]]
            crowd_counter.roi_service.set_roi("0", fence_pts)
            
            # Keep Clear View completely clean while all analytics continue.
            raw_copy = frame.copy()
            
            # Heavy inference (this modifies frame in-place)
            processed = crowd_counter.process_frame(frame, camera_id="0")
            
            with self.lock:
                self.latest_raw_frame = raw_copy
                self.latest_processed_frame = processed.copy() if processed is not None else None

def get_face_identity_manager():
    global face_identity_manager
    with pipeline_lock:
        if face_identity_manager is None:
            face_identity_manager = FaceIdentityManager()
    return face_identity_manager


def ensure_pipeline_started():
    global head_detector, person_detector, face_identity_manager, crowd_counter
    global stream_reader, inference_engine

    if inference_engine is not None:
        return
    with pipeline_lock:
        if inference_engine is not None:
            return

        print("Initializing Architecture...")
        head_detector = HeadDetector(
            "/Volumes/tridelMac/Projects/Dev/ccms/ccms/models/best_head.pt"
        )
        person_detector = YOLOPersonDetector("yolo11m.pt")
        manager = face_identity_manager or FaceIdentityManager()
        face_identity_manager = manager
        crowd_counter = CrowdCounterService(
            head_detector,
            person_detector,
            face_identity_manager=manager,
        )

        default_fence_pts = [
            [160, 90],
            [480, 90],
            [480, 270],
            [160, 270],
        ]
        crowd_counter.roi_service.set_roi("0", default_fence_pts)
        stream_reader = RTSPVideoStream(
            "rtsp://127.0.0.1:8554/mystream1"
        ).start()
        inference_engine = InferenceEngine().start()

# Global state for UI view mode
ai_view_enabled = False

@csrf_exempt
def set_view_mode(request):
    global ai_view_enabled
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            ai_view_enabled = data.get("ai_view", False)
            return JsonResponse({"status": "success", "ai_view": ai_view_enabled})
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=400)
    return JsonResponse({"status": "error", "message": "Invalid method"}, status=405)

def home(request):
    ensure_pipeline_started()
    return render(request, 'dashboard/home.html')

def get_counts(request):
    ensure_pipeline_started()
    stats = crowd_counter.get_global_stats()
    return JsonResponse({
        'inside': stats.get('inside', 0),
        'outside': stats.get('outside', 0),
        'total': stats.get('total', 0),
        'occupancy': stats.get('occupancy', 0),
        'available': stats.get('available_capacity', 200),
        'density_percentage': stats.get('density_percentage', 0),
        'density_level': stats.get('density_level', 'LOW'),
        'alerts': stats.get('alerts', [])
    })


def _media_url(image_path):
    if not image_path:
        return ""
    return f"{settings.MEDIA_URL}{image_path.lstrip('/')}"


def _observation_payload(directory_name, label, limit=50):
    directory = Path(settings.MEDIA_ROOT) / "faces" / directory_name
    directory.mkdir(parents=True, exist_ok=True)
    files = sorted(
        directory.glob("*.jpg"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:limit]
    return [
        {
            "label": label,
            "image_url": _media_url(path.relative_to(settings.MEDIA_ROOT).as_posix()),
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%S%z",
                time.localtime(path.stat().st_mtime),
            ),
        }
        for path in files
    ]


def get_known_faces(request):
    manager = get_face_identity_manager()
    faces = FaceIdentity.objects.filter(is_active=True).order_by("-last_seen")[:50]
    return JsonResponse(
        {
            "faces": [
                {
                    "person_id": face.person_id,
                    "image_url": _media_url(face.image_path),
                    "first_seen": face.first_seen.isoformat(),
                    "last_seen": face.last_seen.isoformat(),
                    "seen_count": face.seen_count,
                }
                for face in faces
            ],
            "recognition_available": manager.recognition_available,
            "backend": manager.backend_name,
        }
    )


def get_unknown_faces(request):
    # Clear quality-approved faces receive PERSON-ID values. Unknown is
    # reserved for masked faces and is exposed through /api/faces/masked/.
    return JsonResponse({"faces": []})


def get_masked_faces(request):
    return JsonResponse(
        {"faces": _observation_payload("masked_unknown", "MASKED")}
    )


def get_face_stats(request):
    known = FaceIdentity.objects.filter(is_active=True)
    faces_root = Path(settings.MEDIA_ROOT) / "faces"
    unknown_count = 0
    masked_count = sum(
        1 for _ in (faces_root / "masked_unknown").glob("*.jpg")
    )
    return JsonResponse(
        {
            "known_count": known.count(),
            "unknown_count": unknown_count,
            "masked_count": masked_count,
            "total_recognized": known.aggregate(total=Sum("seen_count"))["total"] or 0,
        }
    )

def gen_frames():
    ensure_pipeline_started()
    # This just streams the latest frame to the client browser at ~30 FPS
    # Independent of how slow the inference engine is running
    while True:
        time.sleep(0.03) # Cap streaming FPS to ~30
        
        with inference_engine.lock:
            if ai_view_enabled:
                frame_to_send = inference_engine.latest_processed_frame
            else:
                frame_to_send = inference_engine.latest_raw_frame
            
        if frame_to_send is None:
            # Fallback to pure video if inference hasn't run yet
            success, f, _ = stream_reader.read()
            if success and f is not None:
                frame_to_send = f.copy()
            else:
                continue
        
        # Encode and yield frame
        ret, buffer = cv2.imencode('.jpg', frame_to_send)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def video_feed(request):
    return StreamingHttpResponse(gen_frames(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

def landing(request):
    ensure_pipeline_started()
    return render(request, 'dashboard/landing.html')
