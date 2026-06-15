# Face Recognition Setup

The application runs without InsightFace. In that mode it uses a conservative
OpenCV appearance descriptor so clear, quality-approved faces still receive
and reuse `PERSON-ID` values. Masked faces remain `MASKED` and never receive a
permanent identity.

The fallback prefers the existing local `yolov8n-face.pt` model for crowded
scenes, then the OpenCV DNN face detector under `models/`. Haar is used only
when neither model is available.

ArcFace and OpenCV DNN retain the 50x50 minimum. The current 898x506 crowd
camera contains faces roughly 24-43 pixels wide, so the high-confidence YOLO
crowd-face path uses a 24-pixel adaptive minimum and stricter detector-driven
filtering instead of discarding every visible face.

Install the optional backend in the same virtual environment as Django:

```bash
./venv/bin/pip install -r requirements-face.txt
```

On Apple Silicon, `onnxruntime` is the portable CPU option. If InsightFace does
not build under the project's Python version, create the project environment
with a Python version supported by InsightFace and reinstall the existing
project dependencies there. Do not run two Django environments against the
same live video stream.

The first successful InsightFace startup downloads the `buffalo_l` model pack.
The process therefore needs network access once, and the runtime user needs a
writable InsightFace model cache.

After installation:

```bash
./venv/bin/python manage.py migrate
./venv/bin/python manage.py runserver
```

At startup, `insightface-arcface` is preferred. Without it, the application
reports `yolo-face-opencv-appearance`. ArcFace is substantially more reliable across
pose, lighting, and long gaps, so install it for production identity matching.
