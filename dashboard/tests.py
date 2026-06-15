import math
import tempfile
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
from django.test import TestCase, override_settings
from django.utils import timezone

from .models import FaceIdentity
from .services.crowd_counter import CrowdCounterService
from .services.face_identity_manager import FaceIdentityManager
from .views import FrameQualityGuard


class FakeHeadDetector:
    def detect(self, frame):
        return [
            {
                "id": "H_1",
                "bbox": (10, 10, 70, 70),
                "conf": 0.9,
                "center": (40, 40),
            },
            {
                "id": "H_2",
                "bbox": (100, 10, 160, 70),
                "conf": 0.8,
                "center": (130, 40),
            },
        ]


class FakePersonDetector:
    def detect(self, frame):
        return [
            {
                "id": "B_1",
                "bbox": (0, 0, 80, 180),
                "conf": 0.9,
                "center": (40, 90),
            }
        ]


class FaceIdentityManagerTests(TestCase):
    def setUp(self):
        self.temporary_media = tempfile.TemporaryDirectory()
        self.settings_override = override_settings(
            MEDIA_ROOT=Path(self.temporary_media.name)
        )
        self.settings_override.enable()
        with patch.object(FaceIdentityManager, "_load_backend"):
            self.manager = FaceIdentityManager()
        self.manager._face_app = object()
        self.frame = np.random.default_rng(7).integers(
            0, 255, size=(180, 180, 3), dtype=np.uint8
        )
        self.heads = FakeHeadDetector().detect(self.frame)

    def tearDown(self):
        self.settings_override.disable()
        self.temporary_media.cleanup()

    def _recognize(self, embedding, confidence=0.95):
        detection = {
            "bbox": (10, 10, 90, 90),
            "confidence": confidence,
            "embedding": np.asarray(embedding, dtype=np.float32),
        }
        with (
            patch.object(self.manager, "_detect_faces", return_value=[detection]),
            patch.object(self.manager, "_is_probably_masked", return_value=False),
        ):
            return self.manager.process_frame(self.frame, self.heads)[0]

    def test_same_embedding_reuses_person_id(self):
        first = self._recognize([1.0, 0.0])
        second = self._recognize([1.0, 0.0])

        self.assertEqual(first["label"], "PERSON-0001")
        self.assertEqual(second["label"], "PERSON-0001")
        self.assertEqual(FaceIdentity.objects.count(), 1)

    def test_new_strong_face_gets_next_person_id(self):
        first = self._recognize([1.0, 0.0])
        second = self._recognize([0.0, 1.0])

        self.assertEqual(first["label"], "PERSON-0001")
        self.assertEqual(second["label"], "PERSON-0002")
        self.assertEqual(FaceIdentity.objects.count(), 2)

    def test_uncertain_clear_face_gets_new_id_instead_of_wrong_match(self):
        self._recognize([1.0, 0.0])
        uncertain = self._recognize([0.5, math.sqrt(0.75)])

        self.assertEqual(uncertain["label"], "PERSON-0002")
        self.assertEqual(uncertain["status"], "known")
        self.assertEqual(FaceIdentity.objects.count(), 2)

    def test_quality_approved_clear_face_gets_identity(self):
        result = self._recognize([1.0, 0.0], confidence=0.70)

        self.assertEqual(result["label"], "PERSON-0001")
        self.assertEqual(FaceIdentity.objects.count(), 1)

    def test_masked_face_never_creates_identity(self):
        detection = {
            "bbox": (10, 10, 90, 90),
            "confidence": 0.95,
            "embedding": np.asarray([1.0, 0.0], dtype=np.float32),
        }
        with (
            patch.object(self.manager, "_detect_faces", return_value=[detection]),
            patch.object(self.manager, "_is_probably_masked", return_value=True),
        ):
            result = self.manager.process_frame(self.frame, self.heads)[0]

        self.assertEqual(result["label"], "MASKED")
        self.assertEqual(FaceIdentity.objects.count(), 0)

    def test_fallback_embedding_is_stable_for_same_crop(self):
        first = self.manager._fallback_embedding(self.frame)
        second = self.manager._fallback_embedding(self.frame.copy())

        self.assertGreater(self.manager.cosine_similarity(first, second), 0.999)

    def test_same_identity_image_is_throttled_for_five_seconds(self):
        now = timezone.now()

        self.assertTrue(self.manager._should_save("known:PERSON-0001", now))
        self.assertFalse(
            self.manager._should_save(
                "known:PERSON-0001", now + timedelta(seconds=4)
            )
        )
        self.assertTrue(
            self.manager._should_save(
                "known:PERSON-0001", now + timedelta(seconds=5)
            )
        )


class CrowdCountArchitectureTests(TestCase):
    def test_total_count_comes_from_heads_not_faces_or_people(self):
        service = CrowdCounterService(
            FakeHeadDetector(),
            FakePersonDetector(),
            face_identity_manager=None,
        )
        frame = np.zeros((200, 200, 3), dtype=np.uint8)

        service.process_frame(frame, camera_id="test")
        stats = service.get_global_stats()

        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["inside"], 2)
        self.assertEqual(stats["mode"], "head_priority")


class FrameQualityGuardTests(TestCase):
    def setUp(self):
        self.guard = FrameQualityGuard()
        self.clean = np.full((180, 320, 3), 90, dtype=np.uint8)
        cv2 = __import__("cv2")
        cv2.rectangle(self.clean, (30, 30), (120, 150), (130, 100, 80), -1)
        cv2.circle(self.clean, (230, 90), 45, (70, 140, 110), -1)

    def test_accepts_normal_small_frame_changes(self):
        changed = self.clean.copy()
        changed[60:100, 150:190] = (110, 120, 100)

        self.assertTrue(self.guard.accept(self.clean))
        self.assertTrue(self.guard.accept(changed))

    def test_rejects_unstable_corrupted_burst(self):
        rng = np.random.default_rng(123)
        corrupted_one = rng.integers(
            0, 255, size=self.clean.shape, dtype=np.uint8
        )
        corrupted_two = rng.integers(
            0, 255, size=self.clean.shape, dtype=np.uint8
        )

        self.assertTrue(self.guard.accept(self.clean))
        self.assertFalse(self.guard.accept(corrupted_one))
        self.assertFalse(self.guard.accept(corrupted_two))

    def test_rejects_vertical_decoder_smear(self):
        smeared = self.clean.copy()
        source_row = smeared[smeared.shape[0] // 2].copy()
        smeared[smeared.shape[0] // 2:] = source_row

        self.assertFalse(self.guard.accept(smeared))

    def test_accepts_stable_scene_change_after_confirmation(self):
        new_scene = np.full_like(self.clean, 210)

        self.assertTrue(self.guard.accept(self.clean))
        self.assertFalse(self.guard.accept(new_scene))
        self.assertFalse(self.guard.accept(new_scene))
        self.assertTrue(self.guard.accept(new_scene))

# Create your tests here.
