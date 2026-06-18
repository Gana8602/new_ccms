import json
from django.test import TestCase, Client
from django.urls import reverse
from gps_mapping.models import CameraGCPConfiguration
from gps_mapping.geospatial import LocalCartesianProjection, calculate_polygon_area
from gps_mapping.homography import calculate_homography, pixel_to_metric
from gps_mapping.services import save_gcp_config, pixel_to_gps

class GeospatialTests(TestCase):
    def test_local_projection(self):
        # Center around Singapore: lat 1.3521, lon 103.8198
        origin_lat, origin_lon = 1.3521, 103.8198
        proj = LocalCartesianProjection(origin_lat, origin_lon)
        
        # Origin point should map exactly to (0, 0)
        x0, y0 = proj.gps_to_metric(origin_lat, origin_lon)
        self.assertAlmostEqual(x0, 0.0, places=2)
        self.assertAlmostEqual(y0, 0.0, places=2)
        
        # Check round trip conversion
        test_lat, test_lon = 1.36, 103.83
        x, y = proj.gps_to_metric(test_lat, test_lon)
        lat_back, lon_back = proj.metric_to_gps(x, y)
        self.assertAlmostEqual(test_lat, lat_back, places=5)
        self.assertAlmostEqual(test_lon, lon_back, places=5)

    def test_area_calculation(self):
        # 4 points representing a ~100m x ~100m square in degrees
        # (approx 111,320m per degree lat, and approx 111,320 * cos(lat) per degree lon)
        # origin (0, 0)
        # top left, top right, bottom right, bottom left
        points = [
            [0.0, 0.0],
            [0.0, 0.0009],  # ~100m east
            [-0.0009, 0.0009], # ~100m south, ~100m east
            [-0.0009, 0.0],  # ~100m south
        ]
        
        stats = calculate_polygon_area(points)
        self.assertGreater(stats["area_m2"], 9000.0)
        self.assertLess(stats["area_m2"], 11000.0)
        self.assertGreater(stats["perimeter_m"], 380.0)
        self.assertLess(stats["perimeter_m"], 420.0)
        self.assertAlmostEqual(stats["area_hectare"], stats["area_m2"] / 10000.0, places=5)


class HomographyTests(TestCase):
    def test_homography_and_metric_mapping(self):
        # Simple square pixel space (e.g. 1000x1000)
        pixel_points = [
            [0, 0],
            [1000, 0],
            [1000, 1000],
            [0, 1000]
        ]
        
        # Corresponding metric coordinates (e.g. 100mx100m square in meters)
        # Lat/Lon mappings around origin (0.0, 0.0)
        gps_points = [
            [0.0, 0.0],
            [0.0, 0.0009],
            [-0.0009, 0.0009],
            [-0.0009, 0.0]
        ]
        
        # Calculate homography matrix and projection origin
        H_matrix, origin = calculate_homography(pixel_points, gps_points)
        self.assertEqual(len(H_matrix), 3)
        self.assertEqual(len(H_matrix[0]), 3)
        self.assertEqual(origin, (0.0, 0.0))
        
        # Check mapping of center point: pixel (500, 500)
        mx, my = pixel_to_metric(500, 500, H_matrix)
        proj = LocalCartesianProjection(origin[0], origin[1])
        lat, lon = proj.metric_to_gps(mx, my)
        
        # Coordinates should fall roughly in the center of the bounding box
        self.assertLess(lat, 0.0)
        self.assertGreater(lat, -0.0009)
        self.assertGreater(lon, 0.0)
        self.assertLess(lon, 0.0009)


class APITests(TestCase):
    def setUp(self):
        self.client = Client()
        self.camera_id = "test_cam_1"
        self.gcp_payload = {
            "camera_id": self.camera_id,
            "camera_name": "Test Camera One",
            "gcps": {
                "top_left": {"x": 0, "y": 0, "lat": 12.345, "lon": 56.789},
                "top_right": {"x": 640, "y": 0, "lat": 12.345, "lon": 56.790},
                "bottom_right": {"x": 640, "y": 480, "lat": 12.344, "lon": 56.790},
                "bottom_left": {"x": 0, "y": 480, "lat": 12.344, "lon": 56.789}
            }
        }

    def test_save_and_retrieve_gcp(self):
        # 1. Save configuration
        response = self.client.post(
            reverse("save_gcp"),
            data=json.dumps(self.gcp_payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("area_m2", data)
        self.assertTrue(CameraGCPConfiguration.objects.filter(camera_id=self.camera_id).exists())

        # 2. Retrieve configuration
        response = self.client.get(reverse("get_gcp", args=[self.camera_id]))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["configured"])
        self.assertEqual(data["camera_name"], "Test Camera One")
        self.assertEqual(data["gcps"]["top_left"]["x"], 0)

    def test_pixel_to_gps_conversion(self):
        # Save first
        self.client.post(
            reverse("save_gcp"),
            data=json.dumps(self.gcp_payload),
            content_type="application/json"
        )
        
        # Test pixel-to-gps endpoint
        payload = {
            "camera_id": self.camera_id,
            "x": 320,
            "y": 240
        }
        response = self.client.post(
            reverse("pixel_to_gps"),
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("latitude", data)
        self.assertIn("longitude", data)
        self.assertAlmostEqual(data["latitude"], 12.3445, places=3)
        self.assertAlmostEqual(data["longitude"], 56.7895, places=3)

    def test_clear_configuration(self):
        # Save first
        self.client.post(
            reverse("save_gcp"),
            data=json.dumps(self.gcp_payload),
            content_type="application/json"
        )
        
        # Send clear action
        payload = {
            "camera_id": self.camera_id,
            "action": "clear"
        }
        response = self.client.post(
            reverse("save_gcp"),
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(CameraGCPConfiguration.objects.filter(camera_id=self.camera_id).exists())
