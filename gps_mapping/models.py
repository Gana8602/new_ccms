from django.db import models

class CameraGCPConfiguration(models.Model):
    camera_name = models.CharField(max_length=255)
    camera_id = models.CharField(max_length=100, unique=True)

    # Top Left GCP
    top_left_pixel_x = models.FloatField()
    top_left_pixel_y = models.FloatField()
    top_left_lat = models.FloatField()
    top_left_lon = models.FloatField()

    # Top Right GCP
    top_right_pixel_x = models.FloatField()
    top_right_pixel_y = models.FloatField()
    top_right_lat = models.FloatField()
    top_right_lon = models.FloatField()

    # Bottom Right GCP
    bottom_right_pixel_x = models.FloatField()
    bottom_right_pixel_y = models.FloatField()
    bottom_right_lat = models.FloatField()
    bottom_right_lon = models.FloatField()

    # Bottom Left GCP
    bottom_left_pixel_x = models.FloatField()
    bottom_left_pixel_y = models.FloatField()
    bottom_left_lat = models.FloatField()
    bottom_left_lon = models.FloatField()

    # Geospatial stats
    area_m2 = models.FloatField(default=0.0)
    area_hectare = models.FloatField(default=0.0)
    perimeter_m = models.FloatField(default=0.0)

    # Homography matrix
    # Stores 3x3 projection matrix as list of list: [[h11, h12, h13], [h21, h22, h23], [h31, h32, h33]]
    homography_matrix = models.JSONField(null=True, blank=True)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.camera_name} (ID: {self.camera_id})"
