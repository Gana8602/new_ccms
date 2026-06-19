from django.db import models


class Camera(models.Model):
    name = models.CharField(max_length=255)
    url = models.CharField(max_length=500)
    is_active = models.BooleanField(default=True)
    gcp_points = models.JSONField(default=dict, blank=True)
    zones = models.JSONField(default=list, blank=True)

    def __str__(self):
        return self.name

class FaceIdentity(models.Model):
    person_id = models.CharField(max_length=32, unique=True, db_index=True)
    embedding = models.JSONField(default=list)
    image_path = models.CharField(max_length=500, blank=True)
    first_seen = models.DateTimeField(auto_now_add=True)
    last_seen = models.DateTimeField(auto_now=True)
    seen_count = models.PositiveIntegerField(default=1)
    is_masked = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ("-last_seen",)
        verbose_name_plural = "face identities"

    def __str__(self):
        return self.person_id
