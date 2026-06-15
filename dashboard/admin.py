from django.contrib import admin
from .models import FaceIdentity


@admin.register(FaceIdentity)
class FaceIdentityAdmin(admin.ModelAdmin):
    list_display = (
        "person_id",
        "last_seen",
        "seen_count",
        "is_masked",
        "is_active",
    )
    list_filter = ("is_masked", "is_active")
    search_fields = ("person_id",)
    readonly_fields = ("first_seen", "last_seen")
