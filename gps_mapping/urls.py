from django.urls import path
from . import api

urlpatterns = [
    path('save/', api.save_gcp, name='save_gcp'),
    path('calculate-area/', api.calculate_area_api, name='calculate_area'),
    path('pixel-to-gps/', api.pixel_to_gps_api, name='pixel_to_gps'),
    path('cameras/list/', api.list_cameras, name='list_cameras'),
    path('<str:camera_id>/', api.get_gcp, name='get_gcp'),
]
